"""Anthropic /v1/messages 端点测试。

三层覆盖：
1. anthropic_to_openai / openai_to_anthropic 纯函数（请求/响应翻译）；
2. AnthropicStreamTranslator 流式事件序列（Claude Code 依赖的完整事件骨架）；
3. 端到端：mock client 挂进真实 app，验证 x-api-key 鉴权、非流式与
   流式的 /v1/messages 全链路（不打真实上游）。
"""
from __future__ import annotations

import json
import logging

import pytest
from fastapi.testclient import TestClient

from webai2api.anthropic_adapter import (
    AnthropicStreamTranslator,
    anthropic_to_openai,
    openai_to_anthropic,
)
from webai2api.app import build_app
from webai2api.providers import ProviderRegistry
from webai2api.store import Store

TOKEN_HDR = "x-admin-token"
SSE_SEP = "\n\n"


def _sse_types(text):
    out = []
    for e in text.split(SSE_SEP):
        if not e.startswith("event: "):
            continue
        ev = e.split("\n", 1)[0][7:]
        data = json.loads(e.split("data: ", 1)[1]) if "data: " in e else {}
        out.append((ev, data))
    return out


# ── 1. 请求翻译 ──────────────────────────────────────────

def test_request_translation_basic():
    body = {
        "model": "GLM-5.3-Flash", "max_tokens": 1024,
        "system": "你是助手",
        "messages": [{"role": "user", "content": "你好"}],
    }
    out = anthropic_to_openai(body)
    assert out["model"] == "GLM-5.3-Flash"
    assert out["max_tokens"] == 1024
    assert out["messages"][0] == {"role": "system", "content": "你是助手"}
    assert out["messages"][1] == {"role": "user", "content": "你好"}


def test_request_translation_system_blocks_and_content_blocks():
    body = {
        "model": "m",
        "system": [{"type": "text", "text": "规"},
                   {"type": "text", "text": "则"}],
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "第一段"},
            {"type": "text", "text": "第二段"}]}],
    }
    out = anthropic_to_openai(body)
    assert out["messages"][0]["content"] == "规则"
    assert out["messages"][1]["content"] == "第一段第二段"


def test_request_translation_tools_and_tool_result():
    body = {
        "model": "m",
        "tools": [{"name": "write", "description": "写文件",
                   "input_schema": {"type": "object", "properties": {
                       "filePath": {"type": "string"}}}}],
        "tool_choice": "any",
        "messages": [
            {"role": "user", "content": "写文件"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "write",
                 "input": {"filePath": "a.txt"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1",
                 "content": "写入成功"},
                {"type": "text", "text": "继续"}]},
        ],
    }
    out = anthropic_to_openai(body)
    assert out["tools"] == [{"type": "function", "function": {
        "name": "write", "description": "写文件",
        "parameters": {"type": "object", "properties": {
            "filePath": {"type": "string"}}}}}]
    assert out["tool_choice"] == "required"
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["user", "assistant", "tool", "user"]
    assert out["messages"][1]["tool_calls"][0]["id"] == "toolu_1"
    assert json.loads(out["messages"][1]["tool_calls"][0]["function"]
                      ["arguments"]) == {"filePath": "a.txt"}
    assert out["messages"][2] == {"role": "tool", "tool_call_id": "toolu_1",
                                  "content": "写入成功"}


def test_request_translation_tool_choice_variants():
    base = {"model": "m", "messages": [{"role": "user", "content": "x"}]}
    assert anthropic_to_openai({**base, "tool_choice": "auto"})["tool_choice"] == "auto"
    assert anthropic_to_openai({**base, "tool_choice": "none"})["tool_choice"] == "none"
    tc = anthropic_to_openai(
        {**base, "tool_choice": {"type": "tool", "name": "write"}})["tool_choice"]
    assert tc == {"type": "function", "function": {"name": "write"}}


# ── 2. 响应翻译 ──────────────────────────────────────────

def test_response_translation_text_and_thinking():
    oai = {
        "model": "GLM-5.3-Flash",
        "choices": [{"finish_reason": "stop", "message": {
            "role": "assistant", "content": "你好",
            "reasoning_content": "想了一下"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    out = openai_to_anthropic(oai, "GLM-5.3-Flash")
    assert out["type"] == "message" and out["role"] == "assistant"
    assert out["stop_reason"] == "end_turn"
    assert out["content"][0]["type"] == "thinking"
    assert out["content"][0]["thinking"] == "想了一下"
    assert out["content"][0]["signature"].startswith("webai2api_")
    assert out["content"][1] == {"type": "text", "text": "你好"}
    assert out["usage"] == {"input_tokens": 10, "output_tokens": 5}


def test_response_translation_tool_use():
    oai = {"choices": [{"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "call_1", "type": "function",
                        "function": {"name": "write",
                                     "arguments": '{"filePath": "a.txt"}'}}]}}]}
    out = openai_to_anthropic(oai, "m")
    assert out["stop_reason"] == "tool_use"
    assert out["content"] == [{"type": "tool_use", "id": "call_1",
                               "name": "write", "input": {"filePath": "a.txt"}}]


# ── 3. 流式事件序列 ──────────────────────────────────────

def _feed_chunks(tr, chunks):
    events = []
    for c in chunks:
        events += tr.feed(c)
    return events


def test_stream_event_sequence_text_only():
    tr = AnthropicStreamTranslator("m")
    events = _feed_chunks(tr, [
        {"choices": [{"delta": {"role": "assistant", "content": ""}}]},
        {"choices": [{"delta": {"reasoning_content": "思考"}}]},
        {"choices": [{"delta": {"content": "你好"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 4}},
    ]) + tr.finish()
    text = "".join(events)
    typed = _sse_types(text)
    assert typed[0][0] == "message_start"
    deltas = [d["delta"]["type"] for _, d in typed
              if d.get("type") == "content_block_delta"]
    # thinking 闭合前必须带 signature_delta（Claude 系客户端依赖）
    assert deltas == ["thinking_delta", "signature_delta", "text_delta"]
    md = [d for _, d in typed if d.get("type") == "message_delta"][0]
    assert md["delta"]["stop_reason"] == "end_turn"
    assert md["usage"] == {"input_tokens": 3, "output_tokens": 4}
    assert typed[-1][0] == "message_stop"
    assert "event: content_block_delta\ndata: " in text


def test_stream_tool_use_input_json_delta():
    tr = AnthropicStreamTranslator("m")
    events = _feed_chunks(tr, [
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_z",
            "type": "function",
            "function": {"name": "write", "arguments": '{"file'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0,
            "function": {"arguments": 'Path": "a.txt"}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]) + tr.finish()
    text = "".join(events)
    assert "tool_use" in text and "input_json_delta" in text
    parts = [d["delta"]["partial_json"]
             for _, d in _sse_types(text)
             if d.get("type") == "content_block_delta"]
    assert "".join(parts) == '{"filePath": "a.txt"}'
    md = [d for _, d in _sse_types(text) if d.get("type") == "message_delta"][0]
    assert md["delta"]["stop_reason"] == "tool_use"


def test_stream_error_event():
    tr = AnthropicStreamTranslator("m")
    events = _feed_chunks(tr, [{"error": {"message": "上游错误",
                                          "type": "upstream_error"}}])
    text = "".join(events)
    assert "event: error" in text
    assert "上游错误" in text
    assert tr.finish() == []


# ── 4. 端到端（mock client，不打真实上游）────────────────

class _FakeClient:
    def __init__(self, response=None, stream_chunks=None):
        self._resp = response
        self._chunks = stream_chunks or []

    def chat_completion(self, payload):
        return self._resp

    def stream_chat_completion(self, payload):
        for c in self._chunks:
            yield c


def _make_client(tmp_path, fake):
    store = Store(tmp_path / "data" / "webai2api.db")
    logger = logging.getLogger("webai2api.test.anthropic")
    registry = ProviderRegistry(store, logger)
    app = build_app(store, registry, logger)
    store.add_account("chatglm", "假账号", {"auth_token": "tok"})
    registry.refresh()

    class _AC:
        account = {"id": "acc-x", "name": "假账号", "provider": "chatglm",
                   "enabled": 1}
        client = fake

        def mark_ok(self):
            pass

        def cooldown_left(self):
            return 0

    registry.pick = lambda *a, **k: _AC()          # type: ignore[assignment]
    registry.pick_by_id = lambda *a, **k: _AC()    # type: ignore[assignment]
    return TestClient(app, client=("127.0.0.1", 50000))


def _make_key(client) -> str:
    r = client.post("/admin/api/login", json={"password": "admin"})
    tok = r.json()["token"]
    r = client.post("/admin/api/keys", json={"name": "测试Key"},
                    headers={TOKEN_HDR: tok})
    assert r.json().get("ok") is True, r.text
    return r.json()["key"]["key"]


@pytest.fixture()
def e2e_nonstream(tmp_path):
    fake = _FakeClient(response={
        "model": "GLM-5.3-Flash",
        "choices": [{"finish_reason": "stop", "message": {
            "role": "assistant", "content": "你好呀"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    })
    client = _make_client(tmp_path, fake)
    return client, _make_key(client)


def test_e2e_messages_nonstream(e2e_nonstream):
    client, key = e2e_nonstream
    r = client.post("/v1/messages",
                    headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                    json={"model": "GLM-5.3-Flash", "max_tokens": 100,
                          "messages": [{"role": "user", "content": "你好"}]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["type"] == "message" and body["role"] == "assistant"
    assert body["content"] == [{"type": "text", "text": "你好呀"}]
    assert body["stop_reason"] == "end_turn"


def test_e2e_messages_bad_key_anthropic_error(e2e_nonstream):
    client, _ = e2e_nonstream
    r = client.post("/v1/messages", headers={"x-api-key": "sk-wrong"},
                    json={"model": "m", "max_tokens": 1,
                          "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 401
    assert r.json()["type"] == "error"
    assert r.json()["error"]["type"] == "authentication_error"


def test_e2e_messages_stream_event_skeleton(tmp_path):
    chunks = [
        "data: " + json.dumps({"choices": [{"delta": {
            "role": "assistant", "content": ""}}]}) + "\n\n",
        "data: " + json.dumps({"choices": [{"delta": {"content": "你"}}]}) + "\n\n",
        "data: " + json.dumps({"choices": [{"delta": {"content": "好"}}]}) + "\n\n",
        "data: " + json.dumps({"choices": [{"delta": {},
                                            "finish_reason": "stop"}]}) + "\n\n",
        "data: [DONE]\n\n",
    ]
    client = _make_client(tmp_path, _FakeClient(stream_chunks=chunks))
    key = _make_key(client)
    with client.stream("POST", "/v1/messages",
                       headers={"x-api-key": key},
                       json={"model": "m", "max_tokens": 50, "stream": True,
                             "messages": [{"role": "user", "content": "你好"}]}) as r:
        assert r.status_code == 200
        text = "".join(r.iter_text())
    types = [t for t, _ in _sse_types(text)]
    assert types[0] == "message_start"
    assert "content_block_delta" in types
    assert types[-2] == "message_delta" and types[-1] == "message_stop"


def test_e2e_messages_stream_tool_use_roundtrip(tmp_path):
    """工具调用全链路：模型 tool_use → Anthropic 形态的 tool_result 回传。"""
    chunks = [
        "data: " + json.dumps({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_z", "type": "function",
             "function": {"name": "write",
                          "arguments": '{"filePath": "a.txt", "content": "hi"}'}}]}}]}) + "\n\n",
        "data: " + json.dumps({"choices": [{"delta": {},
                                            "finish_reason": "tool_calls"}]}) + "\n\n",
        "data: [DONE]\n\n",
    ]
    client = _make_client(tmp_path, _FakeClient(stream_chunks=chunks, response={"model": "m",
    "choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": "文件写好了"}}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2,
              "total_tokens": 7}}))
    key = _make_key(client)
    with client.stream("POST", "/v1/messages",
                       headers={"x-api-key": key},
                       json={"model": "m", "max_tokens": 50, "stream": True,
                             "tools": [{"name": "write", "description": "w",
                                        "input_schema": {"type": "object"}}],
                             "messages": [{"role": "user", "content": "写文件"}]}) as r:
        text = "".join(r.iter_text())
    data_blocks = [d for _, d in _sse_types(text)]
    starts = [d for d in data_blocks if d["type"] == "content_block_start"]
    assert starts and starts[0]["content_block"]["type"] == "tool_use"
    assert starts[0]["content_block"]["name"] == "write"
    md = [d for d in data_blocks if d["type"] == "message_delta"][0]
    assert md["delta"]["stop_reason"] == "tool_use"

    # 回填 tool_result 再问一轮：请求历史含 assistant.tool_use
    body = {"model": "m", "max_tokens": 50, "stream": False,
            "messages": [
                {"role": "user", "content": "写文件"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "call_z", "name": "write",
                     "input": {"filePath": "a.txt", "content": "hi"}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "call_z",
                     "content": "写入成功"}]},
            ]}
    r2 = client.post("/v1/messages", headers={"x-api-key": key}, json=body)
    assert r2.status_code == 200, r2.text
    assert r2.json()["type"] == "message"


# ── 5. 面板 Anthropic 接入片段（backlog 第 1 项）─────────

def test_panels_have_anthropic_snippet_section():
    """两面板都要有 Anthropic 客户端接入区：片段容器、构建函数、
    与 Key 选择联动（rebuildSnippet 里同步填充）。"""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "webai2api" / "web"
    for name in ("index.html", "lite_panel.html"):
        html = (root / name).read_text(encoding="utf-8")
        assert 'id="anthSnippet"' in html, f"{name} 缺片段容器"
        assert "buildAnthSnippet" in html, f"{name} 缺构建函数"
        assert "ANTHROPIC_BASE_URL" in html, f"{name} 缺 Claude Code 环境变量"
        assert "anthropic-messages" in html, f"{name} 缺 ZCode 协议指引"
        # 随所选 Key 联动
        assert "buildAnthSnippet(k?k.key:'')" in html, \
            f"{name} 片段未跟随 Key 选择刷新"
