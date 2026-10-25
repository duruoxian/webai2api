"""GLM 会话增量协议 e2e（mock 上游 SSE）。

背景（v0.1.33 修复）：智谱适配此前把**全部历史**拼平塞进每轮请求，
而会话缓存又在复用同一个上游 conversation——第二轮起上下文被一遍
遍重复（长会话越拖越慢、模型越来越乱）。修复后：命中缓存只发
「最后一条 assistant 之后」的增量（tool_bridge.increment_text），
未命中（首轮/缓存失效/轮换）才发全量拼平。
"""
from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from webai2api.adapters.chatglm.client import ChatGlmWebClient, _GlmParts
from webai2api.adapters.chatglm.config import GlmConfig

TOOLS = [{"type": "function", "function": {"name": "write", "description": "写文件",
          "parameters": {"type": "object", "properties": {
              "filePath": {"type": "string"}, "content": {"type": "string"}}}}}]


def _sse_parts(text: str) -> bytes:
    """构造一帧 GLM 风格 SSE：单 part（增量 delta）+ finish 全量帧。"""
    events = [
        {"conversation_id": "conv-e2e-1",
         "parts": [{"logic_id": "a", "status": "init",
                    "content": [{"type": "text", "text": text}]}]},
        {"status": "finish",
         "parts": [{"logic_id": "a", "status": "finish",
                    "content": [{"type": "text", "text": text}]}]},
    ]
    out = b"".join(f"data: {json.dumps(e, ensure_ascii=False)}\n\n".encode()
                   for e in events)
    return out + b"data: [DONE]\n\n"


class Handler(BaseHTTPRequestHandler):
    captured: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        if self.path.endswith("/user/refresh"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(
                {"code": 0, "result": {"access_token": "tok-e2e"}}).encode())
            return
        body = json.loads(raw or b"{}")
        type(self).captured.append({"path": self.path, "body": body})
        text = body.get("messages", [{}])[0].get("content", [{}])[0].get("text", "")
        # 携带工具协议的首轮请求 → 回工具调用标记（真实流程形态，
        # 使 store 侧指纹 = 规范化 tool_calls，回填轮才能命中缓存）
        if "用 write" in text:
            reply = ('<<<TOOL_CALL>>>{"name":"write",'
                     '"arguments":{"filePath":"t.txt","content":"hi"}}')
        else:
            reply = f"收到：{text[:8]}"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(_sse_parts(reply))

    do_GET = do_POST


@pytest.fixture()
def glm_server():
    Handler.captured = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _client(base: str) -> ChatGlmWebClient:
    cfg = GlmConfig(refresh_token="rt", base_url=base,
                    anti_rate_limit_enabled=False,
                    session_store_path="",   # 测试内不落盘
                    egress="direct")         # 绕开系统代理（Clash 会劫持 localhost）
    return ChatGlmWebClient(cfg, logging.getLogger("glm.e2e"))


def _consume(c: ChatGlmWebClient, payload: dict) -> dict:
    final = {}
    for ev in c._stream_events(payload):
        if ev.get("__done__"):
            final = ev
    return final


def test_glm_multiturn_sends_increment_on_cache_hit(glm_server):
    c = _client(glm_server)

    # 第一轮（首轮）：全量拼平，不带 conversation_id
    p1 = {"model": "GLM-5.3-Flash", "messages": [
        {"role": "user", "content": "第一问：介绍你自己"}]}
    f1 = _consume(c, p1)
    assert f1["conv_id"] == "conv-e2e-1"
    b1 = Handler.captured[-1]["body"]
    assert b1["conversation_id"] == ""
    assert "第一问" in json.dumps(b1, ensure_ascii=False)

    # 第二轮（真实延续）：命中同一会话 → 只发增量（新 user 文本），
    # 不再携带第一轮内容；conversation_id 复用
    p2 = {"model": "GLM-5.3-Flash", "messages": [
        {"role": "user", "content": "第一问：介绍你自己"},
        {"role": "assistant", "content": f1["text"]},   # 与首轮真实回复一致（指纹锚点）
        {"role": "user", "content": "第二问：继续"}]}
    f2 = _consume(c, p2)
    assert f2["conv_id"] == "conv-e2e-1"
    b2 = Handler.captured[-1]["body"]
    assert b2["conversation_id"] == "conv-e2e-1"
    msgs_text = json.dumps(b2, ensure_ascii=False)
    assert "第二问" in msgs_text
    assert "第一问" not in msgs_text, "命中缓存仍重发了全量历史（增量失效）"


def test_glm_tool_chain_increment_carries_tool_result(glm_server):
    c = _client(glm_server)
    q = "任务xyz：用 write 写文件"
    p1 = {"model": "GLM-5.3-Flash", "tools": TOOLS,
          "messages": [{"role": "user", "content": q}]}
    f1 = _consume(c, p1)
    assert f1["conv_id"] == "conv-e2e-1"

    # 工具链回填：assistant(tool_calls) + tool 结果 + 新 user。
    # 首轮回复是工具标记 → store 指纹为规范化 tool_calls，与回填轮
    # 客户端历史里的 tool_calls assistant 指纹一致 → 命中同一会话。
    assert "<<<TOOL_CALL>>>" in f1["text"]
    p2 = {"model": "GLM-5.3-Flash", "tools": TOOLS, "messages": [
        {"role": "user", "content": q},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_g", "type": "function",
             "function": {"name": "write",
                          "arguments": '{"filePath":"t.txt","content":"hi"}'}}]},
        {"role": "tool", "tool_call_id": "call_g", "content": "写入成功"},
        {"role": "user", "content": "检查一下结果"},
    ]}
    _consume(c, p2)
    b2 = Handler.captured[-1]["body"]
    msgs_text = json.dumps(b2, ensure_ascii=False)
    assert b2["conversation_id"] == "conv-e2e-1"
    # 增量必须带 tool_result 块与新 user 文本，且不带首轮原文
    # （msgs_text 是 JSON 转义文本，断言避开引号字面量）
    assert "<tool_result" in msgs_text and "write" in msgs_text
    assert "写入成功" in msgs_text and "检查一下结果" in msgs_text
    assert q not in msgs_text, "工具链回填不应重发首轮全文"


def test_glm_cache_miss_sends_full_flatten(glm_server):
    """缓存未命中（换 key / 首轮）：发全量拼平历史，新建上游会话。"""
    c = _client(glm_server)
    p = {"model": "GLM-5.3-Flash", "messages": [
        {"role": "user", "content": "独立问题A"},
        {"role": "assistant", "content": "答A"},
        {"role": "user", "content": "追问B"}]}
    _consume(c, p)
    b = Handler.captured[-1]["body"]
    msgs_text = json.dumps(b, ensure_ascii=False)
    assert b["conversation_id"] == ""      # 未命中 → 新会话
    assert "独立问题A" in msgs_text and "追问B" in msgs_text  # 全量历史都在
