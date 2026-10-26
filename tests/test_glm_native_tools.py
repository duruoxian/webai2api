"""GLM 服务端原生 tool_calls 测试。

上游在服务端执行工具时会直接下发 ``content`` 项 ``type=tool_calls``：
    {"type": "tool_calls", "tool_calls": {"id", "name", "arguments"}}
这是上游已确定的调用意图，优先于从正文标记里解析；仅在客户端声明了
工具且名字在清单内时下发（否则属于上游内置能力，混进 tool_calls 会
让客户端收到无法执行的调用）。
"""
from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from webai2api.adapters.chatglm.client import ChatGlmWebClient, _GlmParts
from webai2api.adapters.chatglm.config import GlmConfig

TOOLS = [{"type": "function", "function": {
    "name": "write", "description": "写文件",
    "parameters": {"type": "object", "properties": {
        "filePath": {"type": "string"}, "content": {"type": "string"}}}}}]


def _part(item: dict, status: str = "finish") -> dict:
    return {"logic_id": "t", "status": status, "content": [item]}


def _native(cid: str, name: str = "write", arguments=None) -> dict:
    return {"type": "tool_calls", "tool_calls": {
        "id": cid, "name": name,
        "arguments": arguments if arguments is not None else {
            "filePath": "a.txt", "content": "hi"}}}


def test_collects_native_call():
    acc = _GlmParts(allowed_tools={"write"})
    acc.consume([_part(_native("call_1"))])
    assert len(acc.tool_calls) == 1
    c = acc.tool_calls[0]
    assert c["id"] == "call_1" and c["type"] == "function"
    assert c["function"]["name"] == "write"
    assert json.loads(c["function"]["arguments"])["filePath"] == "a.txt"
    assert acc.text == ""


def test_dedupes_same_id_on_finish_frame():
    acc = _GlmParts(allowed_tools={"write"})
    acc.consume([_part(_native("call_1"), status="init")])
    acc.consume([_part(_native("call_1"), status="finish")])
    assert len(acc.tool_calls) == 1


def test_ignores_tool_not_declared():
    acc = _GlmParts(allowed_tools={"write"})
    acc.consume([_part(_native("call_1", name="rm_rf"))])
    assert acc.tool_calls == []


def test_ignores_when_no_tools_declared():
    acc = _GlmParts(allowed_tools=None)
    acc.consume([_part(_native("call_1"))])
    assert acc.tool_calls == []


def test_accepts_list_payload_and_string_arguments():
    acc = _GlmParts(allowed_tools={"write"})
    item = {"type": "tool_calls", "tool_calls": [
        {"id": "call_1", "name": "write",
         "arguments": '{"filePath":"a","content":"b"}'}]}
    acc.consume([_part(item)])
    assert json.loads(acc.tool_calls[0]["function"]["arguments"])["content"] == "b"


def test_empty_tool_calls_ignored():
    acc = _GlmParts(allowed_tools={"write"})
    acc.consume([_part({"type": "tool_calls", "tool_calls": {}})])
    assert acc.tool_calls == [] and acc.text == ""


def test_text_and_native_call_coexist():
    acc = _GlmParts(allowed_tools={"write"})
    acc.consume([{"logic_id": "a", "status": "init",
                  "content": [{"type": "text", "text": "我来写"}]}])
    acc.consume([_part(_native("call_1"))])
    assert acc.text == "我来写"
    assert len(acc.tool_calls) == 1


def _sse_native(text: str) -> bytes:
    events = [
        {"conversation_id": "conv-native-1",
         "parts": [{"logic_id": "a", "status": "init",
                    "content": [{"type": "text", "text": text}]}]},
        {"status": "finish",
         "parts": [{"logic_id": "t", "status": "finish",
                    "content": [_native("call_native_1")]}]},
    ]
    return b"".join(
        f"data: {json.dumps(e, ensure_ascii=False)}\n\n".encode() for e in events
    ) + b"data: [DONE]\n\n"


class _Handler(BaseHTTPRequestHandler):
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
                {"code": 0, "result": {"access_token": "tok"}}).encode())
            return
        type(self).captured.append(json.loads(raw or b"{}"))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(_sse_native("好的"))

    do_GET = do_POST


@pytest.fixture()
def glm_native_server():
    _Handler.captured = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _client(base: str) -> ChatGlmWebClient:
    cfg = GlmConfig(refresh_token="rt", base_url=base,
                    anti_rate_limit_enabled=False,
                    session_store_path="", egress="direct")
    return ChatGlmWebClient(cfg, logging.getLogger("glm.native"))


def test_chat_completion_prefers_native_tool_calls(glm_native_server):
    c = _client(glm_native_server)
    out = c.chat_completion({
        "model": "GLM-5.3-Flash", "tools": TOOLS,
        "messages": [{"role": "user", "content": "写个文件"}],
    })
    choice = out["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    calls = choice["message"]["tool_calls"]
    assert calls[0]["id"] == "call_native_1"
    assert calls[0]["function"]["name"] == "write"
    assert choice["message"]["content"] is None


def test_stream_completion_emits_native_tool_calls(glm_native_server):
    c = _client(glm_native_server)
    chunks = list(c.stream_chat_completion({
        "model": "GLM-5.3-Flash", "tools": TOOLS,
        "messages": [{"role": "user", "content": "写个文件"}],
    }))
    payloads = [json.loads(x[6:]) for x in chunks
                if x.startswith("data: {")]
    calls = [p for p in payloads
             if p["choices"][0]["delta"].get("tool_calls")]
    assert calls, "流式响应没有下发原生 tool_calls"
    assert calls[0]["choices"][0]["delta"]["tool_calls"][0]["id"] == "call_native_1"
    assert any(p["choices"][0].get("finish_reason") == "tool_calls"
               for p in payloads if p.get("choices"))
