"""GLM 断流自动续写测试（mock 上游 SSE 多轮行为）。

背景（2026-10-27）：上游偶尔在长思考/长回答中途掐流且不发
status=finish 帧。修复后 _stream_events 检测断流，向同一 conversation
发「继续」，续写轮用全新累积器拼接输出（对上层 prefix-diff 保持单调），
最多 2 轮；救不回则 finish=length。
"""
from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from webai2api.adapters.chatglm.client import ChatGlmWebClient
from webai2api.adapters.chatglm.config import GlmConfig


def _sse(obj: dict) -> bytes:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode()


def _round_sse(conv_id: str | None, text: str, finish: bool) -> bytes:
    """一轮上游 SSE：init 帧（可带 conversation_id）+ 可选 finish 帧。"""
    out = b""
    parts = [{"logic_id": "a", "status": "init",
              "content": [{"type": "text", "text": text}]}]
    if conv_id:
        out += _sse({"conversation_id": conv_id, "parts": parts})
    else:
        out += _sse({"parts": parts})
    if finish:
        out += _sse({"status": "finish",
                     "parts": [{"logic_id": "a", "status": "finish",
                                "content": [{"type": "text", "text": text}]}]})
    return out + b"data: [DONE]\n\n"


class Handler(BaseHTTPRequestHandler):
    # 每次上游调用依次取一个 (conv_id, text, has_finish)；超出后重复最后一组
    rounds: list = []
    call_no: int = 0
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
        body = json.loads(raw or b"{}")
        text_in = body.get("messages", [{}])[0].get(
            "content", [{}])[0].get("text", "")
        type(self).captured.append({
            "text": text_in,
            "conversation_id": body.get("conversation_id", ""),
        })
        idx = min(type(self).call_no, len(type(self).rounds) - 1)
        type(self).call_no += 1
        conv_id, text, has_finish = type(self).rounds[idx]
        payload = _round_sse(conv_id, text, has_finish)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(payload)

    do_GET = do_POST


@pytest.fixture()
def glm_server():
    Handler.rounds = []
    Handler.call_no = 0
    Handler.captured = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _client(base: str) -> ChatGlmWebClient:
    cfg = GlmConfig(refresh_token="rt", base_url=base,
                    anti_rate_limit_enabled=False,
                    session_store_path="", egress="direct")
    return ChatGlmWebClient(cfg, logging.getLogger("glm.cont.test"))


def _payload() -> dict:
    return {"model": "GLM-5.3-Flash",
            "messages": [{"role": "user", "content": "写一段介绍"}]}


def _parse_stream(chunks: list[str]) -> list[dict]:
    out = []
    for sse in chunks:
        sse = sse.strip()
        if sse.startswith("data: ") and sse != "data: [DONE]":
            out.append(json.loads(sse[6:]))
    return out


def test_glm_truncated_auto_continues_to_complete_answer(glm_server):
    Handler.rounds = [("conv-e2e-1", "回答开头，", False),
                      ("conv-e2e-1", "剩余部分", True)]
    c = _client(glm_server)
    chunks = _parse_stream(list(c.stream_chat_completion(_payload())))
    contents = "".join(x["choices"][0]["delta"].get("content", "")
                       for x in chunks if x.get("choices"))
    assert contents == "回答开头，剩余部分"
    finishes = [x["choices"][0].get("finish_reason") for x in chunks
                if x.get("choices") and x["choices"][0].get("finish_reason")]
    assert finishes == ["stop"], f"续写成功应返回 stop，实际 {finishes}"
    assert len(Handler.captured) == 2, "应有一次自动续写请求"
    second = Handler.captured[1]
    assert second["text"] == "继续"
    assert second["conversation_id"] == "conv-e2e-1", "续写必须发到同一会话"
    # 存储的是合并后的完整正文（会话缓存应有一条记录）
    assert len(c._conv_cache._cache) == 1


def test_glm_truncated_no_conv_id_falls_back_to_length(glm_server):
    """上游连 conversation_id 都没发就断流：无法续写，finish=length。"""
    Handler.rounds = [(None, "半截", False)]
    c = _client(glm_server)
    chunks = _parse_stream(list(c.stream_chat_completion(_payload())))
    finishes = [x["choices"][0].get("finish_reason") for x in chunks
                if x.get("choices") and x["choices"][0].get("finish_reason")]
    assert finishes == ["length"]
    assert len(Handler.captured) == 1, "无会话 id 不应发起续写"


def test_glm_truncated_continuation_also_truncates(glm_server):
    """续写轮也断流：最多 2 轮续写（共 3 次上游调用），finish=length。"""
    Handler.rounds = [("conv-e2e-1", "第一段", False)]
    c = _client(glm_server)
    chunks = _parse_stream(list(c.stream_chat_completion(_payload())))
    finishes = [x["choices"][0].get("finish_reason") for x in chunks
                if x.get("choices") and x["choices"][0].get("finish_reason")]
    assert finishes == ["length"]
    assert len(Handler.captured) == 3, "1 次原始 + 2 次续写"
    assert Handler.captured[1]["text"] == "继续"
    assert Handler.captured[2]["text"] == "继续"


def test_glm_nonstream_truncated_auto_continues(glm_server):
    Handler.rounds = [("conv-e2e-1", "回答开头，", False),
                      ("conv-e2e-1", "剩余部分", True)]
    c = _client(glm_server)
    out = c.chat_completion(_payload())
    assert out["choices"][0]["message"]["content"] == "回答开头，剩余部分"
    assert out["choices"][0]["finish_reason"] == "stop"


def test_glm_tool_marker_truncated_recovered_by_continuation(glm_server):
    """工具标记被截断：残块扣在缓冲器里不外漏，续写轮重发完整标记 →
    照常解析出 tool_calls。"""
    Handler.rounds = [
        ("conv-e2e-1", '我来写入\n<<<TOOL_CALL>>>{"name": "write", "arg', False),
        ("conv-e2e-1", '<<<TOOL_CALL>>>{"name": "write", '
                       '"arguments": {"filePath": "a.txt"}}', True),
    ]
    tools = [{"type": "function", "function": {
        "name": "write", "parameters": {"type": "object",
        "properties": {"filePath": {"type": "string"}}}}}]
    c = _client(glm_server)
    chunks = _parse_stream(list(c.stream_chat_completion(
        {**_payload(), "tools": tools})))
    calls = []
    for x in chunks:
        if x.get("choices"):
            tcd = x["choices"][0]["delta"].get("tool_calls")
            if tcd:
                calls.extend(tcd)
    assert calls, "续写后应拿到完整 tool_calls"
    assert calls[0]["function"]["name"] == "write"
    # 半截标记绝不能漏进正文
    contents = "".join(x["choices"][0]["delta"].get("content", "")
                       for x in chunks if x.get("choices"))
    assert "<<<TOOL_CALL>>>" not in contents
    finishes = [x["choices"][0].get("finish_reason") for x in chunks
                if x.get("choices") and x["choices"][0].get("finish_reason")]
    assert finishes == ["tool_calls"]
