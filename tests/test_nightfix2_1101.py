"""2026-11-01 夜间批次3回归（遗留项）。

- D-4 拉平：工具名归一匹配（大小写 / - / _ 变体），GLM/kimi/qwen 共享层
  与 deepseek 同语义；命中变体按**声明名**下发。
- K-4 拉平：kimi 续写轮思考接缝剪除（对齐 GLM seam_reason）——续写复读
  断点前思考尾巴不再整段透传。
- K-5 拉平：kimi/GLM 主轮读流中途异常（RST/截断/超时）降级为断流，
  交给自动续写，而不是把已输出的流变报错（对齐 deepseek）。
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from webai2api.adapters import tool_bridge
from webai2api.adapters.chatglm.client import ChatGlmWebClient
from webai2api.adapters.chatglm.config import GlmConfig
from webai2api.adapters.kimi.client import KimiWebClient
from webai2api.adapters.kimi.config import KimiConfig

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "执行命令",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]


# ── D-4：工具名归一匹配 ──────────────────────────────────────────


def test_resolve_tool_name_variants():
    allowed = {"bash", "read_file"}
    assert tool_bridge.resolve_tool_name("bash", allowed) == "bash"
    assert tool_bridge.resolve_tool_name("Bash", allowed) == "bash"
    assert tool_bridge.resolve_tool_name("BASH", allowed) == "bash"
    assert tool_bridge.resolve_tool_name("read-file", allowed) == "read_file"
    assert tool_bridge.resolve_tool_name("read.file", allowed) == "read_file"
    assert tool_bridge.resolve_tool_name("ReadFile", allowed) == ""
    assert tool_bridge.resolve_tool_name("", allowed) == ""
    assert tool_bridge.resolve_tool_name("bash", set()) == ""


def test_parse_tool_calls_case_variant_emits_declared_name():
    text = '<tool_call>{"name": "Bash", "arguments": {"command": "ls"}}</tool_call>'
    calls = tool_bridge.parse_tool_calls(text, TOOLS, tag="tool_call")
    assert calls, "大小写漂移不该丢调用"
    assert calls[0]["function"]["name"] == "bash", "必须按声明名下发"
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "ls"}


def test_parse_tool_calls_hallucinated_name_still_dropped():
    text = '<tool_call>{"name": "rm_rf", "arguments": {"x": 1}}</tool_call>'
    assert tool_bridge.parse_tool_calls(text, TOOLS, tag="tool_call") == []


def test_dsml_parse_case_variant():
    text = (
        "<|DSML|tool_calls>\n"
        '  <|DSML|invoke name="Bash">\n'
        '    <|DSML|parameter name="command"><![CDATA[ls]]></|DSML|parameter>\n'
        "  </|DSML|invoke>\n"
        "</|DSML|tool_calls>"
    )
    calls = tool_bridge.parse_tool_calls(text, TOOLS, dsml=True)
    assert calls and calls[0]["function"]["name"] == "bash"


# ── K-4：kimi 续写轮思考接缝 ─────────────────────────────────────


def _frame(flag: int, payload: bytes) -> bytes:
    return (flag).to_bytes(1, "big") + len(payload).to_bytes(4, "big") + payload


def _json_frame(obj: dict) -> bytes:
    return _frame(0, json.dumps(obj, ensure_ascii=False).encode())


class _KimiHandler(BaseHTTPRequestHandler):
    rounds: list = []  # [(bytes, has_end, cut_mid: bool)]
    call_no = 0

    def log_message(self, *a):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        idx = min(type(self).call_no, len(type(self).rounds) - 1)
        body, has_end, cut_mid = type(self).rounds[idx]
        type(self).call_no += 1
        self.send_response(200)
        self.send_header("Content-Type", "application/connect+json")
        self.send_header("Content-Length", str(len(body) + 4096))
        self.end_headers()
        self.wfile.write(body)
        if has_end:
            self.wfile.write(_frame(0x02, b"{}"))
        elif cut_mid:
            # 故意不发完 Content-Length 声称的字节就关连接：
            # http.client 会抛 IncompleteRead（读流异常型断流）
            try:
                self.wfile.flush()
            except Exception:
                pass
            self.close_connection = True

    do_GET = do_POST


@pytest.fixture()
def kimi_server():
    _KimiHandler.rounds = []
    _KimiHandler.call_no = 0
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _KimiHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def _head(think: str, text: str):
    return (
        _json_frame({"op": "set", "chat": {"id": "chat-seam"}})
        + _json_frame({
            "op": "set",
            "message": {"id": "msg-u", "role": "user", "status": "done"},
        })
        + _json_frame({
            "op": "set",
            "message": {
                "id": "msg-a",
                "role": "assistant",
                "status": "MESSAGE_STATUS_GENERATING",
            },
        })
        + _json_frame({"op": "set", "mask": "block.think",
                       "block": {"id": "t1", "think": {"content": ""}}})
        + _json_frame({"op": "append", "mask": "block.think.content",
                       "block": {"id": "t1", "think": {"content": think}}})
        + _json_frame({"op": "set", "mask": "block.text",
                       "block": {"id": "b1", "text": {"content": ""}}})
        + _json_frame({"op": "append", "mask": "block.text.content",
                       "block": {"id": "b1", "text": {"content": text}}})
    )


def _client(base: str) -> KimiWebClient:
    cfg = KimiConfig(
        base_url=base,
        auth_token="t",
        egress="direct",
        anti_rate_limit_enabled=False,
        session_store_path="",
    )
    return KimiWebClient(cfg, logging.getLogger("test.kimi.nightfix2"))


def test_kimi_continuation_trims_repeated_thinking(kimi_server):
    """续写轮把断点前思考尾巴逐字复读：剪除后 reasoning 无重复。"""
    round1 = _head("ABCDEFGHIJKLMN", "回答开头，")
    # 续写轮思考以「NM」（基线尾巴）复读者开局，再接新内容
    round2 = (
        _json_frame({"op": "set", "mask": "block.think",
                     "block": {"id": "t2", "think": {"content": ""}}})
        + _json_frame({"op": "append", "mask": "block.think.content",
                       "block": {"id": "t2", "think": {"content": "MN"}}})
        + _json_frame({"op": "append", "mask": "block.think.content",
                       "block": {"id": "t2", "think": {"content": "OPQ"}}})
        + _json_frame({"op": "set", "mask": "block.text",
                       "block": {"id": "b2", "text": {"content": ""}}})
        + _json_frame({"op": "append", "mask": "block.text.content",
                       "block": {"id": "b2", "text": {"content": "答案的剩余部分"}}})
        + _json_frame({"done": {}})
    )
    _KimiHandler.rounds = [(round1, False, False), (round2, True, False)]
    out = _client(kimi_server).chat_completion({
        "model": "kimi-k2.6",
        "messages": [{"role": "user", "content": "hi"}],
    })
    msg = out["choices"][0]["message"]
    assert "回答开头" in (msg.get("content") or "")
    assert "剩余部分" in (msg.get("content") or "")
    reasoning = msg.get("reasoning_content") or ""
    assert "ABCDEFGHIJKLMN" in reasoning
    assert "OPQ" in reasoning
    assert reasoning.count("MN") == 1, f"思考复读未被剪除: {reasoning!r}"


def test_kimi_read_error_degrades_to_continuation(kimi_server):
    """主轮读流异常（IncompleteRead）：降级为断流走自动续写，不再报错。"""
    round1 = _head("思考", "回答开头，")
    round2 = (
        _json_frame({"op": "set", "mask": "block.text",
                     "block": {"id": "b2", "text": {"content": ""}}})
        + _json_frame({"op": "append", "mask": "block.text.content",
                       "block": {"id": "b2", "text": {"content": "续写接上了"}}})
        + _json_frame({"done": {}})
    )
    _KimiHandler.rounds = [(round1, False, True), (round2, True, False)]
    out = _client(kimi_server).chat_completion({
        "model": "kimi-k2.6",
        "messages": [{"role": "user", "content": "hi"}],
    })
    msg = out["choices"][0]["message"]
    content = msg.get("content") or ""
    assert "回答开头" in content
    assert "续写接上了" in content
    assert out["choices"][0]["finish_reason"] == "stop"
    assert _KimiHandler.call_no == 2, "续写请求必须发出"


# ── K-5：GLM 主轮读流异常降级 ────────────────────────────────────


class _GlmHandler(BaseHTTPRequestHandler):
    rounds: list = []  # [(raw_sse_bytes, complete: bool)]
    call_no = 0

    def log_message(self, *a):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        if self.path.endswith("/user/refresh"):
            payload = json.dumps({"code": 0, "result": {"access_token": "tok"}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        idx = min(type(self).call_no, len(type(self).rounds) - 1)
        body, complete = type(self).rounds[idx]
        type(self).call_no += 1
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body) + 4096))
        self.end_headers()
        self.wfile.write(body)
        if complete:
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            # 短发 + 关连接 → IncompleteRead（读流异常型断流）
            try:
                self.wfile.flush()
            except Exception:
                pass
            self.close_connection = True

    do_GET = do_POST


def _sse(obj: dict) -> bytes:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode()


@pytest.fixture()
def glm_server():
    _GlmHandler.rounds = []
    _GlmHandler.call_no = 0
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _GlmHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _glm_client(base: str) -> ChatGlmWebClient:
    cfg = GlmConfig(
        refresh_token="rt",
        base_url=base,
        anti_rate_limit_enabled=False,
        session_store_path="",
        egress="direct",
    )
    return ChatGlmWebClient(cfg, logging.getLogger("test.glm.nightfix2"))


def test_glm_read_error_degrades_to_continuation(glm_server):
    round1 = (
        _sse({"conversation_id": "conv-deg",
              "parts": [{"logic_id": "b1", "status": "init",
                         "content": [{"type": "text", "text": "回答开头，"}]}]})
        + b"data: {incomplete"
    )
    round2 = (
        _sse({"conversation_id": "conv-deg",
              "parts": [{"logic_id": "b2", "status": "init",
                         "content": [{"type": "text", "text": "续写接上了"}]}]})
        + _sse({"status": "finish",
                "parts": [{"logic_id": "b2", "status": "finish",
                           "content": [{"type": "text", "text": "续写接上了"}]}]})
    )
    _GlmHandler.rounds = [(round1, False), (round2, True)]
    out = _glm_client(glm_server).chat_completion({
        "model": "GLM-5.3-Flash",
        "messages": [{"role": "user", "content": "hi"}],
    })
    msg = out["choices"][0]["message"]
    content = msg.get("content") or ""
    assert "回答开头" in content
    assert "续写接上了" in content
    assert out["choices"][0]["finish_reason"] == "stop"
    assert _GlmHandler.call_no == 2
