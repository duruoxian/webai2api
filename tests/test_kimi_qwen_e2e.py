"""端到端冒烟：本地 mock Kimi/Qwen 服务器 → 走完整适配器链路（egress=direct）。

不发任何真实网络请求：mock 服务器监听 127.0.0.1，客户端强制直连。
"""
from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from webai2api.adapters.kimi.client import KimiWebClient
from webai2api.adapters.kimi.config import KimiConfig
from webai2api.adapters.qwen.client import QwenWebClient
from webai2api.adapters.qwen.config import QwenConfig

KIMI_SSE = [
    {"chat_id": "chat-777", "type": "think", "content": "思考中…"},
    {"chat_id": "chat-777", "type": "content", "content": "你好，"},
    {"chat_id": "chat-777", "type": "content", "content": "我是 Kimi。"},
    {"chat_id": "chat-777", "type": "finish"},
]
QWEN_SSE = [
    {"chat_id": "q-chat-1", "choices": [{"delta": {"reasoning_content": "先想一下"}}]},
    {"chat_id": "q-chat-1", "choices": [{"delta": {"content": "你好，"}}]},
    {"chat_id": "q-chat-1", "choices": [{"delta": {"content": "我是千问。"}}]},
    {"chat_id": "q-chat-1", "choices": [{"delta": {}, "finish_reason": "stop"}]},
]


class Handler(BaseHTTPRequestHandler):
    captured: dict = {}
    sse_for = {"kimi": KIMI_SSE, "qwen": QWEN_SSE}

    def log_message(self, *a):
        pass

    def _serve_sse(self, key: str):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).captured[key] = {
            "headers": dict(self.headers),
            "path": self.path,
            "body": body,
        }
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for ev in self.sse_for[key]:
            self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def do_POST(self):
        if self.path.startswith("/api/chat"):
            self._serve_sse("kimi")
        elif self.path.startswith("/api/v2/chat/completions"):
            self._serve_sse("qwen")
        else:
            self.send_error(404)


@pytest.fixture()
def mock_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def _payload(text="你好"):
    return {
        "model": "kimi-k3",
        "messages": [{"role": "user", "content": text}],
        "_wapi_key_id": "key-1",
    }


def test_kimi_end_to_end(mock_server):
    Handler.captured = {}
    cfg = KimiConfig(
        base_url=mock_server, auth_token="tok-k", user_agent="UA-1",
        device_id="dev-1", egress="direct", anti_rate_limit_enabled=False,
        session_store_path="",
    )
    c = KimiWebClient(cfg, logging.getLogger("t"))
    out = c.chat_completion(_payload())
    assert "我是 Kimi" in out["choices"][0]["message"]["content"]
    assert out["choices"][0]["message"]["reasoning_content"] == "思考中…"

    cap = Handler.captured["kimi"]
    assert cap["headers"]["Authorization"] == "Bearer tok-k"
    assert cap["headers"]["User-Agent"] == "UA-1"
    assert cap["body"]["name"] == "kimi-k3"
    # 会话 id 复用：真实多轮（首条 user 不变）第二轮 body 带 chat_id
    p2 = {"model": "kimi-k3", "messages": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好，我是 Kimi。"},
        {"role": "user", "content": "继续"}], "_wapi_key_id": "key-1"}
    c2 = c.chat_completion(p2)
    assert Handler.captured["kimi"]["body"]["chat_id"] == "chat-777"
    assert c2["choices"][0]["message"]["content"]


def test_qwen_end_to_end(mock_server):
    Handler.captured = {}
    cfg = QwenConfig(
        base_url=mock_server, auth_token="tok-q", user_agent="UA-2",
        device_id="dev-2", egress="direct", anti_rate_limit_enabled=False,
        session_store_path="",
    )
    c = QwenWebClient(cfg, logging.getLogger("t"))
    out = c.chat_completion(_payload())
    assert "我是千问" in out["choices"][0]["message"]["content"]
    assert out["choices"][0]["message"]["reasoning_content"] == "先想一下"

    cap = Handler.captured["qwen"]
    assert cap["headers"]["Authorization"] == "Bearer tok-q"
    assert cap["body"]["model"] == "kimi-k3"
    assert cap["body"]["thinking_enabled"] is False      # 非 plus 模型恒不思考
    # plus 模型 effort → 思考研究
    p = _payload(); p["model"] = "qwen3.7-plus"; p["reasoning_effort"] = "high"
    c.chat_completion(p)
    assert Handler.captured["qwen"]["body"]["thinking_enabled"] is True
    # plus 但 low → 快速（不思考）
    p2 = _payload(); p2["model"] = "qwen3.7-plus"; p2["reasoning_effort"] = "low"
    c.chat_completion(p2)
    assert Handler.captured["qwen"]["body"]["thinking_enabled"] is False
    # 第二轮带 chat_id query 参数
    assert "chat_id=q-chat-1" in Handler.captured["qwen"]["path"]


def test_kimi_stream_chunks(mock_server):
    Handler.captured = {}
    cfg = KimiConfig(base_url=mock_server, auth_token="t", egress="direct",
                     anti_rate_limit_enabled=False, session_store_path="")
    c = KimiWebClient(cfg, logging.getLogger("t"))
    chunks = list(c.stream_chat_completion(_payload()))
    joined = "".join(chunks)
    assert "data: [DONE]" in joined
    assert "我是 Kimi" in joined
    assert "reasoning_content" in joined


def test_egress_direct_forces_no_proxy(mock_server):
    """direct 模式即使系统有代理变量也应直连（mock 是本机，能通即验证路径）。"""
    import os
    Handler.captured = {}
    old = os.environ.get("HTTP_PROXY")
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:1"  # 故意指向死代理
    try:
        cfg = KimiConfig(base_url=mock_server, auth_token="t", egress="direct",
                         anti_rate_limit_enabled=False, session_store_path="")
        c = KimiWebClient(cfg, logging.getLogger("t"))
        out = c.chat_completion(_payload())
        assert "我是 Kimi" in out["choices"][0]["message"]["content"]
    finally:
        if old is None:
            os.environ.pop("HTTP_PROXY", None)
        else:
            os.environ["HTTP_PROXY"] = old


# ── egress 相对 import 层级回归（防止从 deepseek/services/ 复制成 4 点）──
def test_adapter_egress_import_paths_construct_conn():
    """_conn 里 `from ...egress`（3 点）必须能解析；构造连接即触发，不发网络。"""
    import logging as _lg

    from webai2api.adapters.chatglm.client import ChatGlmWebClient
    from webai2api.adapters.chatglm.config import GlmConfig

    for factory, url in (
        (lambda: ChatGlmWebClient(GlmConfig(refresh_token="t", egress="direct"),
                                  _lg.getLogger("g")),
         "https://chatglm.cn/x"),
        (lambda: KimiWebClient(KimiConfig(auth_token="t", egress="direct"),
                               _lg.getLogger("k")),
         "https://www.kimi.com/api/chat"),
        (lambda: QwenWebClient(QwenConfig(auth_token="t", egress="direct"),
                               _lg.getLogger("q")),
         "https://chat.qwen.ai/api/v2/chat/completions"),
    ):
        client = factory()
        conn = client._conn(url, 10)   # 只建对象不 connect，能过即 import 层级正确
        assert conn is not None
        try:
            conn.close()
        except Exception:
            pass
