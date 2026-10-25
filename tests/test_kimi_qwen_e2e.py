"""端到端冒烟：本地 mock Kimi/Qwen 服务器 → 走完整适配器链路（egress=direct）。

不发任何真实网络请求：mock 服务器监听 127.0.0.1，客户端强制直连。
Kimi 使用 Connect 协议（5 字节帧头 + JSON），Qwen 仍是 SSE。
"""
from __future__ import annotations

import json
import logging
import os
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from webai2api.adapters.kimi.client import KimiWebClient
from webai2api.adapters.kimi.config import KimiConfig
from webai2api.adapters.qwen.client import QwenWebClient
from webai2api.adapters.qwen.config import QwenConfig

KIMI_EVENTS = [
    {"op": "set", "chat": {"id": "chat-777"}},
    {"op": "set", "message": {"id": "msg-u", "role": "user",
                              "status": "MESSAGE_STATUS_COMPLETED"}},
    {"op": "set", "message": {"id": "msg-a", "role": "assistant",
                              "status": "MESSAGE_STATUS_GENERATING"}},
    {"op": "set", "mask": "block.think", "block": {"id": "b0", "think": {"content": ""}}},
    {"op": "append", "mask": "block.think.content",
     "block": {"id": "b0", "think": {"content": "思考中…"}}},
    {"op": "set", "mask": "block.text", "block": {"id": "b1", "text": {"content": ""}}},
    {"op": "append", "mask": "block.text.content",
     "block": {"id": "b1", "text": {"content": "你好，"}}},
    {"op": "append", "mask": "block.text.content",
     "block": {"id": "b1", "text": {"content": "我是 Kimi。"}}},
    {"op": "set", "mask": "message.status",
     "message": {"id": "msg-a", "role": "assistant",
                 "status": "MESSAGE_STATUS_COMPLETED"}},
    {"done": {}},
]
QWEN_SSE = [
    {"chat_id": "q-chat-1", "choices": [{"delta": {"reasoning_content": "先想一下"}}]},
    {"chat_id": "q-chat-1", "choices": [{"delta": {"content": "你好，"}}]},
    {"chat_id": "q-chat-1", "choices": [{"delta": {"content": "我是千问。"}}]},
    {"chat_id": "q-chat-1", "choices": [{"delta": {}, "finish_reason": "stop"}]},
]


def _frame(flag: int, payload: bytes) -> bytes:
    return struct.pack(">BI", flag, len(payload)) + payload


def _json_frame(obj: dict) -> bytes:
    return _frame(0, json.dumps(obj, ensure_ascii=False).encode())


class Handler(BaseHTTPRequestHandler):
    captured: dict = {}
    kimi_requests: list = []

    def log_message(self, *a):
        pass

    def _read_framed(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        if len(raw) < 5:
            return {}
        n = int.from_bytes(raw[1:5], "big")
        try:
            return json.loads(raw[5:5 + n] or b"{}")
        except Exception:
            return {}

    def _serve_kimi(self):
        body = self._read_framed()
        type(self).kimi_requests.append({
            "headers": dict(self.headers),
            "path": self.path,
            "body": body,
        })
        type(self).captured["kimi"] = type(self).kimi_requests[-1]
        internal = ((body.get("options") or {}).get("model") or "")
        self.send_response(200)
        self.send_header("Content-Type", "application/connect+json")
        self.end_headers()
        # 模拟套餐不含 k3：首个模型 invalid_argument，触发降级
        if internal == "k3-agent":
            err = {"error": {"code": "invalid_argument", "message": "Invalid parameter"}}
            self.wfile.write(_frame(0x02, json.dumps(err).encode()))
            return
        for ev in KIMI_EVENTS:
            self.wfile.write(_json_frame(ev))
        self.wfile.write(_frame(0x02, b"{}"))

    def _serve_qwen(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).captured["qwen"] = {
            "headers": dict(self.headers),
            "path": self.path,
            "body": body,
        }
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for ev in QWEN_SSE:
            self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def _serve_relay(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).captured["relay"] = {
            "headers": dict(self.headers),
            "path": self.path,
            "body": body,
        }
        events = [
            {"type": "meta", "chat_id": "q-relay-1", "response_id": "r-relay-1"},
            {"type": "reasoning", "content": "先想一下"},
            {"type": "delta", "content": "你好，"},
            {"type": "delta", "content": "我是千问。"},
            {"type": "done", "chat_id": "q-relay-1", "response_id": "r-relay-1"},
        ]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for ev in events:
            self.wfile.write(f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode())

    def do_GET(self):
        if self.path.startswith("/health"):
            payload = json.dumps({"ok": True, "logged_in": True,
                                  "account_id": "acc-relay"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path.startswith("/apiv2/kimi.gateway.chat.v1.ChatService/"):
            if self.path.endswith("/Chat"):
                self._serve_kimi()
            else:
                self.send_response(200)
                self.send_header("Content-Type", "application/connect+json")
                self.end_headers()
                self.wfile.write(_frame(0x02, b"{}"))
        elif self.path.startswith("/api/v2/chat/completions"):
            self._serve_qwen()
        elif self.path.startswith("/chat"):
            self._serve_relay()
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
        "model": "kimi-k2.6",
        "messages": [{"role": "user", "content": text}],
        "_wapi_key_id": "key-1",
    }


def test_kimi_end_to_end(mock_server):
    Handler.captured = {}
    Handler.kimi_requests = []
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
    assert cap["headers"]["Content-Type"] == "application/connect+json"
    assert cap["body"]["options"]["model"] == "k2d6-chat"
    assert cap["body"]["message"]["blocks"][0]["text"]["content"] == "你好"

    # 会话 id 复用：真实多轮（首条 user 不变）第二轮 body 带 chatId / parentId
    p2 = {"model": "kimi-k2.6", "messages": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好，我是 Kimi。"},
        {"role": "user", "content": "继续"}], "_wapi_key_id": "key-1"}
    c2 = c.chat_completion(p2)
    cap2 = Handler.captured["kimi"]
    assert cap2["body"]["chatId"] == "chat-777"
    assert cap2["body"]["message"]["parentId"] == "msg-a"
    assert c2["choices"][0]["message"]["content"]


def test_kimi_model_fallback(mock_server):
    """k3-agent 不可用时自动降到 k2d6-chat，且不会重复撞同一个模型。"""
    Handler.captured = {}
    Handler.kimi_requests = []
    cfg = KimiConfig(base_url=mock_server, auth_token="t", egress="direct",
                     anti_rate_limit_enabled=False, session_store_path="")
    c = KimiWebClient(cfg, logging.getLogger("t"))
    p0 = _payload(); p0["model"] = "kimi-k3"
    out = c.chat_completion(p0)
    assert "我是 Kimi" in out["choices"][0]["message"]["content"]
    models = [r["body"]["options"]["model"] for r in Handler.kimi_requests]
    assert models == ["k3-agent", "k2d6-chat"]
    # 第二轮直接用已探测成功的模型，不再撞 k3-agent
    c.chat_completion({"model": "kimi-k3",
                       "messages": [{"role": "user", "content": "再来"}],
                       "_wapi_key_id": "key-2"})
    assert Handler.kimi_requests[-1]["body"]["options"]["model"] == "k2d6-chat"


def test_qwen_end_to_end(mock_server):
    Handler.captured = {}
    cfg = QwenConfig(
        base_url=mock_server, auth_token="tok-q", user_agent="UA-2",
        device_id="dev-2", egress="direct", anti_rate_limit_enabled=False,
        session_store_path="",
    )
    c = QwenWebClient(cfg, logging.getLogger("t"))
    p0 = _payload(); p0["model"] = "qwen3.8-max"
    out = c.chat_completion(p0)
    assert "我是千问" in out["choices"][0]["message"]["content"]
    assert out["choices"][0]["message"]["reasoning_content"] == "先想一下"

    cap = Handler.captured["qwen"]
    assert cap["headers"]["Authorization"] == "Bearer tok-q"
    assert cap["body"]["model"] == "qwen3.8-max"
    assert cap["body"]["thinking_enabled"] is False      # 未指定档位 → 自动
    assert cap["body"]["auto_thinking"] is True
    # 显式"思考"
    p = _payload(); p["model"] = "qwen3.7-plus"; p["reasoning_effort"] = "high"
    c.chat_completion(p)
    assert Handler.captured["qwen"]["body"]["thinking_enabled"] is True
    assert Handler.captured["qwen"]["body"]["auto_thinking"] is False
    # 显式"快速"
    p2 = _payload(); p2["model"] = "qwen3.7-plus"; p2["reasoning_effort"] = "low"
    c.chat_completion(p2)
    assert Handler.captured["qwen"]["body"]["thinking_enabled"] is False
    assert Handler.captured["qwen"]["body"]["auto_thinking"] is False
    # 同一 plus 会话多轮延续（带 assistant 历史）才复用 chat_id query 参数；
    # 首轮（无 assistant）必须新建会话——首轮指纹常与旧会话相同，复用会串上下文
    p_multi = _payload(); p_multi["model"] = "qwen3.7-plus"
    p_multi["messages"] = p_multi["messages"] + [
        {"role": "assistant", "content": "你好，我是千问。"},
        {"role": "user", "content": "继续"},
    ]
    c.chat_completion(p_multi)
    assert "chat_id=q-chat-1" in Handler.captured["qwen"]["path"]
    # 首轮不复用：新会话发同样的开场白也拿不到旧 chat_id
    p_fresh = _payload(); p_fresh["model"] = "qwen3.7-plus"
    c.chat_completion(p_fresh)
    assert "chat_id=" not in Handler.captured["qwen"]["path"]
    # flash 无自动档：medium→思考
    p3 = _payload(); p3["model"] = "qwen3.8-omni-flash"; p3["reasoning_effort"] = "medium"
    c.chat_completion(p3)
    assert Handler.captured["qwen"]["body"]["thinking_enabled"] is True


def test_qwen_relay_client(mock_server, tmp_path, monkeypatch):
    """中继模式：适配器把 OpenAI 请求交给本机中继，并复用 chat_id/response_id。"""
    Handler.captured = {}
    port = int(mock_server.rsplit(":", 1)[1])
    # 把中继状态目录指到 tmp_path，并伪造一个"已就绪"的状态（pid 用本进程，保证存活）
    from webai2api import qwen_relay
    monkeypatch.setattr(qwen_relay, "user_data_dir", lambda: tmp_path)
    qwen_relay.write_state("acc-relay", {
        "pid": os.getpid(), "port": port, "provider": "qwen",
        "started_at": time.time(),
    })
    cfg = QwenConfig(
        base_url="https://chat.qwen.ai", auth_token="tok-q", user_agent="UA-2",
        device_id="dev-2", egress="direct", anti_rate_limit_enabled=False,
        session_store_path="", account_id="acc-relay", relay_enabled=True,
    )
    c = QwenWebClient(cfg, logging.getLogger("t"))
    # 真正跑一遍 _ensure_relay：同时覆盖 from ...qwen_relay 的 import 层级
    # 与 RelayManager.ensure（已就绪端口时直接返回，不启动子进程）
    assert c._ensure_relay() == port
    p = _payload(); p["model"] = "qwen3.7-plus"; p["reasoning_effort"] = "high"
    out = c.chat_completion(p)
    assert "我是千问" in out["choices"][0]["message"]["content"]
    assert out["choices"][0]["message"]["reasoning_content"] == "先想一下"
    cap = Handler.captured["relay"]["body"]
    assert cap["model"] == "qwen3.7-plus"
    assert cap["prompt"] == "你好"
    assert cap["chat_id"] == ""
    assert cap["parent_id"] == ""
    assert cap["thinking_enabled"] is True
    assert cap["thinking_mode"] == "Thinking"

    # 第二轮：带上一轮的 chat_id / response_id
    p2 = _payload(); p2["model"] = "qwen3.7-plus"
    p2["messages"] = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好，我是千问。"},
        {"role": "user", "content": "继续"},
    ]
    c.chat_completion(p2)
    cap2 = Handler.captured["relay"]["body"]
    assert cap2["chat_id"] == "q-relay-1"
    assert cap2["parent_id"] == "r-relay-1"


def test_kimi_stream_chunks(mock_server):
    Handler.captured = {}
    Handler.kimi_requests = []
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
    Handler.kimi_requests = []
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
         "https://www.kimi.com/apiv2/kimi.gateway.chat.v1.ChatService/Chat"),
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
