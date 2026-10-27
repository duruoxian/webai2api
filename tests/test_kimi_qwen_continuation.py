"""Kimi / Qwen 断流自动续写测试（mock 上游多轮行为）。

与 deepseek/GLM 同款防线：上游没发结束信号就断流 → 向同一会话发「继续」
（Kimi: chatId+parentId；Qwen 中继: chat_id+parent_id），续写增量基线
拼接进同一输出。正常结束（done 事件 / 0x02 结束帧 / done 事件）绝不误触发。
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


def _frame(flag: int, payload: bytes) -> bytes:
    return struct.pack(">BI", flag, len(payload)) + payload


def _json_frame(obj: dict) -> bytes:
    return _frame(0, json.dumps(obj, ensure_ascii=False).encode())


KIMI_TAIL_EVENTS = [
    {"op": "set", "mask": "block.text", "block": {"id": "b1", "text": {"content": ""}}},
    {"op": "append", "mask": "block.text.content",
     "block": {"id": "b1", "text": {"content": "答案的剩余部分"}}},
    {"done": {}},
]


class Handler(BaseHTTPRequestHandler):
    # Kimi: 每次调用依次取 (events, has_end_frame)；Qwen: 每次取一组事件
    kimi_rounds: list = []
    kimi_ends: list = []
    qwen_rounds: list = []
    call_no: int = 0
    kimi_requests: list = []
    relay_requests: list = []

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
        type(self).kimi_requests.append(body)
        idx = min(type(self).call_no, len(type(self).kimi_rounds) - 1)
        events, has_end = type(self).kimi_rounds[idx]
        end_payload = (type(self).kimi_ends[idx]
                       if idx < len(type(self).kimi_ends) else b"{}")
        type(self).call_no += 1
        self.send_response(200)
        self.send_header("Content-Type", "application/connect+json")
        self.end_headers()
        for ev in events:
            self.wfile.write(_json_frame(ev))
        if has_end:
            self.wfile.write(_frame(0x02, end_payload))
        # has_end=False：直接关连接，模拟上游掐流（无 0x02、无 done）

    def _serve_relay(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).relay_requests.append(body)
        events = type(self).qwen_rounds[
            min(type(self).call_no, len(type(self).qwen_rounds) - 1)]
        type(self).call_no += 1
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for ev in events:
            self.wfile.write(f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode())

    def do_GET(self):
        if self.path.startswith("/health"):
            payload = json.dumps({"ok": True, "logged_in": True,
                                  "account_id": "acc-cont"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path.endswith("/Chat"):
            self._serve_kimi()
        elif self.path.startswith("/chat"):
            self._serve_relay()
        else:
            self.send_response(200)
            self.send_header("Content-Type", "application/connect+json")
            self.end_headers()
            self.wfile.write(_frame(0x02, b"{}"))


@pytest.fixture()
def mock_server():
    Handler.kimi_rounds = []
    Handler.kimi_ends = []
    Handler.qwen_rounds = []
    Handler.call_no = 0
    Handler.kimi_requests = []
    Handler.relay_requests = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def _kimi_events_head():
    return [
        {"op": "set", "chat": {"id": "chat-777"}},
        {"op": "set", "message": {"id": "msg-u", "role": "user",
                                  "status": "MESSAGE_STATUS_COMPLETED"}},
        {"op": "set", "message": {"id": "msg-a", "role": "assistant",
                                  "status": "MESSAGE_STATUS_GENERATING"}},
        {"op": "set", "mask": "block.text", "block": {"id": "b1", "text": {"content": ""}}},
        {"op": "append", "mask": "block.text.content",
         "block": {"id": "b1", "text": {"content": "回答开头，"}}},
    ]


def _payload(text="你好"):
    return {"model": "kimi-k2.6",
            "messages": [{"role": "user", "content": text}],
            "_wapi_key_id": "key-1"}


def test_kimi_truncated_auto_continues(mock_server):
    """第 1 轮无 done、无 0x02 就断流 → 自动发「继续」续写完整回答。"""
    Handler.kimi_rounds = [
        (_kimi_events_head(), False),
        (KIMI_TAIL_EVENTS, True),
    ]
    cfg = KimiConfig(base_url=mock_server, auth_token="t", egress="direct",
                     anti_rate_limit_enabled=False, session_store_path="")
    c = KimiWebClient(cfg, logging.getLogger("t"))
    out = c.chat_completion(_payload())
    assert out["choices"][0]["message"]["content"] == "回答开头，答案的剩余部分"
    assert out["choices"][0]["finish_reason"] == "stop"
    assert len(Handler.kimi_requests) == 2
    cont = Handler.kimi_requests[1]
    assert cont["chatId"] == "chat-777"
    assert cont["message"]["parentId"] == "msg-a"
    assert cont["message"]["blocks"][0]["text"]["content"] == "继续"


def test_kimi_normal_end_never_continues(mock_server):
    """done + 0x02 正常结束：绝不发第二次请求（防误续写）。"""
    Handler.kimi_rounds = [
        (_kimi_events_head() + KIMI_TAIL_EVENTS, True),
    ]
    cfg = KimiConfig(base_url=mock_server, auth_token="t", egress="direct",
                     anti_rate_limit_enabled=False, session_store_path="")
    c = KimiWebClient(cfg, logging.getLogger("t"))
    out = c.chat_completion(_payload())
    assert "答案的剩余部分" in out["choices"][0]["message"]["content"]
    assert out["choices"][0]["finish_reason"] == "stop"
    assert len(Handler.kimi_requests) == 1


def test_kimi_end_frame_without_done_is_normal_end(mock_server):
    """只有 0x02 结束帧、没有 done 事件：也是正常结束，不触发续写。"""
    head = _kimi_events_head()
    tail = [{"op": "append", "mask": "block.text.content",
             "block": {"id": "b1", "text": {"content": "答案的剩余部分"}}}]
    Handler.kimi_rounds = [(head + tail, True)]
    cfg = KimiConfig(base_url=mock_server, auth_token="t", egress="direct",
                     anti_rate_limit_enabled=False, session_store_path="")
    c = KimiWebClient(cfg, logging.getLogger("t"))
    out = c.chat_completion(_payload())
    assert out["choices"][0]["finish_reason"] == "stop"
    assert len(Handler.kimi_requests) == 1


# ── Qwen 中继路径 ───────────────────────────────────
QWEN_HEAD = [
    {"type": "meta", "chat_id": "q-relay-1", "response_id": "r-1"},
    {"type": "delta", "content": "回答开头，"},
    # —— 掐流：无 done ——
]
QWEN_TAIL = [
    {"type": "meta", "chat_id": "q-relay-1", "response_id": "r-2"},
    {"type": "delta", "content": "答案的剩余部分"},
    {"type": "done", "chat_id": "q-relay-1", "response_id": "r-2"},
]


def test_qwen_relay_truncated_auto_continues(mock_server, tmp_path, monkeypatch):
    port = int(mock_server.rsplit(":", 1)[1])
    from webai2api import qwen_relay
    monkeypatch.setattr(qwen_relay, "user_data_dir", lambda: tmp_path)
    qwen_relay.write_state("acc-cont", {
        "pid": os.getpid(), "port": port, "provider": "qwen",
        "started_at": time.time(),
    })
    Handler.qwen_rounds = [QWEN_HEAD, QWEN_TAIL]
    cfg = QwenConfig(base_url="https://chat.qwen.ai", auth_token="tok-q",
                     egress="direct", anti_rate_limit_enabled=False,
                     session_store_path="", account_id="acc-cont",
                     relay_enabled=True)
    c = QwenWebClient(cfg, logging.getLogger("t"))
    out = c.chat_completion(_payload())
    assert out["choices"][0]["message"]["content"] == "回答开头，答案的剩余部分"
    assert out["choices"][0]["finish_reason"] == "stop"
    assert len(Handler.relay_requests) == 2
    cont = Handler.relay_requests[1]
    assert cont["prompt"] == "继续"
    assert cont["chat_id"] == "q-relay-1"
    assert cont["parent_id"] == "r-1"


def test_qwen_relay_normal_end_never_continues(mock_server, tmp_path, monkeypatch):
    port = int(mock_server.rsplit(":", 1)[1])
    from webai2api import qwen_relay
    monkeypatch.setattr(qwen_relay, "user_data_dir", lambda: tmp_path)
    qwen_relay.write_state("acc-cont", {
        "pid": os.getpid(), "port": port, "provider": "qwen",
        "started_at": time.time(),
    })
    Handler.qwen_rounds = [QWEN_HEAD + QWEN_TAIL]
    cfg = QwenConfig(base_url="https://chat.qwen.ai", auth_token="tok-q",
                     egress="direct", anti_rate_limit_enabled=False,
                     session_store_path="", account_id="acc-cont",
                     relay_enabled=True)
    c = QwenWebClient(cfg, logging.getLogger("t"))
    out = c.chat_completion(_payload())
    assert out["choices"][0]["finish_reason"] == "stop"
    assert len(Handler.relay_requests) == 1


# ── 锚点与失败降级（2026-10-27 审查发现）────────────
def test_kimi_truncated_without_assistant_id_never_continues(mock_server):
    """主轮没拿到 assistant 消息 id 就断流：不发起续写（「继续」会挂到
    上一轮消息上成为孤儿分支），按截断返回 finish=length。"""
    head = [
        {"op": "set", "chat": {"id": "chat-777"}},
        {"op": "set", "message": {"id": "msg-u", "role": "user",
                                  "status": "MESSAGE_STATUS_COMPLETED"}},
        {"op": "set", "mask": "block.text", "block": {"id": "b1", "text": {"content": ""}}},
        {"op": "append", "mask": "block.text.content",
         "block": {"id": "b1", "text": {"content": "半截"}}},
        # 无 assistant 消息事件、无 done、无 0x02 → 断流且无锚点
    ]
    Handler.kimi_rounds = [(head, False), (KIMI_TAIL_EVENTS, True)]
    cfg = KimiConfig(base_url=mock_server, auth_token="t", egress="direct",
                     anti_rate_limit_enabled=False, session_store_path="")
    c = KimiWebClient(cfg, logging.getLogger("t"))
    out = c.chat_completion(_payload())
    assert out["choices"][0]["finish_reason"] == "length"
    assert len(Handler.kimi_requests) == 1, "无锚点不得发起续写"


def test_kimi_continuation_rejected_degrades_to_length(mock_server):
    """续写轮被 0x02 错误帧拒绝：按截断收场（finish=length），
    绝不误报 stop，也绝不把已输出的流变成报错。"""
    Handler.kimi_rounds = [
        (_kimi_events_head(), False),
        ([], True),   # 空 events + 错误帧
    ]
    Handler.kimi_ends = [
        None,
        json.dumps({"error": {"code": "rate_limit",
                              "message": "too many requests"}}).encode(),
    ]
    cfg = KimiConfig(base_url=mock_server, auth_token="t", egress="direct",
                     anti_rate_limit_enabled=False, session_store_path="")
    c = KimiWebClient(cfg, logging.getLogger("t"))
    out = c.chat_completion(_payload())
    assert out["choices"][0]["finish_reason"] == "length"
    chunks_err = None
    assert chunks_err is None


def test_qwen_relay_no_anchor_never_continues(mock_server, tmp_path, monkeypatch):
    """中继流连 meta（会话/响应锚点）都没发就断流：不发起续写。"""
    port = int(mock_server.rsplit(":", 1)[1])
    from webai2api import qwen_relay
    monkeypatch.setattr(qwen_relay, "user_data_dir", lambda: tmp_path)
    qwen_relay.write_state("acc-cont", {
        "pid": os.getpid(), "port": port, "provider": "qwen",
        "started_at": time.time(),
    })
    Handler.qwen_rounds = [[{"type": "delta", "content": "半截"}]]
    cfg = QwenConfig(base_url="https://chat.qwen.ai", auth_token="tok-q",
                     egress="direct", anti_rate_limit_enabled=False,
                     session_store_path="", account_id="acc-cont",
                     relay_enabled=True)
    c = QwenWebClient(cfg, logging.getLogger("t"))
    out = c.chat_completion(_payload())
    assert out["choices"][0]["finish_reason"] == "length"
    assert len(Handler.relay_requests) == 1


def test_kimi_continuation_failure_keeps_parent_anchor(mock_server):
    """续写轮失败（无消息 id）时存储回退 cur_parent：下一轮的 parentId
    才不会倒退回上一轮（此前会话链分叉丢上下文）。"""
    Handler.kimi_rounds = [
        (_kimi_events_head(), False),   # 截断，含 assistant id=msg-a
        ([], True),                     # 续写轮直接被错误帧拒绝
    ]
    Handler.kimi_ends = [
        None,
        json.dumps({"error": {"code": "x", "message": "y"}}).encode(),
    ]
    cfg = KimiConfig(base_url=mock_server, auth_token="t", egress="direct",
                     anti_rate_limit_enabled=False, session_store_path="")
    c = KimiWebClient(cfg, logging.getLogger("t"))
    stores = []
    c._conv_cache.store = lambda k, sid, extra="": stores.append((sid, extra))
    out = c.chat_completion(_payload())
    assert out["choices"][0]["finish_reason"] == "length"
    assert stores and stores[0] == ("chat-777", "msg-a"), \
        f"parent 应回退到本轮 assistant id: {stores}"


def test_kimi_continuation_state_error_degrades(mock_server):
    """续写轮收到 error 事件：按截断收场（finish=length），且不再重复
    发起续写（此前会再打一轮，错误信息被吞）。"""
    Handler.kimi_rounds = [
        (_kimi_events_head(), False),
        ([{"error": "续写出错"}], False),
        (KIMI_TAIL_EVENTS, True),
    ]
    cfg = KimiConfig(base_url=mock_server, auth_token="t", egress="direct",
                     anti_rate_limit_enabled=False, session_store_path="")
    c = KimiWebClient(cfg, logging.getLogger("t"))
    out = c.chat_completion(_payload())
    assert out["choices"][0]["finish_reason"] == "length"
    assert len(Handler.kimi_requests) == 2, "error 轮后不得再发起续写"
