"""2026-11-02 优化批次（A/B 组）回归。

- /v1/models 能力元数据（与 opencode 生成器同源）
- 网关请求体上限 413
- https 代理真 TLS 隧道（proxy._apply_https）
- SSE keepalive 帧格式与网关容忍性
- 健康检查并发化后结果顺序保持
"""

from __future__ import annotations

import json
import logging
import socket
import ssl
import threading
import types
import urllib.error
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from webai2api.app import build_app
from webai2api.providers import ProviderRegistry, model_capabilities
from webai2api.store import Store

TOKEN_HDR = "X-Admin-Token"


def _make_admin(tmp_path: Path):
    store = Store(tmp_path / "data" / "webai2api.db")
    registry = ProviderRegistry(store, logging.getLogger("t"))
    app = build_app(store, registry, logging.getLogger("t"))
    client = TestClient(app, base_url="http://127.0.0.1:8800", client=("127.0.0.1", 50000))
    tok = client.post("/admin/api/login", json={"password": "admin"}).json()["token"]
    return store, registry, client, {TOKEN_HDR: tok}


def _msg(content):
    return {"role": "user", "content": content}


# ── A1：/v1/models 能力元数据 ───────────────────────────────

def test_models_carry_capabilities(tmp_path):
    _store, registry, client, hdr = _make_admin(tmp_path)
    r = client.get("/v1/models", headers={"Authorization": "Bearer bad"})
    assert r.status_code == 401

    # 需要有效 key
    r = client.post("/admin/api/keys", json={"name": "K"}, headers=hdr)
    key = r.json()["key"]["key"]
    r = client.get("/v1/models", headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 200
    data = {m["id"]: m for m in r.json()["data"]}
    assert data, "模型清单不应为空"
    for mid, m in data.items():
        caps = m.get("capabilities")
        assert isinstance(caps, dict) and caps.get("output") == ["text"]
        assert isinstance(m.get("limit"), dict)
    # 真实能力：deepseek/GLM 有图片，kimi/qwen 没有；deepseek-chat 无思考
    assert data["deepseek-chat"]["capabilities"]["input"] == ["image", "text"]
    assert data["GLM-5.3"]["capabilities"]["input"] == ["image", "text"]
    assert data["kimi-k3"]["capabilities"]["input"] == ["text"]
    assert data["qwen3.7-plus"]["capabilities"]["input"] == ["text"]
    assert data["deepseek-chat"]["capabilities"]["reasoning"]["supported"] is False
    assert data["deepseek-reasoner"]["capabilities"]["reasoning"]["supported"] is True


def test_capabilities_source_shared_with_snippet(tmp_path):
    """生成器与元数据同源：opencode 片段里的声明必须与 capabilities 一致。"""
    from webai2api.opencode_config import _build_opencode_snippet

    store, registry, client, hdr = _make_admin(tmp_path)
    r = client.post("/admin/api/keys", json={"name": "K"}, headers=hdr)
    key = r.json()["key"]["key"]
    snippet = _build_opencode_snippet("http://127.0.0.1:8800", key,
                                      ["deepseek-chat", "kimi-k3"])
    assert '"deepseek-chat"' in snippet and '"kimi-k3"' in snippet
    # deepseek-chat：图片+无思考；kimi-k3：无图片+思考
    assert '["image","text"]' in snippet and '["text"]' in snippet
    assert '"supported": false' in snippet and '"supported": true' in snippet


def test_model_capabilities_shape():
    caps = model_capabilities("deepseek-chat")
    assert caps == {
        "input": ["image", "text"],
        "output": ["text"],
        "reasoning": {"supported": False},
    }


# ── A2：请求体上限 ─────────────────────────────────────────

def test_body_size_limit_413(tmp_path, monkeypatch):
    import webai2api.gateway as gw

    monkeypatch.setattr(gw, "MAX_BODY_BYTES", 64)
    class _AC:
        account = {"id": "a", "name": "a", "provider": "deepseek", "enabled": 1}
        client = types.SimpleNamespace(
            chat_completion=lambda p: {"choices": [{"message": {"content": "x"}}]})

        def mark_ok(self):
            pass

        def mark_bad(self, seconds=0, exc=None):
            pass

        def cooldown_left(self):
            return 0

    store, registry, client, hdr = _make_admin(tmp_path)
    registry.pick = lambda *a, **k: _AC()  # type: ignore[assignment]
    registry.pick_by_id = registry.pick  # type: ignore[assignment]
    r = client.post("/admin/api/keys", json={"name": "K"}, headers=hdr)
    key = r.json()["key"]["key"]
    big = "x" * 200
    r = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "deepseek-chat", "messages": [_msg(big)]},
    )
    assert r.status_code == 413


# ── B8：https 代理真 TLS 隧道 ──────────────────────────────

class _FakeTLSSock:
    def __init__(self, wrapped_from, resp=b""):
        self._from = wrapped_from
        self._resp = resp
        self.sent = b""
        self.server_hostname = None

    def sendall(self, data):
        self.sent += data

    def recv(self, n):
        if self._resp:
            r, self._resp = self._resp, b""
            return r
        return b""

    def close(self):
        pass


def test_https_proxy_tunnel(monkeypatch):
    from webai2api.proxy import Proxy

    wraps = []
    fakes = []

    def fake_wrap(ctx, sock, **kw):
        fake = _FakeTLSSock(sock, b"HTTP/1.1 200 Connection established\r\n\r\n")
        wraps.append(kw.get("server_hostname"))
        fakes.append(fake)
        return fake

    created = []

    def fake_create_connection(addr, timeout=None):
        created.append((addr, timeout))
        return socket.socket()

    monkeypatch.setattr("ssl.SSLContext.wrap_socket", fake_wrap)
    monkeypatch.setattr("socket.create_connection", fake_create_connection)

    px = Proxy.parse("https://u:p@proxy.example.com:8443")
    conn = __import__("http.client", fromlist=["x"]).HTTPSConnection("chat.deepseek.com")
    px.apply(conn, "chat.deepseek.com", 443)

    assert created and created[0][0] == ("proxy.example.com", 8443)
    # 第一层 wrap=代理，第二层 wrap=目标（SNI）
    assert wraps == ["proxy.example.com", "chat.deepseek.com"]
    assert conn.sock is fakes[-1]
    # CONNECT 在代理 TLS 层（第一层 wrap）里发出
    sent = fakes[0].sent.decode("latin-1")
    assert sent.startswith("CONNECT chat.deepseek.com:443 HTTP/1.1\r\n")
    assert "Proxy-Authorization: Basic" in sent


def test_https_proxy_rejected_connect(monkeypatch):
    from webai2api.proxy import Proxy, ProxyConfigError

    def fake_wrap(ctx, sock, **kw):
        return _FakeTLSSock(sock, b"HTTP/1.1 407 Denied\r\n\r\n")

    monkeypatch.setattr("ssl.SSLContext.wrap_socket", fake_wrap)
    monkeypatch.setattr("socket.create_connection",
                        lambda addr, timeout=None: socket.socket())
    px = Proxy.parse("https://proxy.example.com:8443")
    conn = __import__("http.client", fromlist=["x"]).HTTPConnection("example.com", 80)
    with pytest.raises(ProxyConfigError):
        px.apply(conn, "example.com", 80)


# ── B5：SSE keepalive ──────────────────────────────────────

def test_keepalive_sleep_frames(monkeypatch):
    from webai2api.adapters import tool_bridge

    class _Clock:
        t = 0.0

        def time(self):
            return _Clock.t

        def sleep(self, s):
            _Clock.t += s

    clock = _Clock()
    monkeypatch.setattr(tool_bridge, "time", clock)
    frames = list(tool_bridge.keepalive_sleep(2.5, step=1.0))
    assert frames == [": keepalive\n\n"] * 3
    assert list(tool_bridge.keepalive_sleep(0)) == []
    assert list(tool_bridge.keepalive_sleep(-1)) == []


def test_gateway_tolerates_keepalive_lines():
    from webai2api.gateway import _chunk_has_payload, _event_has_payload

    line = ": keepalive\n\n"
    assert not _chunk_has_payload(line)
    assert not _event_has_payload(line)


def test_glm_stream_emits_keepalive_during_throttle(monkeypatch):
    """GLM 生成器：防风控等待期间必须发出 : keepalive 注释帧。"""
    import webai2api.adapters.chatglm.client as gc

    sleeps = []
    monkeypatch.setattr(gc.time, "sleep", lambda s: sleeps.append(s))

    c = gc.ChatGlmWebClient.__new__(gc.ChatGlmWebClient)
    c.config = types.SimpleNamespace(
        anti_rate_limit_enabled=True,
        min_request_interval=0.0,
        anti_rate_limit_min_delay=1.5,
        anti_rate_limit_max_delay=1.5,
        long_pause_chance=0.0,
    )
    c.logger = logging.getLogger("t")
    c._last_call_lock = threading.Lock()
    c._last_call_at = 0.0

    def gen():
        _wait = c._throttle_wait()
        if _wait > 0:
            yield from __import__("webai2api.adapters.tool_bridge",
                                  fromlist=["x"]).keepalive_sleep(_wait)
        yield "data: real\n\n"

    frames = list(gen())
    assert frames[0] == ": keepalive\n\n"
    assert frames[-1] == "data: real\n\n"
    assert len(frames) >= 3  # ~1.5s → 至少 2 帧
    assert sleeps, "分片睡眠应发生"


# ── A4：check_all 并发顺序 ─────────────────────────────────

def test_check_all_parallel_preserves_order(monkeypatch):
    import webai2api.health as H

    class _Store:
        def __init__(self):
            self.writes = []

        def list_accounts(self):
            return [{"id": f"a{i}", "enabled": 1, "checked_at": 0.0} for i in range(6)]

        def set_account_health(self, aid, status, *a, **k):
            self.writes.append((aid, status))

    st = _Store()
    mon = H.HealthMonitor(st, logging.getLogger("t"), interval=300)

    def _ok(acc):
        import time as _t
        _t.sleep(0.05 * (5 - int(acc["id"][1])))  # 故意乱序耗时
        return "ok", {}, "", ""

    monkeypatch.setattr(H, "check_account", _ok)
    res = mon.check_all()
    assert [r["id"] for r in res] == [f"a{i}" for i in range(6)]
    assert len(st.writes) == 6
