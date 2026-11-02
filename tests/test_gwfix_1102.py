"""2026-11-02 网关层审查修复回归。

覆盖：
- P1 socks 代理 + HTTPS：_apply_socks 必须当场 wrap TLS（此前 send() 跳过
  connect()，TLS 握手永不发生，请求明文打 443）
- P2 gateway：touch_key/记账失败不挡请求；换号重试关掉废弃生成器；
  "usage" 子串误判（正文含 usage 的块要进 completion 估算）
- P2 health 周期检查：网络抖动不动 health，只有 auth 类写 expired
- P2 qwen_relay：X-Relay-Token 鉴权 + 请求体上限
- P3：/health 账号池空降级、anthropic 零内容流补空 text 块、错误类型白名单、
  conv_cache 淘汰跳过写入 key、_mask_key 短值不漏、login 状态文件路径校验、
  egress bypass 通配
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import socket
import sys
import threading
import types
import urllib.error
import urllib.request
from pathlib import Path

import http.client
import pytest
from fastapi.testclient import TestClient

from webai2api.app import build_app
from webai2api.providers import ProviderRegistry
from webai2api.store import Store

TOKEN_HDR = "X-Admin-Token"


def _msg(content):
    return {"role": "user", "content": content}


def _resp(text: str) -> dict:
    return {
        "model": "deepseek-chat",
        "choices": [
            {"finish_reason": "stop", "message": {"role": "assistant", "content": text}}
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


class _AC:
    def __init__(self, name: str, client):
        self.account = {
            "id": f"acc-{name}",
            "name": name,
            "provider": "deepseek",
            "enabled": 1,
        }
        self.client = client
        self.calls = 0

    def mark_ok(self):
        pass

    def mark_bad(self, seconds=0, exc=None):
        pass

    def cooldown_left(self):
        return 0


class _Good:
    def __init__(self, ac: _AC, text: str = "回答"):
        self.ac = ac
        self._text = text

    def chat_completion(self, payload):
        self.ac.calls += 1
        return _resp(self._text)

    def stream_chat_completion(self, payload):
        self.ac.calls += 1
        for piece in self._text:
            yield (
                "data: "
                + json.dumps(
                    {"choices": [{"index": 0, "delta": {"content": piece}}]},
                    ensure_ascii=False,
                )
                + "\n\n"
            )
        yield (
            "data: "
            + json.dumps(
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            )
            + "\n\n"
        )
        yield "data: [DONE]\n\n"


def _make_app(tmp_path: Path, ac: _AC):
    store = Store(tmp_path / "data" / "webai2api.db")
    logger = logging.getLogger("webai2api.test.gwfix")
    registry = ProviderRegistry(store, logger)
    app = build_app(store, registry, logger)
    store.add_account("deepseek", "号", {"auth_token": "tok"})
    registry.refresh()

    def _pick(*a, **k):
        return ac

    registry.pick = _pick  # type: ignore[assignment]
    registry.pick_by_id = _pick  # type: ignore[assignment]
    client = TestClient(app, base_url="http://127.0.0.1:8800", client=("127.0.0.1", 50000))
    r = client.post("/admin/api/login", json={"password": "admin"})
    tok = r.json()["token"]
    r = client.post("/admin/api/keys", json={"name": "K"}, headers={TOKEN_HDR: tok})
    assert r.json().get("ok") is True, r.text
    return store, client, r.json()["key"]["key"]


# ── P1：socks 代理 TLS ──────────────────────────────────────

def test_socks_https_wraps_tls(monkeypatch):
    fake = types.ModuleType("socks")

    class _Sock(socket.socket):
        def set_proxy(self, *a, **k):
            pass

        def connect(self, addr):
            self._connected_addr = addr

    fake.socksocket = _Sock
    fake.SOCKS5 = 2
    fake.SOCKS4 = 1
    monkeypatch.setitem(sys.modules, "socks", fake)

    from webai2api.proxy import Proxy

    px = Proxy.parse("socks5://127.0.0.1:1080")
    conn = http.client.HTTPSConnection("chat.deepseek.com")
    seen = {}

    def fake_wrap(ctx, sock, **kw):
        seen.update(kw)
        seen["sock"] = sock
        return ("WRAPPED", sock)

    monkeypatch.setattr("ssl.SSLContext.wrap_socket", fake_wrap)
    px.apply(conn, "chat.deepseek.com")
    assert seen.get("server_hostname") == "chat.deepseek.com"
    assert seen["sock"]._connected_addr == ("chat.deepseek.com", 443)
    assert conn.sock[0] == "WRAPPED"


def test_socks_http_target_no_tls_and_port(monkeypatch):
    fake = types.ModuleType("socks")

    class _Sock(socket.socket):
        def set_proxy(self, *a, **k):
            pass

        def connect(self, addr):
            self._connected_addr = addr

    fake.socksocket = _Sock
    fake.SOCKS5 = 2
    fake.SOCKS4 = 1
    monkeypatch.setitem(sys.modules, "socks", fake)

    from webai2api.proxy import Proxy

    px = Proxy.parse("socks5://127.0.0.1:1080")
    conn = http.client.HTTPConnection("example.test", 8080)
    px.apply(conn, "example.test", 80)
    assert isinstance(conn.sock, _Sock)
    assert conn.sock._connected_addr == ("example.test", 80)


# ── P2：gateway 记账失败不挡请求 ────────────────────────────

def test_touch_key_failure_does_not_break_chat(tmp_path):
    ac = _AC("号", None)
    ac.client = _Good(ac)
    store, client, key = _make_app(tmp_path, ac)

    def _boom(*a, **k):
        raise RuntimeError("database is locked")

    store.touch_key = _boom
    r = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "deepseek-chat", "messages": [_msg("hi")]},
    )
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["message"]["content"] == "回答"


def test_log_usage_failure_keeps_200(tmp_path):
    ac = _AC("号", None)
    ac.client = _Good(ac)
    store, client, key = _make_app(tmp_path, ac)

    def _boom(**kw):
        raise RuntimeError("database is locked")

    store.log_usage = _boom
    r = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "deepseek-chat", "messages": [_msg("hi")]},
    )
    assert r.status_code == 200, r.text


def test_usage_substring_content_still_counted(tmp_path):
    """正文里出现 "usage" 字样的流式块，必须照常进 completion 估算。"""
    ac = _AC("号", None)
    text = "本词 usage 出现在正文里"
    ac.client = _Good(ac, text=text)
    store, client, key = _make_app(tmp_path, ac)

    calls = []
    real_log = store.log_usage

    def _log(**kw):
        calls.append(kw)
        return real_log(**kw)

    store.log_usage = _log
    r = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "deepseek-chat", "messages": [_msg("hi")], "stream": True},
    )
    assert r.status_code == 200
    assert calls, "流结束必须记账"
    # 逐字符流式下整句不会原样出现在响应里；关键断言是「含 usage 字样的
    # 正文块照常进了估算」（旧代码该块被 usage 子串分支吞掉，估算≈1）
    assert calls[-1].get("completion_tokens", 0) > 1


# ── P2：换号重试关掉废弃生成器 ──────────────────────────────

def test_stream_retry_closes_abandoned_generator():
    from webai2api.gateway import _stream_with_retry

    closed = []

    def _gen():
        try:
            yield (
                "data: "
                + json.dumps(
                    {"error": {"message": "Authorization Failed (invalid token)"}}
                )
                + "\n\n"
            )
        finally:
            closed.append(True)

    class _Reg:
        logger = None

        def note_auth_ok(self, *a):
            pass

        def note_auth_failure(self, *a):
            pass

        def pick(self, *a, **k):
            return None

    ac = _AC("死号", None)
    ac.client = types.SimpleNamespace(stream_chat_completion=lambda p: _gen())
    _, _, _, last_exc = asyncio.run(
        _stream_with_retry(
            _Reg(), None, {}, "deepseek-chat", ac, {}, "", None, logging.getLogger("t")
        )
    )
    assert closed == [True]
    assert last_exc is not None


# ── P2：health 周期检查口径 ─────────────────────────────────

def _monitor():
    import webai2api.health as H

    class _Store:
        def __init__(self):
            self.writes = []

        def list_accounts(self):
            return [{"id": "a1", "enabled": 1, "checked_at": 0.0}]

        def set_account_health(self, aid, status, *a, **k):
            self.writes.append((aid, status))

    st = _Store()
    mon = H.HealthMonitor(st, logging.getLogger("t"), interval=300)
    return H, mon, st


def test_health_periodic_network_error_keeps_health(monkeypatch):
    H, mon, st = _monitor()

    def _boom(acc):
        raise urllib.error.URLError("connect timeout")

    monkeypatch.setattr(H, "check_account", _boom)
    res = mon.check_all()
    assert st.writes == [], "网络抖动不能改 health"
    assert res[0].get("health_kept") is True


def test_health_periodic_403_keeps_401_expired(monkeypatch):
    H, mon, st = _monitor()

    def _forbidden(acc):
        raise urllib.error.HTTPError("http://x", 403, "Forbidden", None, io.BytesIO(b""))

    monkeypatch.setattr(H, "check_account", _forbidden)
    mon.check_all()
    assert st.writes == [], "403 可能是风控，不能当凭证过期"

    def _unauth(acc):
        raise urllib.error.HTTPError("http://x", 401, "Unauthorized", None, io.BytesIO(b""))

    monkeypatch.setattr(H, "check_account", _unauth)
    mon.check_all()
    assert st.writes == [("a1", "expired")]


def test_health_periodic_auth_text_marks_expired(monkeypatch):
    H, mon, st = _monitor()

    def _expired(acc):
        raise RuntimeError("accessToken 已过期，且续期失败，请重新登录 Kimi")

    monkeypatch.setattr(H, "check_account", _expired)
    mon.check_all()
    assert st.writes == [("a1", "expired")]


# ── P2：qwen_relay 鉴权 ─────────────────────────────────────

@pytest.fixture()
def relay_server():
    from webai2api.qwen_relay import _Handler, _Runtime
    from http.server import ThreadingHTTPServer

    rt = _Runtime("t")
    orig_runtime = getattr(_Handler, "runtime", None)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.daemon_threads = True
    _Handler.runtime = rt
    server.relay_token = "tok123456"
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    yield server, rt
    server.shutdown()
    server.server_close()
    _Handler.runtime = orig_runtime


def test_relay_rejects_missing_token(relay_server):
    server, _rt = relay_server
    url = f"http://127.0.0.1:{server.server_port}/shutdown"
    with pytest.raises(urllib.error.HTTPError) as ei:
        urllib.request.urlopen(urllib.request.Request(url, data=b"{}", method="POST"), timeout=5)
    assert ei.value.code == 403


def test_relay_accepts_valid_token(relay_server):
    server, rt = relay_server
    url = f"http://127.0.0.1:{server.server_port}/shutdown"
    with urllib.request.urlopen(
        urllib.request.Request(
            url, data=b"{}", method="POST", headers={"X-Relay-Token": "tok123456"}
        ),
        timeout=5,
    ) as r:
        assert r.status == 200
    assert rt.stop.is_set()


def test_relay_body_size_limit(relay_server):
    server, _rt = relay_server
    # Content-Length 谎报超大、实体只发一小部分：服务端在读 body 前就应回 413
    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
    conn.putrequest("POST", "/delete")
    conn.putheader("X-Relay-Token", "tok123456")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Content-Length", str(3_000_000))
    conn.endheaders()
    conn.send(b'{"chat_id": "x"}')
    resp = conn.getresponse()
    assert resp.status == 413
    resp.read()
    conn.close()


def test_relay_health_stays_open(relay_server):
    server, _rt = relay_server
    with urllib.request.urlopen(
        f"http://127.0.0.1:{server.server_port}/health", timeout=5
    ) as r:
        assert r.status == 200


# ── P3 批次 ────────────────────────────────────────────────

def test_health_endpoint_degrades_without_accounts(tmp_path):
    store = Store(tmp_path / "data" / "webai2api.db")
    registry = ProviderRegistry(store, logging.getLogger("t"))
    app = build_app(store, registry, logging.getLogger("t"))
    client = TestClient(app, base_url="http://127.0.0.1:8800", client=("127.0.0.1", 50000))
    assert client.get("/health").json()["status"] == "degraded"
    store.add_account("deepseek", "号", {"auth_token": "tok"})
    registry.refresh()
    assert client.get("/health").json()["status"] == "ok"


def test_anthropic_empty_stream_emits_text_block():
    from webai2api.anthropic_adapter import AnthropicStreamTranslator

    tr = AnthropicStreamTranslator("GLM-5.3")
    text = "\n".join(tr.finish())
    assert "content_block_start" in text
    assert "message_stop" in text
    assert text.index("content_block_start") < text.index("message_stop")


def test_anthropic_error_type_whitelist():
    from webai2api.anthropic_adapter import normalize_error_type

    assert normalize_error_type("internal_error") == "api_error"
    assert normalize_error_type("upstream_error") == "api_error"
    assert normalize_error_type("") == "api_error"
    for ok in (
        "invalid_request_error",
        "authentication_error",
        "permission_error",
        "rate_limit_error",
        "api_error",
        "overloaded_error",
    ):
        assert normalize_error_type(ok) == ok


def test_conv_cache_eviction_skips_written_key():
    from webai2api.adapters.conv_cache import ConversationCache

    class _Cfg:
        session_cache_enabled = True
        session_cache_max_size = 2
        session_cache_ttl = 9999
        max_turns_per_session = 40
        max_tokens_per_session = 0

    cc = ConversationCache(
        config=_Cfg(), logger=logging.getLogger("t"), tag="t", id_field="sid",
        persist_path="",
    )
    cc.store("k1", "s1", "", 0)
    cc.store("k2", "s2", "", 0)
    # k1 恰好最旧且正在被写入：必须淘汰 k2，保住 k1 的轮次
    cc._cache["k1"] = ("s1", 0.0, 1, "", 0)
    cc.store("k1", "s1", "", 0)
    assert "k1" in cc._cache and "k2" not in cc._cache
    assert cc._cache["k1"][2] == 2


def test_mask_key_never_leaks_short_values():
    from webai2api.gateway import _mask_key

    assert _mask_key("") == "(empty)"
    m = _mask_key("short")
    assert m == "(len=5)"
    m2 = _mask_key("sk-abcdefghij123456")
    assert m2.startswith("sk-abcde...") and "3456 (len=19)" in m2


def test_login_state_rejects_path_traversal():
    from webai2api.login_webview import write_state

    with pytest.raises(ValueError):
        write_state("../evil", "ok")
    with pytest.raises(ValueError):
        write_state("a/b", "ok")


def test_default_port_falls_back_on_garbage(monkeypatch):
    import run as run_mod

    monkeypatch.setenv("WEBAI2API_PORT", "abc")
    assert run_mod._default_port() == 8800
    monkeypatch.setenv("WEBAI2API_PORT", "9100")
    assert run_mod._default_port() == 9100


def test_egress_bypass_wildcards(monkeypatch):
    import webai2api.egress as eg

    monkeypatch.setattr(
        urllib.request,
        "getproxies",
        lambda: {
            "https": "http://1.2.3.4:8080",
            "no": "127.*, *.foo.com, localhost",
        },
    )
    assert eg.system_proxy_for("127.0.0.1") == ""
    assert eg.system_proxy_for("foo.com") == ""
    assert eg.system_proxy_for("sub.foo.com") == ""
    assert eg.system_proxy_for("notfoo.com") != ""
    assert "1.2.3.4:8080" in eg.system_proxy_for("example.com")


def test_admin_panel_requires_local_host_header(tmp_path):
    """DNS 重绑定防线：Host 非本机一律 403（即便 client IP 是 127.0.0.1）。"""
    store = Store(tmp_path / "data" / "webai2api.db")
    registry = ProviderRegistry(store, logging.getLogger("t"))
    app = build_app(store, registry, logging.getLogger("t"))
    client = TestClient(
        app,
        base_url="http://attacker.example.com",
        client=("127.0.0.1", 50000),
    )
    r = client.post("/admin/api/login", json={"password": "admin"})
    assert r.status_code == 403
