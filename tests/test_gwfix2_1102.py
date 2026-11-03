"""2026-11-02 本地批次（第二批）回归：store 共享连接、relay 超时 error 帧、
kimi/qwen refresh 网络/凭证分流、admin enabled 字符串真值、opencode 备份上限。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from webai2api.app import build_app
from webai2api.providers import ProviderRegistry
from webai2api.store import Store

TOKEN_HDR = "X-Admin-Token"


# ── store 共享单连接 ────────────────────────────────────────

def test_store_shared_conn_thread_safety(tmp_path):
    store = Store(tmp_path / "db" / "webai2api.db")
    errors: list[Exception] = []

    def _worker(i: int):
        try:
            for j in range(5):
                acc = store.add_account("deepseek", f"号{i}-{j}", {"auth_token": "t"})
                got = store.get_account(acc["id"])
                assert got is not None and got["name"] == f"号{i}-{j}"
                store.log_usage(
                    api_key_id="k", api_key_name="k", account_id="a",
                    account_name="a", provider="deepseek", model="m",
                    prompt_tokens=1, completion_tokens=1, total_tokens=2,
                    latency_ms=1, status="ok",
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(store.list_accounts()) == 40


def test_store_failed_op_rolls_back_and_keeps_working(tmp_path):
    store = Store(tmp_path / "db" / "webai2api.db")
    with pytest.raises(sqlite3.IntegrityError):
        with store._conn() as c:
            c.execute(
                "INSERT INTO accounts (id, provider, name) VALUES ('x','deepseek','a')"
            )
            c.execute(
                "INSERT INTO accounts (id, provider, name) VALUES ('x','deepseek','b')"
            )
    # 同一事务整体回滚：不留半截数据；共享连接继续可用
    assert store.get_account("x") is None
    acc = store.add_account("deepseek", "正常", {"auth_token": "t"})
    assert store.get_account(acc["id"]) is not None


def test_store_pragma_persists_on_shared_conn(tmp_path):
    store = Store(tmp_path / "db" / "webai2api.db")
    row = store._db.execute("PRAGMA synchronous").fetchone()
    assert int(row[0]) == 1  # NORMAL；旧实现每次新连接回退 FULL(2)
    mode = store._db.execute("PRAGMA journal_mode").fetchone()
    assert str(mode[0]).lower() == "wal"


def test_store_close_then_reuse(tmp_path):
    store = Store(tmp_path / "db" / "webai2api.db")
    acc = store.add_account("deepseek", "号", {"auth_token": "t"})
    store.close()
    assert store.get_account(acc["id"]) is not None  # close 后按需重连


# ── relay 超时 error 帧 ─────────────────────────────────────

def test_relay_timeout_pushes_error_frame(monkeypatch):
    import webai2api.qwen_relay as qr
    from http.server import ThreadingHTTPServer

    class _IdleRuntime:
        """start_request 成功但永不回帧：模拟上游挂着不吐字。"""

        def __init__(self):
            self.lock = threading.Lock()
            self.queues = {}
            self.stop = threading.Event()

        def start_request(self, req_id, req):
            pass

    monkeypatch.setattr(qr, "_RELAY_STREAM_TIMEOUT", 2)
    rt = _IdleRuntime()
    orig_runtime = getattr(qr._Handler, "runtime", None)
    server = ThreadingHTTPServer(("127.0.0.1", 0), qr._Handler)
    server.daemon_threads = True
    qr._Handler.runtime = rt
    server.relay_token = "tok"
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/chat"
        req = urllib.request.Request(
            url,
            data=json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode(),
            method="POST",
            headers={"X-Relay-Token": "tok", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            body = r.read().decode("utf-8", "replace")
        assert '"type": "error"' in body or '"type":"error"' in body
        assert "中继超时" in body
    finally:
        server.shutdown()
        server.server_close()
        if orig_runtime is not None:
            qr._Handler.runtime = orig_runtime


# ── kimi/qwen refresh 网络/凭证分流 ────────────────────────

def test_kimi_refresh_network_error_raises_not_auth(monkeypatch):
    import webai2api.health as H

    def _netfail(account, req, timeout):
        raise urllib.error.URLError("connect timeout")

    monkeypatch.setattr(H, "_open_account", _netfail)
    with pytest.raises(RuntimeError) as ei:
        H._kimi_refresh({"settings_token": "r", "id": "a"})
    assert "网络失败" in str(ei.value)

    def _unauth(account, req, timeout):
        raise urllib.error.HTTPError("http://x", 401, "Unauthorized", None, None)

    monkeypatch.setattr(H, "_open_account", _unauth)
    assert H._kimi_refresh({"settings_token": "r", "id": "a"}) == ("", "")


def test_qwen_refresh_network_error_raises_not_auth(monkeypatch):
    import webai2api.health as H

    def _netfail(account, req, timeout):
        raise urllib.error.URLError("dns failure")

    monkeypatch.setattr(H, "_open_account", _netfail)
    with pytest.raises(RuntimeError) as ei:
        H._qwen_refresh({"cookie": "c", "id": "a"})
    assert "网络失败" in str(ei.value)


def test_gateway_classifies_chinese_network_word():
    from webai2api.gateway import _classify_error

    assert _classify_error(RuntimeError("kimi refresh 网络失败: timeout")) == "network"
    assert _classify_error(RuntimeError("qwen refresh 网络失败: refused")) == "network"


# ── admin enabled 字符串真值 ────────────────────────────────

def _make_admin(tmp_path: Path):
    store = Store(tmp_path / "data" / "webai2api.db")
    registry = ProviderRegistry(store, logging.getLogger("t"))
    app = build_app(store, registry, logging.getLogger("t"))
    client = TestClient(app, base_url="http://127.0.0.1:8800", client=("127.0.0.1", 50000))
    tok = client.post("/admin/api/login", json={"password": "admin"}).json()["token"]
    hdr = {TOKEN_HDR: tok}
    return store, client, hdr


def test_key_enabled_string_false_disables(tmp_path):
    store, client, hdr = _make_admin(tmp_path)
    r = client.post(
        "/admin/api/keys", json={"name": "K"}, headers=hdr)
    key_id = r.json()["key"]["id"]
    # 前端透传字符串 "false"：必须禁用而不是启用
    r2 = client.put(f"/admin/api/keys/{key_id}", json={"enabled": "false"}, headers=hdr)
    assert r2.json().get("ok") is True, r2.text
    row = store.find_key_by_id(key_id)
    assert not row["enabled"], "字符串 'false' 不得被当成真值启用 Key"


def test_account_enabled_string_false_disables(tmp_path):
    store, client, hdr = _make_admin(tmp_path)
    acc = store.add_account("deepseek", "号", {"auth_token": "t"})
    r = client.put(f"/admin/api/accounts/{acc['id']}", json={"enabled": "false"}, headers=hdr)
    assert r.json().get("ok") is True, r.text
    assert not store.get_account(acc["id"])["enabled"]


# ── opencode 备份上限 ──────────────────────────────────────

def test_prune_opencode_backups_keeps_recent(tmp_path):
    from webai2api.opencode_config import _prune_opencode_backups

    cfg = tmp_path / "opencode.jsonc"
    for i in range(15):
        (tmp_path / f"opencode.jsonc.bak_2026110{i % 10}0000_{i:06d}").write_text("x")
    _prune_opencode_backups(cfg, keep=10)
    left = sorted(tmp_path.glob("opencode.jsonc.bak_*"))
    assert len(left) == 10


def test_to_bool():
    from webai2api.admin_api import _to_bool

    for falsy in (False, 0, "", "false", "False", " 0 ", "off", None):
        assert _to_bool(falsy) is False, repr(falsy)
    for truthy in (True, 1, "true", "True", "1", "yes", "on"):
        assert _to_bool(truthy) is True, repr(truthy)
