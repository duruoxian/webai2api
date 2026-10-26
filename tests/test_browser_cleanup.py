"""清浏览器痕迹功能测试。

背景（2026-10-27 用户报障）：旧版本（<0.1.21）的登录窗口共用默认
%APPDATA%\\pywebview 存储，旧 cookie（如微信授权）永久留在机器上，
别人电脑上会出现开发者的微信登录残留。清理范围：
- data/browser_profiles/_session/*（一次性登录目录残留）
- %APPDATA%/pywebview（旧版默认 WebView2 存储，新版面板已改用
  data/panel_profile，此目录成为纯残留）
绝不碰 browser_profiles/<acc_id>（千问中继常驻登录态）。
"""
from __future__ import annotations

import logging

import pytest

from webai2api import login_webview
from webai2api.login_webview import cleanup_browser_residue


@pytest.fixture
def residue(tmp_path, monkeypatch):
    """构造三类目录：一次性残留 / 千问常驻 / 旧版默认存储。"""
    data = tmp_path / "data"
    stale = data / "browser_profiles" / "_session" / "acc-1-abcdef123456"
    (stale / "EBWebView" / "Default" / "Network").mkdir(parents=True)
    qwen = data / "browser_profiles" / "acc-qwen-1"
    (qwen / "EBWebView").mkdir(parents=True)
    legacy = tmp_path / "appdata" / "pywebview"
    (legacy / "EBWebView" / "Default" / "Network").mkdir(parents=True)

    killed: list = []
    monkeypatch.setattr(login_webview, "user_data_dir", lambda: data)
    monkeypatch.setattr(
        login_webview, "_kill_webview_processes_for",
        lambda p: killed.append(str(p)))
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    return {"data": data, "stale": stale, "qwen": qwen,
            "legacy": legacy, "killed": killed}


def test_cleanup_removes_session_and_legacy_keeps_qwen(residue):
    out = cleanup_browser_residue(logging.getLogger("test.cleanup"))

    assert out["ok"] is True
    assert not residue["stale"].exists(), "一次性残留目录应被删除"
    assert not residue["legacy"].exists(), "旧版默认 WebView2 存储应被删除"
    assert residue["qwen"].exists(), "千问中继常驻登录态绝不能动"
    assert any("_session" in p for p in out["removed"])
    assert any("pywebview" in p for p in out["removed"])
    # 杀进程只针对被清目录，千问 profile 不在其中
    killed = " | ".join(residue["killed"])
    assert "pywebview" in killed
    assert str(residue["qwen"]) not in killed


def test_cleanup_idempotent_when_clean(tmp_path, monkeypatch):
    monkeypatch.setattr(login_webview, "user_data_dir", lambda: tmp_path / "data")
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    out = cleanup_browser_residue(logging.getLogger("test.cleanup"))
    assert out == {"ok": True, "removed": []}


def test_cleanup_endpoint_auth_and_result(tmp_path, monkeypatch):
    """/admin/api/browser/cleanup：需要管理员 token，返回 ok+removed。"""
    from fastapi.testclient import TestClient

    from webai2api.app import build_app
    from webai2api.providers import ProviderRegistry
    from webai2api.store import Store

    monkeypatch.setattr(login_webview, "user_data_dir", lambda: tmp_path / "data")
    monkeypatch.setattr(login_webview, "_kill_webview_processes_for", lambda p: None)
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))

    store = Store(tmp_path / "data" / "webai2api.db")
    logger = logging.getLogger("test.cleanup.endpoint")
    app = build_app(store, ProviderRegistry(store, logger), logger)
    client = TestClient(app, client=("127.0.0.1", 50000))

    r = client.post("/admin/api/browser/cleanup")
    assert r.status_code in (401, 403), "未登录必须被拦截"

    tok = client.post("/admin/api/login", json={"password": "admin"}).json()["token"]
    r = client.post("/admin/api/browser/cleanup",
                    headers={"x-admin-token": tok})
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert "removed" in r.json()
