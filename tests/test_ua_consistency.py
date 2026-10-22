"""UA 一致性测试：登录（WebView2/Edge）与 DeepSeek API / 健康检查共用同一浏览器画像。"""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from webai2api.health import DEFAULT_DEEPSEEK_UA, deepseek_headers

EDGE_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0"
)


def _account(profile: object = None, **extra: object) -> dict:
    acc = {
        "id": "acc_test",
        "auth_token": "tok",
        "device_id": "dev",
        "cookie": "",
    }
    if profile is not None:
        acc["profile"] = profile
    acc.update(extra)
    return acc


# ── health.deepseek_headers ──
def test_headers_use_profile_ua_and_platform() -> None:
    profile = json.dumps({"uid": "1", "ua": EDGE_UA, "platform": "Win32"})
    h = deepseek_headers(_account(profile))
    assert h["User-Agent"] == EDGE_UA
    assert h["x-device-model"] == "Win32"
    assert h["Authorization"] == "Bearer tok"
    assert h["x-device-id"] == "dev"


def test_headers_fallback_when_profile_missing() -> None:
    for profile in (None, "", "{}", "not-json", "[]", "null", 123):
        h = deepseek_headers(_account(profile))
        assert h["User-Agent"] == DEFAULT_DEEPSEEK_UA, profile
        assert h["x-device-model"] == "", profile


def test_headers_keep_profile_without_ua() -> None:
    h = deepseek_headers(_account(json.dumps({"uid": "1", "email": "a@b.c"})))
    assert h["User-Agent"] == DEFAULT_DEEPSEEK_UA
    assert h["x-device-model"] == ""


# ── deepseek_auth.get_browser_headers ──
def _auth_manager(config: object):
    from webai2api.adapters.deepseek.services.deepseek_auth import DeepSeekAuthManager

    return DeepSeekAuthManager(config=config, logger=SimpleNamespace())


def _fake_config(**kv: object) -> SimpleNamespace:
    base = {
        "cookie": "",
        "user_agent": "",
        "device_model": "",
        "auth_token": "tok",
        "device_id": "dev",
        "settings_token": "",
    }
    base.update(kv)
    return SimpleNamespace(**base)


def test_browser_headers_use_config_ua() -> None:
    mgr = _auth_manager(_fake_config(user_agent=EDGE_UA))
    assert mgr.get_browser_headers()["User-Agent"] == EDGE_UA


def test_browser_headers_fallback_without_attr() -> None:
    """config 没有 user_agent 字段（旧版 AppConfig）时回退 Firefox。"""
    mgr = _auth_manager(SimpleNamespace(cookie=""))
    assert mgr.get_browser_headers()["User-Agent"] == DEFAULT_DEEPSEEK_UA


def test_auth_headers_device_model() -> None:
    mgr = _auth_manager(_fake_config(user_agent=EDGE_UA, device_model="Win32"))
    h = mgr.get_auth_headers()
    assert h["User-Agent"] == EDGE_UA
    assert h["x-device-model"] == "Win32"

    mgr2 = _auth_manager(SimpleNamespace(
        cookie="", auth_token="tok", device_id="dev", settings_token=""))
    assert mgr2.get_auth_headers()["x-device-model"] == ""


# ── login_webview 抓取 JS 内含浏览器画像字段 ──
def test_login_js_reports_ua_and_platform() -> None:
    from webai2api.login_webview import DEEPSEEK_EXTRACT_JS

    assert "navigator.userAgent" in DEEPSEEK_EXTRACT_JS
    assert "navigator.platform" in DEEPSEEK_EXTRACT_JS
