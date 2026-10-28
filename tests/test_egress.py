"""出口模式（直连 / 跟随系统 / 自定义代理）测试。"""
from __future__ import annotations

import urllib.request

from webai2api.egress import (
    EGRESS_CUSTOM,
    EGRESS_DIRECT,
    EGRESS_SYSTEM,
    egress_opener,
    normalize_egress,
    system_proxy_enabled,
)


def test_normalize_egress() -> None:
    assert normalize_egress("direct") == EGRESS_DIRECT
    assert normalize_egress("CUSTOM") == EGRESS_CUSTOM
    assert normalize_egress(" system ") == EGRESS_SYSTEM
    # 未知/空值回退到 system（保持旧行为）
    assert normalize_egress("") == EGRESS_SYSTEM
    assert normalize_egress(None) == EGRESS_SYSTEM
    assert normalize_egress("bogus") == EGRESS_SYSTEM


def test_direct_opener_has_no_proxy() -> None:
    """direct 模式必须显式禁用代理（否则 Windows 会读注册表系统代理）。"""
    from webai2api.egress import proxy_map_for

    assert proxy_map_for(EGRESS_DIRECT, "http://1.2.3.4:1") == {}
    assert proxy_map_for(EGRESS_SYSTEM, "http://1.2.3.4:1") == {}


def test_custom_opener_maps_proxy() -> None:
    from webai2api.egress import proxy_map_for

    m = proxy_map_for(EGRESS_CUSTOM, "http://u:p@1.2.3.4:8080")
    assert m.get("https") and "1.2.3.4:8080" in m["https"]
    assert m.get("http")


def test_custom_opener_invalid_falls_back_to_direct() -> None:
    from webai2api.egress import proxy_map_for
    from webai2api.proxy import parse_proxy

    assert parse_proxy("ftp://bad:1") is None
    assert proxy_map_for(EGRESS_CUSTOM, "ftp://bad:1") == {}
    assert proxy_map_for(EGRESS_CUSTOM, "") == {}


def test_system_opener_uses_resolved_proxy(monkeypatch) -> None:
    """system 模式要显式解析系统代理（http.client/urllib 都可能读不到注册表）。"""
    import webai2api.egress as eg

    monkeypatch.setattr(eg, "system_proxy_for", lambda host: "http://127.0.0.1:7897")
    opener = eg.egress_opener(EGRESS_SYSTEM, "", "chat.deepseek.com")
    handlers = [h for h in opener.handlers
                if isinstance(h, urllib.request.ProxyHandler)]
    assert handlers and "7897" in handlers[0].proxies.get("https", "")

    monkeypatch.setattr(eg, "system_proxy_for", lambda host: "")
    opener2 = eg.egress_opener(EGRESS_SYSTEM, "", "chat.deepseek.com")
    assert not [h for h in opener2.handlers
                if isinstance(h, urllib.request.ProxyHandler)]


def test_system_proxy_enabled_returns_bool() -> None:
    assert isinstance(system_proxy_enabled(), bool)


# ── 账号 API 与存储 ────────────────────────────────────
def test_store_accepts_egress(tmp_path) -> None:
    from webai2api.store import Store

    s = Store(tmp_path / "t.db")
    acc = s.add_account("deepseek", "a", {"auth_token": "t", "egress": "direct"})
    assert acc["egress"] == "direct"
    s.update_account(acc["id"], egress="custom", proxy="http://1.2.3.4:1")
    got = s.get_account(acc["id"])
    assert got["egress"] == "custom"
    assert got["proxy"] == "http://1.2.3.4:1"


def test_webview_proxy_args() -> None:
    from webai2api.login_webview import _webview_proxy_args

    assert "direct://" in _webview_proxy_args({"egress": "direct"})
    args = _webview_proxy_args(
        {"egress": "custom", "proxy": "http://u:p@1.2.3.4:8080"})
    assert "http://1.2.3.4:8080" in args
    # 系统模式不干预（由系统代理决定）
    assert _webview_proxy_args({"egress": "system"}) == ""
    # socks5 代理
    socks = _webview_proxy_args(
        {"egress": "custom", "proxy": "socks5://127.0.0.1:1080"})
    assert "socks5://127.0.0.1:1080" in socks
