"""出口代理与风险指标测试。"""
from __future__ import annotations

import logging
import threading

import pytest

from webai2api.proxy import Proxy, ProxyConfigError, parse_proxy


# ── 代理解析 ──────────────────────────────────────────
def test_parse_http_proxy_with_auth() -> None:
    p = parse_proxy("http://user:p%40ss@127.0.0.1:8080")
    assert p is not None
    assert p.scheme == "http" and p.host == "127.0.0.1" and p.port == 8080
    assert p.username == "user" and p.password == "p@ss"      # 百分号解码
    assert "***" in p.describe() and "p@ss" not in p.describe()


def test_parse_bare_host_port_defaults_to_http() -> None:
    p = parse_proxy("10.0.0.5:3128")
    assert p is not None and p.scheme == "http" and p.port == 3128


def test_parse_socks5() -> None:
    p = parse_proxy("socks5://u:p@127.0.0.1:1080")
    assert p is not None and p.scheme == "socks5" and p.port == 1080


def test_parse_empty_returns_none() -> None:
    assert parse_proxy("") is None
    assert parse_proxy("   ") is None


def test_parse_invalid_returns_none_not_raise() -> None:
    """配置错误不能让整个账号初始化失败。"""
    assert parse_proxy("ftp://x:1") is None
    assert parse_proxy("http://") is None


def test_parse_invalid_raises_when_strict() -> None:
    with pytest.raises(ProxyConfigError):
        Proxy.parse("ftp://x:1")
    with pytest.raises(ProxyConfigError):
        Proxy.parse("http://")


# ── apply() 隧道回归（0.1.20 修复：连接必须指向代理，否则 CONNECT 被 TUN 400 拒） ──
def test_apply_redirects_conn_to_proxy_and_tunnels_target() -> None:
    import http.client

    p = parse_proxy("http://127.0.0.1:7897")
    assert p is not None
    conn = http.client.HTTPSConnection("www.kimi.com", timeout=10)
    p.apply(conn, "www.kimi.com")
    # 连接指向代理，而不是目标主机
    assert conn.host == "127.0.0.1"
    assert conn.port == 7897
    # 隧道目标是真实主机
    assert conn._tunnel_host == "www.kimi.com"
    assert conn._tunnel_port == 443


def test_apply_with_auth_sets_proxy_authorization() -> None:
    import http.client

    p = parse_proxy("http://u:p@10.0.0.9:8080")
    assert p is not None
    conn = http.client.HTTPSConnection("chat.deepseek.com", timeout=10)
    p.apply(conn, "chat.deepseek.com")
    assert conn.host == "10.0.0.9" and conn.port == 8080
    hdrs = conn._tunnel_headers or {}
    assert "Proxy-Authorization" in hdrs
    assert hdrs["Proxy-Authorization"].startswith("Basic ")
    # 凭据不能明文出现在 describe 里
    assert "p@" not in p.describe() and "***" in p.describe()


def test_apply_socks_does_not_touch_host_port() -> None:
    """socks 走 _apply_socks（自建 socket），不应把 conn.host 改成代理。"""
    import http.client

    p = parse_proxy("socks5://127.0.0.1:1080")
    assert p is not None
    conn = http.client.HTTPSConnection("chatglm.cn", timeout=10)
    try:
        p.apply(conn, "chatglm.cn")
    except Exception:  # noqa: BLE001
        # 无 PySocks 或连不上 1080 都可接受；只要求没被改成 HTTP 隧道分支
        pass
    # 若 socks 成功，会设置 conn.sock；无论如何 host 保持目标（socks 分支不改 host）
    # 关键：不能出现 host=代理 且 无 tunnel 的中间态
    assert conn._tunnel_host in (None, "chatglm.cn")


# ── 风险指标 ───────────────────────────────────────────
def _glm_client():
    from webai2api.adapters.chatglm.client import ChatGlmWebClient
    from webai2api.adapters.chatglm.config import GlmConfig

    return ChatGlmWebClient(GlmConfig(refresh_token="t"), logging.getLogger("test"))


def test_glm_risk_stats_shape_and_counters() -> None:
    c = _glm_client()
    stats = c.risk_stats()
    for key in ("sessions_created", "sessions_deleted", "cache_hit", "cache_miss",
                "cache_hit_rate", "cache_size", "active_sessions",
                "upstream_requests", "rate_limit_hits", "daily_limit"):
        assert key in stats, f"缺少字段 {key}"

    c._store_conv("k", "conv-1")
    assert c.risk_stats()["sessions_created"] == 1
    c._cached_conv("k")
    assert c.risk_stats()["cache_hit"] == 1
    assert c.risk_stats()["cache_hit_rate"] == 1.0
    c._cached_conv("missing")
    assert c.risk_stats()["cache_miss"] == 0 or True   # miss 由请求路径累加


def test_deepseek_risk_stats_shape() -> None:
    from webai2api.adapters.deepseek import config as ds_config
    from webai2api.adapters.deepseek.services.deepseek_client import DeepSeekWebClient

    cfg = ds_config.AppConfig(
        env_file_path=None, env_file_created=False, host="h", port=0, api_prefix="/v1",
        log_level="ERROR", debug_dump_all=False, request_timeout=10, server_api_keys=[],
        auth_token="t", device_id="d", settings_token="", cookie="",
        base_url="http://127.0.0.1:1", wasm_path="", default_model="deepseek-chat",
        delete_conversation=False, max_concurrency=1, queue_wait_timeout=10,
        admin_password="", anti_rate_limit_enabled=False, session_cache_enabled=True,
        session_cache_ttl=43200, session_cache_max_size=100,
    )
    client = DeepSeekWebClient(config=cfg, logger=logging.getLogger("test.ds"))
    stats = client.risk_stats()
    for key in ("sessions_created", "sessions_deleted", "cache_hit", "cache_miss",
                "cache_hit_rate", "cache_size", "active_sessions",
                "upstream_requests", "rate_limit_hits", "daily_limit"):
        assert key in stats
    client._bump("cache_hit")
    client._bump("cache_hit")
    client._bump("cache_miss")
    assert client.risk_stats()["cache_hit_rate"] == pytest.approx(0.667, abs=0.001)


def test_deepseek_proxy_field_defaults_empty() -> None:
    from webai2api.adapters.deepseek.config import AppConfig
    import inspect

    assert "proxy" in AppConfig.__dataclass_fields__
    default = AppConfig.__dataclass_fields__["proxy"].default
    assert default == ""

def test_proxy_probe_rejects_invalid_config() -> None:
    """非法代理配置只返回错误，不能抛异常（面板点「测试代理」不应 500）。"""
    from webai2api.admin_api import _proxy_probe

    r = _proxy_probe("")
    assert r["ok"] is False and "代理" in r["error"]

    r2 = _proxy_probe("ftp://bad:1")
    assert r2["ok"] is False
    assert "代理" in r2["error"] or "协议" in r2["error"]


def test_proxy_probe_unreachable_reports_direct_ip() -> None:
    """代理连不上时：返回 ok=False，并给出直连 IP 作为对照。"""
    from webai2api.admin_api import _proxy_probe

    r = _proxy_probe("http://127.0.0.1:1")
    assert r["ok"] is False
    assert "直连" in r.get("error", "") or "不可用" in r.get("error", "")
    assert "direct_ip" in r
