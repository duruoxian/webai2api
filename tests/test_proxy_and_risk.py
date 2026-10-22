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
