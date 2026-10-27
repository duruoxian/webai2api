"""deepseek 账号保护（防风控）回归测试。

对应 2026-10-27 封号治理四项改动：
  P1 选号负载分散：优先无在途请求的账号
  P2 封禁熔断：封禁信号 → 7 天长冷却，绝不重试/续写
  P3 日额度收紧：默认 600 → 150
  P4 浏览器指纹传输：curl_cffi 优先，失败静默回退 http.client
"""

from __future__ import annotations

import urllib.error

import pytest

from webai2api.adapters.deepseek.services import browser_transport as bt
from webai2api.adapters.deepseek.services.deepseek_client import (
    AccountBannedError,
    ConcurrentRequestQueue,
    _hint_is_banned,
)


# ── P2 封禁熔断 ──────────────────────────────────────────


@pytest.mark.parametrize(
    "reason,content",
    [
        ("banned", ""),
        ("", "账号已被封禁"),
        ("account_suspended", ""),
        ("", "该账号存在异常，永久限制使用"),
        ("", "您的账号已被限制"),
    ],
)
def test_banned_hint_detected(reason, content):
    assert _hint_is_banned(reason, content) is True


@pytest.mark.parametrize(
    "reason,content",
    [
        ("rate_limit_reached", "消息发送过于频繁"),
        ("context_length_exceeded", ""),
        ("file_parse_error", ""),
        ("", "内容涉嫌违规"),
        ("", ""),
    ],
)
def test_non_banned_hint_ignored(reason, content):
    """限流/超长/文件/审核都不能被误判成封禁。"""
    assert _hint_is_banned(reason, content) is False


def test_banned_error_is_upstream_error_with_403():
    err = AccountBannedError("账号已被封禁")
    assert err.status_code == 403
    assert "封禁" in str(err)


# ── P3 日额度 ────────────────────────────────────────────


def test_daily_limit_default_is_150():
    import dataclasses

    from webai2api.adapters.deepseek.config import AppConfig

    defaults = {f.name: f.default for f in dataclasses.fields(AppConfig)}
    assert defaults["daily_request_limit"] == 150


# ── P1 队列水位 ──────────────────────────────────────────


def test_queue_pending_counts_acquired_and_released():
    import logging

    q = ConcurrentRequestQueue(
        logging.getLogger("t"), wait_timeout=5, max_concurrency=1
    )
    assert q.pending() == 0
    lease = q.acquire("a")
    assert q.pending() == 1
    lease.release()
    assert q.pending() == 0


# ── P4 传输层 ────────────────────────────────────────────


def test_transport_reports_availability():
    assert isinstance(bt.available(), bool)


def test_resolve_proxy_direct_and_custom():
    assert bt.resolve_proxy("direct", "") == ("", False)
    assert bt.resolve_proxy("custom", "http://127.0.0.1:7897") == (
        "http://127.0.0.1:7897",
        False,
    )


def test_resolve_proxy_custom_empty_falls_back_to_direct():
    assert bt.resolve_proxy("custom", "") == ("", False)


def test_transport_http_error_is_not_swallowed(monkeypatch):
    """HTTP 层错误是业务结果，必须原样抛出，不能被当成传输失败。"""

    class _Boom:
        def request(self, **kwargs):
            raise urllib.error.HTTPError("http://x", 403, "Forbidden", {}, None)

    sess = bt.BrowserSession()
    monkeypatch.setattr(sess, "_session_for", lambda proxy: _Boom())
    with pytest.raises(urllib.error.HTTPError):
        sess.request("http://x")
