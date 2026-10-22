"""会话清理线程 / 日配额 / 账号汇总 的轻量单测（不触网、不起真实线程循环）。"""
from __future__ import annotations

import threading

from webai2api.adapters.base import AccountClient
from webai2api.app import start_session_cleaner
from webai2api.providers import ProviderRegistry


class _FakeConfig:
    def __init__(self, limit: int = 3) -> None:
        self.daily_request_limit = limit


def _mk(limit: int = 3) -> AccountClient:
    return AccountClient({"id": "a1", "auth_token": "t"}, client=object(), config=_FakeConfig(limit))


def test_daily_not_exhausted_initially():
    ac = _mk(3)
    assert ac.daily_exhausted() is False
    ac.mark_ok()
    ac.mark_ok()
    assert ac.daily_exhausted() is False


def test_daily_exhausted_after_limit():
    ac = _mk(2)
    ac.mark_ok()
    ac.mark_ok()
    assert ac.daily_exhausted() is True
    # 超限后继续计数仍然是耗尽
    ac.mark_ok()
    assert ac.daily_exhausted() is True


def test_daily_unlimited_when_zero():
    ac = _mk(0)
    for _ in range(5):
        ac.mark_ok()
    assert ac.daily_exhausted() is False


def test_daily_reset_next_day():
    ac = _mk(1)
    ac.mark_ok()
    assert ac.daily_exhausted() is True
    ac._daily_day = "1970-01-01"  # 模拟跨天
    assert ac.daily_exhausted() is False
    assert ac._daily_count == 0


# ── start_session_cleaner ──

class _Client:
    def __init__(self) -> None:
        self.calls = 0

    def cleanup_stale_sessions(self) -> None:
        self.calls += 1


class _Ac:
    def __init__(self, client) -> None:
        self.client = client
        self.account = {"id": "x", "name": "x"}


class _FakeRegistry:
    def __init__(self, accounts) -> None:
        self._accounts = accounts

    def all_accounts(self) -> list:
        return list(self._accounts)


def test_start_session_cleaner_returns_daemon_thread():
    reg = _FakeRegistry([_Ac(_Client()), _Ac(object())])
    th = start_session_cleaner(reg, __import__("logging").getLogger("t"), interval=3600)
    assert isinstance(th, threading.Thread)
    assert th.daemon is True
    # 线程首轮先 sleep，不会立刻真正执行；确认已启动后结束
    assert th.is_alive() or th.ident is not None


# ── providers.all_accounts ──

class _FakePool:
    def __init__(self, items) -> None:
        self._items = items

    def available(self):
        return list(self._items)


def test_all_accounts_aggregates_pools():
    reg = ProviderRegistry.__new__(ProviderRegistry)
    a, b, c = object(), object(), object()
    reg._pools = {
        "deepseek": _FakePool([a, b]),
        "chatglm": _FakePool([c]),
    }
    out = reg.all_accounts()
    assert out == [a, b, c]


def test_all_accounts_skips_broken_pool():
    class _Boom:
        def available(self):
            raise RuntimeError("boom")

    reg = ProviderRegistry.__new__(ProviderRegistry)
    a = object()
    reg._pools = {"bad": _Boom(), "ok": _FakePool([a])}
    assert reg.all_accounts() == [a]
