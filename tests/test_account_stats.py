"""acc_stats 持久化：日配额跨重启 + 风险指标当日累计（2026-10-28 第六批）。

修复的实质漏洞：此前每日请求上限是纯内存计数，重启一次配额就重新拉满，
日限流形同虚设。现在 mark_ok 写 acc_stats，daily_used 读持久值。
"""
from __future__ import annotations

from types import SimpleNamespace

from webai2api.store import Store
from webai2api.adapters.base import AccountClient


def _store(tmp_path) -> Store:
    return Store(tmp_path / "t.db")


def _cfg(limit: int = 0):
    return SimpleNamespace(daily_request_limit=limit)


def test_bump_stat_roundtrip(tmp_path):
    s = _store(tmp_path)
    assert s.get_stat("a1", "2026-10-28", "requests") == 0
    s.bump_stat("a1", "2026-10-28", "requests", 2)
    s.bump_stat("a1", "2026-10-28", "requests", 3)
    assert s.get_stat("a1", "2026-10-28", "requests") == 5
    # 其它账号/指标/日期互不串
    assert s.get_stat("a2", "2026-10-28", "requests") == 0
    assert s.get_stat("a1", "2026-10-28", "cache_hit") == 0
    assert s.get_stat("a1", "2026-10-27", "requests") == 0


def test_daily_quota_persists_across_restart(tmp_path):
    s = _store(tmp_path)
    ac = AccountClient({"id": "a1", "name": "x"}, None, _cfg(5), store=s)
    for _ in range(3):
        ac.mark_ok()
    assert ac.daily_used() == 3
    # 重启：新建 AccountClient（内存计数归零），持久值仍为 3
    ac2 = AccountClient({"id": "a1", "name": "x"}, None, _cfg(5), store=s)
    assert ac2.daily_used() == 3
    assert not ac2.daily_exhausted()


def test_daily_quota_exhausted_persists(tmp_path):
    s = _store(tmp_path)
    ac = AccountClient({"id": "a1"}, None, _cfg(2), store=s)
    ac.mark_ok()
    ac.mark_ok()
    assert ac.daily_exhausted()
    ac2 = AccountClient({"id": "a1"}, None, _cfg(2), store=s)
    assert ac2.daily_exhausted(), "重启后日配额不得重置"


def test_risk_delta_persist_and_display(tmp_path):
    s = _store(tmp_path)
    mem = {"sessions_created": 2, "sessions_deleted": 0, "cache_hit": 6,
           "cache_miss": 2, "cache_hit_rate": 0.75, "cache_stale": 0,
           "cache_size": 3, "active_sessions": 1, "upstream_requests": 8,
           "rate_limit_hits": 0, "daily_count": 0, "daily_limit": 0}
    client = SimpleNamespace(risk_stats=lambda: dict(mem))
    ac = AccountClient({"id": "a1"}, client, _cfg(), store=s)

    st1 = ac.persist_and_display_risk()
    assert st1["sessions_created"] == 2
    assert st1["cache_hit"] == 6 and st1["cache_hit_rate"] == 0.75

    mem["sessions_created"] = 5
    mem["cache_hit"] = 9
    st2 = ac.persist_and_display_risk()
    assert st2["sessions_created"] == 5 and st2["cache_hit"] == 9

    # 重启：内存归零，展示持久累计值（跨重启保留）
    zero = {k: 0 for k in ("sessions_created", "sessions_deleted", "cache_hit",
                           "cache_miss", "upstream_requests", "rate_limit_hits")}
    client2 = SimpleNamespace(risk_stats=lambda: {
        **zero, "cache_hit_rate": 0.0, "active_sessions": 0, "cache_size": 0,
        "daily_count": 0, "daily_limit": 0})
    ac2 = AccountClient({"id": "a1"}, client2, _cfg(), store=s)
    st3 = ac2.persist_and_display_risk()
    assert st3["sessions_created"] == 5 and st3["cache_hit"] == 9


def test_deepseek_accountclient_uses_base_persistence():
    """回归护栏（第二轮审查 P1）：deepseek 池的 AccountClient 必须是 base
    子类且接受 store 形参——此前独立类不接受 store= → TypeError 被
    _sync 的 except 静默吞掉 → DeepSeek 账号池为空、全平台 503。"""
    from webai2api.adapters.base import AccountClient as BaseAC
    from webai2api.adapters.deepseek.pool import AccountClient as DSAC

    ac = DSAC({"id": "a1"}, None, _cfg(3), store=None)
    assert isinstance(ac, BaseAC)


def test_update_account_health_whitelisted(tmp_path):
    """回归护栏（第二轮审查 P2）：凭证变更后的 health/checked_at 重置
    必须真正落库（此前白名单不含这两列，赋值被静默丢弃）。"""
    s = _store(tmp_path)
    s.add_account("deepseek", "x", {"auth_token": "t"})
    acc = s.list_accounts("deepseek")[0]
    s.update_account(acc["id"], health="expired", checked_at=123.0)
    acc2 = s.get_account(acc["id"])
    assert acc2["health"] == "expired" and acc2["checked_at"] == 123.0
