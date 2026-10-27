"""会话 token 上限轮换（「快满了就换」）回归测试。"""

from __future__ import annotations

import logging
from types import SimpleNamespace

from webai2api.adapters.conv_cache import ConversationCache


def _cache(max_tokens=1000, max_turns=80):
    cfg = SimpleNamespace(
        session_cache_enabled=True,
        session_cache_ttl=43200,
        session_cache_max_size=100,
        max_turns_per_session=max_turns,
        max_tokens_per_session=max_tokens,
    )
    return ConversationCache(
        config=cfg, logger=logging.getLogger("t"), tag="test", id_field="session_id"
    )


def test_tokens_accumulate_across_turns():
    c = _cache(max_tokens=1000)
    c.store("k", "s1", used_tokens=300)
    c.store("k", "s1", used_tokens=200)
    item = c._cache["k"]
    assert item[2] == 2  # 两轮
    assert item[4] == 500  # 累计 token


def test_rotate_when_token_cap_reached():
    c = _cache(max_tokens=1000)
    c.store("k", "s1", used_tokens=900)
    assert c.cached("k") == ("s1", "")  # 未达上限，继续复用
    c.store("k", "s1", used_tokens=200)  # 累计 1100
    assert c.cached("k") == ("", "")  # 达上限 → 新建


def test_token_cap_disabled_when_zero():
    c = _cache(max_tokens=0)
    for _ in range(5):
        c.store("k", "s1", used_tokens=10_000)
    assert c.cached("k")[0] == "s1"


def test_turn_limit_still_works():
    c = _cache(max_tokens=10**9, max_turns=2)
    c.store("k", "s1")
    assert c.cached("k")[0] == "s1"
    c.store("k", "s1")
    assert c.cached("k") == ("", "")  # 2 轮达上限


def test_collect_stale_includes_token_exhausted():
    c = _cache(max_tokens=100)
    c.store("k", "s1", used_tokens=500)
    stale = c.collect_stale()
    assert ("k", "s1") in stale


def test_legacy_four_tuple_still_readable():
    """旧版本落盘是四元组，读取不能炸。"""
    c = _cache(max_tokens=1000)
    c._cache["old"] = ("s-legacy", 9e9, 1, "")
    assert c.cached("old")[0] == "s-legacy"
