"""conv_fingerprint / SessionCache 持久化 单元测试（不联网、不启上游）。"""

from __future__ import annotations

import json

from webai2api.adapters.deepseek.services.session_cache import (
    SessionCache,
    SessionState,
    conv_fingerprint,
)


SYS = {"role": "system", "content": "You are a helpful assistant.\n\n  Keep it short."}


def _conv(turns: int, key: str = "") -> str:
    messages = [SYS, {"role": "user", "content": "Help me write a function"}]
    for i in range(turns):
        messages.append({"role": "assistant", "content": f"answer {i}"})
        messages.append({"role": "user", "content": f"follow up {i}"})
    return conv_fingerprint(messages, "deepseek-chat", key)


def test_fingerprint_stable_across_turns() -> None:
    keys = {_conv(n) for n in range(6)}
    assert len(keys) == 1


def test_fingerprint_differs_on_key_id() -> None:
    assert _conv(0, "key-a") != _conv(0, "key-b")
    assert _conv(0, "key-a") != _conv(0, "")


def test_fingerprint_differs_on_model_and_prefix() -> None:
    base = [SYS, {"role": "user", "content": "Hello there"}]
    a = conv_fingerprint(base, "deepseek-chat", "k")
    b = conv_fingerprint(base, "deepseek-reasoner", "k")
    c = conv_fingerprint(
        [{"role": "system", "content": "Other system"}, base[1]], "deepseek-chat", "k"
    )
    assert a != b != c != a


def test_fingerprint_missing_system_or_user() -> None:
    # 不应抛异常，且能稳定返回
    assert conv_fingerprint([], "m", "") == conv_fingerprint([], "m", "")
    only_user = [{"role": "user", "content": "hi"}]
    assert conv_fingerprint(only_user, "m", "") != conv_fingerprint([], "m", "")


def test_fingerprint_whitespace_normalised() -> None:
    a = conv_fingerprint(
        [{"role": "system", "content": "a  b\n c"}, {"role": "user", "content": "u"}],
        "m", "",
    )
    b = conv_fingerprint(
        [{"role": "system", "content": " a b c "}, {"role": "user", "content": "u"}],
        "m", "",
    )
    assert a == b


def test_msg_count_guard_rule() -> None:
    """守卫规则：msg_count < 缓存值 → 历史被改写；相等（重试重发）与增长都属正常延续。"""
    cache = SessionCache()
    cache.put("conv", SessionState(session_id="s1", turn_count=1, msg_count=5))
    cached = cache.get("conv")
    assert cached is not None

    def stale(msg_count: int) -> bool:
        return bool(cached.msg_count and msg_count < cached.msg_count)

    assert stale(4), "消息数减少应判定为历史改写"
    assert not stale(5), "消息数相等（网络重试重发）不应误判"
    assert not stale(6), "消息数增长是正常延续"


def test_conv_key_store_and_hit_end_to_end(tmp_path) -> None:
    """模拟 _resolve_conversation 的 key 计算 + _store_after_response 的写回：
    第 1 轮存进去，第 2 轮用相同对话（消息更多）算出的 key 必须能命中。
    """
    cache = SessionCache(persist_path=str(tmp_path / "s.json"))
    system = {"role": "system", "content": "sys " + "x" * 300}
    u1 = {"role": "user", "content": "第一条需求"}

    # 第 1 轮：请求 messages = [system, u1]
    key1 = conv_fingerprint([system, u1], "deepseek-chat", "key-1")
    cache.put(key1, SessionState(session_id="sess-a", turn_count=1, msg_count=2))

    # 第 2 轮：客户端回传历史 + 新消息，key 必须相同
    t2 = [system, u1, {"role": "assistant", "content": "好的"},
          {"role": "user", "content": "继续"}]
    key2 = conv_fingerprint(t2, "deepseek-chat", "key-1")
    assert key1 == key2
    hit = cache.get(key2)
    assert hit is not None and hit.session_id == "sess-a", "第 2 轮应命中同一上游会话"

    # 另一个 key 的相同对话不能命中（租户隔离）
    assert conv_fingerprint(t2, "deepseek-chat", "key-2") != key1


def test_persist_throttle_but_flush_forces(tmp_path) -> None:
    """落盘节流：短时间内连续 put 不写盘，flush() 强制落盘后新实例可见。"""
    path = tmp_path / "s.json"
    cache = SessionCache(persist_path=str(path))
    cache.put("k1", SessionState(session_id="s1", turn_count=1, msg_count=1))
    first_size = path.stat().st_size if path.exists() else 0
    # 立刻再 put 一条：应被节流（文件不变）
    cache.put("k2", SessionState(session_id="s2", turn_count=1, msg_count=1))
    assert (path.stat().st_size if path.exists() else 0) == first_size, "节流期内不应重复写盘"
    cache.flush()
    assert SessionCache(persist_path=str(path)).get("k2") is not None, "flush 后应落盘"


def test_persist_roundtrip(tmp_path) -> None:
    path = tmp_path / "nested" / "sessions.json"
    c1 = SessionCache(persist_path=path)
    c1.put(
        "k1",
        SessionState(
            session_id="sid-1", turn_count=3, last_message_id="mid",
            msg_count=7, history_hash="h",
        ),
    )
    assert path.exists()

    c2 = SessionCache(persist_path=path)
    st = c2.get("k1")
    assert st is not None
    assert (st.session_id, st.turn_count, st.last_message_id) == ("sid-1", 3, "mid")
    assert (st.msg_count, st.history_hash) == (7, "h")

    c2.delete("k1")
    c3 = SessionCache(persist_path=path)
    assert c3.get("k1") is None


def test_persist_delete_by_session_id_and_clear(tmp_path) -> None:
    path = tmp_path / "s.json"
    c = SessionCache(persist_path=path)
    c.put("a", SessionState(session_id="x"))
    c.put("b", SessionState(session_id="x"))
    c.put("c", SessionState(session_id="y"))
    assert c.delete_by_session_id("x") == 2
    assert SessionCache(persist_path=path).size == 1

    c.clear()
    assert SessionCache(persist_path=path).size == 0


def test_corrupt_json_tolerated(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    cache = SessionCache(persist_path=path)
    assert cache.size == 0
    cache.put("k", SessionState(session_id="s"))
    assert cache.get("k") is not None


def test_missing_fields_skipped(tmp_path) -> None:
    path = tmp_path / "mix.json"
    path.write_text(
        json.dumps({
            "ok": {"session_id": "s1", "turn_count": 1},
            "no_id": {"turn_count": 2},
            "bad_type": {"session_id": 123},
            "not_dict": "oops",
        }),
        encoding="utf-8",
    )
    cache = SessionCache(persist_path=path)
    assert cache.size == 1
    st = cache.get("ok")
    assert st is not None and st.msg_count == 0 and st.history_hash == ""
