"""GLM（智谱清言）防风控：设备指纹一致性、会话复用、轮次与节奏配置。"""

from __future__ import annotations

import logging
import time

from webai2api.adapters.chatglm.client import (
    ChatGlmWebClient,
    _build_headers,
    _headers_for,
)
from webai2api.adapters.chatglm.config import GlmConfig, _stable_device_id


def test_headers_device_id_and_ua_are_stable() -> None:
    """同一账号多次构造请求头：device_id 与 UA 必须一致（这是防封核心）。"""
    h1 = _build_headers(
        "secret", "tok", device_id="dev-abc", user_agent="UA-X", device_model="Win32"
    )
    h2 = _build_headers(
        "secret", "tok", device_id="dev-abc", user_agent="UA-X", device_model="Win32"
    )
    assert h1["X-Device-Id"] == h2["X-Device-Id"] == "dev-abc"
    assert h1["User-Agent"] == h2["User-Agent"] == "UA-X"
    assert h1["X-Device-Model"] == "Win32"
    # 每请求变化的字段保持变化（否则反而不像真实客户端）
    assert h1["X-Request-Id"] != h2["X-Request-Id"]
    assert h1["X-Sign"] != h2["X-Sign"]


def test_headers_device_id_isolated_between_accounts() -> None:
    h1 = _build_headers("s", "t", device_id="dev-abc", user_agent="UA")
    h2 = _build_headers("s", "t", device_id="dev-xyz", user_agent="UA")
    assert h1["X-Device-Id"] != h2["X-Device-Id"]


def test_headers_without_device_id_still_stable() -> None:
    """未显式提供 device_id 时按 UA 稳定派生，绝不每次随机。"""
    a = _build_headers("s", "t", user_agent="UA-Y")
    b = _build_headers("s", "t", user_agent="UA-Y")
    assert a["X-Device-Id"] == b["X-Device-Id"]


def test_stable_device_id_from_profile_or_account() -> None:
    acc = {"id": "acc-A", "profile": '{"device_id":"from-profile"}'}
    assert _stable_device_id(acc) == "from-profile"

    broken = {"id": "acc-B", "profile": "bad json"}
    empty = {"id": "acc-B", "profile": "{}"}
    assert _stable_device_id(broken) == _stable_device_id(empty)
    assert _stable_device_id(empty) != _stable_device_id(
        {"id": "acc-Z", "profile": "{}"}
    )


def test_headers_for_uses_account_config() -> None:
    cfg = GlmConfig(
        refresh_token="t",
        device_id="cfg-dev",
        user_agent="cfg-UA",
        device_model="Win32",
    )
    h = _headers_for(cfg, "tok")
    assert h["X-Device-Id"] == "cfg-dev"
    assert h["User-Agent"] == "cfg-UA"
    assert h["Authorization"] == "Bearer tok"


def _client(cfg: GlmConfig) -> ChatGlmWebClient:
    return ChatGlmWebClient(cfg, logging.getLogger("test.glm"))


def test_conv_key_stable_across_turns_and_isolated() -> None:
    cfg = GlmConfig(refresh_token="t")
    c = _client(cfg)
    system = {"role": "system", "content": "sys " + "x" * 300}
    u1 = {"role": "user", "content": "第一条需求"}
    t2 = [
        system,
        u1,
        {"role": "assistant", "content": "好"},
        {"role": "user", "content": "继续"},
    ]

    twin2 = [
        system,
        u1,
        {"role": "assistant", "content": "不同的回复"},
        {"role": "user", "content": "继续"},
    ]
    k1 = c._conv_key(
        {"model": "GLM-5.3", "messages": [system, u1], "_wapi_key_id": "key-A"}
    )
    k2 = c._conv_key({"model": "GLM-5.3", "messages": t2, "_wapi_key_id": "key-A"})
    k3 = c._conv_key({"model": "GLM-5.3", "messages": t2, "_wapi_key_id": "key-B"})
    k4 = c._conv_key(
        {"model": "GLM-5.3-Flash", "messages": t2, "_wapi_key_id": "key-A"}
    )
    k5 = c._conv_key({"model": "GLM-5.3", "messages": twin2, "_wapi_key_id": "key-A"})

    assert k2 == c._conv_key(
        {"model": "GLM-5.3", "messages": t2, "_wapi_key_id": "key-A"}
    )  # 同对话多轮稳定
    assert k3 != k2  # 不同 Key 隔离
    assert k4 != k2  # 不同模型隔离
    assert k5 != k2  # 关键防线：同开场白不同首轮回复 → 不同会话（防串台）


def test_conv_cache_turn_limit_and_ttl() -> None:
    cfg = GlmConfig(refresh_token="t", max_turns_per_session=2, session_cache_ttl=43200)
    c = _client(cfg)
    c._store_conv("k", "conv-1")  # turns=1
    assert c._cached_conv("k") == "conv-1"
    c._store_conv("k", "conv-1")  # turns=2
    assert c._cached_conv("k") == ""  # 达上限 → 新建

    c2 = _client(cfg)
    c2._store_conv("k2", "conv-2")
    with c2._conv_cache._lock:
        c2._conv_cache._cache["k2"] = ("conv-2", 0.0, 1, "")
    assert c2._cached_conv("k2") == ""  # 过期 → 新建


def test_body_carries_conversation_id() -> None:
    """回归：曾经漏传 conv_id，导致多轮复用实际不生效。"""
    cfg = GlmConfig(refresh_token="t")
    c = _client(cfg)
    body = c._body("内容", "GLM-5.3", conv_id="conv-9")
    assert body["conversation_id"] == "conv-9"
    assert c._body("内容", "GLM-5.3")["conversation_id"] == ""


def test_default_antiban_knobs() -> None:
    cfg = GlmConfig()
    assert cfg.session_cache_ttl == 43200
    assert cfg.session_cache_max_size == 2000
    assert cfg.long_pause_chance == 0.02
    assert cfg.min_request_interval == 2.5
    assert cfg.max_turns_per_session == 80
    assert cfg.max_tokens_per_session == 200000


def test_conv_cache_persists_across_instances(tmp_path) -> None:
    """会话缓存落盘：新实例（模拟重启）仍能复用同一 conversation_id。"""
    path = str(tmp_path / "glm.json")
    cfg = GlmConfig(refresh_token="t", session_store_path=path)
    c1 = _client(cfg)
    c1._store_conv("conv-key", "conv-abc")
    c1.flush_cache()

    c2 = _client(cfg)
    assert c2._cached_conv("conv-key") == "conv-abc", "重启后应复用同一会话"


def test_conv_cache_corrupt_file_is_tolerated(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    cfg = GlmConfig(refresh_token="t", session_store_path=str(path))
    c = _client(cfg)  # 不应抛异常
    assert c._cached_conv("nope") == ""


def test_cleanup_stale_sessions_removes_expired(tmp_path) -> None:
    cfg = GlmConfig(
        refresh_token="t",
        session_store_path=str(tmp_path / "s.json"),
        session_cache_ttl=10,
        max_turns_per_session=40,
    )
    c = _client(cfg)
    c._store_conv("fresh", "conv-fresh")
    with c._conv_cache._lock:
        c._conv_cache._cache["old"] = ("conv-old", 0.0, 1, "")
        c._conv_cache._cache["full"] = ("conv-full", time.time(), 40, "")
    # 打桩：不真的发删除请求
    deleted: list[str] = []
    c._access = lambda: "tok"  # type: ignore[assignment]
    c._delete_conversation = lambda cid, tok: deleted.append(cid)  # type: ignore[assignment]
    c.cleanup_stale_sessions()
    assert "old" not in c._conv_cache._cache
    assert "full" not in c._conv_cache._cache
    assert "fresh" in c._conv_cache._cache
    assert set(deleted) == {"conv-old", "conv-full"}
