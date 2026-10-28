"""deepseek 用量口径回归（2026-10-28 用户报障：面板「十几秒产出几万 token」）。

根因：上游 BATCH 事件的 accumulated_token_usage 是**会话累计值**
（含历史各轮 prompt+completion，随轮次单调增长——usage_log 实测同一
会话 7136 → 14864 → … → 97171），旧实现把它当单轮 completion 下发，
且在会话缓存里与 prev 二次相加（轮换上限被毒化、提前触发）。

修复：completion 一律本地估算（正文+思考）；会话累计值只作「快满了
就换」的满量信号，到货时整体替换、绝不累加。
"""

from __future__ import annotations

import logging

import pytest

from webai2api.adapters.deepseek import config as ds_config
from webai2api.adapters.deepseek.services import tool_calling as tc
from webai2api.adapters.deepseek.services.deepseek_client import (
    DeepSeekWebClient,
    _estimate_tokens,
)
from webai2api.adapters.deepseek.services.session_cache import SessionState


@pytest.fixture()
def client():
    cfg = ds_config.AppConfig(
        env_file_path=None, env_file_created=False, host="h", port=0,
        api_prefix="/v1", log_level="ERROR", debug_dump_all=False,
        request_timeout=10, server_api_keys=[], auth_token="t", device_id="d",
        settings_token="", cookie="", base_url="http://127.0.0.1:1",
        wasm_path="", default_model="deepseek-chat", delete_conversation=False,
        max_concurrency=1, queue_wait_timeout=10, admin_password="",
        anti_rate_limit_enabled=False, session_cache_enabled=True,
        session_cache_ttl=43200, session_cache_max_size=100,
    )
    return DeepSeekWebClient(config=cfg, logger=logging.getLogger("test.ds.usage"))


class _FakeCache:
    def __init__(self, state: SessionState | None = None):
        self.state = state
        self.put_calls: list[tuple[str, SessionState]] = []

    def get(self, key):
        return self.state

    def put(self, key, state):
        self.put_calls.append((key, state))
        self.state = state


def _resolved() -> dict:
    return {
        "mode": "cached",
        "model": "deepseek-chat",
        "cache_key": "conv-test",
        "full_prefix": [{"role": "user", "content": "q"}],
        "last_message": {"role": "user", "content": "q"},
        "last_content": "q",
        "turn_count": 3,
        "msg_count": 7,
        "ds_payload": {"prompt": "历史+增量"},
    }


def test_store_upstream_total_replaces_not_accumulates(client):
    """上游会话累计值到货：整体替换，绝不与 prev 或本轮估算相加。"""
    cache = _FakeCache(SessionState(
        session_id="sess-old", turn_count=3, used_tokens=7000,
    ))
    client.session_cache = cache
    client._store_after_response(
        _resolved(), "sess-1", "答", "123",
        used_tokens=1234, upstream_total=50000,
    )
    key, st = cache.put_calls[-1]
    assert key == "conv-test"
    assert st.used_tokens == 50000
    assert st.turn_count == 4


def test_store_falls_back_to_prev_plus_estimate(client):
    cache = _FakeCache(SessionState(
        session_id="sess-old", turn_count=3, used_tokens=7000,
    ))
    client.session_cache = cache
    client._store_after_response(
        _resolved(), "sess-1", "答", "123",
        used_tokens=1234, upstream_total=0,
    )
    assert cache.put_calls[-1][1].used_tokens == 8234


def test_store_first_turn_no_prev(client):
    cache = _FakeCache(None)
    client.session_cache = cache
    client._store_after_response(
        _resolved(), "sess-1", "答", None,
        used_tokens=300, upstream_total=0,
    )
    assert cache.put_calls[-1][1].used_tokens == 300


def test_non_stream_completion_counts_text_and_reasoning(client):
    """非流式 completion = 正文估算 + 思考估算（不再看上游累计值）。"""
    resolved = _resolved()
    resp = client._build_non_stream_result(
        {"model": "deepseek-chat"}, resolved,
        full_text="a" * 400,          # 400 ASCII 字符 ≈ 100 token
        full_reasoning="思" * 300,    # 300 CJK 字符 ≈ 200 token
        completion_id="cid",
    )
    usage = resp["usage"]
    assert usage["completion_tokens"] == 100 + 200
    assert usage["prompt_tokens"] == max(1, _estimate_tokens("历史+增量"))


def test_tool_call_completion_counts_args_and_reasoning(client):
    calls = [tc.ParsedToolCall(
        id="c1", name="shell",
        arguments='{"command": "' + "x" * 200 + '"}',  # ≈ 200+/4 token
    )]
    resp = client._build_tool_call_result(
        {"model": "deepseek-chat"}, _resolved(), calls,
        full_reasoning="思" * 150,  # 150 CJK ≈ 100 token
        completion_id="cid",
    )
    usage = resp["usage"]
    assert usage["completion_tokens"] == (
        max(1, (len(calls[0].arguments)) // 4) + 100
    )
    assert resp["choices"][0]["finish_reason"] == "tool_calls"
