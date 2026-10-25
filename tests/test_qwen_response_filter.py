"""千问多路候选回答过滤器测试（qwen3.8-max 语无伦次的根治）。

背景（移植自 Rfym21/Qwen2API 的 createUpstreamResponseFilter）：
上游偶尔对同一次请求开启多路候选回答——多个 response.created
（response_index 0/1、response_id 各不同），随后两路增量帧交错到达。
不区分 response_id 拼接 = 两路内容搅在一起（语无伦次）。
"""
from __future__ import annotations

from webai2api.adapters.qwen.client import UpstreamResponseFilter


def _created(rid: str, index) -> dict:
    ev: dict = {"response.created": {"response_id": rid}}
    if index is not None:
        ev["response.created"]["response_index"] = index
    return ev


def _delta(rid: str, text: str) -> dict:
    return {"response_id": rid,
            "choices": [{"delta": {"content": text, "phase": "answer"}}]}


def test_interleaved_candidates_only_primary_kept():
    """两路候选交错到达：只保留 index=0 那一路，另一路整条丢弃。"""
    f = UpstreamResponseFilter()
    assert f.accept(_created("A", 0))
    assert not f.accept(_created("B", 1))         # A 已锁定，B 的 created 也整条丢
    assert f.accept(_delta("A", "巴黎"))
    assert not f.accept(_delta("B", "柏林"))     # 备用候选 → 丢
    assert f.accept(_delta("A", "是法国首都"))
    assert not f.accept(_delta("B", "是德国首都"))
    assert f.accept({"choices": [{"delta": {"content": "。", "phase": "answer"}}]})


def test_lock_on_index_zero_even_if_backup_speaks_first():
    """备用候选先吐 delta 也不能抢锁：index=0 才是官方主回答。"""
    f = UpstreamResponseFilter()
    assert f.accept(_created("B", 1))
    assert not f.accept(_delta("B", "抢跑的备用内容"))   # 已知非 0 → 拒绝
    assert f.accept(_created("A", 0))
    assert f.accept(_delta("A", "真正的主回答"))
    assert not f.accept(_delta("B", "迟到的备用内容"))


def test_old_protocol_locks_first_stream_with_response_id():
    """没有 response.created 元数据：锁定第一路带 response_id 的正文。"""
    f = UpstreamResponseFilter()
    assert f.accept(_delta("A", "第一路"))
    assert not f.accept(_delta("B", "第二路"))
    assert f.accept(_delta("A", "继续"))


def test_no_response_id_passes_all():
    """上游完全不带 response_id：一律放行（老行为兜底）。"""
    f = UpstreamResponseFilter()
    assert f.accept({"choices": [{"delta": {"content": "a"}}]})
    assert f.accept({"choices": [{"delta": {"content": "b"}}]})


def test_created_without_index_locks_first_then_filters():
    """created 事件缺 response_index：锁第一路，后续不同 id 丢弃。"""
    f = UpstreamResponseFilter()
    assert f.accept(_created("A", None))
    assert f.accept(_delta("A", "内容"))
    assert not f.accept(_delta("B", "别的内容"))


def test_error_frames_unaffected():
    """错误帧不带 response_id，永远放行（错误处理在上层）。"""
    f = UpstreamResponseFilter()
    assert f.accept(_created("A", 0))
    assert f.accept({"error": {"code": "RATE_LIMIT"}})
    assert f.accept({"success": False, "data": {"code": "CHAT_NOT_FOUND"}})
    assert f.accept(_delta("A", "x"))
    assert not f.accept(_delta("B", "y"))


def test_non_dict_passes():
    f = UpstreamResponseFilter()
    assert f.accept("garbage")  # type: ignore[arg-type]
