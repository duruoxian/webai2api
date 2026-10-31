"""deepseek translator 多 fragment 事件回归（2026-11-01 审计 D3）。

- APPEND response/fragments 一次携带多个 fragment（THINK+RESPONSE 同帧）：
  此前循环覆盖 out_delta 只留最后一个，前面的内容整体丢失 → 合并进同一
  delta（reasoning_content / content 两键并存，事件粒度顺序保真）。
- 嵌套完整 response 快照内多个 fragment：此前处理完第一个就 return，
  同快照其余 fragment 不再重发即永久丢字 → 全量合并。
- 快照 fragment 全部发过但状态 FINISHED：补发 finish（旧实现落空）。
"""

from __future__ import annotations

from webai2api.adapters.deepseek.services.translator import (
    StreamState,
    convert_deepseek_to_openai_chunk,
)


def _delta(chunk: dict) -> dict:
    return chunk["choices"][0]["delta"]


def _finish(chunk: dict):
    return chunk["choices"][0]["finish_reason"]


def test_append_fragments_multi_types_merged():
    st = StreamState()
    chunk = convert_deepseek_to_openai_chunk(
        {
            "p": "response/fragments",
            "o": "APPEND",
            "v": [
                {"id": "f1", "type": "THINK", "content": "先想想"},
                {"id": "f2", "type": "RESPONSE", "content": "答案正文"},
            ],
        },
        "deepseek-chat",
        st,
        "cmpl-x",
    )
    d = _delta(chunk)
    assert d.get("reasoning_content") == "先想想", f"THINK 不能丢: {d}"
    assert d.get("content") == "答案正文", f"RESPONSE 不能丢: {d}"
    assert st.last_fragment_type == "RESPONSE", "last_fragment_type 取最后一个"


def test_append_fragments_single_unchanged():
    st = StreamState()
    chunk = convert_deepseek_to_openai_chunk(
        {
            "p": "response/fragments",
            "o": "APPEND",
            "v": [{"id": "f1", "type": "RESPONSE", "content": "只有一段"}],
        },
        "deepseek-chat",
        st,
        "cmpl-x",
    )
    assert _delta(chunk).get("content") == "只有一段"
    assert "reasoning_content" not in _delta(chunk)


def test_nested_snapshot_all_fragments_emitted():
    st = StreamState()
    chunk = convert_deepseek_to_openai_chunk(
        {
            "v": {
                "response": {
                    "status": "FINISHED",
                    "fragments": [
                        {"id": "f1", "type": "THINK", "content": "思考A"},
                        {"id": "f2", "type": "RESPONSE", "content": "正文B"},
                        {"id": "f3", "type": "RESPONSE", "content": "续C"},
                    ],
                }
            }
        },
        "deepseek-chat",
        st,
        "cmpl-x",
    )
    d = _delta(chunk)
    assert d.get("reasoning_content") == "思考A"
    assert d.get("content") == "正文B续C", f"快照内全部 fragment 必须合并: {d}"
    assert _finish(chunk) == "stop", "快照带 FINISHED 要出 finish"
    # 已发快照不重发
    st2 = StreamState()
    st2.emitted_fragment_ids = {"f1", "f2", "f3"}
    st2.finished_sent = True
    assert convert_deepseek_to_openai_chunk(
        {
            "v": {
                "response": {
                    "status": "FINISHED",
                    "fragments": [
                        {"id": "f1", "type": "THINK", "content": "思考A"},
                        {"id": "f2", "type": "RESPONSE", "content": "正文B"},
                    ],
                }
            }
        },
        "deepseek-chat",
        st2,
        "cmpl-x",
    ) is None


def test_nested_snapshot_all_emitted_but_finished_still_stops():
    """快照 fragment 全部发过但状态 FINISHED：旧实现直接落 None，finish 丢。"""
    st = StreamState()
    st.emitted_fragment_ids = {"f1"}
    chunk = convert_deepseek_to_openai_chunk(
        {
            "v": {
                "response": {
                    "status": "FINISHED",
                    "fragments": [{"id": "f1", "type": "RESPONSE", "content": "已发"}],
                }
            }
        },
        "deepseek-chat",
        st,
        "cmpl-x",
    )
    assert chunk is not None
    assert _finish(chunk) == "stop"
    assert _delta(chunk).get("content") in (None, "")
