"""deepseek 并行工具调用 index 回归测试。

真机事故（2026-10-27）：流式检测器按 `</invoke>` 逐条切块、一次 action
只含一条调用；chunk 组装层若每次都从 index=0 起编，客户端会把 N 条调用
全部堆到第 0 槽——arguments 变成 N 个 JSON 首尾相接的非法串、name 全
变成第一条的名字。修复后 chunk 的 tool_calls[].index 必须全局递增。
"""

from __future__ import annotations

import json

from webai2api.adapters.deepseek.services import tool_calling as tc
from webai2api.adapters.deepseek.services.deepseek_client import (
    _make_tool_call_chunks,
)

D = "｜｜DSML｜｜"  # 全角竖线，与提示词同先验


def _dsml_three_invokes() -> str:
    return (
        f"<{D} calls>\n"
        f'<{D} invoke name="glob">\n'
        f'<{D} parameter name="pattern" string="true">**/*.py</{D} parameter>\n'
        f"</{D} invoke>\n"
        f'<{D} invoke name="grep">\n'
        f'<{D} parameter name="pattern" string="true">hello</{D} parameter>\n'
        f"</{D} invoke>\n"
        f'<{D} invoke name="shell">\n'
        f'<{D} parameter name="command" string="true">ls -la</{D} parameter>\n'
        f"</{D} invoke>\n"
        f"</{D} calls>\n"
    )


def _tools():
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in ("glob", "grep", "shell")
    ]


def _merge_like_client(chunks):
    """按 OpenAI 流式口径合并：以 tool_calls[].index 累加 id/name/arguments。"""
    slots = {}
    for raw in chunks:
        data = json.loads(raw[len("data: ") :].strip())
        for choice in data.get("choices") or []:
            for t in (choice.get("delta") or {}).get("tool_calls") or []:
                idx = t["index"]
                slot = slots.setdefault(
                    idx, {"id": None, "name": None, "arguments": ""}
                )
                if t.get("id"):
                    slot["id"] = t["id"]
                fn = t.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                slot["arguments"] += fn.get("arguments") or ""
    return slots


def test_chunk_index_continues_from_base():
    a = tc.ParsedToolCall(id="c1", name="glob", arguments='{"pattern": "**/*.py"}')
    b = tc.ParsedToolCall(id="c2", name="grep", arguments='{"pattern": "hello"}')
    first = _make_tool_call_chunks([a], "m", "cid", base_index=0)
    second = _make_tool_call_chunks([b], "m", "cid", base_index=1)
    slots = _merge_like_client(first + second)
    assert sorted(slots) == [0, 1]
    assert slots[0]["name"] == "glob"
    assert slots[1]["name"] == "grep"
    json.loads(slots[0]["arguments"])
    json.loads(slots[1]["arguments"])


def test_streaming_three_invokes_keep_separate_slots():
    detector = tc.StreamingToolDetector(_tools(), require_marker=False)
    emitted = []
    chunks = []
    for action in detector.feed(_dsml_three_invokes()):
        if action.kind == "tool_calls" and action.tool_calls:
            chunks.extend(
                _make_tool_call_chunks(
                    action.tool_calls, "m", "cid", base_index=len(emitted)
                )
            )
            emitted.extend(action.tool_calls)
    for action in detector.finalize():
        if action.kind == "tool_calls" and action.tool_calls:
            chunks.extend(
                _make_tool_call_chunks(
                    action.tool_calls, "m", "cid", base_index=len(emitted)
                )
            )
            emitted.extend(action.tool_calls)

    assert [c.name for c in emitted] == ["glob", "grep", "shell"]
    slots = _merge_like_client(chunks)
    assert sorted(slots) == [0, 1, 2]
    names = [slots[i]["name"] for i in (0, 1, 2)]
    assert names == ["glob", "grep", "shell"]
    args = [json.loads(slots[i]["arguments"]) for i in (0, 1, 2)]
    assert args[0] == {"pattern": "**/*.py"}
    assert args[1] == {"pattern": "hello"}
    assert args[2] == {"command": "ls -la"}


def test_single_action_multi_calls_indices_monotonic():
    """finalize 兜底一次给出多条调用时，index 也要从 base 起连续递增。"""
    calls = [
        tc.ParsedToolCall(id="a", name="glob", arguments="{}"),
        tc.ParsedToolCall(id="b", name="shell", arguments="{}"),
    ]
    chunks = _make_tool_call_chunks(calls, "m", "cid", base_index=5)
    slots = _merge_like_client(chunks)
    assert sorted(slots) == [5, 6]
