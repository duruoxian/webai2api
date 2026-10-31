"""2026-11-01 Anthropic 适配层夜间修复回归。

- A1：并行 tool_use 流（GW-1 残余）——_switch 同类短路使第二条调用的
  content_block_start 不发、参数拼进第一条。Claude Code agent 循环高频触发。
- D8：args 被思考/正文打断后重开无名块 → 用记住的 meta 重开。
- A2：纯图片 user 消息整条蒸发 → 降级占位文本保住轮次。
- B6：tool_choice 官方对象形态（{"type":"any"}）此前静默丢失。
- B2：tool_result 乱序回传时结果互换 → id 精确配对。
"""

from __future__ import annotations

import json

from webai2api.anthropic_adapter import (
    AnthropicStreamTranslator,
    anthropic_to_openai,
)


def _chunk(delta: dict, finish=None) -> dict:
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def _parse(events: list[str]) -> list[tuple[str, dict]]:
    out = []
    for e in events:
        for line in e.strip().splitlines():
            if line.startswith("data: "):
                out.append((json.loads(line[6:]).get("type"), json.loads(line[6:])))
    return out


def test_parallel_tool_use_two_blocks():
    tr = AnthropicStreamTranslator("m")
    events: list[str] = []
    events += tr.feed(_chunk({"tool_calls": [
        {"index": 0, "id": "call_A", "type": "function",
         "function": {"name": "f", "arguments": ""}},
    ]}))
    events += tr.feed(_chunk({"tool_calls": [
        {"index": 0, "id": "call_A", "type": "function",
         "function": {"arguments": "{\"x\":1}"}},
    ]}))
    events += tr.feed(_chunk({"tool_calls": [
        {"index": 1, "id": "call_B", "type": "function",
         "function": {"name": "g", "arguments": ""}},
    ]}))
    events += tr.feed(_chunk({"tool_calls": [
        {"index": 1, "id": "call_B", "type": "function",
         "function": {"arguments": "{\"y\":2}"}},
    ]}))
    events += tr.finish()
    ev = _parse(events)

    starts = [d for t, d in ev if t == "content_block_start"]
    assert len(starts) == 2, f"必须有两条 tool_use 块: {starts}"
    assert starts[0]["content_block"]["name"] == "f"
    assert starts[1]["content_block"]["name"] == "g", "第二条调用 name 不能丢"
    assert starts[0]["index"] == 0 and starts[1]["index"] == 1

    stops = [d for t, d in ev if t == "content_block_stop"]
    assert [s["index"] for s in stops] == [0, 1], "旧块必须先闭合"

    # 参数归位：x 进块 0，y 进块 1
    args: dict[int, str] = {}
    for t, d in ev:
        if t == "content_block_delta" and d["delta"].get("type") == "input_json_delta":
            args[d["index"]] = args.get(d["index"], "") + d["delta"]["partial_json"]
    assert json.loads(args[0]) == {"x": 1}
    assert json.loads(args[1]) == {"y": 2}


def test_args_interrupted_by_text_reopens_named():
    tr = AnthropicStreamTranslator("m")
    events: list[str] = []
    events += tr.feed(_chunk({"tool_calls": [
        {"index": 0, "id": "call_A", "type": "function",
         "function": {"name": "shell", "arguments": "{\"command\": \"ls"}},
    ]}))
    events += tr.feed(_chunk({"reasoning_content": "想一想"}))
    events += tr.feed(_chunk({"tool_calls": [
        {"index": 0, "id": "call_A", "type": "function",
         "function": {"arguments": " -la\"}"}},
    ]}))
    events += tr.finish()
    ev = _parse(events)
    starts = [d for t, d in ev if t == "content_block_start"]
    tool_blocks = [s for s in starts if s["content_block"]["type"] == "tool_use"]
    assert len(tool_blocks) == 2, "打断后必须重开一个 tool_use 块"
    reopened = tool_blocks[1]["content_block"]
    assert reopened["name"] == "shell", f"重开块必须带 name: {reopened}"
    assert reopened["id"] == "call_A", "重开块必须带原 id"


def test_image_only_message_becomes_placeholder():
    body = {
        "model": "m",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "data": "xxx"}},
            ]},
        ],
    }
    out = anthropic_to_openai(body)
    user_msgs = [m for m in out["messages"] if m["role"] == "user"]
    assert len(user_msgs) == 1, "纯图片消息绝不能整条蒸发"
    assert "[图片]" in user_msgs[0]["content"]


def test_mixed_image_text_message_keeps_text_and_placeholder():
    body = {
        "model": "m",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "看这张图"},
                {"type": "image", "source": {"type": "base64", "data": "xxx"}},
            ]},
        ],
    }
    out = anthropic_to_openai(body)
    content = out["messages"][-1]["content"]
    assert "看这张图" in content and "[图片]" in content


def test_tool_choice_official_dict_forms():
    base = {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]}
    assert anthropic_to_openai({**base, "tool_choice": {"type": "any"}})[
        "tool_choice"] == "required"
    assert anthropic_to_openai({**base, "tool_choice": {"type": "auto"}})[
        "tool_choice"] == "auto"
    tc = anthropic_to_openai({
        **base, "tool_choice": {"type": "tool", "name": "shell"}
    })["tool_choice"]
    assert tc == {"type": "function", "function": {"name": "shell"}}


def test_tool_result_out_of_order_exact_pairing():
    body = {
        "model": "m",
        "max_tokens": 10,
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tA", "name": "f", "input": {}},
                {"type": "tool_use", "id": "tB", "name": "g", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tB", "content": "B的结果"},
                {"type": "tool_result", "tool_use_id": "tA", "content": "A的结果"},
            ]},
        ],
    }
    out = anthropic_to_openai(body)
    tool_msgs = [m for m in out["messages"] if m["role"] == "tool"]
    by_id = {m["tool_call_id"]: m["content"] for m in tool_msgs}
    assert by_id["tA"] == "A的结果", "乱序回传不得互换结果"
    assert by_id["tB"] == "B的结果"
