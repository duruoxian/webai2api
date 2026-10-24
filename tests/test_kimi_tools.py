"""Kimi 工具调用（OpenAI tools 协议）测试。

网页版 Kimi 没有原生 function calling，靠 system prompt 约定标记 + 解析实现。
这些测试保证：工具清单能注入、模型输出能被解析成标准 tool_calls、
非法工具名不被放行、普通回答不会被误伤。
"""
from __future__ import annotations

import json

from webai2api.adapters.kimi.client import (
    TOOL_MARKER,
    _inject_tools,
    _parse_tool_calls,
    _strip_tool_block,
)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "write",
        "description": "把内容写入本地文件",
        "parameters": {
            "type": "object",
            "properties": {
                "filePath": {"type": "string", "description": "绝对路径"},
                "content": {"type": "string", "description": "文件内容"},
            },
            "required": ["filePath", "content"],
        },
    },
}]


def _call_line(name: str, args: dict) -> str:
    return TOOL_MARKER + json.dumps({"name": name, "arguments": args},
                                    ensure_ascii=False)


def test_inject_tools_creates_system_message():
    out = _inject_tools([{"role": "user", "content": "写个文件"}], TOOLS)
    assert out[0]["role"] == "system"
    assert "write" in out[0]["content"]
    assert TOOL_MARKER in out[0]["content"]
    assert out[-1]["content"] == "写个文件"


def test_inject_tools_merges_existing_system():
    msgs = [{"role": "system", "content": "你是助手"},
            {"role": "user", "content": "hi"}]
    out = _inject_tools(msgs, TOOLS)
    assert len(out) == 2
    assert out[0]["content"].startswith("你是助手")
    assert "write" in out[0]["content"]


def test_inject_tools_noop_without_tools():
    msgs = [{"role": "user", "content": "hi"}]
    assert _inject_tools(msgs, []) == msgs


def test_parse_tool_call_returns_openai_shape():
    text = "好的\n" + _call_line("write", {"filePath": "D:\\a.md", "content": "hey"})
    calls = _parse_tool_calls(text, TOOLS)
    assert len(calls) == 1
    c = calls[0]
    assert c["type"] == "function"
    assert c["function"]["name"] == "write"
    assert c["id"].startswith("call_")
    args = json.loads(c["function"]["arguments"])
    assert args["filePath"] == "D:\\a.md"


def test_parse_rejects_unknown_tool():
    text = _call_line("rm_rf", {"path": "C:\\"})
    assert _parse_tool_calls(text, TOOLS) == []


def test_parse_ignores_plain_answer():
    assert _parse_tool_calls("这是一段普通回答", TOOLS) == []


def test_parse_only_first_call():
    text = _call_line("write", {"filePath": "a", "content": "1"}) + "\n" + \
        _call_line("write", {"filePath": "b", "content": "2"})
    assert len(_parse_tool_calls(text, TOOLS)) == 1


def test_strip_tool_block_removes_marker_line():
    text = "好的\n" + _call_line("write", {"filePath": "a", "content": "1"})
    assert TOOL_MARKER not in _strip_tool_block(text)
    assert _strip_tool_block("普通回答") == "普通回答"
