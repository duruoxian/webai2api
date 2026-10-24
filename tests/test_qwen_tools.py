"""千问工具调用（OpenAI tools 协议）测试。

千问走 Hermes 风格 <tool_call> 标签（Qwen 系列训练时见过该格式），
<<<TOOL_CALL>>> 行内标记作兜底。这些测试保证：
提示词注入正确、增量组装带 <tool_response>、标签/兜底都能解析、
非法工具名被拒绝、普通回答不受影响。
"""
from __future__ import annotations

import json

from webai2api.adapters.qwen.client import (
    QWEN_TOOL_MARKER,
    QWEN_TOOL_TAG,
    _active_tools,
    _inject_tools_messages,
    _inject_tools_system,
    _tools_prompt,
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


def _tag_call(name: str, args: dict) -> str:
    return ("<tool_call>\n"
            + json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
            + "\n</tool_call>")


def _marker_line(name: str, args: dict) -> str:
    return QWEN_TOOL_MARKER + json.dumps({"name": name, "arguments": args},
                                         ensure_ascii=False)


# ── 提示词注入 ────────────────────────────────────────────

def test_tools_prompt_renders_json_schema():
    spec = _tools_prompt(TOOLS)
    assert '"write"' in spec
    assert "filePath" in spec


def test_inject_system_creates_instruction():
    out = _inject_tools_system("", TOOLS)
    assert out.startswith("# Tools")
    assert QWEN_TOOL_TAG in out and "tool_response" in out


def test_inject_system_merges_existing_system():
    out = _inject_tools_system("你是本地助手", TOOLS)
    assert out.startswith("你是本地助手")
    assert "<tool_call>" in out


def test_inject_messages_merges_first_system():
    msgs = [{"role": "system", "content": "你是本地助手"},
            {"role": "user", "content": "hi"}]
    out = _inject_tools_messages(msgs, TOOLS)
    assert len(out) == 2
    assert out[0]["content"].startswith("你是本地助手")
    assert "<tool_call>" in out[0]["content"]


def test_active_tools_respects_tool_choice_none():
    assert _active_tools({"tools": TOOLS}) == TOOLS
    assert _active_tools({"tools": TOOLS, "tool_choice": "none"}) == []


# ── 解析（走共享桥接，校验千问的标签 + 兜底标记组合） ────

def _parse(text: str):
    from webai2api.adapters import tool_bridge
    return tool_bridge.parse_tool_calls(
        text, TOOLS, tag=QWEN_TOOL_TAG, marker=QWEN_TOOL_MARKER)


def test_parse_tag_call():
    calls = _parse("我来写入文件\n" + _tag_call("write", {"filePath": "D:\\a.md"}))
    assert len(calls) == 1
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "write"
    assert json.loads(calls[0]["function"]["arguments"]) == {"filePath": "D:\\a.md"}


def test_parse_marker_fallback():
    calls = _parse("……\n" + _marker_line("write", {"filePath": "a"}))
    assert calls and calls[0]["function"]["name"] == "write"


def test_parse_rejects_unknown_tool():
    assert _parse(_tag_call("rm_rf", {})) == []
    assert _parse(_marker_line("shutdown", {})) == []


def test_parse_ignores_plain_answer():
    assert _parse("这是普通回答，没有工具。") == []
