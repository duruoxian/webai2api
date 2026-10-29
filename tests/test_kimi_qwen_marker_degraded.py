"""kimi/qwen 工具标记降级形态零残留回归（对齐 GLM/deepseek 同族加固）。

共享机制（碎片正则/宽松锚/半截扣留）的端到端行为由 test_glm_marker_degraded
覆盖；本文件锁定两家适配器的接入点：
- kimi：_strip_tool_block 归一+清扫、_parse_tool_calls 归一后解析；
- qwen：标签残迹清扫（<tool_call>/</tool_call>/tool_call>）、剥除、解析；
- tool_bridge：标签漏写闭合时从 opener 截断。
"""
from __future__ import annotations

from webai2api.adapters.kimi.client import (
    TOOL_MARKER as KIMI_MARKER,
    _parse_tool_calls as kimi_parse,
    _strip_tool_block as kimi_strip,
)
from webai2api.adapters.qwen.client import (
    QWEN_TOOL_MARKER,
    _parse_tool_calls as qwen_parse,
    _strip_tool_block as qwen_strip,
)
from webai2api.adapters import tool_bridge

_TOOLS = [{
    "type": "function",
    "function": {
        "name": "write",
        "description": "写文件",
        "parameters": {
            "type": "object",
            "properties": {"filePath": {"type": "string"},
                           "content": {"type": "string"}},
            "required": ["filePath", "content"],
        },
    },
}]


def _kimi_dsml(path: str, content: str) -> str:
    return (
        "<<<TOOL_CALL>>>\n<|DSML|tool_calls>\n"
        '<|DSML|invoke name="write">\n'
        f'<|DSML|parameter name="filePath"><![CDATA[{path}]]></|DSML|parameter>\n'
        f'<|DSML|parameter name="content"><![CDATA[{content}]]></|DSML|parameter>\n'
        "</|DSML|invoke>\n</|DSML|tool_calls>"
    )


def test_kimi_degraded_marker_stripped_and_parsed():
    degraded = _kimi_dsml("D:/x.txt", "hi").replace(
        "<<<TOOL_CALL>>>", "<<TOOL_CALL>>", 1
    )
    text = "好的\n" + degraded
    calls = kimi_parse(text, _TOOLS)
    assert calls and calls[0]["function"]["name"] == "write"
    assert "TOOL_CALL" not in kimi_strip(text)


def test_kimi_orphan_closer_stripped():
    assert kimi_strip("正文\n</|DSML|tool_calls>") == "正文"
    assert kimi_strip("正文\n<<<TOOL_CA") == "正文"


def test_kimi_canonical_unchanged():
    text = _kimi_dsml("D:/x.txt", "hi")
    calls = kimi_parse(text, _TOOLS)
    assert calls and calls[0]["function"]["name"] == "write"
    assert kimi_strip(text) == ""


def test_qwen_tag_block_stripped_and_parsed():
    block = '<tool_call>{"name": "write", "arguments": {"filePath": "D:/x", "content": "hi"}}</tool_call>'
    text = "好的\n" + block
    calls = qwen_parse(text, _TOOLS)
    assert calls and calls[0]["function"]["name"] == "write"
    assert qwen_strip(text) == "好的"


def test_qwen_degraded_tag_fragments_scrubbed():
    """opener 的 < 被吃掉：残迹 tool_call> 与孤立闭合都剥掉（调用走裸 JSON 兜底）。"""
    body = '{"name": "write", "arguments": {"filePath": "D:/x", "content": "hi"}}'
    text = f"正文\ntool_call>{body}</tool_call>"
    out = qwen_strip(text)
    assert "tool_call>" not in out and "</tool_call>" not in out
    calls = qwen_parse(text, _TOOLS)
    assert calls and calls[0]["function"]["name"] == "write"


def test_qwen_degraded_marker_stripped_and_parsed():
    block = '<<TOOL_CALL>>{"name": "write", "arguments": {"filePath": "D:/x", "content": "hi"}}'
    calls = qwen_parse("好的\n" + block, _TOOLS)
    assert calls and calls[0]["function"]["name"] == "write"
    assert "TOOL_CALL" not in qwen_strip("好的\n" + block)


def test_qwen_tag_without_closer_truncated():
    """漏写闭合标签：从 opener 截断，半截 JSON 不漏正文。"""
    out = tool_bridge.strip_tool_markup(
        '正文\n<tool_call>{"name": "write", "arguments": {"filePath": "D',
        tag="tool_call", marker=QWEN_TOOL_MARKER,
    )
    assert out == "正文"
    assert "<tool_call>" not in out


def test_qwen_marker_fragments_in_prose_survive():
    """正文里的普通 tool_call 字样（无尖括号结构）不误伤。"""
    assert "调用 tool_call 工具" in qwen_strip("调用 tool_call 工具")
