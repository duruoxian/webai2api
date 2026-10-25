"""tool_bridge 共享桥接逻辑测试（Kimi / 智谱 / 千问三家共用的核心）。

覆盖四块：
1. increment_text 增量组装（多轮工具链的命脉）；
2. parse_tool_calls 宽松解析（标签 / 行内标记 / JSON 修复）；
3. ToolStreamBuffer 流式扣留（标记不漏进 content）；
4. strip_tool_markup 纯文本回退。
"""
from __future__ import annotations

import json

from webai2api.adapters.tool_bridge import (
    ToolStreamBuffer,
    increment_text,
    is_first_turn,
    is_forced_tool_choice,
    parse_tool_calls,
    render_tool_result,
    required_nudge,
    strip_tool_markup,
    text_of,
    tool_name_by_id,
    tool_names,
)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "write",
        "description": "写入本地文件",
        "parameters": {"type": "object", "properties": {
            "filePath": {"type": "string"}, "content": {"type": "string"},
        }, "required": ["filePath", "content"]},
    },
}]


def _call_id(name: str, n: int = 1) -> str:
    return f"call_{name}_{n}"


def _assistant_tool_call(name: str, args: dict, call_id: str) -> dict:
    return {"role": "assistant", "content": None, "tool_calls": [{
        "id": call_id, "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }]}


# ── increment_text ────────────────────────────────────────

def test_increment_first_turn_returns_last_user():
    msgs = [{"role": "system", "content": "你是助手"},
            {"role": "user", "content": "你好"}]
    assert increment_text(msgs) == "你好"


def test_increment_plain_multiturn_returns_last_user():
    msgs = [{"role": "user", "content": "第一问"},
            {"role": "assistant", "content": "第一答"},
            {"role": "user", "content": "第二问"}]
    assert increment_text(msgs) == "第二问"


def test_increment_renders_tool_result_after_assistant():
    call_id = _call_id("write")
    msgs = [
        {"role": "user", "content": "帮我写文件"},
        _assistant_tool_call("write", {"filePath": "a.md"}, call_id),
        {"role": "tool", "tool_call_id": call_id, "content": "写入成功"},
    ]
    out = increment_text(msgs)
    assert '<tool_result tool="write">' in out
    assert "写入成功" in out


def test_increment_tool_result_plus_followup_user():
    call_id = _call_id("write")
    msgs = [
        {"role": "user", "content": "帮我写文件"},
        _assistant_tool_call("write", {}, call_id),
        {"role": "tool", "tool_call_id": call_id, "content": "OK"},
        {"role": "user", "content": "顺便再说一句"},
    ]
    out = increment_text(msgs)
    assert "OK" in out and "顺便再说一句" in out


def test_increment_unknown_tool_id_falls_back_to_message_name():
    msgs = [
        {"role": "user", "content": "q"},
        _assistant_tool_call("write", {}, _call_id("write")),
        {"role": "tool", "tool_call_id": "call_missing", "name": "read",
         "content": "内容"},
    ]
    out = increment_text(msgs)
    assert 'tool="read"' in out


def test_increment_skips_empty_tool_results():
    call_id = _call_id("write")
    msgs = [
        {"role": "user", "content": "q"},
        _assistant_tool_call("write", {}, call_id),
        {"role": "tool", "tool_call_id": call_id, "content": "   "},
    ]
    assert increment_text(msgs) == "q"


def test_increment_multimodal_content_parts():
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "图片"},
        {"type": "text", "text": "里有什么"},
    ]}]
    assert increment_text(msgs) == "图片里有什么"


def test_text_of_and_name_maps():
    assert text_of({"content": None}) == ""
    assert tool_names(TOOLS) == {"write"}
    assert tool_name_by_id([
        _assistant_tool_call("write", {}, "call_x")]) == {"call_x": "write"}


# ── parse_tool_calls ──────────────────────────────────────

def _marker_line(name, args):
    return "<<<TOOL_CALL>>>" + json.dumps({"name": name, "arguments": args},
                                          ensure_ascii=False)


def test_parse_tag_format():
    text = '好的\n<tool_call>\n{"name": "write", "arguments": {"filePath": "a"}}\n</tool_call>'
    calls = parse_tool_calls(text, TOOLS, tag="tool_call")
    assert calls[0]["function"]["name"] == "write"
    assert json.loads(calls[0]["function"]["arguments"]) == {"filePath": "a"}


def test_parse_marker_format():
    text = "先看看\n" + _marker_line("write", {"filePath": "a"})
    calls = parse_tool_calls(text, TOOLS, marker="<<<TOOL_CALL>>>")
    assert calls[0]["function"]["name"] == "write"


def test_parse_rejects_unknown_tool():
    text = "<tool_call>{\"name\": \"rm_rf\", \"arguments\": {}}</tool_call>"
    assert parse_tool_calls(text, TOOLS, tag="tool_call") == []


def test_parse_repairs_fenced_and_trailing_comma():
    text = '```json\n<tool_call>{"name": "write", "arguments": {"a": 1,},}</tool_call>```'
    calls = parse_tool_calls(text, TOOLS, tag="tool_call")
    assert calls and calls[0]["function"]["name"] == "write"


def test_parse_repairs_fullwidth_punct():
    text = '<tool_call>{"name"："write"，"arguments"：{"a": 1}}</tool_call>'
    calls = parse_tool_calls(text, TOOLS, tag="tool_call")
    assert calls and calls[0]["function"]["name"] == "write"


def test_parse_plain_answer_returns_empty():
    assert parse_tool_calls("这是一段普通回答", TOOLS, tag="tool_call",
                            marker="<<<TOOL_CALL>>>") == []


def test_parse_limit_one_call():
    text = ("<tool_call>{\"name\": \"write\", \"arguments\": {}}</tool_call>\n"
            "<tool_call>{\"name\": \"write\", \"arguments\": {}}</tool_call>")
    assert len(parse_tool_calls(text, TOOLS, tag="tool_call")) == 1


# ── 多行 / 裸 JSON（v0.1.28 解析重构）─────────────────────

def test_parse_marker_multiline_json():
    """标记后 JSON 写成多行：配平扫描要取全，不能只拿第一个 '{'。"""
    text = ('我来看看\n<<<TOOL_CALL>>>{\n'
            '  "name": "write",\n'
            '  "arguments": {"filePath": "a.md", "content": "x"}\n'
            '}\n后续说明')
    calls = parse_tool_calls(text, TOOLS, marker="<<<TOOL_CALL>>>")
    assert calls and calls[0]["function"]["name"] == "write"
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "filePath": "a.md", "content": "x"}


def test_parse_marker_json_with_braces_in_string():
    """JSON 字符串值里的花括号不能干扰配平。"""
    text = '<<<TOOL_CALL>>>{"name": "write", "arguments": {"content": "a{b}c"}}'
    calls = parse_tool_calls(text, TOOLS, marker="<<<TOOL_CALL>>>")
    assert calls and json.loads(calls[0]["function"]["arguments"])["content"] == "a{b}c"


def test_parse_bare_json_fallback():
    """模型不带任何标记直接把调用 JSON 写进正文 → 兜底要能解析。"""
    text = '好的，我来写入文件：\n{"name": "write", "arguments": {"filePath": "a"}}\n完成。'
    calls = parse_tool_calls(text, TOOLS, marker="<<<TOOL_CALL>>>")
    assert calls and calls[0]["function"]["name"] == "write"


def test_parse_bare_json_multiline():
    text = ('{"name": "write",\n'
            ' "arguments": {"filePath": "a"}}')
    calls = parse_tool_calls(text, TOOLS, tag="tool_call")
    assert calls and calls[0]["function"]["name"] == "write"


def test_parse_bare_json_rejects_unknown_name():
    """普通文本里恰好有 {"name": "xxx"}（不在工具清单）→ 不误判。"""
    text = '配置文件内容：{"name": "settings", "arguments": {"verbose": true}}'
    assert parse_tool_calls(text, TOOLS, marker="<<<TOOL_CALL>>>") == []


def test_parse_bare_json_requires_arguments_key():
    """工具定义回显（有 name 无 arguments）不算调用。"""
    text = '工具清单：{"name": "write", "description": "写入文件", "parameters": {}}'
    assert parse_tool_calls(text, TOOLS, marker="<<<TOOL_CALL>>>") == []


def test_parse_truncated_json_returns_empty():
    """流截断（JSON 没写完）→ 宁可不调，不能产出半个调用。"""
    text = '<<<TOOL_CALL>>>{"name": "write", "arguments": {"filePath": "a"'
    assert parse_tool_calls(text, TOOLS, marker="<<<TOOL_CALL>>>") == []


def test_parse_marker_preferred_over_bare():
    """标记命中时优先标记；正文的裸 JSON 不抢调用。"""
    text = ('参考 {"name": "write", "arguments": {"filePath": "正文里的"}}\n'
            '<<<TOOL_CALL>>>{"name": "write", "arguments": {"filePath": "标记里的"}}')
    calls = parse_tool_calls(text, TOOLS, marker="<<<TOOL_CALL>>>")
    assert json.loads(calls[0]["function"]["arguments"])["filePath"] == "标记里的"


def test_strip_multiline_marker_json():
    """剥离时多行 JSON 的续行不能残留在正文里。"""
    text = ('正文一\n<<<TOOL_CALL>>>{\n'
            '  "name": "write",\n'
            '  "arguments": {}\n'
            '}\n正文二')
    out = strip_tool_markup(text, marker="<<<TOOL_CALL>>>")
    assert out == "正文一\n正文二"


# ── tool_choice 强校验辅助 ───────────────────────────────

def test_is_forced_tool_choice():
    assert not is_forced_tool_choice({})
    assert not is_forced_tool_choice({"tool_choice": "auto"})
    assert not is_forced_tool_choice({"tool_choice": "none"})
    assert is_forced_tool_choice({"tool_choice": "required"})
    assert is_forced_tool_choice(
        {"tool_choice": {"type": "function", "function": {"name": "write"}}})


def test_required_nudge_contains_platform_example():
    assert "<<<TOOL_CALL>>>" in required_nudge(marker="<<<TOOL_CALL>>>")
    n = required_nudge(tag="tool_call")
    assert "<tool_call>" in n and "必须调用一个工具" in n


# ── ToolStreamBuffer ──────────────────────────────────────

def test_buffer_no_opener_flushes_everything():
    buf = ToolStreamBuffer(["<<<TOOL_CALL>>>"])
    out = buf.feed("普通回答第一段")
    out += buf.feed("普通回答第一段第二段")
    out += buf.flush()
    assert out == "普通回答第一段第二段"


def test_buffer_holds_marker_text():
    buf = ToolStreamBuffer(["<<<TOOL_CALL>>>"])
    buf.feed("好的，我来写入文件\n")
    out = buf.feed("好的，我来写入文件\n<<<TOOL_CALL>>>{\"name\":\"write\"}")
    assert "<<<TOOL_CALL>>>" not in out
    assert buf.opener_found


def test_buffer_partial_opener_never_leaks():
    """逐字喂入：完整标记的任何一个字符都不能在被解析前漏出去。"""
    buf = ToolStreamBuffer(["<<<TOOL_CALL>>>"])
    released = ""
    for i in range(1, len("答案<<<TOOL_CALL>>>{") + 1):
        released += buf.feed("答案<<<TOOL_CALL>>>{"[:i])
    assert "<<<TOOL_CALL>>>" not in released


def test_buffer_strips_only_from_opener_onwards():
    buf = ToolStreamBuffer(["<tool_call>"])
    buf.feed("前面的正文")
    tail = buf.feed("前面的正文<tool_call>{bad json")
    assert tail == "前面的正文"
    # opener 之后的扣留内容留在缓冲里，flush 会给出来（解析失败的回退路径）
    assert "<tool_call>" in buf.flush()


# ── strip_tool_markup ─────────────────────────────────────

def test_strip_removes_tag_blocks_and_marker_lines():
    text = "正文一\n<tool_call>{...}</tool_call>\n正文二\n<<<TOOL_CALL>>>{...}\n"
    out = strip_tool_markup(text, tag="tool_call", marker="<<<TOOL_CALL>>>")
    assert out == "正文一\n正文二"


def test_render_tool_result_shape():
    out = render_tool_result("write", "OK", tag="tool_response")
    assert out == '<tool_response tool="write">\nOK\n</tool_response>'


# ── is_first_turn（会话缓存复用的守门员）──────────────────

def test_first_turn_detection():
    """首轮（无 assistant）必须新建上游会话：不同客户端会话的首轮
    指纹常相同，复用会让模型接着旧会话聊（opencode 实测串上下文）。"""
    first = [{"role": "system", "content": "opencode 助手"},
             {"role": "user", "content": "你好"}]
    followup = first + [{"role": "assistant", "content": "你好！"},
                        {"role": "user", "content": "继续"}]
    assert is_first_turn(first) is True
    assert is_first_turn(followup) is False
    assert is_first_turn([]) is True
    assert is_first_turn(None) is True
    # 工具链多轮（assistant.tool_calls + tool 结果）不算首轮
    chain = first + [_assistant_tool_call("write", {}, "call_x"),
                     {"role": "tool", "tool_call_id": "call_x", "content": "ok"}]
    assert is_first_turn(chain) is False
