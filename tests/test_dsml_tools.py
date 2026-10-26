"""DSML 工具协议测试（Kimi/智谱原生 XML 格式）。

格式来源：glm2api 的 tool_protocol / tool_parser——
<|DSML|tool_calls><|DSML|invoke name="x"><|DSML|parameter name="k">
<![CDATA[value]]></|DSML|parameter></|DSML|invoke></|DSML|tool_calls>
字符串走 CDATA（换行/引号/花括号免转义），对象用嵌套 parameter，
数组用 <item>。旧 JSON 标记继续作为兜底解析。
"""
from __future__ import annotations

import json

from webai2api.adapters.tool_bridge import (
    DSML_TOOL_CALLS_OPEN,
    DeltaToolBuffer,
    increment_text,
    parse_dsml_tool_calls,
    parse_tool_calls,
    render_dsml_assistant_tool_calls,
    render_dsml_tool_result,
    strip_tool_markup,
)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "write",
        "description": "写文件",
        "parameters": {
            "type": "object",
            "properties": {
                "filePath": {"type": "string"},
                "content": {"type": "string"},
                "options": {"type": "object"},
            },
        },
    },
}]


def _call_block(params: str, name: str = "write") -> str:
    return (
        f"{DSML_TOOL_CALLS_OPEN}\n"
        f'  <|DSML|invoke name="{name}">\n'
        f"{params}\n"
        f"  </|DSML|invoke>\n"
        f"</|DSML|tool_calls>"
    )


def _param(name: str, raw: str) -> str:
    return f'    <|DSML|parameter name="{name}">{raw}</|DSML|parameter>'


def test_parse_basic_cdata_multiline():
    text = "好的，我来写。\n" + _call_block(
        _param("filePath", "<![CDATA[D:\\a.md]]>")
        + "\n"
        + _param("content", '<![CDATA[第一行\n{"json": "not parsed"}\n]]>')
    )
    calls = parse_dsml_tool_calls(text, TOOLS)
    assert len(calls) == 1
    c = calls[0]
    assert c["type"] == "function"
    assert c["function"]["name"] == "write"
    assert c["id"].startswith("call_")
    args = json.loads(c["function"]["arguments"])
    assert args["filePath"] == "D:\\a.md"
    # CDATA 内的换行、引号、花括号原样保留，不经过 JSON 转义
    assert args["content"] == '第一行\n{"json": "not parsed"}\n'


def test_parse_cdata_escaped_terminator():
    """参数里含 ]]> 时按 CDATA 转义拼接还原。"""
    text = _call_block(_param("content", "<![CDATA[a]]]]><![CDATA[>b]]>"))
    args = json.loads(parse_dsml_tool_calls(text, TOOLS)[0]["function"]["arguments"])
    assert args["content"] == "a]]>b"


def test_parse_nested_object_and_item_array():
    text = _call_block(_param(
        "options",
        _param("depth", "3")
        + '<|DSML|parameter name="tags"><item>a</item><item>b</item>'
        '</|DSML|parameter>',
    ))
    args = json.loads(parse_dsml_tool_calls(text, TOOLS)[0]["function"]["arguments"])
    assert args["options"] == {"depth": 3, "tags": ["a", "b"]}


def test_parse_json_literals():
    text = _call_block(
        _param("filePath", "a.txt")
        + _param("content", "true")
        + _param("options", '{"k": [1, 2]}')
    )
    args = json.loads(parse_dsml_tool_calls(text, TOOLS)[0]["function"]["arguments"])
    assert args["content"] is True
    assert args["options"] == {"k": [1, 2]}


def test_parse_plain_text_value_fallback():
    text = _call_block(_param("filePath", "D:\\没有 CDATA.md"))
    args = json.loads(parse_dsml_tool_calls(text, TOOLS)[0]["function"]["arguments"])
    assert args["filePath"] == "D:\\没有 CDATA.md"


def test_reject_unknown_tool():
    text = _call_block(_param("filePath", "a"), name="rm_rf")
    assert parse_dsml_tool_calls(text, TOOLS) == []


def test_only_first_call():
    text = _call_block(_param("filePath", "a")) + _call_block(_param("filePath", "b"))
    assert len(parse_dsml_tool_calls(text, TOOLS)) == 1


def test_no_tools_no_parse():
    text = _call_block(_param("filePath", "a"))
    assert parse_dsml_tool_calls(text, []) == []


def test_parse_via_parse_tool_calls_dsml_flag():
    text = _call_block(_param("filePath", "a"))
    assert parse_tool_calls(text, TOOLS, dsml=True)
    assert parse_tool_calls(text, TOOLS) == []


def test_legacy_marker_still_parsed_with_dsml_flag():
    text = '<<<TOOL_CALL>>>{"name":"write","arguments":{"filePath":"a"}}'
    calls = parse_tool_calls(text, TOOLS, marker="<<<TOOL_CALL>>>", dsml=True)
    assert calls[0]["function"]["name"] == "write"


def test_strip_dsml_block_keeps_prose():
    text = "前文\n" + _call_block(_param("filePath", "a")) + "\n后文"
    out = strip_tool_markup(text, dsml=True)
    assert "<|DSML|" not in out
    assert "前文" in out and "后文" in out
    assert out.strip() == "前文\n\n后文"


def test_strip_unclosed_dsml_truncates():
    text = "前文\n<|DSML|tool_calls>\n<|DSML|invoke name=\"write\">"
    out = strip_tool_markup(text, dsml=True)
    assert out == "前文"


def test_render_assistant_tool_calls_roundtrip():
    calls = [{
        "id": "call_a",
        "type": "function",
        "function": {"name": "write", "arguments": json.dumps(
            {"filePath": "a.txt", "content": "多行\n内容"}, ensure_ascii=False)},
    }]
    rendered = render_dsml_assistant_tool_calls(calls)
    back = parse_dsml_tool_calls(rendered, TOOLS)
    args = json.loads(back[0]["function"]["arguments"])
    assert args["filePath"] == "a.txt"
    assert args["content"] == "多行\n内容"


def test_render_tool_result_escapes_cdata_terminator():
    block = render_dsml_tool_result("write", "a]]>b", "call_1")
    assert 'call_id="call_1"' in block and 'name="write"' in block
    assert "<![CDATA[a]]]]><![CDATA[>b]]>" in block


def test_increment_text_dsml_renders_tool_result():
    msgs = [
        {"role": "user", "content": "写文件"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "write", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "写入成功"},
        {"role": "user", "content": "继续"},
    ]
    out = increment_text(msgs, dsml=True)
    assert out.startswith('<|DSML|tool_result call_id="call_1" name="write">')
    assert "写入成功" in out
    assert out.endswith("继续")
    assert "<tool_result" not in out


def test_delta_buffer_holds_dsml_opener_split_across_chunks():
    buf = DeltaToolBuffer([DSML_TOOL_CALLS_OPEN])
    pieces = ["正文", "\n<|DSM", "L|tool_calls>", "\n<|DSML|invoke"]
    emitted = "".join(buf.feed_delta(p) for p in pieces)
    # opener 之前的正文（含换行）照常下发，opener 及之后全部扣住
    assert emitted == "正文\n"
    assert buf.opener_found is True
    assert buf.text.endswith("<|DSML|invoke")


# ── 病态输入护栏（2026-10-27 审查发现：可被提示注入诱导）──────
def test_parse_dsml_deep_nesting_does_not_crash():
    """~600 层嵌套参数：按文本折叠返回，绝不 RecursionError
    （此前 ~500 层即爆栈 → 上游成功回复被当 502）。"""
    import time
    deep = "<|DSML|parameter name=\"a\">" * 600 + "v" + "</|DSML|parameter>" * 600
    text = (DSML_TOOL_CALLS_OPEN
            + f'<|DSML|invoke name="write">{deep}</|DSML|invoke>'
            + "</|DSML|tool_calls>")
    t0 = time.time()
    calls = parse_dsml_tool_calls(text, TOOLS)
    assert isinstance(calls, list)
    assert time.time() - t0 < 2


def test_parse_dsml_opener_flood_guard():
    """两万个未闭合开标签：护栏直接放弃解析（此前 460KB 需 34 秒）。"""
    import time
    flood = "<|DSML|invoke name=\"write\">" * 20000
    text = DSML_TOOL_CALLS_OPEN + flood
    t0 = time.time()
    calls = parse_dsml_tool_calls(text, TOOLS)
    assert calls == []
    assert time.time() - t0 < 2


def test_strip_dsml_opener_flood_guard():
    import time
    flood = "<|DSML|invoke>" * 20000
    t0 = time.time()
    out = strip_tool_markup("前文" + flood, dsml=True)
    assert time.time() - t0 < 2
    assert out.startswith("前文")
