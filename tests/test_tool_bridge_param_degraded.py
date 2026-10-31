"""tool_bridge DSML 参数降级形态回归（2026-11-01，对齐 deepseek 同族修复 e38ccc7）。

GLM/Kimi 共用的 webai2api/adapters/tool_bridge.py 从未对齐 deepseek
2026-10-30 的降级形态修复，已知真实事故：GLM 路径上 shell 工具的 command
参数值被污染成 `python make_yueyanglou.py<｜｜DSML｜｜ parameter
name="workdir" string="true">D:\\study`（漏闭合 → `_dsml_param_end` 吃到
invoke 末尾，把下一条标签吞进参数值）。

本文件锁定：

1. `_dsml_param_end` 缺闭合时截到下一条 DSML 标签边界（半角规范
   `<|DSML|...>` 与全角降级 `<｜｜DSML｜｜...>` / `</｜｜DSML｜｜...>` /
   `<` 被吃掉的 `｜｜DSML｜｜...>` 都认；必须是「标签形状」——单个 `|` 与
   Markdown 表格行绝不当边界）；
2. `_dsml_value` 字符串结果（CDATA 分支与原样文本分支）统一过降级残片
   清洗（对等移植 deepseek `_strip_marker_residue`）；
3. 普通 `<`（`if a < b`）与 Markdown 表格值不误伤；
4. 半角规范形态解析结果与改动前完全一致（防回归）。
"""
from __future__ import annotations

import json

from webai2api.adapters.tool_bridge import (
    DSML_TOOL_CALLS_OPEN,
    _dsml_param_end,
    _strip_marker_residue,
    parse_dsml_tool_calls,
)

FW = "\uff5c\uff5cDSML\uff5c\uff5c"  # 全角降级标记：｜｜DSML｜｜

SHELL_TOOLS = [{
    "type": "function",
    "function": {
        "name": "shell",
        "description": "执行命令",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "workdir": {"type": "string"},
            },
        },
    },
}]

WRITE_TOOLS = [{
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


def _args(calls: list) -> dict:
    assert len(calls) == 1, calls
    return json.loads(calls[0]["function"]["arguments"])


# ── 1. 半角规范形态：漏写 </parameter> + 下一条参数标签 ──────────

def test_missing_close_next_param_tag_not_swallowed():
    """command 漏写闭合、后面跟 workdir 参数标签（闭合也被上游吞掉）：
    下一条不被吞进前值，两个参数各自还原。"""
    text = (
        f"{DSML_TOOL_CALLS_OPEN}\n"
        '<|DSML|invoke name="shell">\n'
        '<|DSML|parameter name="command" string="true">'
        "python make_yueyanglou.py\n"
        '<|DSML|parameter name="workdir" string="true">D:\\study\n'
        "</|DSML|invoke>\n"
        "</|DSML|tool_calls>"
    )
    calls = parse_dsml_tool_calls(text, SHELL_TOOLS)
    args = _args(calls)
    assert calls[0]["function"]["name"] == "shell"
    assert args["command"] == "python make_yueyanglou.py"
    assert args["workdir"] == "D:\\study"


# ── 2. 全角降级形态：漏闭合 + 全角下一条标签（真实事故形态）─────

def test_fullwidth_boundary_cut_incident():
    """实测事故形态：command 漏闭合 + 全角 workdir 标签（< 还在）。
    修复前 command == 'python make_yueyanglou.py<｜｜DSML｜｜ parameter
    name="workdir" string="true">D:\\study'；修复后截到全角标签边界。"""
    text = (
        f"{DSML_TOOL_CALLS_OPEN}\n"
        '<|DSML|invoke name="shell">\n'
        '<|DSML|parameter name="command" string="true">'
        "python make_yueyanglou.py<"
        f'{FW} parameter name="workdir" string="true">D:\\study'
        "</|DSML|invoke>\n"
        "</|DSML|tool_calls>"
    )
    args = _args(parse_dsml_tool_calls(text, SHELL_TOOLS))
    assert args["command"] == "python make_yueyanglou.py"
    # 全角降级标签不解析成参数，也不得残留在任何参数值里
    assert "workdir" not in args
    assert "DSML" not in json.dumps(args, ensure_ascii=False)


def test_fullwidth_boundary_cut_eaten_angle_bracket():
    """`<` 被上游吃掉的降级形态（只剩 ｜｜DSML｜｜ 开头）同样认作边界。"""
    text = (
        f"{DSML_TOOL_CALLS_OPEN}\n"
        '<|DSML|invoke name="shell">\n'
        '<|DSML|parameter name="command" string="true">'
        "python make_yueyanglou.py"
        f'{FW} parameter name="workdir" string="true">D:\\study'
        "</|DSML|invoke>\n"
        "</|DSML|tool_calls>"
    )
    args = _args(parse_dsml_tool_calls(text, SHELL_TOOLS))
    assert args["command"] == "python make_yueyanglou.py"
    assert "DSML" not in json.dumps(args, ensure_ascii=False)


def test_fullwidth_close_tag_boundary_cut():
    """全角闭合标签 </｜｜DSML｜｜ ...> 也是合法边界（漏闭合时截断）。"""
    text = (
        f"{DSML_TOOL_CALLS_OPEN}\n"
        '<|DSML|invoke name="shell">\n'
        '<|DSML|parameter name="command" string="true">'
        "python run.py"
        f"</{FW} parameter>\n"
        "</|DSML|invoke>\n"
        "</|DSML|tool_calls>"
    )
    args = _args(parse_dsml_tool_calls(text, SHELL_TOOLS))
    assert args["command"] == "python run.py"
    assert "DSML" not in json.dumps(args, ensure_ascii=False)


# ── 3. 值清洗：内嵌全角降级残片 ─────────────────────────────────

def test_fullwidth_residue_cleaned_plain_text_value():
    """原样文本分支：值内嵌降级残片 → 清洗干净（对齐 deepseek
    workdir `D:\\study<｜｜DSML｜｜>` → `D:\\study`）。"""
    text = (
        f"{DSML_TOOL_CALLS_OPEN}\n"
        '<|DSML|invoke name="shell">\n'
        '<|DSML|parameter name="workdir" string="true">'
        f"D:\\study<{FW}>"
        "</|DSML|parameter>\n"
        '<|DSML|parameter name="command" string="true">'
        f'python x.py<{FW} parameter name="w" string="true">'
        "</|DSML|parameter>\n"
        "</|DSML|invoke>\n"
        "</|DSML|tool_calls>"
    )
    args = _args(parse_dsml_tool_calls(text, SHELL_TOOLS))
    assert args["workdir"] == "D:\\study"
    assert args["command"] == "python x.py"


def test_fullwidth_residue_cleaned_cdata_value():
    """CDATA 分支同样过残片清洗（含孤立全角闭合标签）。"""
    text = (
        f"{DSML_TOOL_CALLS_OPEN}\n"
        '<|DSML|invoke name="write">\n'
        '<|DSML|parameter name="filePath"><![CDATA[D:\\a.md]]></|DSML|parameter>\n'
        '<|DSML|parameter name="content"><![CDATA['
        f"done<{FW} invoke>]]></|DSML|parameter>\n"
        '<|DSML|parameter name="options">'
        f"<![CDATA[result</{FW} invoke>]]></|DSML|parameter>\n"
        "</|DSML|invoke>\n"
        "</|DSML|tool_calls>"
    )
    args = _args(parse_dsml_tool_calls(text, WRITE_TOOLS))
    assert args["filePath"] == "D:\\a.md"
    assert args["content"] == "done"
    assert args["options"] == "result"


# ── 4. 保守规则：普通 < 与 Markdown 表格不误伤 ──────────────────

def test_plain_lt_not_mangled():
    """普通 `<`（if a < b）与只有 DSML 字样、无标签形状的文本都不动。"""
    text = (
        f"{DSML_TOOL_CALLS_OPEN}\n"
        '<|DSML|invoke name="write">\n'
        '<|DSML|parameter name="filePath"><![CDATA[D:\\a.md]]></|DSML|parameter>\n'
        '<|DSML|parameter name="content"><![CDATA['
        "if a < b and x > y]]></|DSML|parameter>\n"
        '<|DSML|parameter name="options">if a < b</|DSML|parameter>\n'
        "</|DSML|invoke>\n"
        "</|DSML|tool_calls>"
    )
    args = _args(parse_dsml_tool_calls(text, WRITE_TOOLS))
    assert args["content"] == "if a < b and x > y"
    assert args["options"] == "if a < b"
    # 值里出现 DSML 字样但无「竖线+DSML+竖线」标签形状：不清洗
    text2 = text.replace("if a < b</|DSML|parameter>",
                         "DSML 格式说明见 a < b</|DSML|parameter>")
    args2 = _args(parse_dsml_tool_calls(text2, WRITE_TOOLS))
    assert args2["options"] == "DSML 格式说明见 a < b"


def test_markdown_table_value_not_mangled():
    """Markdown 表格式参数值不误伤：有闭合时原样保留；缺闭合时也不得
    把表格行当边界截断（截到的是真正的 DSML 标签）。"""
    table = "| a | b |\n|---|---|\n| 1 | 2 |"
    closed = (
        f"{DSML_TOOL_CALLS_OPEN}\n"
        '<|DSML|invoke name="write">\n'
        '<|DSML|parameter name="filePath"><![CDATA[D:\\t.md]]></|DSML|parameter>\n'
        f'<|DSML|parameter name="content"><![CDATA[{table}]]></|DSML|parameter>\n'
        "</|DSML|invoke>\n"
        "</|DSML|tool_calls>"
    )
    args = _args(parse_dsml_tool_calls(closed, WRITE_TOOLS))
    assert args["content"] == table

    unclosed = (
        f"{DSML_TOOL_CALLS_OPEN}\n"
        '<|DSML|invoke name="write">\n'
        '<|DSML|parameter name="content" string="true">'
        "| a | b |\n| c | d |\n"
        "</|DSML|invoke>\n"
        "</|DSML|tool_calls>"
    )
    args2 = _args(parse_dsml_tool_calls(unclosed, WRITE_TOOLS))
    assert "| a | b |" in args2["content"]
    assert "| c | d |" in args2["content"]
    assert "DSML" not in args2["content"]


# ── 5. 半角规范形态零回归 ────────────────────────────────────────

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


def test_canonical_cdata_parse_unchanged():
    text = "好的，我来写。\n" + _call_block(
        _param("filePath", "<![CDATA[D:\\a.md]]>")
        + "\n"
        + _param("content", '<![CDATA[第一行\n{"json": "not parsed"}\n]]>')
    )
    args = _args(parse_dsml_tool_calls(text, WRITE_TOOLS))
    assert args["filePath"] == "D:\\a.md"
    assert args["content"] == '第一行\n{"json": "not parsed"}\n'


def test_canonical_cdata_escaped_terminator_unchanged():
    text = _call_block(_param("content", "<![CDATA[a]]]]><![CDATA[>b]]>"))
    assert _args(parse_dsml_tool_calls(text, WRITE_TOOLS))["content"] == "a]]>b"


def test_canonical_nested_object_and_items_unchanged():
    text = _call_block(_param(
        "options",
        _param("depth", "3")
        + '<|DSML|parameter name="tags"><item>a</item><item>b</item>'
        "</|DSML|parameter>",
    ))
    args = _args(parse_dsml_tool_calls(text, WRITE_TOOLS))
    assert args["options"] == {"depth": 3, "tags": ["a", "b"]}


def test_canonical_json_literals_unchanged():
    text = _call_block(
        _param("filePath", "a.txt")
        + _param("content", "true")
        + _param("options", '{"k": [1, 2]}')
    )
    args = _args(parse_dsml_tool_calls(text, WRITE_TOOLS))
    assert args["content"] is True
    assert args["options"] == {"k": [1, 2]}


# ── 6. 单元级：边界识别与残片清洗的保守规则 ─────────────────────

def test_param_end_boundary_units():
    # 缺闭合：截到下一条半角参数标签
    text = 'v1<|DSML|parameter name="b">v2'
    assert _dsml_param_end(text, 2) == (2, 2)
    # 缺闭合：全角降级形态（< 在 / < 被吃 / 全角闭合）都是边界
    assert _dsml_param_end(f"v1<{FW} parameter", 2) == (2, 2)
    assert _dsml_param_end(f"v1{FW} parameter", 2) == (2, 2)
    assert _dsml_param_end(f"v1</{FW} invoke>", 2) == (2, 2)
    # 保守规则：Markdown 表格行不是边界（无 DSML 标签形状 → 吃到文末）
    assert _dsml_param_end("v1\n| a | b |\n", 2) == (len("v1\n| a | b |\n"),) * 2
    # 保守规则：半角 |DSML|（无 <，单个 | 开头）不当边界
    assert _dsml_param_end('v1|DSML|parameter name="b">v2', 2) == \
        (len('v1|DSML|parameter name="b">v2'),) * 2


def test_strip_marker_residue_units():
    # 实测事故形态：D:\study<｜｜DSML｜｜> → D:\study
    assert _strip_marker_residue(f"D:\\study<{FW}>") == "D:\\study"
    # 半角残片同样清（无 < 的孤立闭合）
    assert _strip_marker_residue(f"done</{FW} invoke>") == "done"
    # 残片 + 行尾换行：一并收敛（\Z 收尾，不吃正文尾巴）
    assert _strip_marker_residue(f"done</{FW} invoke>\n") == "done"
    # 普通 <、表格行、无标签形状的 DSML 字样：一律不动
    assert _strip_marker_residue("if a < b") == "if a < b"
    assert _strip_marker_residue("| a | b |") == "| a | b |"
    assert _strip_marker_residue("含 DSML 字样但没有标签形状") == \
        "含 DSML 字样但没有标签形状"
