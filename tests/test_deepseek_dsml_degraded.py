"""deepseek DSML 降级形态回归（2026-10-30，用户报障：正文漏出 `| | DSML | | calls>`）。

上游（DeepSeek 网页端）会把工具标记写成降级形态，实测有三类：

1. **吃掉 `<`**：下发的正文是 `｜｜DSML｜｜ calls>`，客户端渲染成
   `| | DSML | | calls>`（用户截图）。上游断流续写轮同样只剩后半截。
2. **开标签写成 `</`**：`</｜｜DSML｜｜ invoke name="read">`，参数照常跟在后面。
3. **孤立闭合标签**：`</｜｜DSML｜｜ calls>` 单独漏出来（调用方没声明 tools 时
   旧实现把它当普通文本吐出去）。

旧实现只认 `<` + 全角竖线的规范形态：降级形态既不进 DSML 状态、也不被剥离，
于是原样漏进 content。本文件锁定「零残留 + 调用仍能还原」两条底线。
"""

from __future__ import annotations

import json

import pytest

from webai2api.adapters.deepseek.services.tool_calling import (
    StreamingToolDetector,
    parse_tool_calls,
    strip_dsml_residue,
)

D = "｜｜DSML｜｜"  # 全角竖线，DeepSeek 原生先验形态

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "glob",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
]

CALL_BLOCK = (
    f"\n\n<{D} calls>\n"
    f'<{D} invoke name="glob">\n'
    f'<{D} parameter name="pattern" string="true">*.md</{D} parameter>\n'
    f"</{D} invoke>\n"
    f"</{D} calls>\n\n"
)


def _drive(chunks, tools=TOOLS, require_marker=False):
    det = StreamingToolDetector(tools, require_marker=require_marker)
    text, calls = [], []
    for ch in chunks:
        for a in det.feed(ch):
            if a.kind == "text":
                text.append(a.text)
            elif a.kind == "tool_calls":
                calls.extend(a.tool_calls)
    for a in det.finalize():
        if a.kind == "text":
            text.append(a.text)
        elif a.kind == "tool_calls":
            calls.extend(a.tool_calls)
    return "".join(text), calls


def _drive_rounds(rounds, tools=TOOLS, require_marker=False):
    """多轮 = 多检测器（模拟断流续写/空响应重试换新检测器）。"""
    text, calls = [], []
    for chunks in rounds:
        t, c = _drive(chunks, tools=tools, require_marker=require_marker)
        text.append(t)
        calls.extend(c)
    return "".join(text), calls


# ── 1. 上游吃掉 `<` ────────────────────────────────────────


def test_missing_angle_bracket_does_not_leak():
    """实测形态：`｜｜DSML｜｜ calls>` 没有 `<`，正文一字不漏、调用照常还原。"""
    text, calls = _drive([f"\n\n{D} calls>\n" + CALL_BLOCK])
    assert "DSML" not in text
    assert [c.name for c in calls] == ["glob"]


def test_missing_angle_bracket_split_char_by_char():
    """标记被切碎成单字符（上游按 token 切片）也不能漏。"""
    raw = f"\n\n{D} calls>\n" + CALL_BLOCK
    text, calls = _drive(list(raw))
    assert "DSML" not in text
    assert [c.name for c in calls] == ["glob"]


def test_continuation_round_drops_lone_angle_bracket():
    """断流时检测器只扣住一个 `<`，换新检测器续写：裸 `<` 不许漏。"""
    text, calls = _drive_rounds(
        [
            ["\n\n<"],
            [f"{D} calls>\n" + CALL_BLOCK],
        ]
    )
    assert "DSML" not in text
    assert "<" not in text
    assert [c.name for c in calls] == ["glob"]


# ── 2. 开标签写成 `</` ────────────────────────────────────

MALFORMED = (
    f'\n\n</{D} invoke name="read">\n'
    f'<{D} parameter name="path" string="true">D:\\a.md\n'
)


def test_malformed_open_slash_tag_does_not_leak():
    text, calls = _drive([MALFORMED])
    assert "DSML" not in text
    # 参数照样能救回来
    assert [c.name for c in calls] == ["read"]
    assert (
        calls[0].arguments == '{"path": "D:\\\\a.md"}' or "a.md" in calls[0].arguments
    )


def test_malformed_open_slash_tag_non_stream():
    parsed = parse_tool_calls(MALFORMED, TOOLS)
    assert [c.name for c in parsed.calls] == ["read"]
    assert "DSML" not in parsed.clean_text


# ── 3. 孤立闭合标签 ───────────────────────────────────────


def test_orphan_closer_stripped_with_tools():
    text, calls = _drive([CALL_BLOCK], require_marker=False)
    assert "DSML" not in text
    assert [c.name for c in calls] == ["glob"]


def test_orphan_closer_stripped_without_tools_after_marker_seen():
    """调用方没声明 tools（require_marker=True）也不能把闭合标签当正文。"""
    text, calls = _drive([CALL_BLOCK], require_marker=True)
    assert "DSML" not in text
    assert [c.name for c in calls] == ["glob"]


# ── 4. 残留剥离与误伤 ─────────────────────────────────────


@pytest.mark.parametrize(
    "residue",
    [
        f"\n\n{D} calls>\n\n",
        f"\n\n</{D} calls>\n\n",
        f"\n\n|{D} calls>\n",
        f'\n\n<{D} calls attr="x">\n',
    ],
)
def test_strip_dsml_residue_handles_degraded_forms(residue):
    assert "DSML" not in strip_dsml_residue(residue)


def test_plain_answer_not_swallowed():
    """普通正文（含 `<`、`|`、表格）必须原样吐出，不能被标记规则误吞。"""
    plain = "结论：a < b 且 c | d。\n\n| 列1 | 列2 |\n|---|---|\n| 1 | 2 |\n"
    text, calls = _drive([plain])
    assert text == plain
    assert calls == []


@pytest.mark.parametrize(
    "tail",
    ["| a |", "a |", "结尾是竖线 |", "calls", "a < b"],
)
def test_plain_tail_not_dropped(tail):
    """答案尾巴刚好是 `|` / `<` 这类字符时也不能被当成标记碎片吞掉。"""
    text, calls = _drive([tail])
    assert text == tail
    assert calls == []


def test_bare_trailing_angle_bracket_is_dropped():
    """流尾只剩一个 `<`：按「上游把标记切在 `<` 之后」处理，丢弃而不是漏出去。

    这是有意的取舍——协议下模型只会把 `<` 用在 DSML 标签上，正文以裸 `<`
    结尾的概率远低于漏标记；丢一个字符，换客户端不见 `| | DSML | |` 残片。
    """
    text, _ = _drive(["结尾是尖括号 <"])
    assert text == "结尾是尖括号 "


def test_normal_dsml_still_parsed():
    """规范形态不受影响（防回归）。"""
    text, calls = _drive([CALL_BLOCK])
    assert "DSML" not in text
    assert [c.name for c in calls] == ["glob"]


# ── 5. 残片污染参数值（用户报障：`Working directory does not exist: D:\study<…DSML…>`）──


def test_parameter_missing_closer_does_not_swallow_tags():
    """缺 `</parameter>` 时吃到下一条 DSML 标签为止，不把标签吃进值里。"""
    block = (
        f"\n\n<{D} calls>\n"
        f'<{D} invoke name="shell">\n'
        f'<{D} parameter name="command" string="true">python -c "print(1)"</{D} parameter>\n'
        f'<{D} parameter name="workdir" string="true">D:\\study<\\{D}>\n'
        f"</{D} invoke>\n"
        f"</{D} calls>\n"
    )
    parsed = parse_tool_calls(
        block,
        TOOLS + [{"type": "function", "function": {"name": "shell", "parameters": {}}}],
    )
    assert [c.name for c in parsed.calls] == ["shell"]
    args = json.loads(parsed.calls[0].arguments)
    assert args["workdir"] == "D:\\study"
    assert "DSML" not in parsed.calls[0].arguments


def test_residue_inside_value_is_scrubbed():
    """值里直接粘着残片（有闭合）也要清掉。"""
    block = (
        f"\n\n<{D} calls>\n"
        f'<{D} invoke name="shell">\n'
        f'<{D} parameter name="workdir" string="true">D:\\study<\\{D}></{D} parameter>\n'
        f"</{D} invoke>\n"
        f"</{D} calls>\n"
    )
    parsed = parse_tool_calls(
        block,
        TOOLS + [{"type": "function", "function": {"name": "shell", "parameters": {}}}],
    )
    assert json.loads(parsed.calls[0].arguments)["workdir"] == "D:\\study"


def test_plain_angle_bracket_value_kept():
    """普通参数值里的 `<` 不能被误删（没有 DSML 字样就原样保留）。"""
    block = (
        f"\n\n<{D} calls>\n"
        f'<{D} invoke name="shell">\n'
        f'<{D} parameter name="command" string="true">if a < b; then echo ok</{D} parameter>\n'
        f"</{D} invoke>\n"
        f"</{D} calls>\n"
    )
    parsed = parse_tool_calls(
        block,
        TOOLS + [{"type": "function", "function": {"name": "shell", "parameters": {}}}],
    )
    assert json.loads(parsed.calls[0].arguments)["command"] == "if a < b; then echo ok"

