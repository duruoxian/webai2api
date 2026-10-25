"""reasoning_bridge 测试：空正文兜底 + LEGACY_REASONING_IN_CONTENT。

覆盖两块：
1. finalize_reasoning 非流式收尾（兜底置换 / legacy 合并 / 保持原样）；
2. ReasoningStream 流式分发（正常模式走 reasoning_content、legacy 模式
   <think> 并入 content 且顺序保证、收尾兜底、工具路径只闭合不兜底）。
"""
from __future__ import annotations

from webai2api.adapters.reasoning_bridge import (
    ReasoningStream,
    finalize_reasoning,
)


# ── finalize_reasoning（非流式）──────────────────────────

def test_finalize_rescues_empty_content_from_reasoning():
    """正文空白而思考非空：思考转正为正文，思考清空（Kimi 全系暗病）。"""
    content, reasoning = finalize_reasoning("", "你好！有什么我可以帮你的吗？")
    assert content == "你好！有什么我可以帮你的吗？"
    assert reasoning == ""


def test_finalize_rescues_whitespace_content():
    content, reasoning = finalize_reasoning("   \n", "答案在思考里")
    assert content == "答案在思考里"
    assert reasoning == ""


def test_finalize_keeps_normal_content_untouched():
    """正文非空：两字段原样保留（默认行为不变）。"""
    content, reasoning = finalize_reasoning("正文", "思考")
    assert content == "正文"
    assert reasoning == "思考"


def test_finalize_no_reasoning_noop():
    content, reasoning = finalize_reasoning("正文", "")
    assert (content, reasoning) == ("正文", "")


def test_finalize_legacy_merges_think_into_content():
    """legacy=true：思考以 <think> 块并入 content，reasoning 清空。"""
    content, reasoning = finalize_reasoning("正文", "思考", legacy=True)
    assert content.startswith("<think>\n思考\n</think>")
    assert content.endswith("正文")
    assert reasoning == ""


def test_finalize_legacy_with_empty_content():
    content, reasoning = finalize_reasoning("", "只有思考", legacy=True)
    assert "<think>\n只有思考\n</think>" in content
    assert reasoning == ""


# ── ReasoningStream（流式）───────────────────────────────

def test_stream_normal_mode_reasoning_goes_to_field():
    rs = ReasoningStream()
    out_r, out_c = rs.feed("思考一段", "")
    assert out_r == "思考一段" and out_c == ""
    out_r, out_c = rs.feed("", "正文一段")
    assert out_r == "" and out_c == "正文一段"


def test_stream_normal_mode_rescue_at_finish():
    """流结束正文全空而思考非空 → 补发一发 content 增量。"""
    rs = ReasoningStream()
    rs.feed("完整答案在思考里", "")
    out_r, out_c = rs.finish()
    assert out_r == ""
    assert out_c == "完整答案在思考里"


def test_stream_normal_mode_no_rescue_when_content_exists():
    rs = ReasoningStream()
    rs.feed("思考", "正文")
    assert rs.finish() == ("", "")


def test_stream_legacy_mode_think_wraps_content_in_order():
    """legacy 模式：<think> 必须在正文之前，收尾闭合。"""
    rs = ReasoningStream(legacy=True)
    emitted = ""
    for r, c in (rs.feed("思考A", ""), rs.feed("思考B", ""),
                 rs.feed("", "正文"), rs.finish()):
        emitted += (r or "") + (c or "")
    assert emitted.startswith("<think>\n思考A思考B\n</think>\n正文")


def test_stream_legacy_reasoning_only_closes_at_finish():
    rs = ReasoningStream(legacy=True)
    _, c1 = rs.feed("思考", "")
    assert c1 == "<think>\n思考"
    out_r, out_c = rs.finish()
    assert out_r == "" and out_c == "\n</think>\n"


def test_stream_close_think_for_tool_path():
    """工具调用路径 content 为空是正常形态：只闭合 <think>，不触发兜底。"""
    rs = ReasoningStream()
    rs.feed("思考", "")
    assert rs.close_think() == ""      # 正常模式无需闭合
    assert rs.finish() == ("", "思考")  # 兜底仍由 finish 负责（普通收尾路径）

    rs2 = ReasoningStream(legacy=True)
    rs2.feed("思考", "")
    assert rs2.close_think() == "\n</think>\n"
    assert rs2.finish() == ("", "")    # 已闭合，不再补发
