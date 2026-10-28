"""C-3 面板性能与泄漏防线（方案 C 第三层，2026-10-29）。

把人工审计的结论固化为静态不变量（当时实测全部干净）：

1. innerHTML 注入面：模板插值必须过 esc()、是纯数字/受控表达式，
   或属于已人工核对的豁免清单（dataset 自设值 / fmt 数字格式化 /
   长度统计）。donut 悬浮提示的 data-name 在渲染处已 esc。
2. 轮询定时器：全面板只允许一个 setInterval（15s 数据轮询），
   且必须带 document.hidden / 主视图隐藏 / 弹窗打开三重守卫。
3. 监听器纪律：render* 函数体内只允许出现悬停提示类监听
   （挂载在每次重渲染都新建的节点上，旧节点随 GC 释放=无泄漏）
   与一次性 DOMContentLoaded 引导；文档级监听不允许出现在 render 内。
4. 大宗渲染：render* 的行渲染使用 map+join 重建模式（整段替换），
   不存在逐行 innerHTML+= 累加的 O(n²) 模式。
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "webai2api" / "web"
HTML = (ROOT / "index.html").read_text(encoding="utf-8")


def _render_spans():
    spans = []
    for m in re.finditer(r"function (render[A-Za-z]+)\(", HTML):
        start = m.start()
        nxt = HTML.find("\nfunction ", start + 10)
        spans.append((start, nxt if nxt > 0 else len(HTML), m.group(1)))
    return spans


_SPANS = _render_spans()


def _in_render(pos: int) -> bool:
    return any(s <= pos < e for s, e, _ in _SPANS)


def test_innerHTML_interpolations_are_escaped_or_allowlisted():
    """未过 esc() 的插值必须命中人工核对的豁免模式（数值/自设 dataset/长度）。"""
    allow_number = re.compile(
        r"^[0-9a-zA-Z_ .,()?:!&|=\-+*/\"\']*$"  # 无 <> 与反引号的表达式
    )
    allow_prefix = (
        "esc(",
        "fmt(",  # 数字格式化
        "fmtTok(",
        "s.accounts.filter",  # 长度统计
        "s.keys.length",
        "seg.dataset.",  # 面板自设 dataset（data-name 渲染处已 esc）
        "j.elapsed_ms",  # 自家后端返回的数值
        "t.n?",  # 成功率表达式
    )
    bad = []
    for m in re.finditer(r"innerHTML\s*=\s*`(.*?)`", HTML, re.S):
        line = HTML[: m.start()].count("\n") + 1
        for interp in re.findall(r"\$\{([^}]{1,120})\}", m.group(1)):
            raw = interp.strip()
            if raw.startswith(allow_prefix):
                continue
            if allow_number.fullmatch(raw) and "<" not in raw and "`" not in raw:
                continue
            bad.append((line, raw[:70]))
    assert not bad, f"innerHTML 出现未审定的插值（先加 esc 或扩充豁免并说明理由）: {bad}"


def test_donut_tooltip_data_source_is_escaped():
    """donut 悬浮提示读 dataset.name，其渲染处必须已过 esc()。"""
    seg_render = re.search(r'data-name="\$\{(.*?)\}"', HTML)
    assert seg_render, "未找到 donut data-name 渲染点"
    assert seg_render.group(1).startswith("esc("), (
        "donut data-name 必须过 esc()（模型名来自请求记录，tooltip 以 innerHTML 呈现）"
    )


def test_single_polling_interval_with_guards():
    intervals = list(re.finditer(r"setInterval\(", HTML))
    assert len(intervals) == 1, f"轮询定时器应只有一个，实际 {len(intervals)}"
    body = HTML[intervals[0].start() : intervals[0].start() + 400]
    assert "document.hidden" in body, "轮询必须跳过后台隐藏页"
    assert "modal-open" in body, "弹窗打开时轮询必须挂起"
    assert "mainView" in body, "登录页阶段轮询必须挂起"


def test_no_document_level_listeners_inside_render_functions():
    """render* 内允许的监听：悬停提示（挂在新节点上随 GC 释放）与一次性
    DOMContentLoaded 引导；文档级（click/keydown/scroll/resize）不允许。"""
    forbidden = ("click", "keydown", "scroll", "resize", "mousedown", "blur", "change")
    bad = []
    for m in re.finditer(r"addEventListener\('([a-z]+)'", HTML):
        if _in_render(m.start()) and m.group(1) in forbidden:
            line = HTML[: m.start()].count("\n") + 1
            bad.append((line, m.group(1)))
    assert not bad, f"render 函数内出现文档级监听（重渲染会重复注册）: {bad}"


def test_no_string_concat_row_rendering():
    """行渲染必须是「模板数组 + join」或「整段 innerHTML 替换」，
    禁止循环里 `x.innerHTML +=` 逐行累加（O(n²) 且强制同步布局）。"""
    bad = []
    for m in re.finditer(r"\.innerHTML\s*\+=", HTML):
        if _in_render(m.start()):
            line = HTML[: m.start()].count("\n") + 1
            bad.append((line, HTML[m.start() : m.start() + 40]))
    assert not bad, f"render 内出现 innerHTML += 累加模式: {bad}"
