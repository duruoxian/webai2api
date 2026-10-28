"""行菜单（⋯ 下拉）浮层回归测试（2026-10-28 用户截图：菜单被行内按钮压住、
只露一条窄缝）。

根因：菜单原本 absolute 定位在行内，被 .tbl-scroll 的 overflow:auto 裁剪，
且 z-index 被邻近行的按钮盖住。修法：body 级 fixed 浮层 + 按按钮位置定位。
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "webai2api" / "web"
HTML = (ROOT / "index.html").read_text(encoding="utf-8")
CSS = (ROOT / "app.css").read_text(encoding="utf-8")


def test_row_menu_is_fixed_floating_layer():
    block = re.search(r"\.row-menu\{([^}]*)\}", CSS, re.S)
    assert block, ".row-menu 样式缺失"
    body = block.group(1)
    assert "position:fixed" in body, "行菜单必须是 fixed 浮层，不能被表格 overflow 裁剪"
    assert "z-index:900" in CSS, "行菜单层级需高于页面其它元素"


def test_open_row_menu_moves_to_body_and_measures_before_show():
    fn = HTML[HTML.index("function openRowMenu") : HTML.index("function closeRowMenus")]
    assert "document.body.appendChild(menu)" in fn, "菜单应挂到 body 上"
    # display:none 时量出的尺寸是 0，必须先显示再测
    assert "visibility" in fn and fn.index("getBoundingClientRect") > fn.index(
        "document.body.appendChild(menu)"
    )
    assert "innerHeight" in fn and "innerWidth" in fn, "需做视口内翻转/夹取"


def test_row_menu_closes_on_outside_click_and_escape():
    assert "closeRowMenus" in HTML
    assert re.search(r"keydown.*Escape", HTML, re.S), "Escape 应能收起行菜单"
    assert "_guardClick" in HTML, "打开时的同一次点击不应立刻收起"


def test_key_rows_restore_open_menu_after_poll():
    """轮询重渲染 Key 表后要恢复已打开的菜单（账号表早已有，Key 表曾漏）。"""
    seg = HTML[HTML.index("function renderKeys") : HTML.index("function renderUsage")]
    assert 'data-menu-id="key:${k.id}"' in seg
    assert "openRowMenu(b,true)" in seg
