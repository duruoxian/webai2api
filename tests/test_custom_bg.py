"""背景颜色自定义回归测试（2026-10-28 用户报障：点了色板完全没反应）。

根因：极简主题把 `body{background-color:#fafafa}` 写死，特异性高于
`body{background-color:var(--bg)}`，所以只改 `--bg` 变量对底色无效。
修法：自定义时给 <html> 打 data-bgmode（light/dark），由
`html[data-bgmode] body{background-color:var(--bg)}` 抬高特异性并适配明暗。
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "webai2api" / "web"
HTML = (ROOT / "index.html").read_text(encoding="utf-8")
CSS = (ROOT / "app.css").read_text(encoding="utf-8")


def test_custom_bg_sets_bgmode_attribute():
    assert "data-bgmode" in HTML
    # data-bgmode 现在由 _applyCustomTheme 统一设置（底色 + 按钮色一并推导）
    seg = HTML[
        HTML.index("function _applyCustomTheme") : HTML.index(
            "function _clearCustomTheme"
        )
    ]
    assert "setAttribute('data-bgmode'" in seg, "自定义主题必须打 data-bgmode"
    assert "removeAttribute('data-bgmode')" in HTML, (
        "切回预设主题时必须摘掉 data-bgmode"
    )


def test_bgmode_css_overrides_hardcoded_mono_body():
    assert re.search(r"html\[data-bgmode\] body\{background-color:var\(--bg\)", CSS), (
        "必须用 data-bgmode 抬高特异性覆盖极简主题写死的 body 底色"
    )
    assert "html[data-bgmode=dark]{" in CSS, "暗底需要独立的变量块"


def test_bgmode_variables_cover_text_and_surfaces():
    """暗底自定义时，文字/输入框/分隔线都要翻转，否则白字黑字看不清。"""
    block = re.search(r"html\[data-bgmode=dark\]\{(.*?)\}", CSS, re.S).group(1)
    for var in ("--text:", "--text2:", "--text3:", "--hairline:", "--surface-top:"):
        assert var in block, f"暗底变量块缺 {var}"


def test_bootstrap_applies_saved_custom_bg_with_mode():
    """首绘 IIFE 也要恢复自定义底色并算好明暗，避免开面板闪一下默认色。"""
    seg = HTML[HTML.index("(function(){") : HTML.index("})();")]
    assert "wapi_custom_bg" in seg
    assert "data-bgmode" in seg


def test_hex_input_validates():
    seg = HTML[
        HTML.index("function _hexToRgb") : HTML.index("function currentCustomBg")
    ]
    assert "/^[0-9a-fA-F]{6}$/" in seg, "十六进制输入必须校验，非法值不写入"
    assert "function applyCustomHex" in HTML


def test_appearance_panel_is_structured():
    """外观区（2026-10-28 第三版）：行式布局，主题列表 + 自定义背景 + 纹理。

    用户拍板：取消强调色与预设色板，主题只留四个 + 自定义背景，
    按钮颜色一律跟随主题（不设独立强调色）。
    """
    for marker in (
        'class="setup-row"',
        'class="theme-list" id="themeChips"',
        'id="customBgRow"',
        'class="chip-seg" id="patternChips"',
        'id="pickerFace"',
        'id="bgResetBtn"',
    ):
        assert marker in HTML, f"外观区缺结构标记 {marker}"
    # 强调色与预设色板已下线
    assert 'id="accentChips"' not in HTML, "强调色选择器应已移除"
    assert "renderAccentChips" not in HTML, "强调色渲染函数应已移除"
    assert 'id="bgDots"' not in HTML, "预设色板应已移除"
    # 历史的 data-accent 存值要清掉，否则按钮仍被旧色染
    assert "localStorage.removeItem('wapi_accent')" in HTML
