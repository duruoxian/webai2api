"""自定义主题：底色推导按钮色 的回归测试（2026-10-28 用户要求五种主题）。"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "webai2api" / "web"
HTML = (ROOT / "index.html").read_text(encoding="utf-8")
CSS = (ROOT / "app.css").read_text(encoding="utf-8")


def test_five_themes_with_custom_entry():
    seg = HTML[HTML.index("const THEMES=[") : HTML.index("const CUSTOM_THEME")]
    ids = [x for x in ("mono-light", "mono-dark", "light", "dark") if f"'{x}'" in seg]
    assert len(ids) == 4, "四个预设主题应保留"
    assert "CUSTOM_THEME={id:'custom',name:'自定义'}" in HTML, "第五个主题是「自定义」"
    seg2 = HTML[HTML.index("function renderThemeChips") : HTML.index("/* ── 液玻璃")]
    assert "THEMES.map" in seg2 and "applyTheme('custom')" in seg2, (
        "主题列表需渲染五项（四个预设 + 自定义）"
    )


def test_custom_theme_derives_accent_from_background():
    """按钮色必须由底色推导（同色系 + 对比度），不能沿用固定强调色。"""
    assert "function _accentFromBg" in HTML
    fn = HTML[HTML.index("function _accentFromBg") : HTML.index("function _accentRgb")]
    assert "_rgbToHsl" in fn and "_hslToHex" in fn, "需走 HSL 同色相推导"
    assert "_contrast" in fn, "需按对比度迭代保证可读"
    assert "s < 10" in fn, "近灰底应退化为黑白按钮"


def test_custom_theme_sets_button_foreground():
    fn = HTML[
        HTML.index("function _applyCustomTheme") : HTML.index(
            "function _clearCustomTheme"
        )
    ]
    for var in (
        "--accent",
        "--accent-hover",
        "--accent-grad",
        "--accent-fg",
        "--accent-soft",
    ):
        assert var in fn, f"自定义主题需写入 {var}"
    assert "html[data-theme=custom] .btn{color:var(--accent-fg" in CSS


def test_switching_back_to_preset_clears_custom_state():
    fn = HTML[
        HTML.index("function _clearCustomTheme") : HTML.index(
            "function currentCustomBg"
        )
    ]
    assert "removeAttribute('data-bgmode')" in fn
    assert "removeAttribute('data-theme')" in fn
    seg = HTML[
        HTML.index("function applyTheme") : HTML.index("function renderThemeChips")
    ]
    assert "_clearCustomTheme()" in seg, "切回预设主题必须摘掉自定义内联色"


def test_custom_bg_editable_only_in_custom_theme():
    seg = HTML[HTML.index("function renderThemeChips") : HTML.index("/* ── 液玻璃")]
    assert "is-active" in seg
    assert "#customBgRow:not(.is-active)" in CSS
