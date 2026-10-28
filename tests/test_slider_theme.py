"""滑块驱动自定义主题：「不动滑块就不改背景」的回归测试。"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "webai2api" / "web"
HTML = (ROOT / "index.html").read_text(encoding="utf-8")
CSS = (ROOT / "app.css").read_text(encoding="utf-8")


def test_mp_presets_removed_from_ui():
    """预设色块已按用户要求移除，只保留两条滑杆。"""
    assert 'id="mpPresets" role="radiogroup" aria-label="预设背景色" hidden' in HTML
    assert ".mp-presets[hidden]{display:none}" in CSS


def test_opening_picker_does_not_touch_background():
    """打开调色板/切主题都不得改背景；只有滑块 input 才接管。"""
    ok = HTML[
        HTML.index("function toggleMiniPicker") : HTML.index("function closeMiniPicker")
    ]
    assert "applyCustomBg" not in ok and "applyCustomHex" not in ok
    # applyTheme 只写主题名，自定义只在 custom 分支里发生
    theme = HTML[
        HTML.index("function applyTheme") : HTML.index("function renderThemeChips")
    ]
    assert "_clearCustomTheme()" in theme
    custom_branch = theme.split("if(t==='custom')")[1].split("return;")[0]
    assert "applyCustomBg" not in custom_branch, "选到自定义主题也不得自动写底色"


def test_slider_input_switches_to_custom_theme():
    assert "function _ensureCustomTheme" in HTML
    seg = HTML[HTML.index("function _mpBind") : HTML.index("if(document.readyState")]
    assert seg.count("_ensureCustomTheme()") == 2, (
        "明度与色相滑杆都要先确保进入自定义主题"
    )


def test_accent_follows_background_hue():
    """按钮色必须来自自定义底色（同色相），所以底色一变按钮就要变。"""
    fn = HTML[HTML.index("function _accentFromBg") : HTML.index("function _accentRgb")]
    assert "_rgbToHsl" in fn and "_hslToHex" in fn
    assert "s < 10" in fn, "近中性底色退化为黑白按钮"
    apply_fn = HTML[
        HTML.index("function _applyCustomTheme") : HTML.index(
            "function _clearCustomTheme"
        )
    ]
    assert "--accent" in apply_fn, "自定义底色必须重算 --accent"
