"""迷你调色板（替代原生取色器）的回归测试。"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "webai2api" / "web"
HTML = (ROOT / "index.html").read_text(encoding="utf-8")
CSS = (ROOT / "app.css").read_text(encoding="utf-8")


def test_native_color_input_is_hidden_behind_custom_picker():
    """原生 <input type=color> 只做兜底，不能直接当点击入口。"""
    assert 'class="sr-color"' in HTML
    assert ".sr-color{display:none}" in CSS, (
        "隐藏的原生取色器不能占位（会露出 1px 黑边）"
    )
    assert 'id="bgSwatch"' in HTML
    assert 'onpointerdown="toggleMiniPicker(event)"' in HTML, (
        "必须用 pointerdown + preventDefault，否则同一次点击会被 document 的收起监听吃掉"
    )


def test_opening_picker_consumes_the_click():
    seg = HTML[
        HTML.index("function toggleMiniPicker") : HTML.index(
            "function openNativePicker"
        )
    ]
    assert "preventDefault()" in seg and "stopPropagation()" in seg
    bind = HTML[HTML.index("function _mpBind") : HTML.index("if(document.readyState")]
    assert "e.target.closest('#bgSwatch')" in bind, "点色块本身不应走「点外面收起」"
    assert "box.contains(e.target)" in bind, "面板内部点击不应收起"


def test_mini_picker_has_presets_and_ramps():
    for marker in (
        'id="mpPresets"',
        'id="mpLight"',
        'id="mpHue"',
        'id="mpHex"',
        'id="miniPicker"',
        "openNativePicker()",
    ):
        assert marker in HTML, f"迷你调色板缺 {marker}"
    seg = HTML[HTML.index("const MP_PRESETS=") : HTML.index("let _mpHsl")]
    assert seg.count("['#") == 16, "预设应为 16 个"


def test_both_ramps_drive_theme_color():
    """两条滑杆都必须真正改底色（并连带重算按钮色）。"""
    seg = HTML[HTML.index("function _mpBind") : HTML.index("if(document.readyState")]
    assert "mpHue" in seg and "mpLight" in seg
    assert seg.count("pickMiniColor(") >= 2
    # 明度滑杆画成当前色相的渐变
    assert "--mp-from" in HTML and "#mpLight{background:linear-gradient" in CSS


def test_mini_picker_closes_on_outside_and_escape():
    seg = HTML[HTML.index("function _mpBind") : HTML.index("if(document.readyState")]
    assert "closeMiniPicker()" in seg
    assert "Escape" in seg
