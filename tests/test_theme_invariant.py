"""一致性铁律：只有「自定义」主题才允许自定义底色生效（脏数据自愈）。"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "webai2api" / "web"
HTML = (ROOT / "index.html").read_text(encoding="utf-8")
CSS = (ROOT / "app.css").read_text(encoding="utf-8")


def test_bootstrap_drops_custom_bg_under_preset_theme():
    """启动时若主题不是 custom 却残留 wapi_custom_bg，必须清掉（用户现场事故）。"""
    seg = HTML[HTML.index("(function(){") : HTML.index("})();")]
    assert "'custom'" in seg.split("KNOWN")[1][:120], "启动白名单要包含 custom"
    assert "t !== 'custom' && cb" in seg, "预设主题下必须忽略并清除残留自定义色"
    assert "localStorage.removeItem('wapi_custom_bg')" in seg


def test_switching_to_custom_clears_inline_colors_before_writing():
    """applyTheme('custom') 会清内联色，所以必须先切主题、后写颜色。"""
    fn = HTML[
        HTML.index("function _ensureCustomTheme") : HTML.index("function pickMiniColor")
    ]
    assert "applyTheme('custom')" in fn
    assert "applyCustomBg" not in fn, "顺序反了会把刚写的底色清掉"


def test_hex_box_always_matches_real_background():
    seg = HTML[
        HTML.index("function renderBgPick") : HTML.index("function renderMiniPicker")
    ]
    assert "customTheme ? (cur || themeBaseBg()) : themeBaseBg()" in seg
    assert "readOnly = false" in seg, "输入框应随时可改（改了就进自定义主题）"


def test_no_locked_state_blocks_picker():
    seg = HTML[
        HTML.index("function toggleMiniPicker") : HTML.index("function closeMiniPicker")
    ]
    assert "return;" not in seg.split("if(!box)")[0], "不允许在预设主题下拦截打开调色板"
