"""面板主题系统一致性测试（静态检查，不打上游）。

2026-10-28 起新规格（用户拍板）：只做开发端 index.html（lite 面板冻结退役）；
主题收敛为 4 个（液态浅/深、极简白/黑）；右上角独立深浅切换按钮删除，
深浅语义由「液态深/液态浅」主题自表达。

约束链：THEMES 注册表 ↔ app.css 变量块 ↔ 启动白名单 ↔ 深色家族选择器组，
四处必须完全对齐——任何一处漏改都会表现为「设置里点了没反应」或
「深色组件闪白」。
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "webai2api" / "web"
PANEL = "index.html"
CORE_VARS = ("--bg", "--text", "--accent", "--surface-top",
             "--input-bg", "--shadow")

EXPECTED = {"light", "dark", "mono-light", "mono-dark"}
DARK_FAMILY = {"dark", "mono-dark"}
RETIRED_THEMES = {"paper", "sakura", "neon", "terminal"}


def _panel() -> str:
    return (ROOT / PANEL).read_text(encoding="utf-8")


def _themes_table(html: str) -> list[dict]:
    m = re.search(r"const THEMES=\[(.*?)\];", html, re.S)
    assert m, "THEMES 注册表缺失"
    ids = re.findall(r"\{id:'([a-z-]+)'", m.group(1))
    darks = re.findall(r"\{id:'([a-z-]+)'[^}]*dark:(true|false)", m.group(1))
    return [{"id": i, "dark": dict(darks)[i] == "true"} for i in ids]


def test_theme_registry_is_new_four():
    table = _themes_table(_panel())
    assert {t["id"] for t in table} == EXPECTED, "主题注册表应为 4 主题新规格"
    for retired in RETIRED_THEMES:
        assert retired not in {t["id"] for t in table}, f"已退役主题 {retired} 不应回归"


def test_theme_dark_flags():
    for t in _themes_table(_panel()):
        assert t["dark"] == (t["id"] in DARK_FAMILY), t


def test_css_has_variable_block_for_every_theme():
    css = (ROOT / "app.css").read_text(encoding="utf-8")
    for theme in EXPECTED - {"light"}:  # light 由 :root 承担
        m = re.search(rf"html\[data-theme={theme}\]\{{(.*?)\}}", css, re.S)
        assert m, f"app.css 缺 {theme} 变量块"
        for v in CORE_VARS:
            assert v in m.group(1), f"{theme} 块缺核心变量 {v}"
    # 每套主题都要有自己的背景渐变（light/dark 已有）
    for theme in EXPECTED - {"light"}:
        assert f"html[data-theme={theme}] body" in css, f"{theme} 缺背景渐变"


def test_dark_component_rules_cover_dark_family():
    """每一条组件级深色规则，选择器组必须同时覆盖 mono-dark。

    历史教训：曾出现新深色主题漏进家族导致组件闪白、选择器组
    「只有最后一项带后代」的批量替换事故。逐规则校验杜绝回归。
    （neon/terminal 的 CSS 块作为死代码保留，不要求新规则覆盖。）"""
    css = (ROOT / "app.css").read_text(encoding="utf-8")
    selector_groups = re.findall(r"([^{}]+)\{", css)
    dark_rules = []
    for group in selector_groups:
        sels = [s.strip() for s in group.strip().split(",")]
        if not any(s.startswith("html[data-theme=dark]") for s in sels):
            continue
        if sels == ["html[data-theme=dark]"]:   # 变量定义块，允许单独存在
            continue
        dark_rules.append(group.strip())
        for theme in sorted(DARK_FAMILY - {"dark"}):
            assert f"html[data-theme={theme}]" in group, \
                f"深色组件规则未覆盖 {theme}: {group[:70]}"
        # 家族后代的组件部分必须逐项一致（防「只给最后一项带后代」）
        for s in sels:
            m = re.match(r"html\[data-theme=([a-z-]+)\](.*)", s)
            if m and m.group(1) == "dark":
                desc = m.group(2)
                for theme in sorted(DARK_FAMILY - {"dark"}):
                    assert f"html[data-theme={theme}]{desc}" in group, \
                        f"家族后代不一致: {group[:70]}"
    assert len(dark_rules) >= 15, f"深色组件规则只有 {len(dark_rules)} 条"


def test_boot_whitelist_and_picker_ui():
    html = _panel()
    m = re.search(r"var KNOWN = \[([^\]]+)\];", html)
    assert m, "启动白名单缺失"
    known = set(re.findall(r"'([a-z-]+)'", m.group(1)))
    assert known == EXPECTED, f"白名单 {known} != {EXPECTED}"
    assert 'id="themeChips"' in html, "缺主题选择器容器"
    assert "renderThemeChips" in html, "缺选择器渲染函数"
    assert "isDarkTheme(currentTheme())" in html, "玻璃特效未用主题家族判定"


def test_no_standalone_light_dark_toggle():
    """新规格：独立深浅切换按钮已删除，深浅语义由「液态深/液态浅」表达。"""
    html = _panel()
    assert "toggleTheme" not in html, "toggleTheme 应已删除（切换按钮嵌入主题列表）"
    assert "login-theme" not in html and 'id="themeBtn"' not in html, \
        "登录页与顶栏的月亮按钮应已删除"


def test_mono_themes_restyle_buttons_but_keep_semantic_variants():
    """极简主题把主按钮染成黑白，但 .btn.gray / .btn.danger 的语义色
    必须显式恢复（mono .btn 覆写特异性更高，不恢复会连带染黑）。"""
    css = (ROOT / "app.css").read_text(encoding="utf-8")
    for theme in ("mono-light", "mono-dark"):
        assert f"html[data-theme={theme}] .btn{{" in css
        assert f"html[data-theme={theme}] .btn.gray{{" in css,             f"{theme} 缺 .btn.gray 语义恢复"
        assert f"html[data-theme={theme}] .btn.danger{{" in css,             f"{theme} 缺 .btn.danger 语义恢复"
    # 极简 = 无彩色背景渐变
    for theme in ("mono-light", "mono-dark"):
        blocks = re.findall(rf"html\[data-theme={theme}\] body\{{(.*?)\}}", css, re.S)
        assert any("background-image:none" in b for b in blocks),             f"{theme} 背景必须是纯色（极简无渐变）"


def test_panel_window_persists_localstorage():
    """主题选择存在 localStorage，面板窗口必须关闭 pywebview 隐私模式。

    事故：run.py 的 webview.start 未传 private_mode（默认 True），
    WebView2 以 InPrivate 运行 → localStorage 每次启动被清空 →
    换的主题重开软件就回到默认（同时每次都要重新输管理密码）。
    """
    run_py = (ROOT.parent.parent / "run.py").read_text(encoding="utf-8")
    assert "private_mode=False" in run_py,             "面板窗口必须显式关闭 pywebview 隐私模式"
    assert "panel_profile" in run_py,             "面板存储必须指向数据目录的 panel_profile"


def test_panel_critical_markers():
    pro = _panel()
    for marker in ("cleanupBrowserResidue", "/browser/cleanup",
                   "wapi_theme", "browserLogin", "openAccModal",
                   "renderKeys", "THEMES"):
        assert marker in pro, f"开发端面板缺关键标记: {marker}"


def test_primary_button_follows_theme_accent():
    """主按钮渐变必须走 --accent-grad 变量（:root 定义；mono 系按钮被 .btn
    覆写为纯黑白，无需各自定义渐变）。"""
    css = (ROOT / "app.css").read_text(encoding="utf-8")
    assert "background:var(--accent-grad)" in css, ".btn 必须使用 --accent-grad"
    root_block = re.search(r":root\{(.*?)\}", css, re.S)
    assert root_block and "--accent-grad:" in root_block.group(1), ":root 缺 --accent-grad"


def test_dark_family_inputs_have_visible_border():
    """深色家族输入框必须有可见描边（无边框时与卡片融成低对比团块）。"""
    css = (ROOT / "app.css").read_text(encoding="utf-8")
    for theme in ("dark", "mono-dark"):
        assert (f"html[data-theme={theme}] input" in css
                and "border-color:rgba(255,255,255,.17)" in css), \
            f"{theme} 输入框描边缺失"


def test_pattern_system_exists_and_off_by_default():
    """纹理系统：8 种（含第四批新增 cross/waves/weave/veil），默认 display:none。"""
    css = (ROOT / "app.css").read_text(encoding="utf-8")
    for pat in ("dots", "grid", "lines", "cross", "waves", "weave", "veil"):
        assert f"html[data-pattern={pat}] body::before" in css, f"纹理 {pat} 缺失"
    assert "body::before{content:\"\";position:fixed;inset:0;pointer-events:none;z-index:0;display:none}" in css


def test_accent_override_presets_exist():
    """调色板换色：强调色预设必须同时覆盖 --accent 与 --accent-grad。"""
    css = (ROOT / "app.css").read_text(encoding="utf-8")
    for acc in ("violet", "pink", "red", "green", "orange", "cyan"):
        block = re.search(rf"html\[data-accent={acc}\]\{{(.*?)\}}", css, re.S)
        assert block, f"强调色预设 {acc} 缺失"
        assert "--accent:" in block.group(1) and "--accent-grad:" in block.group(1)


def test_custom_background_color_support():
    """第四批新增：自定义背景调色盘——写 --bg 内联样式 + localStorage 持久化。"""
    html = _panel()
    assert "applyCustomBg" in html and "wapi_custom_bg" in html, "缺自定义背景支持"
    assert 'id="bgPick"' in html, "缺调色盘 input[type=color]"
    assert "--bg', cb" in html.replace('"', "'") or "setProperty('--bg'" in html, \
        "启动时应把持久化的自定义底色应用到 --bg"
