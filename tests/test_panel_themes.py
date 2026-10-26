"""面板主题系统一致性测试（静态检查，不打上游）。

约束链：两面板的 THEMES 注册表 ↔ app.css 变量块 ↔ 启动白名单 ↔
深色家族选择器组，四处必须完全对齐——任何一处漏加主题都会表现为
「设置里点了没反应」或「深色组件在霓虹主题下闪白」。
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "webai2api" / "web"
PANELS = ["index.html", "lite_panel.html"]
CORE_VARS = ("--bg", "--text", "--accent", "--surface-top",
             "--input-bg", "--shadow")

EXPECTED = {"light", "dark", "paper", "sakura", "neon", "terminal",
             "mono-light", "mono-dark"}
DARK_FAMILY = {"dark", "neon", "terminal", "mono-dark"}
LIGHT_FAMILY = {"light", "paper", "sakura", "mono-light"}


def _panel(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def _themes_table(html: str) -> list[dict]:
    m = re.search(r"const THEMES=\[(.*?)\];", html, re.S)
    assert m, "THEMES 注册表缺失"
    ids = re.findall(r"\{id:'([a-z-]+)'", m.group(1))
    darks = re.findall(r"\{id:'([a-z-]+)'[^}]*dark:(true|false)", m.group(1))
    return [{"id": i, "dark": dict(darks)[i] == "true"} for i in ids]


def test_both_panels_have_identical_theme_registry():
    tables = [_themes_table(_panel(p)) for p in PANELS]
    assert {t["id"] for t in tables[0]} == EXPECTED
    assert tables[0] == tables[1], "Pro 与 Lite 的主题表不一致"


def test_theme_dark_flags():
    for t in _themes_table(_panel(PANELS[0])):
        assert t["dark"] == (t["id"] in DARK_FAMILY), t


def test_css_has_variable_block_for_every_theme():
    css = (ROOT / "app.css").read_text(encoding="utf-8")
    for theme in EXPECTED - {"light"}:  # light 由 :root 承担
        m = re.search(rf"html\[data-theme={theme}\]\{{(.*?)\}}", css, re.S)
        assert m, f"app.css 缺 {theme} 变量块"
        for v in CORE_VARS:
            assert v in m.group(1), f"{theme} 块缺核心变量 {v}"
    # 每套主题都要有自己的背景渐变（light/dark 已有，新主题追加）
    for theme in EXPECTED - {"light"}:
        assert f"html[data-theme={theme}] body" in css, f"{theme} 缺背景渐变"


def test_dark_component_rules_cover_dark_family():
    """每一条组件级深色规则，选择器组必须同时覆盖 dark/neon/terminal。

    历史教训：曾用批量替换生成「只有最后一项带后代」的错误选择器组
    （html[A],html[B],html[C] body → 前两项命中整个 html），也出现过
    新深色主题漏进家族导致组件闪白。逐规则校验杜绝这两类回归。
    """
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
        # 家族三项的后代部分必须逐项一致（防「只给最后一项带后代」）
        for s in sels:
            m = re.match(r"html\[data-theme=([a-z-]+)\](.*)", s)
            if m and m.group(1) == "dark":
                desc = m.group(2)
                for theme in sorted(DARK_FAMILY - {"dark"}):
                    assert f"html[data-theme={theme}]{desc}" in group, \
                        f"家族后代不一致: {group[:70]}"
    assert len(dark_rules) >= 15, f"深色组件规则只有 {len(dark_rules)} 条"


def test_boot_whitelist_and_picker_ui_in_both_panels():
    for p in PANELS:
        html = _panel(p)
        m = re.search(r"var KNOWN = \[([^\]]+)\];", html)
        assert m, f"{p} 启动白名单缺失"
        known = set(re.findall(r"'([a-z-]+)'", m.group(1)))
        assert known == EXPECTED, f"{p} 白名单 {known} != {EXPECTED}"
        assert 'id="themeChips"' in html, f"{p} 缺主题选择器容器"
        assert "renderThemeChips" in html, f"{p} 缺选择器渲染函数"
        assert "isDarkTheme(currentTheme())" in html, f"{p} 玻璃特效未用主题家族判定"


def test_toggle_keeps_quick_light_dark_semantics():
    html = _panel(PANELS[0])
    m = re.search(r"function toggleTheme\(\)\{(.*?)\n\}", html, re.S)
    assert m and "isDarkTheme(currentTheme())?'light':'dark'" in m.group(1), \
        "顶栏按钮应保持深色家族↔浅色的快速切换语义"


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
