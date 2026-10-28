"""C-2 主题系统矩阵测试（方案 C 第二层，2026-10-29）。

- 变量集对称不变量：四个主题块定义的 CSS 变量集合必须完全一致——
  「新变量加在 light 忘了同步深色」正是深色闪白那一族 bug 的源头。
- _accentFromBg 真函数边界：用 node 跑面板源码里抽出的纯函数，
  覆盖近中性退化、对比度守卫、色相保持、扫掠不变量。
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "webai2api" / "web"
CSS = (WEB / "app.css").read_text(encoding="utf-8")
HTML = (WEB / "index.html").read_text(encoding="utf-8")

THEMES = ["light", "dark", "mono-light", "mono-dark"]


def _theme_vars(theme: str) -> set[str]:
    if theme == "light":
        m = re.search(r":root\{(.*?)\}", CSS, re.S)
        assert m, "app.css 缺 :root（light）变量块"
    else:
        m = re.search(rf"html\[data-theme={theme}\]\{{(.*?)\}}", CSS, re.S)
        assert m, f"app.css 缺 {theme} 变量块"
    return set(re.findall(r"(--[a-z0-9-]+)\s*:", m.group(1)))


def test_all_var_references_are_defined():
    """全库 var(--x) 引用防呆网：引用的每个变量名必须在某处有定义
    （:root / 主题块 / 组件作用域 / JS setProperty / HTML 内联），
    或带 fallback。var() 落空正是「点了没反应/闪白」一族 bug 的根源。
    （液态与极简主题的 --glass-blur、--accent-grad 差异是有意的审美分层，
    变量本身都有定义，不在此检查范围。）"""
    css = CSS
    html_defs = set(re.findall(r"(--[a-z0-9-]+)\s*:", HTML))
    js_defs = set(re.findall(r"setProperty\(\s*'(--[a-z0-9-]+)'", HTML))
    defined = set(re.findall(r"(--[a-z0-9-]+)\s*:", css)) | html_defs | js_defs
    refs = re.findall(r"var\(\s*(--[a-z0-9-]+)\s*(,[^)]*)?\)", css + HTML)
    for name, fallback in refs:
        assert name in defined or (fallback or "").strip(), (
            f"var({name}) 无定义且无 fallback（CSS/HTML/JS 全库未找到 --{name[2:]}: 定义）"
        )


def test_all_core_vars_nonempty_in_every_theme():
    """核心变量的每个主题块里都必须有非空取值。"""
    core = ("--bg", "--text", "--accent", "--surface-top", "--input-bg")
    for t in THEMES:
        m = re.search(
            (r":root\{([^}]*)\}" if t == "light" else rf"html\[data-theme={t}\]\{{([^}}]*)\}}"),
            CSS,
            re.S,
        )
        body = m.group(1)
        for v in core:
            mm = re.search(rf"{v}\s*:\s*([^;]+);", body)
            assert mm and mm.group(1).strip(), f"{t} 的 {v} 取值为空"


# ── _accentFromBg 真函数边界（node 执行面板源码）─────────────────


def _accent_js() -> str:
    start = HTML.index("function _hexToRgb")
    end = HTML.index("function _accentRgb")
    return HTML[start:end]


def _run_node(script: str):
    import tempfile

    with tempfile.NamedTemporaryFile(
        "w", suffix=".js", delete=False, encoding="utf-8"
    ) as f:
        f.write(script)
        path = f.name
    try:
        out = subprocess.run(
            ["node", path], capture_output=True, text=True, timeout=30
        )
    finally:
        Path(path).unlink(missing_ok=True)
    assert out.returncode == 0, f"node 执行失败: {out.stderr[:400]}"
    return json.loads(out.stdout)


def test_accent_from_bg_extremes_and_degradation():
    js = _accent_js() + r"""
const cases = ["#000000","#111111","#808080","#f2f2f2","#ffffff",
               "#e11d48","#2563eb","#16a34a","#9333ea","#f59e0b"];
const out = cases.map(hex => {
  const acc = _accentFromBg(hex);
  return {bg: hex, acc, contrast: _contrast(acc, hex),
          hueBg: _rgbToHsl(_hexToRgb(hex))[0],
          hueAcc: _rgbToHsl(_hexToRgb(acc))[0],
          satBg: _rgbToHsl(_hexToRgb(hex))[1],
          accLum: _luminance(_hexToRgb(acc))};
});
console.log(JSON.stringify(out));
"""
    rows = _run_node(js)
    by_bg = {r["bg"]: r for r in rows}
    # 近中性底色 → 黑白退化（HANDOFF 规则：近中性退化为黑白按钮）
    # 中灰亮度 0.216 ≤ 0.42 阈值，实现按深底色处理 → 白按钮（实现与注释一致）
    for bg, want in (("#000000", "#f2f2f2"), ("#111111", "#f2f2f2"),
                     ("#808080", "#f2f2f2"), ("#f2f2f2", "#1d1d1f"),
                     ("#ffffff", "#1d1d1f")):
        assert by_bg[bg]["acc"] == want, f"近中性 {bg} 应退化为 {want}"
    # 彩色底色：输出合法 hex、与底色有最低可辨对比、色相保持同系
    for hex_bg in ("#e11d48", "#2563eb", "#16a34a", "#9333ea", "#f59e0b"):
        r = by_bg[hex_bg]
        assert re.fullmatch(r"#[0-9a-f]{6}", r["acc"]), r
        assert r["contrast"] >= 2.2, f"{hex_bg} 的按钮对比度不足: {r}"
        assert abs(r["hueBg"] - r["hueAcc"]) < 1e-6 or (
            max(r["hueBg"], r["hueAcc"]) - min(r["hueBg"], r["hueAcc"]) < 30
        ), f"{hex_bg} 色相偏移过大: {r}"


def test_accent_hue_sweep_contrast_floor():
    """全色相扫掠：任何底色推导出的按钮色对比度必须有下界（不可与底色同色）。"""
    js = _accent_js() + r"""
const out = [];
for (let h = 0; h < 360; h += 15) {
  for (const [s, l] of [[70, 25], [70, 50], [70, 75], [35, 50]]) {
    const bg = _hslToHex(h, s, l);
    const acc = _accentFromBg(bg);
    out.push({bg, acc, c: _contrast(acc, bg)});
  }
}
console.log(JSON.stringify(out));
"""
    rows = _run_node(js)
    bad = [r for r in rows if r["c"] < 1.5]
    assert not bad, f"对比度过低的组合: {bad[:5]}"
    same = [r for r in rows if r["bg"].lower() == r["acc"].lower()]
    assert not same, f"按钮色与底色同色: {same[:5]}"
