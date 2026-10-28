"""单版本规格测试（2026-10-28 用户拍板：lite 退役，只留开发端；
2026-10-29 物理删除完成——edition 机制、lite spec/面板/安装器/图标全灭）。

约束：全仓库不得再出现 lite 版分流——无 edition.py、无 lite spec、
无 lite 面板、无 -lite tag 分支、无 lite 图标；面板只伺服 index.html。
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(p: str) -> str:
    return (ROOT / p).read_text(encoding="utf-8")


def test_admin_route_serves_index_only():
    src = _read("webai2api/app.py")
    assert 'panel = "index.html"' in src
    assert "lite_panel" not in src, "app.py 不应再引用 lite 面板"
    assert "is_lite" not in src, "面板分派不应再由 edition 决定"


def test_edition_module_exterminated():
    assert not (ROOT / "webai2api" / "edition.py").exists(), "edition.py 应已物理删除"
    for f in (
        "run.py",
        "webai2api/tray.py",
        "webai2api/paths.py",
    ):
        assert "edition" not in _read(f), f"{f} 不应再引用 edition 机制"
    for f in ("webai2api/paths.py", "webai2api/tray.py", "run.py"):
        assert "webai2api-lite" not in _read(f), f"{f} 不应再有 lite 产物名"


def test_lite_artifacts_gone():
    for p in (
        "webai2api-lite.spec",
        "webai2api/web/lite_panel.html",
        "installer/webai2api-lite.iss",
        "scripts/usage-lite.txt",
        "assets/webai2api-lite.ico",
        "assets/webai2api-lite.png",
        "tests/test_paths_edition.py",
    ):
        assert not (ROOT / p).exists(), f"{p} 应已物理删除"


def test_workflow_single_edition():
    yml = _read(".github/workflows/build-release.yml")
    assert "webai2api-lite" not in yml, "CI 不应再有 lite 分支"
    assert "EDITION" not in yml, "CI 不应再有版本选择变量"
    assert '"$env:DIST/$env:APPNAME"' in yml, "打包步骤必须按变量取产物目录"


def test_main_spec_still_edition_free():
    spec = _read("webai2api.spec")
    assert "edition" not in spec
    assert "lite_panel" not in spec
