"""单面板规格测试（2026-10-28 用户拍板：lite 面板冻结退役）。

约束：面板只伺服 index.html；lite 的差异仅剩数据目录分流（edition.txt）、
图标与安装包名，不得再从 edition 分派面板文件。
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


def test_main_spec_no_longer_needs_edition():
    spec = _read("webai2api.spec")
    assert "webai2api.edition" not in spec, "专业版 spec 不再动态 import edition"


def test_lite_spec_drops_lite_panel_and_renames_app():
    spec = _read("webai2api-lite.spec")
    assert "lite_panel.html" not in spec, "lite spec 不再打包 lite 面板"
    assert 'name="webai2api-lite"' in spec, "lite 产物应与专业版目录名区分"
    assert "webai2api.edition" in spec, "lite spec 仍需 edition（数据目录分流）"


def test_workflow_uses_per_edition_appname():
    yml = _read(".github/workflows/build-release.yml")
    assert "APPNAME=webai2api-lite" in yml and "APPNAME=webai2api" in yml
    assert '"$env:DIST/$env:APPNAME"' in yml, "打包步骤必须按版本取产物目录"
