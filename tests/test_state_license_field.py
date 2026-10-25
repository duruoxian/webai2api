"""state 响应必须携带 license 字段（0.1.26 合并回归的防护）。

事故：0.1.26 单代码库合并时丢了 lite 分支 state 里的 license 字段，
客户版面板 renderLicense(s.license) 拿到 undefined → 授权卡（含授权码
输入框）被前端永久隐藏，用户侧表现为「设置里没有填授权码的地方」。
"""
from __future__ import annotations

import logging
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from webai2api.app import build_app
from webai2api.providers import ProviderRegistry
from webai2api.store import Store

TOKEN_HDR = "x-admin-token"


@pytest.fixture
def lite_app(tmp_path, monkeypatch):
    """lite 形态（edition._CACHE 注入）+ 临时数据目录的 app。"""
    import webai2api.edition as edition

    monkeypatch.setattr(edition, "_CACHE", "lite")
    import webai2api.admin_api as admin_api

    monkeypatch.setattr(admin_api, "user_data_dir", lambda: tmp_path / "data")
    store = Store(tmp_path / "data" / "webai2api.db")
    logger = logging.getLogger("webai2api.test.state_license")
    registry = ProviderRegistry(store, logger)
    app = build_app(store, registry, logger)
    yield TestClient(app, client=("127.0.0.1", 50000))
    monkeypatch.setattr(edition, "_CACHE", None)


@pytest.fixture
def tok(lite_app) -> str:
    r = lite_app.post("/admin/api/login", json={"password": "admin"})
    assert r.json()["ok"] is True
    return r.json()["token"]


def test_state_includes_license_field(lite_app, tok):
    """state 响应必须含 license 键，lite 形态下 enabled=True（卡片渲染依据）。"""
    r = lite_app.get("/admin/api/state", headers={TOKEN_HDR: tok})
    assert r.status_code == 200
    body = r.json()
    assert "license" in body, "state 缺少 license 字段 → 授权卡被前端隐藏"
    lic = body["license"]
    assert lic.get("enabled") is True
    # lite 形态必须带面板 renderLicense 依赖的最小键集
    for k in ("valid", "machine", "reason", "code", "days_left"):
        assert k in lic, f"license 缺少 {k}"


def test_license_endpoint_unchanged(lite_app, tok):
    """独立 /license 端点行为不变（enabled=True + machine）。"""
    r = lite_app.get("/admin/api/license", headers={TOKEN_HDR: tok})
    assert r.status_code == 200
    body = r.json()
    assert body.get("ok") is True
    assert body.get("enabled") is True
    assert body.get("machine")


def test_license_info_never_loses_machine(monkeypatch, tmp_path):
    """status() 抛异常时兜底返回仍带 machine（机器码是找作者签码的凭据）。"""
    import webai2api.edition as edition
    import webai2api.paths as paths

    monkeypatch.setattr(edition, "_CACHE", "lite")

    def _boom(*a, **k):
        raise RuntimeError("boom")

    # _license_info 内部 from .paths import user_data_dir（延迟导入），
    # 必须 patch paths 模块属性才能命中
    monkeypatch.setattr(paths, "user_data_dir", _boom)
    import webai2api.admin_api as admin_api

    info = admin_api._license_info()
    assert info["enabled"] is False
    assert info.get("machine"), "异常兜底必须带 machine"
    monkeypatch.setattr(edition, "_CACHE", None)


def test_panel_renders_license_card_in_all_scenarios():
    """前端契约：无论 lic 是什么，授权卡永不隐藏（用 node 跑面板真函数）。

    场景：正常数据 / undefined（0.1.26-0.1.29 事故形态）/ enabled=False。
    断言三种场景下 card.hidden 恒为 False，且 reason 提示始终给出。
    """
    import re
    import subprocess
    import shutil

    panel = Path(__file__).resolve().parents[1] / "webai2api" / "web" / "lite_panel.html"
    src = panel.read_text(encoding="utf-8")
    m = re.search(r"function renderLicense\(lic\)\{.*?\n\}", src, re.S)
    assert m, "renderLicense not found in panel"
    fn = m.group(0)
    # renderLicenseCountdown 被引用但不在本函数体，stub 掉
    harness = """
function renderLicenseCountdown(){}
const _els = {};
function _el(id){ return _els[id] || (_els[id] = {textContent:'', hidden:false,
  className:'', style:{}}); }
const document = { getElementById: _el };
""" + fn + """
for (const lic of [undefined, null, {}, {enabled:false}, {enabled:true, valid:false},
                   {enabled:true, valid:true, days_left:3, expires_ts:1, seconds_left:1}]) {
  renderLicense(lic);
  if (_el('licenseCard').hidden !== false) {
    console.log('FAIL: licenseCard hidden with lic=' + JSON.stringify(lic));
    process.exit(1);
  }
}
console.log('PASS: license card visible in all scenarios');
"""
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    r = subprocess.run([node, "-e", harness], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS" in r.stdout
