"""state 响应必须携带 license 字段（0.1.26 合并回归的防护）。

事故：0.1.26 单代码库合并时丢了 lite 分支 state 里的 license 字段，
客户版面板 renderLicense(s.license) 拿到 undefined → 授权卡（含授权码
输入框）被前端永久隐藏，用户侧表现为「设置里没有填授权码的地方」。
"""
from __future__ import annotations

import logging
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
