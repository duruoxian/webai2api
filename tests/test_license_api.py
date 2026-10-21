"""GET /admin/api/license 与 /admin/api/license/issued 的契约。

main 版（客户版）没有授权功能：
    GET  /admin/api/license         → enabled=False
    GET  /admin/api/license/issued  → 本机有签发器时列出签发记录，否则正常空列表
    POST /admin/api/license/issue   → 无签发器环境必须拒绝
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from webai2api import issuer
from webai2api.app import build_app
from webai2api.providers import ProviderRegistry
from webai2api.store import Store

TOKEN_HDR = "x-admin-token"


@pytest.fixture
def _app_singleton(tmp_path_factory):
    d = tmp_path_factory.mktemp("license_app")
    store = Store(d / "data" / "webai2api.db")
    logger = logging.getLogger("webai2api.test.license")
    registry = ProviderRegistry(store, logging.getLogger("webai2api.test.license.providers"))
    return build_app(store, registry, logger), store, d


@pytest.fixture
def client(_app_singleton):
    return TestClient(_app_singleton[0], client=("127.0.0.1", 50000))


@pytest.fixture
def tok(client) -> str:
    r = client.post("/admin/api/login", json={"password": "admin"})
    assert r.json()["ok"] is True
    return r.json()["token"]


def _hdr(tok: str) -> dict:
    return {TOKEN_HDR: tok}


@pytest.fixture
def no_issuer(monkeypatch, tmp_path):
    """把作者私钥查找路径全部指向不存在的位置。"""
    monkeypatch.setenv("WEBAI2API_AUTHOR_KEY", str(tmp_path / "no-such-key.txt"))
    monkeypatch.setattr(issuer, "author_key_path", lambda: tmp_path / "no-such-key.txt")
    assert issuer.load_author_seed() is None
    return tmp_path


# ── /admin/api/license ─────────────────────────────
def test_license_requires_login(client):
    r = client.get("/admin/api/license")
    assert r.status_code == 401
    assert r.json() == {"ok": False, "error": "未登录"}


def test_license_disabled_on_main(client, tok):
    r = client.get("/admin/api/license", headers=_hdr(tok))
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["enabled"] is False


# ── /admin/api/license/issued ──────────────────────
def test_issued_requires_login(client):
    r = client.get("/admin/api/license/issued")
    assert r.status_code == 401 and r.json()["ok"] is False


def test_issued_without_issuer(client, tok, no_issuer):
    """没有签发器时列表接口不能报错：返回结构完整的（空或既有）列表。"""
    r = client.get("/admin/api/license/issued", headers=_hdr(tok))
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"ok", "items", "total", "matched"}
    assert body["ok"] is True
    assert isinstance(body["items"], list)
    assert body["total"] >= len(body["items"]) or body["total"] == 0
    assert body["matched"] == len(body["items"])


def test_issued_reads_log_without_issuer(client, tok, no_issuer, monkeypatch, tmp_path):
    """签发记录文件存在时，即使没有私钥也应能读（历史记录只是展示）。"""
    log = tmp_path / "issued_codes.json"
    log.write_text(json.dumps([
        {"code": "WB2.x.y", "id": "AAAA0001", "days": 30, "issued": "2026-01-01",
         "note": "老客户", "created": "2026-01-01 10:00:00"},
        {"code": "WB2.p.q", "id": "BBBB0002", "days": 7, "issued": "2026-01-02",
         "note": "", "created": "2026-01-02 10:00:00"},
    ], ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr("webai2api.admin_api._issuer_log_path", lambda: log)

    body = client.get("/admin/api/license/issued", headers=_hdr(tok)).json()
    assert body["ok"] is True and body["total"] == 2
    assert body["items"][0]["id"] == "BBBB0002"  # 倒序：最新在前

    body = client.get("/admin/api/license/issued?q=bbbb", headers=_hdr(tok)).json()
    assert body["matched"] == 1 and body["total"] == 2
    assert body["items"][0]["id"] == "BBBB0002"

    body = client.get("/admin/api/license/issued?q=老客户", headers=_hdr(tok)).json()
    assert body["matched"] == 1 and body["items"][0]["id"] == "AAAA0001"


def test_issued_malformed_log_returns_empty(client, tok, monkeypatch, tmp_path):
    log = tmp_path / "issued_codes.json"
    log.write_text("{not a list", encoding="utf-8")
    monkeypatch.setattr("webai2api.admin_api._issuer_log_path", lambda: log)
    body = client.get("/admin/api/license/issued", headers=_hdr(tok)).json()
    assert body["ok"] is True and body["items"] == [] and body["total"] == 0


# ── 签发：无签发器必须拒绝 ─────────────────────────
def test_issuer_state_false_without_key(client, tok, no_issuer):
    body = client.get("/admin/api/license/issuer", headers=_hdr(tok)).json()
    assert body == {"ok": True, "available": False}
    assert body["available"] is False


def test_issue_rejected_without_key(client, tok, no_issuer, monkeypatch, tmp_path):
    log = tmp_path / "issued_codes.json"
    monkeypatch.setattr("webai2api.admin_api._issuer_log_path", lambda: log)

    r = client.post("/admin/api/license/issue", json={"days": 30, "count": 2},
                    headers=_hdr(tok))
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["error"] == "未找到作者私钥，无法签发"
    assert "items" not in body
    assert not log.exists()  # 不能留下半条记录


def test_issue_and_export_requires_login(client):
    assert client.post("/admin/api/license/issue", json={"days": 1}).status_code == 401
    assert client.get("/admin/api/license/issuer").status_code == 401


# ── 有签发器时（作者本机）走一遍真实链路 ────────────────
def test_issue_when_author_key_present(client, tok, monkeypatch, tmp_path):
    seed = issuer.load_author_seed()
    if seed is None:
        pytest.skip("本机没有作者私钥")
    log = tmp_path / "issued_codes.json"
    monkeypatch.setattr("webai2api.admin_api._issuer_log_path", lambda: log)

    body = client.post("/admin/api/license/issue",
                       json={"days": 15, "count": 2, "note": "回归测试"},
                       headers=_hdr(tok)).json()
    assert body["ok"] is True
    assert len(body["items"]) == 2
    for it in body["items"]:
        assert issuer.verify_code(it["code"], __import__(
            "webai2api.license", fromlist=["x"]).public_key()) is not None
        assert it["days"] == 15
        assert it["note"] == "回归测试"

    items = json.loads(log.read_text(encoding="utf-8"))
    assert len(items) == 2 and all("created" in it for it in items)
    listed = client.get("/admin/api/license/issued", headers=_hdr(tok)).json()
    assert listed["total"] == 2


def test_issue_clamps_days_and_count(client, tok, monkeypatch, tmp_path):
    seed = issuer.load_author_seed()
    if seed is None:
        pytest.skip("本机没有作者私钥")
    monkeypatch.setattr("webai2api.admin_api._issuer_log_path",
                        lambda: tmp_path / "issued_codes.json")
    body = client.post("/admin/api/license/issue",
                       json={"days": 99999, "count": 500}, headers=_hdr(tok)).json()
    assert body["ok"] is True
    assert len(body["items"]) == 100  # count 上限
    assert body["items"][0]["days"] == 3650  # days 上限


def test_issue_bad_params(client, tok, monkeypatch, tmp_path):
    seed = issuer.load_author_seed()
    if seed is None:
        pytest.skip("本机没有作者私钥")
    monkeypatch.setattr("webai2api.admin_api._issuer_log_path",
                        lambda: tmp_path / "issued_codes.json")
    body = client.post("/admin/api/license/issue",
                       json={"days": "abc"}, headers=_hdr(tok)).json()
    assert body["ok"] is False
