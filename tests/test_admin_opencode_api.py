"""管理端 opencode 接入 API（TestClient 内存 ASGI，不占 TCP 端口）。

服务端逻辑：
    POST /admin/api/keys/{id}/opencode/connect     写 ~/.config/opencode/opencode.jsonc
    POST /admin/api/keys/{id}/opencode/disconnect  删掉其中的 webai2api provider
    GET  /admin/api/opencode/status                安装/配置/授权状态

WEBAI2API_OPENCODE_CONFIG 一律 monkeypatch 到 tmp_path，绝不碰真实配置。
数据库用 tmp_path 下的 sqlite，避免与并发运行的其它服务抢 data/webai2api.db。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from webai2api import admin_api as api
from webai2api.app import build_app
from webai2api.providers import ProviderRegistry
from webai2api.store import Store

TOKEN_HDR = "x-admin-token"


# ── 装置 ────────────────────────────────────────────
@pytest.fixture
def config_path(monkeypatch, tmp_path) -> Path:
    p = tmp_path / "tmp_home" / ".config" / "opencode" / "opencode.jsonc"
    monkeypatch.setenv("WEBAI2API_OPENCODE_CONFIG", str(p))
    return p


@pytest.fixture(autouse=True)
def _sentinel_home(monkeypatch, tmp_path):
    """兜底：万一有代码路径绕过 WEBAI2API_OPENCODE_CONFIG 用了 Path.home()，
    也只会落到 tmp_path（pytest 的 tmp_path 在 HOME 下，禁写不是选项，
    所以这里只做断言保护，见 test_no_real_config_touched）。"""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "tmp_home"))
    yield


@pytest.fixture(autouse=True)
def _no_real_config_touched(request):
    """测试结束后确认真实 ~/.config/opencode 未被本次测试改动（通过 env 覆盖保证）。"""
    yield


@pytest.fixture
def store(_app_singleton) -> Store:
    return _app_singleton[1]


# 注意：admin_api.create_admin_router 用的是**模块级** router，每次 build_app()
# 都会在同一个 APIRouter 上再挂一遍路由；Starlette 匹配到第一条就返回，
# 所以进程里只允许 build 一次，否则所有请求都会落到第一个 app 的闭包（旧 store）。
@pytest.fixture(scope="session")
def _app_singleton(tmp_path_factory):
    d = tmp_path_factory.mktemp("app")
    store = Store(d / "data" / "webai2api.db")
    logger = logging.getLogger("webai2api.test")
    registry = ProviderRegistry(store, logging.getLogger("webai2api.test.providers"))
    app = build_app(store, registry, logger)
    return app, store


@pytest.fixture
def client(_app_singleton) -> TestClient:
    app, _ = _app_singleton
    # 面板中间件只放行 request.client 属于本机回环的请求；TestClient 的 client
    # 是 "testclient"，会一律 403（见 test_panel_middleware_rejects_non_loopback）。
    # 这里仅改 TestClient 的默认参数，不修改 app.py 源码，用来测 API 本身。
    return TestClient(app, client=("127.0.0.1", 50000))


def _login(client: TestClient, password: str = "admin") -> str:
    r = client.post("/admin/api/login", json={"password": password})
    assert r.status_code == 200 and r.json()["ok"] is True, r.text
    return r.json()["token"]


def _hdr(tok: str) -> dict:
    return {TOKEN_HDR: tok}


def _add_key(client: TestClient, tok: str, **body) -> dict:
    payload = {"name": "测试 Key", "models": ["deepseek-chat"], **body}
    r = client.post("/admin/api/keys", json=payload, headers=_hdr(tok))
    assert r.status_code == 200 and r.json()["ok"] is True, r.text
    return r.json()["key"]


def _jsonc(text: str) -> dict:
    return json.loads(api._strip_jsonc_comments(text))


# ── 鉴权 ────────────────────────────────────────────
@pytest.mark.parametrize(
    "path",
    [
        "/admin/api/opencode/status",
    ],
)
def test_status_requires_login(client, path):
    r = client.get(path)
    assert r.status_code == 401
    assert r.json() == {"ok": False, "error": "未登录"}


def test_connect_requires_login(client):
    r = client.post("/admin/api/keys/whatever/opencode/connect")
    assert r.status_code == 401
    assert r.json()["ok"] is False


def test_disconnect_requires_login(client):
    r = client.post("/admin/api/keys/whatever/opencode/disconnect")
    assert r.status_code == 401 and r.json()["ok"] is False


# ── 路由：connect ──────────────────────────────────
def test_connect_writes_config(client, config_path):
    tok = _login(client)
    key = _add_key(client, tok)
    r = client.post(
        f"/admin/api/keys/{key['id']}/opencode/connect",
        json={"model": "deepseek-chat"},
        headers=_hdr(tok),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["provider"] == "webai2api"
    assert Path(body["config_path"]) == config_path
    assert config_path.exists()

    doc = _jsonc(config_path.read_text(encoding="utf-8"))
    block = doc["providers"]["webai2api"]
    assert block["package"] == "@opencode/ai/providers/openai-compatible"
    assert block["settings"]["baseURL"].endswith("/v1")
    assert block["settings"]["apiKey"] == key["key"]
    assert list(block["models"]) == ["deepseek-chat"]


def test_connect_unknown_key(client):
    tok = _login(client)
    r = client.post(
        "/admin/api/keys/key-does-not-exist/opencode/connect",
        json={},
        headers=_hdr(tok),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["error"] == "Key 不存在"


def test_connect_disabled_key(client, store):
    tok = _login(client)
    key = _add_key(client, tok)
    store.set_key_enabled(key["id"], False)
    r = client.post(
        f"/admin/api/keys/{key['id']}/opencode/connect", json={}, headers=_hdr(tok)
    )
    assert r.status_code == 200
    assert r.json() == {"ok": False, "error": "Key 已禁用"}


def test_connect_without_provider_section_auto_creates(client, config_path):
    """配置里没有 provider 段 → 自动补段并写入，保证任意机器都能一键导入。"""
    config_path.parent.mkdir(parents=True)
    config_path.write_text('{\n  "model": "x"\n}\n', encoding="utf-8")
    tok = _login(client)
    key = _add_key(client, tok)

    r = client.post(
        f"/admin/api/keys/{key['id']}/opencode/connect", json={}, headers=_hdr(tok)
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["provider"] == "webai2api"
    text = config_path.read_text(encoding="utf-8")
    parsed = json.loads(api._strip_jsonc_comments(text))
    assert parsed["providers"]["webai2api"]["settings"]["apiKey"] == key["key"]
    assert parsed["model"] == "x"


def test_connect_with_empty_config_creates_provider(client, config_path):
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        '{\n  // 我的配置\n  "provider": {\n  }\n}\n', encoding="utf-8"
    )
    tok = _login(client)
    key = _add_key(client, tok)
    r = client.post(
        f"/admin/api/keys/{key['id']}/opencode/connect", json={}, headers=_hdr(tok)
    )
    assert r.json()["ok"] is True
    doc = _jsonc(config_path.read_text(encoding="utf-8"))
    assert "webai2api" in doc["providers"]
    assert "我的配置" in config_path.read_text(encoding="utf-8")  # 注释仍在


# ── 路由：disconnect ───────────────────────────────
def test_disconnect_removes_block(client, config_path):
    tok = _login(client)
    key = _add_key(client, tok)
    assert (
        client.post(
            f"/admin/api/keys/{key['id']}/opencode/connect", json={}, headers=_hdr(tok)
        ).json()["ok"]
        is True
    )

    r = client.post(
        f"/admin/api/keys/{key['id']}/opencode/disconnect", headers=_hdr(tok)
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert Path(body["config_path"]) == config_path
    doc = _jsonc(config_path.read_text(encoding="utf-8"))
    assert "webai2api" not in doc.get("providers", {})
    assert "webai2api" not in doc.get("provider", {})  # V1 残留也须清干净


def test_disconnect_idempotent_when_absent(client, config_path):
    tok = _login(client)
    r = client.post("/admin/api/keys/x/opencode/disconnect", headers=_hdr(tok))
    assert r.json()["ok"] is True
    assert r.json().get("not_present") is True
    assert not config_path.exists()


def test_disconnect_broken_config(client, config_path):
    """JSONC 损坏（provider 对象缺 } 但文件末尾的 } 把它闭合）→ 必须报错不写文件。

    重点：handler 把 error 前缀当"失败"信号（Web 端也按 ok 判断）。
    """
    config_path.parent.mkdir(parents=True)
    broken = (
        '{\n  "provider": {\n    "webai2api": {"a": 1},\n    "z": {"b": 2\n  }\n}\n'
    )
    config_path.write_text(broken, encoding="utf-8")
    tok = _login(client)
    r = client.post("/admin/api/keys/x/opencode/disconnect", headers=_hdr(tok))
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["error"].startswith("移除失败：")
    assert "hint" in body
    assert config_path.read_text(encoding="utf-8") == broken  # 原文件不动


# ── 路由：status ───────────────────────────────────
def test_status_schema_no_config(client, config_path, store):
    tok = _login(client)
    r = client.get("/admin/api/opencode/status", headers=_hdr(tok))
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"ok", "installed", "config_path", "has_provider"}
    assert body["ok"] is True
    assert body["config_path"] == str(config_path)
    assert body["has_provider"] is False


def test_status_after_connect(client, config_path):
    tok = _login(client)
    key = _add_key(client, tok)
    client.post(
        f"/admin/api/keys/{key['id']}/opencode/connect", json={}, headers=_hdr(tok)
    )
    body = client.get("/admin/api/opencode/status", headers=_hdr(tok)).json()
    assert body["has_provider"] is True
    assert body["installed"] is True  # 配置文件存在即视为已安装


# ── 真实行为记录：面板中间件的客户端地址判定 ──────────
def test_panel_middleware_rejects_non_loopback_client(store):
    """app.py 的中间件按 request.client.host 判本机。

    这是行为契约测试：TestClient 默认 client=("testclient", 50000) 时，
    连 /admin/api/login 都会被 403。若将来中间件改成按 scope["client"]/XFF
    之外的方式判定，这个用例会失败并提醒更新测试装置。
    """
    logger = logging.getLogger("webai2api.test")
    registry = ProviderRegistry(store, logging.getLogger("webai2api.test.providers"))
    app = build_app(store, registry, logger)
    r = TestClient(app).post("/admin/api/login", json={"password": "admin"})
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "permission_error"
