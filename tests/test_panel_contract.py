"""C-1 面板↔后端契约测试（方案 C 第一层，2026-10-29）。

思路：把 index.html 里各渲染函数**实际解引用的字段**用正则提取出来，
与 TestClient 真实响应逐字段比对——前端读的每个字段，后端必须真的给。
此前这类漂移只能靠用户点开面板才发现（授权卡消失事故就是 state 响应
丢字段导致前端整卡隐藏）。

覆盖：/admin/api/state（accounts/keys/settings/overview/risk）、
/admin/api/system、/admin/api/risk、/admin/api/keys/{id}/detail。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from webai2api.app import build_app
from webai2api.providers import ProviderRegistry
from webai2api.store import Store

TOKEN_HDR = "X-Admin-Token"
ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "webai2api" / "web" / "index.html").read_text(encoding="utf-8")


def _seg(start: str, end: str) -> str:
    """截取 HTML 中 start 到 end 标记之间的 JS 段。"""
    return HTML[HTML.index(start) : HTML.index(end)]


def _fields(seg: str, obj: str) -> set[str]:
    """提取 JS 段里对某对象的字段解引用（obj.field）。"""
    return set(re.findall(rf"\b{obj}\.([a-zA-Z_][a-zA-Z_0-9]*)", seg))


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("panel_contract")
    store = Store(tmp / "data" / "webai2api.db")
    registry = ProviderRegistry(store, logging := __import__("logging").getLogger("t"))
    app = build_app(store, registry, logging)
    store.add_account("chatglm", "契约账号", {"auth_token": "tok"})
    c = TestClient(app, client=("127.0.0.1", 50000))
    r = c.post("/admin/api/login", json={"password": "admin"})
    tok = r.json()["token"]
    c.headers.update({TOKEN_HDR: tok})
    r = c.post("/admin/api/keys", json={"name": "契约Key"})
    assert r.json().get("ok") is True, r.text
    return c


# ── /state 顶层字段 ──────────────────────────────────────────────


def _state(client):
    r = client.get("/admin/api/state?range=today&limit=50&offset=0")
    assert r.status_code == 200, r.text
    s = r.json()
    assert s.get("ok") is True, "state 响应必须带 ok（前端 refreshData 判它）"
    return s


def test_state_top_level_fields(client):
    """整个面板对 s.* 的全部解引用（含守卫式访问）都必须存在于响应。"""
    s = _state(client)
    used = _fields(HTML, "s")
    # 过滤明显非 state 的同名变量（局部循环变量/字符串方法误入）
    noise = {
        "style", "classList", "add", "remove", "toggle", "contains",
        "get", "children", "inner", "insertAdjacent", "sort", "slice",
        "trim", "split", "length", "map", "filter", "forEach", "join",
        "includes", "replace", "padStart", "toLowerCase", "ok",  # ok 单独验
        "startsWith", "endsWith", "now", "name",  # 局部字符串/日期对象的方法
    }
    missing = {f for f in used - noise if f not in s}
    assert not missing, f"前端读取了 state 响应中不存在的字段: {sorted(missing)}"


def test_state_accounts_fields(client):
    s = _state(client)
    assert s["accounts"], "测试前置：至少一个账号"
    used = _fields(_seg("function renderAccounts(s)", "function renderKeys(s)"), "a")
    noise = {"style", "classList", "add", "remove", "toggle", "contains", "get"}
    missing = {f for f in used - noise if f not in s["accounts"][0]}
    assert not missing, f"renderAccounts 读取了账号对象上不存在的字段: {sorted(missing)}"


def test_state_keys_fields(client):
    s = _state(client)
    assert s["keys"], "测试前置：至少一个 Key"
    used = _fields(_seg("function renderKeys(s)", "function renderUsage(s)"), "k")
    noise = {"style", "classList", "add", "remove", "toggle", "get"}
    missing = {f for f in used - noise if f not in s["keys"][0]}
    assert not missing, f"renderKeys 读取了 Key 对象上不存在的字段: {sorted(missing)}"


def test_state_settings_fields(client):
    s = _state(client)
    used = _fields(_seg("function renderSettings(s)", "function renderAccounts"), "st")
    settings = s.get("settings") or {}
    noise = {"get"}  # Map 误入
    missing = {f for f in used - noise if f not in settings}
    assert not missing, f"renderSettings 读取了 settings 中不存在的键: {sorted(missing)}"


def test_state_overview_fields(client):
    s = _state(client)
    used = _fields(_seg("function renderOverview(s)", "function renderUsage(s)"), "s")
    noise = {"style", "classList", "add", "remove", "toggle", "get", "keys",
             "accounts", "settings", "ok", "recent", "today", "yesterday",
             "totals", "stats", "risk"}
    missing = {f for f in used - noise if f not in s}
    assert not missing, f"renderOverview 读取了 state 中不存在的字段: {sorted(missing)}"


# ── /system 与 /risk ─────────────────────────────────────────────


def test_system_fields(client):
    r = client.get("/admin/api/system")
    assert r.status_code == 200, r.text
    sysinfo = r.json()
    seg = _seg("async function loadSystem()", "function ")
    used = _fields(seg, "j")
    noise = {"ok"}
    missing = {f for f in used - noise if f not in sysinfo}
    assert not missing, f"loadSystem 读取了 system 响应中不存在的字段: {sorted(missing)}"


def test_risk_fields(client):
    """loadRisk() 读 j.ok + j.accounts[].name/provider/stats.*。"""
    r = client.get("/admin/api/risk")
    assert r.status_code == 200, r.text
    risk = r.json()
    seg = _seg("async function loadRisk()", "function ")
    used = _fields(seg, "j")
    noise = {"ok"}
    missing = {f for f in used - noise if f not in risk}
    assert not missing, f"loadRisk 读取了 risk 响应中不存在的字段: {sorted(missing)}"
    assert risk.get("accounts"), "risk 响应必须有 accounts 数组（前端 forEach）"
    st_used = _fields(seg, "st") - {"get"}
    stats = risk["accounts"][0].get("stats") or {}
    missing = {f for f in st_used if f not in stats}
    assert not missing, f"loadRisk 读取了 stats 中不存在的字段: {sorted(missing)}"


# ── Key 详情 ─────────────────────────────────────────────────────


def test_key_detail_fields(client):
    s = _state(client)
    kid = s["keys"][0]["id"]
    r = client.get(f"/admin/api/keys/{kid}/detail")
    assert r.status_code == 200, r.text
    detail = r.json()
    assert detail.get("ok") is True
    # openKeyDetail() 实际读取的字段（逐个从 JS 提取核对）
    for f in ("total", "daily", "by_model"):
        assert f in detail, f"key detail 响应缺 {f}（openKeyDetail 直接解引用）"
    for f in ("n", "t", "ok"):
        assert f in (detail.get("total") or {}), f"key detail.total 缺 {f}"


# ── 前端调用的端点必须全部存在（方法级契约）──────────────────────


def test_every_frontend_api_path_exists_in_backend(client):
    """把面板里全部 api('/xxx') 字面量提出来，逐个按前端用的方法打一遍，
    断言不是 404/405（参数不足导致的 400/422 可接受——契约在路径+方法）。"""
    paths = set(re.findall(r"api\('(/[a-z_0-9.${}]*)'", HTML))
    literal = {p for p in paths if not p.endswith("/")}  # 截断的动态拼接丢弃
    assert literal, "未提取到任何 api 调用（提取器失效）"
    method_re = re.compile(r"api\('(/[a-z_0-9.]*)'\s*,\s*'([A-Z]+)'")
    for p in sorted(literal):
        m = method_re.search(HTML)
        method = "GET"
        for mm in re.finditer(rf"api\('{re.escape(p)}'\s*,\s*'([A-Z]+)'", HTML):
            method = mm.group(1)
        r = client.request(method, f"/admin/api{p}")
        assert r.status_code not in (404, 405), (
            f"前端调用 /admin/api{p} [{method}] 后端不存在: HTTP {r.status_code}"
        )
