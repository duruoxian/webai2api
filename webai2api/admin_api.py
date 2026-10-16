"""管理面板后端 API（/admin/api/*），中文界面用。

鉴权：密码登录后发放随机 token，放在 X-Admin-Token 头里。
token 有有效期（默认 12 小时），并且登录失败有简单限速。
"""
from __future__ import annotations

import hmac
import secrets
import time
from logging import Logger

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from .providers import ProviderRegistry
from .store import Store

router = APIRouter(prefix="/admin/api")

TOKEN_TTL = 12 * 3600          # token 有效期
_MAX_FAILS = 8                 # 连续失败上限
_FAIL_WINDOW = 300             # 失败统计窗口（秒）


class _TokenStore:
    """进程内 token 表 + 登录失败限速。"""

    def __init__(self) -> None:
        self._tokens: dict[str, float] = {}
        self._fails: list[float] = []

    def issue(self) -> str:
        tok = secrets.token_hex(24)
        self._tokens[tok] = time.time() + TOKEN_TTL
        # 顺手清理过期 token
        now = time.time()
        for k in [k for k, exp in self._tokens.items() if exp < now]:
            self._tokens.pop(k, None)
        return tok

    def valid(self, tok: str) -> bool:
        exp = self._tokens.get(tok)
        if not exp:
            return False
        if exp < time.time():
            self._tokens.pop(tok, None)
            return False
        return True

    def revoke(self, tok: str) -> None:
        self._tokens.pop(tok, None)

    def rate_limited(self) -> bool:
        now = time.time()
        self._fails = [t for t in self._fails if now - t < _FAIL_WINDOW]
        return len(self._fails) >= _MAX_FAILS

    def note_fail(self) -> None:
        self._fails.append(time.time())

    def note_ok(self) -> None:
        self._fails.clear()


def create_admin_router(store: Store, registry: ProviderRegistry, login_mgr, logger: Logger) -> APIRouter:
    toks = _TokenStore()

    def _guard(request: Request):
        if not toks.valid(request.headers.get("x-admin-token", "")):
            return JSONResponse(status_code=401, content={"ok": False, "error": "未登录"})
        return None

    async def _json(request: Request) -> dict | None:
        try:
            b = await request.json()
            return b if isinstance(b, dict) else None
        except Exception:
            return None

    @router.post("/login")
    async def login(request: Request):
        if toks.rate_limited():
            return {"ok": False, "error": "尝试过于频繁，请稍后再试"}
        body = await _json(request)
        if body is None:
            return {"ok": False, "error": "请求格式错误"}
        pw = str(body.get("password", ""))
        expect = await run_in_threadpool(store.get_setting, "admin_password", "admin")
        if not hmac.compare_digest(pw, expect):
            toks.note_fail()
            return {"ok": False, "error": "密码错误"}
        toks.note_ok()
        return {"ok": True, "token": toks.issue()}

    @router.post("/logout")
    async def logout(request: Request):
        toks.revoke(request.headers.get("x-admin-token", ""))
        return {"ok": True}

    @router.get("/state")
    async def state(request: Request):
        g = _guard(request)
        if g:
            return g
        accounts = await run_in_threadpool(store.list_accounts)
        return {
            "ok": True,
            "accounts": _safe_accounts(accounts),
            "login": {a["id"]: login_mgr.state(a["id"]) for a in accounts},
            "keys": await run_in_threadpool(store.list_keys),
            "usage": await run_in_threadpool(store.usage_summary),
            "recent": await run_in_threadpool(store.recent_usage, 50),
            "settings": {
                "deepseek_wasm_path": await run_in_threadpool(
                    store.get_setting, "deepseek_wasm_path", ""),
                "deepseek_session_cache": await run_in_threadpool(
                    store.get_setting, "deepseek_session_cache", "true"),
                "deepseek_anti_min": await run_in_threadpool(
                    store.get_setting, "deepseek_anti_min", "1.5"),
                "deepseek_anti_max": await run_in_threadpool(
                    store.get_setting, "deepseek_anti_max", "3.0"),
                "glm_base_url": await run_in_threadpool(
                    store.get_setting, "glm_base_url", "https://chatglm.cn"),
                "glm_default_model": await run_in_threadpool(
                    store.get_setting, "glm_default_model", "GLM-5.3"),
                "glm_assistant_id": await run_in_threadpool(
                    store.get_setting, "glm_assistant_id", ""),
                "glm_anti_min": await run_in_threadpool(
                    store.get_setting, "glm_anti_min", "1.5"),
                "glm_anti_max": await run_in_threadpool(
                    store.get_setting, "glm_anti_max", "3.0"),
                "glm_search": await run_in_threadpool(
                    store.get_setting, "glm_search", "false"),
                "admin_password": "******",
            },
        }

    # ── 账号 ──
    @router.post("/accounts")
    async def add_account(request: Request):
        g = _guard(request)
        if g:
            return g
        b = await _json(request)
        if b is None:
            return {"ok": False, "error": "请求格式错误"}
        acc = await run_in_threadpool(
            store.add_account,
            b.get("provider", "deepseek"),
            b.get("name", "未命名账号"),
            {
                "auth_token": b.get("auth_token", ""),
                "device_id": b.get("device_id", ""),
                "settings_token": b.get("settings_token", ""),
                "cookie": b.get("cookie", ""),
            },
            b.get("note", ""),
        )
        registry.refresh()
        return {"ok": True, "account": _safe_account(acc)}

    @router.put("/accounts/{account_id}")
    async def update_account(account_id: str, request: Request):
        g = _guard(request)
        if g:
            return g
        b = await _json(request)
        if b is None:
            return {"ok": False, "error": "请求格式错误"}
        fields = {k: b[k] for k in
                  ("name", "auth_token", "device_id", "settings_token",
                   "cookie", "enabled", "note") if k in b}
        if "enabled" in fields:
            fields["enabled"] = 1 if fields["enabled"] else 0
        await run_in_threadpool(store.update_account, account_id, **fields)
        registry.refresh()
        return {"ok": True}

    @router.delete("/accounts/{account_id}")
    async def delete_account(account_id: str, request: Request):
        g = _guard(request)
        if g:
            return g
        await run_in_threadpool(store.delete_account, account_id)
        registry.refresh()
        return {"ok": True}

    # ── 浏览器登录 ──
    @router.post("/accounts/{account_id}/login")
    async def browser_login(account_id: str, request: Request):
        g = _guard(request)
        if g:
            return g
        return login_mgr.start(account_id)

    @router.get("/accounts/{account_id}/login")
    async def login_state(account_id: str, request: Request):
        g = _guard(request)
        if g:
            return g
        return {"ok": True, "state": login_mgr.state(account_id)}

    # ── Key ──
    @router.post("/keys")
    async def add_key(request: Request):
        g = _guard(request)
        if g:
            return g
        b = await _json(request) or {}
        k = await run_in_threadpool(store.add_key, b.get("name", "未命名 Key"))
        return {"ok": True, "key": k}

    @router.put("/keys/{key_id}")
    async def update_key(key_id: str, request: Request):
        g = _guard(request)
        if g:
            return g
        b = await _json(request) or {}
        if "enabled" in b:
            await run_in_threadpool(store.set_key_enabled, key_id, bool(b["enabled"]))
        return {"ok": True}

    @router.delete("/keys/{key_id}")
    async def delete_key(key_id: str, request: Request):
        g = _guard(request)
        if g:
            return g
        await run_in_threadpool(store.delete_key, key_id)
        return {"ok": True}

    # ── 设置 ──
    @router.post("/settings")
    async def save_settings(request: Request):
        g = _guard(request)
        if g:
            return g
        b = await _json(request)
        if b is None:
            return {"ok": False, "error": "请求格式错误"}
        for k in ("deepseek_wasm_path", "admin_password",
                  "deepseek_anti_min", "deepseek_anti_max",
                  "glm_base_url", "glm_default_model", "glm_assistant_id",
                  "glm_anti_min", "glm_anti_max"):
            if k in b and str(b[k]).strip():
                await run_in_threadpool(store.set_setting, k, str(b[k]).strip())
        if "deepseek_session_cache" in b:
            v = b["deepseek_session_cache"]
            s = "true" if (v is True or str(v).strip().lower() in ("1", "true", "yes", "on")) else "false"
            await run_in_threadpool(store.set_setting, "deepseek_session_cache", s)
        if "glm_search" in b:
            v = b["glm_search"]
            s = "true" if (v is True or str(v).strip().lower() in ("1", "true", "yes", "on")) else "false"
            await run_in_threadpool(store.set_setting, "glm_search", s)
        try:
            registry.refresh()
        except Exception:
            pass
        return {"ok": True}

    return router


def _safe_account(a: dict) -> dict:
    """返回账号（凭证脱敏）。"""
    return {
        "id": a["id"], "provider": a["provider"], "name": a["name"],
        "enabled": a["enabled"], "note": a["note"],
        "has_token": bool(a.get("auth_token")),
        "has_device": bool(a.get("device_id")),
        "has_settings": bool(a.get("settings_token")),
        "has_cookie": bool(a.get("cookie")),
        "created_at": a["created_at"], "updated_at": a["updated_at"],
    }


def _safe_accounts(items: list[dict]) -> list[dict]:
    return [_safe_account(a) for a in items]

