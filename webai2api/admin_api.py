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

from . import __version__
from .providers import MAX_LIMITS, ProviderRegistry
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


def _range_window(range_key: str, start: str = "", end: str = "") -> tuple[float | None, float | None]:
    """把面板选择的时间范围换算成 (起始, 结束) 时间戳（本地时区）。"""
    import datetime as _dt

    now = _dt.datetime.now()
    key = (range_key or "all").strip().lower()

    def _end_of(day: _dt.datetime) -> float:
        return (day + _dt.timedelta(days=1)).timestamp() - 0.001

    if key == "today":
        d = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return d.timestamp(), None
    if key == "7d":
        return (now - _dt.timedelta(days=7)).timestamp(), None
    if key == "30d":
        return (now - _dt.timedelta(days=30)).timestamp(), None
    if key == "month":
        d = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return d.timestamp(), None
    if key == "custom":
        try:
            s = _dt.datetime.strptime(start.strip(), "%Y-%m-%d")
        except Exception:
            return None, None
        e = None
        if end.strip():
            try:
                e = _end_of(_dt.datetime.strptime(end.strip(), "%Y-%m-%d"))
            except Exception:
                e = None
        return s.timestamp(), e
    return None, None


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

    async def _incompatible_models(accounts: list[str], models: list[str]) -> str:
        """勾选了账号时，模型必须属于这些账号的平台。"""
        if not accounts or not models:
            return ""
        accs = await run_in_threadpool(store.list_accounts)
        chosen = set(accounts)
        provs = {a["provider"] for a in accs if a["id"] in chosen}
        bad = [
            m for m in models
            if ("chatglm" if m.strip().lower().startswith("glm") else "deepseek") not in provs
        ]
        if not bad:
            return ""
        return ("模型 " + "、".join(bad) +
                " 需要对应平台的账号；请先勾选该平台的账号，或取消这些模型。")

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
    async def state(request: Request, range: str = "all", start: str = "", end: str = "",
                    limit: int = 50, offset: int = 0,
                    key_id: str = "", account_id: str = ""):
        g = _guard(request)
        if g:
            return g
        accounts = await run_in_threadpool(store.list_accounts)
        since_ts, until_ts = _range_window(range, start, end)
        limit = max(1, min(int(limit or 50), 500))
        offset = max(0, int(offset or 0))
        return {
            "ok": True,
            "range": range,
            "version": __version__,
            "accounts": _safe_accounts(accounts),
            "login": {a["id"]: login_mgr.state(a["id"]) for a in accounts},
            "keys": await run_in_threadpool(store.list_keys),
            "models": [m["id"] for m in await run_in_threadpool(registry.model_catalog)],
            "model_providers": {
                m["id"]: ("chatglm" if m["id"].strip().lower().startswith("glm") else "deepseek")
                for m in await run_in_threadpool(registry.model_catalog)
            },
            "usage": await run_in_threadpool(store.usage_summary, since_ts, until_ts),
            "recent": await run_in_threadpool(
                store.recent_usage, limit, offset, "", "", key_id, account_id),
            "recent_total": await run_in_threadpool(
                store.count_usage, "", "", key_id, account_id),
            "recent_limit": limit,
            "recent_offset": offset,
            "settings": {
                "deepseek_wasm_path": await run_in_threadpool(
                    store.get_setting, "deepseek_wasm_path", ""),
                "deepseek_session_cache": await run_in_threadpool(
                    store.get_setting, "deepseek_session_cache", "true"),
                "deepseek_default_model": await run_in_threadpool(
                    store.get_setting, "deepseek_default_model", "deepseek-chat"),
                "deepseek_search": await run_in_threadpool(
                    store.get_setting, "deepseek_search", "false"),
                "deepseek_anti_min": await run_in_threadpool(
                    store.get_setting, "deepseek_anti_min", "1.5"),
                "deepseek_anti_max": await run_in_threadpool(
                    store.get_setting, "deepseek_anti_max", "3.0"),
                "glm_base_url": await run_in_threadpool(
                    store.get_setting, "glm_base_url", "https://chatglm.cn"),
                "glm_default_model": await run_in_threadpool(
                    store.get_setting, "glm_default_model", "GLM-5.3"),
                "glm_default_effort": await run_in_threadpool(
                    store.get_setting, "glm_default_effort", "high"),
                "glm_anti_min": await run_in_threadpool(
                    store.get_setting, "glm_anti_min", "1.5"),
                "glm_anti_max": await run_in_threadpool(
                    store.get_setting, "glm_anti_max", "3.0"),
                "glm_search": await run_in_threadpool(
                    store.get_setting, "glm_search", "false"),
                "glm_session_cache": await run_in_threadpool(
                    store.get_setting, "glm_session_cache", "true"),
                "usage_retain_days": await run_in_threadpool(
                    store.get_setting, "usage_retain_days", "0"),
                "deepseek_context_limit": await run_in_threadpool(
                    store.get_setting, "deepseek_context_limit", "1000000"),                "deepseek_output_limit": await run_in_threadpool(
                    store.get_setting, "deepseek_output_limit", "384000"),
                "glm_context_limit": await run_in_threadpool(
                    store.get_setting, "glm_context_limit", "1000000"),
                "glm_output_limit": await run_in_threadpool(
                    store.get_setting, "glm_output_limit", "128000"),
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
        accounts = [str(x) for x in (b.get("accounts") or []) if str(x).strip()]
        models = [str(x).strip() for x in (b.get("models") or []) if str(x).strip()]
        bad = await _incompatible_models(accounts, models)
        if bad:
            return {"ok": False, "error": bad}
        k = await run_in_threadpool(
            store.add_key, b.get("name", "未命名 Key"),
            None, accounts, models,
        )
        return {"ok": True, "key": k}

    @router.put("/keys/{key_id}")
    async def update_key(key_id: str, request: Request):
        g = _guard(request)
        if g:
            return g
        b = await _json(request) or {}
        if "enabled" in b:
            await run_in_threadpool(store.set_key_enabled, key_id, bool(b["enabled"]))
        name = b.get("name")
        accounts = b.get("accounts")
        models = b.get("models")
        if accounts is not None or models is not None:
            cur = next((k for k in await run_in_threadpool(store.list_keys)
                        if k["id"] == key_id), None)
            nacc = ([str(x) for x in accounts if str(x).strip()]
                    if accounts is not None else (cur or {}).get("accounts", []))
            nmod = ([str(x).strip() for x in models if str(x).strip()]
                    if models is not None else (cur or {}).get("models", []))
            bad = await _incompatible_models(nacc, nmod)
            if bad:
                return {"ok": False, "error": bad}
        if name is not None or accounts is not None or models is not None:
            await run_in_threadpool(
                store.update_key, key_id,
                name=(str(name).strip() or "未命名 Key") if name is not None else None,
                accounts=([str(x) for x in accounts if str(x).strip()]
                          if accounts is not None else None),
                models=([str(x).strip() for x in models if str(x).strip()]
                        if models is not None else None),
            )
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
                  "glm_base_url", "glm_default_model", "glm_default_effort",
                  "glm_anti_min", "glm_anti_max", "usage_retain_days",
                  "deepseek_default_model"):
            if k in b and str(b[k]).strip():
                await run_in_threadpool(store.set_setting, k, str(b[k]).strip())
        # 额度类：夹到该模型允许的上限内，防止填出荒谬值
        for k in ("deepseek_context_limit", "deepseek_output_limit",
                  "glm_context_limit", "glm_output_limit"):
            if k not in b:
                continue
            prov = "chatglm" if k.startswith("glm") else "deepseek"
            kind = "context" if "context" in k else "output"
            cap = MAX_LIMITS[prov][kind]
            try:
                val = int(float(str(b[k]).strip()))
            except Exception:
                continue
            val = max(1, min(val, cap))
            await run_in_threadpool(store.set_setting, k, str(val))
        if "deepseek_session_cache" in b:
            v = b["deepseek_session_cache"]
            s = "true" if (v is True or str(v).strip().lower() in ("1", "true", "yes", "on")) else "false"
            await run_in_threadpool(store.set_setting, "deepseek_session_cache", s)
        if "deepseek_search" in b:
            v = b["deepseek_search"]
            s = "true" if (v is True or str(v).strip().lower() in ("1", "true", "yes", "on")) else "false"
            await run_in_threadpool(store.set_setting, "deepseek_search", s)
        if "glm_search" in b:
            v = b["glm_search"]
            s = "true" if (v is True or str(v).strip().lower() in ("1", "true", "yes", "on")) else "false"
            await run_in_threadpool(store.set_setting, "glm_search", s)
        if "glm_session_cache" in b:
            v = b["glm_session_cache"]
            s = "true" if (v is True or str(v).strip().lower() in ("1", "true", "yes", "on")) else "false"
            await run_in_threadpool(store.set_setting, "glm_session_cache", s)
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

