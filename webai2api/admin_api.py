"""管理面板后端 API（/admin/api/*），中文界面用。

鉴权：密码登录后发放随机 token，放在 X-Admin-Token 头里。
token 有有效期（默认 12 小时），并且登录失败有简单限速。
"""
from __future__ import annotations

import hmac
import json
import os
import re
import secrets
import shutil
import time
from datetime import datetime
from logging import Logger
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from . import __version__
from .providers import DEFAULT_LIMITS, MAX_LIMITS, ProviderRegistry, infer_provider
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


_TOKENS = _TokenStore()


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
    toks = _TOKENS

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
            if infer_provider(m) not in provs
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
        return {"ok": True, "token": toks.issue(),
                "default_password": hmac.compare_digest(expect, "admin")}

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
                m["id"]: infer_provider(m["id"])
                for m in await run_in_threadpool(registry.model_catalog)
            },
            "usage": await run_in_threadpool(store.usage_summary, since_ts, until_ts),
            "today": await run_in_threadpool(store.today_summary),
            "cooldowns": await run_in_threadpool(registry.cooldowns),
            "network": await run_in_threadpool(_network_status),
            "key_usage": await run_in_threadpool(store.key_usage_map),
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
                "tray_enabled": await run_in_threadpool(
                    store.get_setting, "tray_enabled", "true"),
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
                "kimi_base_url": await run_in_threadpool(
                    store.get_setting, "kimi_base_url", "https://www.kimi.com"),
                "kimi_default_model": await run_in_threadpool(
                    store.get_setting, "kimi_default_model", "kimi-k3"),
                "kimi_default_effort": await run_in_threadpool(
                    store.get_setting, "kimi_default_effort", ""),
                "kimi_anti_min": await run_in_threadpool(
                    store.get_setting, "kimi_anti_min", "1.5"),
                "kimi_anti_max": await run_in_threadpool(
                    store.get_setting, "kimi_anti_max", "3.0"),
                "kimi_search": await run_in_threadpool(
                    store.get_setting, "kimi_search", "false"),
                "kimi_session_cache": await run_in_threadpool(
                    store.get_setting, "kimi_session_cache", "true"),
                "kimi_context_limit": await run_in_threadpool(
                    store.get_setting, "kimi_context_limit", "262144"),
                "kimi_output_limit": await run_in_threadpool(
                    store.get_setting, "kimi_output_limit", "32768"),
                "qwen_base_url": await run_in_threadpool(
                    store.get_setting, "qwen_base_url", "https://chat.qwen.ai"),
                "qwen_default_model": await run_in_threadpool(
                    store.get_setting, "qwen_default_model", "qwen3.8-max"),
                "qwen_default_effort": await run_in_threadpool(
                    store.get_setting, "qwen_default_effort", ""),
                "qwen_anti_min": await run_in_threadpool(
                    store.get_setting, "qwen_anti_min", "1.5"),
                "qwen_anti_max": await run_in_threadpool(
                    store.get_setting, "qwen_anti_max", "3.0"),
                "qwen_search": await run_in_threadpool(
                    store.get_setting, "qwen_search", "false"),
                "qwen_session_cache": await run_in_threadpool(
                    store.get_setting, "qwen_session_cache", "true"),
                "qwen_context_limit": await run_in_threadpool(
                    store.get_setting, "qwen_context_limit", "262144"),
                "qwen_output_limit": await run_in_threadpool(
                    store.get_setting, "qwen_output_limit", "65536"),
                "admin_password": "******",
            },
            "license": await run_in_threadpool(_license_info),
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
                "proxy": b.get("proxy", ""),
                "egress": b.get("egress", "system"),
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
                   "cookie", "enabled", "note", "proxy", "egress") if k in b}
        if "egress" in fields:
            from .egress import normalize_egress

            fields["egress"] = normalize_egress(str(fields["egress"]))
        # 凭证有变化 → 重新检查
        if any(k in fields for k in ("auth_token", "device_id", "settings_token", "cookie")):
            fields["health"] = ""
            fields["checked_at"] = 0.0
        if "enabled" in fields:
            fields["enabled"] = 1 if fields["enabled"] else 0
        await run_in_threadpool(store.update_account, account_id, **fields)
        registry.refresh()
        return {"ok": True}

    @router.post("/accounts/{account_id}/check")
    async def check_account(account_id: str, request: Request):
        g = _guard(request)
        if g:
            return g
        from .health import check_account as _check

        acc = await run_in_threadpool(store.get_account, account_id)
        if not acc:
            return {"ok": False, "error": "账号不存在"}
        try:
            status, profile, new_token = await run_in_threadpool(_check, acc)
            await run_in_threadpool(
                store.set_account_health, account_id, status, profile, new_token or None)
            return {"ok": True, "status": status, "profile": profile}
        except Exception as exc:  # noqa: BLE001
            await run_in_threadpool(store.set_account_health, account_id, "expired")
            return {"ok": False, "error": str(exc)[:300]}

    @router.post("/accounts/check-all")
    async def check_all_accounts(request: Request):
        g = _guard(request)
        if g:
            return g
        monitor = getattr(request.app.state, "health_monitor", None)
        if monitor is None:
            return {"ok": False, "error": "健康检查未启动"}
        results = await run_in_threadpool(monitor.check_all)
        return {"ok": True, "results": results}

    @router.delete("/accounts/{account_id}")
    async def delete_account(account_id: str, request: Request):
        g = _guard(request)
        if g:
            return g
        await run_in_threadpool(store.delete_account, account_id)
        registry.refresh()
        return {"ok": True}

    # ── 风险指标 ──
    @router.post("/proxy/test")
    async def proxy_test(request: Request):
        """测试代理连通性并返回出口 IP（用于配代理时自检）。"""
        g = _guard(request)
        if g:
            return g
        b = await _json(request)
        if b is None:
            return {"ok": False, "error": "请求格式错误"}
        proxy_url = str(b.get("proxy") or "").strip()
        if not proxy_url:
            return {"ok": False, "error": "请先填写代理地址"}
        return await run_in_threadpool(_proxy_probe, proxy_url)

    @router.get("/risk")
    async def risk_metrics(request: Request):
        """每账号的风控相关指标：会话创建/删除、缓存命中率、限流次数、日用量。"""
        g = _guard(request)
        if g:
            return g
        items: list[dict] = []
        for ac in await run_in_threadpool(registry.all_accounts):
            client = getattr(ac, "client", None)
            stats = {}
            if client is not None and hasattr(client, "risk_stats"):
                try:
                    stats = client.risk_stats()
                except Exception:  # noqa: BLE001
                    stats = {}
            stats["daily_count"] = int(getattr(ac, "_daily_count", 0) or 0)
            stats["daily_limit"] = int(getattr(ac, "daily_limit", 0) or 0)
            stats["cooldown_left"] = int(ac.cooldown_left()) if hasattr(ac, "cooldown_left") else 0
            items.append({
                "id": ac.account.get("id", ""),
                "name": ac.account.get("name", ""),
                "provider": ac.account.get("provider", ""),
                "proxy": bool(ac.account.get("proxy")),
                "egress": ac.account.get("egress", "system"),
                "stats": stats,
            })
        return {"ok": True, "accounts": items}

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
        owner = str(b.get("owner") or "").strip()
        note = str(b.get("note") or "").strip()
        exp = float(b.get("expires_at") or 0)
        qd = int(b.get("quota_daily") or 0)
        qm = int(b.get("quota_monthly") or 0)
        if owner or note or exp or qd or qm:
            await run_in_threadpool(
                store.update_key, k["id"], owner=owner, note=note,
                expires_at=exp, quota_daily=qd, quota_monthly=qm)
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
        owner = b.get("owner")
        note = b.get("note")
        expires_at = b.get("expires_at")
        quota_daily = b.get("quota_daily")
        quota_monthly = b.get("quota_monthly")
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
        if (name is not None or accounts is not None or models is not None
                or owner is not None or note is not None or expires_at is not None):
            await run_in_threadpool(
                store.update_key, key_id,
                name=(str(name).strip() or "未命名 Key") if name is not None else None,
                accounts=([str(x) for x in accounts if str(x).strip()]
                          if accounts is not None else None),
                models=([str(x).strip() for x in models if str(x).strip()]
                        if models is not None else None),
                owner=str(owner or "").strip() if owner is not None else None,
                note=str(note or "").strip() if note is not None else None,
                expires_at=float(expires_at or 0) if expires_at is not None else None,
                quota_daily=int(quota_daily or 0) if quota_daily is not None else None,
                quota_monthly=int(quota_monthly or 0) if quota_monthly is not None else None,
            )
        return {"ok": True}

    @router.get("/keys/{key_id}/detail")
    async def key_detail(key_id: str, request: Request, range: str = "all"):
        g = _guard(request)
        if g:
            return g
        since, _ = _range_window(range)
        data = await run_in_threadpool(store.key_detail, key_id, since)
        return {"ok": True, **data}

    @router.get("/keys/{key_id}/accounts-today")
    async def key_accounts_today(key_id: str, request: Request):
        g = _guard(request)
        if g:
            return g
        import datetime as _dt
        now = _dt.datetime.now()
        day0 = now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        rows = await run_in_threadpool(store.key_daily_by_account, key_id, day0)
        return {"ok": True, "rows": rows}

    @router.get("/usage/export")
    async def export_usage(request: Request, range: str = "all",
                           start: str = "", end: str = ""):
        g = _guard(request)
        if g:
            return g
        since, until = _range_window(range, start, end)
        rows = await run_in_threadpool(store.export_usage, since, until)
        import csv
        import io

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["时间", "Key", "账号", "平台", "模型", "输入", "输出",
                         "合计", "状态", "总耗时ms", "首Token ms", "错误码"])
        for r in rows:
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["ts"])),
                r.get("api_key_name", ""), r.get("account_name", ""),
                r.get("provider", ""), r.get("model", ""),
                r.get("prompt_tokens", 0), r.get("completion_tokens", 0),
                r.get("total_tokens", 0), r.get("status", ""),
                r.get("latency_ms", 0), r.get("first_token_ms", 0),
                r.get("error_code", ""),
            ])
        from fastapi.responses import Response

        data = "\ufeff" + buf.getvalue()
        return Response(
            content=data.encode("utf-8"),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="webai2api-usage.csv"'},
        )

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
                  "deepseek_default_model",
                  "kimi_base_url", "kimi_default_model", "kimi_default_effort",
                  "kimi_anti_min", "kimi_anti_max",
                  "qwen_base_url", "qwen_default_model", "qwen_default_effort",
                  "qwen_anti_min", "qwen_anti_max"):
            if k in b and str(b[k]).strip():
                await run_in_threadpool(store.set_setting, k, str(b[k]).strip())
        # 额度类：夹到该模型允许的上限内，防止填出荒谬值
        for k in ("deepseek_context_limit", "deepseek_output_limit",
                  "glm_context_limit", "glm_output_limit",
                  "kimi_context_limit", "kimi_output_limit",
                  "qwen_context_limit", "qwen_output_limit"):
            if k not in b:
                continue
            prov = ("chatglm" if k.startswith("glm")
                    else "kimi" if k.startswith("kimi")
                    else "qwen" if k.startswith("qwen")
                    else "deepseek")
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
        for bk in ("kimi_search", "kimi_session_cache",
                   "qwen_search", "qwen_session_cache"):
            if bk in b:
                v = b[bk]
                s = "true" if (v is True or str(v).strip().lower() in ("1", "true", "yes", "on")) else "false"
                await run_in_threadpool(store.set_setting, bk, s)
        if "tray_enabled" in b:
            v = b["tray_enabled"]
            s = "true" if (v is True or str(v).strip().lower() in ("1", "true", "yes", "on")) else "false"
            await run_in_threadpool(store.set_setting, "tray_enabled", s)
        try:
            registry.refresh()
        except Exception:
            pass
        return {"ok": True}

    # ── 授权（Lite） ──
    @router.get("/license")
    async def get_license(request: Request):
        g = _guard(request)
        if g:
            return g
        info = await run_in_threadpool(_license_info)
        info["machine"] = await run_in_threadpool(lambda: __import__(
            "webai2api.license", fromlist=["machine_code"]).machine_code())
        info["issuer"] = await run_in_threadpool(_issuer_available)
        return {"ok": True, **info}

    @router.post("/license/install")
    async def install_license(request: Request):
        g = _guard(request)
        if g:
            return g
        b = await _json(request)
        if b is None:
            return {"ok": False, "error": "请求格式错误"}
        text = str(b.get("content") or "").strip()
        if not text:
            return {"ok": False, "error": "请粘贴授权码"}
        from .license import activate
        from .paths import user_data_dir
        res = await run_in_threadpool(activate, user_data_dir(), text)
        if not res.get("ok"):
            return {"ok": False, "error": res.get("error") or "授权码无效"}
        return {"ok": True, **res}

    @router.post("/license/issue")
    async def issue_license(request: Request):
        """仅开发端且本机存在作者私钥时可用。"""
        g = _guard(request)
        if g:
            return g
        if not await run_in_threadpool(_issuer_available):
            return {"ok": False, "error": "当前版本不提供签发功能"}
        from . import issuer
        seed = await run_in_threadpool(issuer.load_author_seed)
        if seed is None:
            return {"ok": False, "error": "未找到作者私钥，无法签发"}
        b = await _json(request)
        if b is None:
            return {"ok": False, "error": "请求格式错误"}
        try:
            days = int(b.get("days") or 7)
            count = max(1, min(int(b.get("count") or 1), 100))
        except Exception:
            return {"ok": False, "error": "参数错误"}
        note = str(b.get("note") or "").strip()
        out = []
        for _ in range(count):
            info = issuer.make_code(seed, days)
            out.append({
                "code": info["code"], "id": info["id"], "days": info["days"],
                "issued": info["issued"], "note": note,
            })
        _issuer_log(out)
        return {"ok": True, "items": out}

    @router.get("/license/issued")
    async def issued_list(request: Request):
        g = _guard(request)
        if g:
            return g
        if not await run_in_threadpool(_issuer_available):
            return {"ok": False, "error": "当前版本不提供签发功能"}
        return {"ok": True, "items": _issuer_read_log()}

    @router.post("/license/activate")
    async def activate_license(request: Request):
        """专业版自用：给自己开一个授权（不绑定机器码校验失败时用）。"""
        return await install_license(request)  # type: ignore[misc]


    # ── opencode 接入 ──
    @router.post("/keys/{key_id}/opencode/connect")
    async def opencode_connect(key_id: str, request: Request):
        g = _guard(request)
        if g:
            return g
        keys = await run_in_threadpool(store.list_keys)
        k = next((x for x in keys if x["id"] == key_id), None)
        if not k:
            return {"ok": False, "error": "Key 不存在"}
        if not k.get("enabled"):
            return {"ok": False, "error": "Key 已禁用"}
        body = await _json(request) or {}
        model_override = str(body.get("model") or "").strip()
        raw_key = await run_in_threadpool(lambda: store.find_key_by_id(key_id))
        if not raw_key:
            return {"ok": False, "error": "无法读取 Key 明文"}
        key_text = raw_key.get("key", "")
        models = [model_override] if model_override else [str(m) for m in (k.get("models") or [])]
        if not models:
            models = [m["id"] for m in await run_in_threadpool(registry.model_catalog)]
        port = request.url.port or _read_current_port() or 8800
        host = f"http://127.0.0.1:{port}"
        snippet = _build_opencode_snippet(host, key_text, models)
        try:
            res = await run_in_threadpool(_write_opencode_config, snippet)
        except Exception as exc:
            logger.warning("写入 opencode 配置失败: %s", exc)
            return {"ok": False, "error": f"自动写入失败：{exc}", "snippet": snippet}
        return {"ok": True, "config_path": res.get("path"), "provider": "webai2api"}

    @router.post("/keys/{key_id}/opencode/disconnect")
    async def opencode_disconnect(key_id: str, request: Request):
        g = _guard(request)
        if g:
            return g
        try:
            res = await run_in_threadpool(_remove_opencode_provider)
        except Exception as exc:
            return {"ok": False, "error": f"移除失败：{exc}", "hint": "请手动删除 opencode.jsonc 中 provider 下的 \"webai2api\" 块"}
        out = {"ok": True, "config_path": res.get("path")}
        if res.get("not_present"):
            out["not_present"] = True
        return out

    @router.get("/opencode/status")
    async def opencode_status(request: Request):
        g = _guard(request)
        if g:
            return g
        info = await run_in_threadpool(_opencode_status_io)
        lic = await run_in_threadpool(_local_license_status)
        return {"ok": True, "installed": info["installed"], "config_path": info["config_path"],
                "has_provider": info["has_provider"], "license_ok": lic.get("valid", False)}

    return router


# ── 签发记录（仅作者本机） ──────────────────────────
def _issuer_log_path():
    from .paths import user_data_dir
    return user_data_dir() / "issued_codes.json"


def _issuer_available() -> bool:
    try:
        from .edition import is_lite
        if is_lite():
            return False  # 客户版永远不显示、不提供签发
        from . import issuer
        return issuer.load_author_seed() is not None
    except Exception:
        return False


def _issuer_read_log() -> list[dict]:
    try:
        p = _issuer_log_path()
        if not p.exists():
            return []
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _issuer_log(items: list[dict]) -> None:
    try:
        p = _issuer_log_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        existing = _issuer_read_log()
        existing.extend({**it, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
                        for it in items)
        p.write_text(json.dumps(existing[-5000:], ensure_ascii=False, indent=2),
                     encoding="utf-8")
    except Exception:
        pass


def _proxy_probe(proxy_url: str) -> dict:
    """通过指定代理请求一个回显 IP 的服务，返回出口 IP 与耗时。

    用 http://ip-api.com/json（HTTP，走普通代理）与 https://api.ipify.org（HTTPS，走 CONNECT）
    双通道，任一成功即可判断代理是否可用。
    """
    import json as _json
    import time as _time
    import urllib.error
    import urllib.request

    from .proxy import ProxyConfigError, parse_proxy

    started = _time.perf_counter()
    try:
        px = parse_proxy(proxy_url)
    except ProxyConfigError as exc:
        return {"ok": False, "error": str(exc)}
    if px is None:
        return {"ok": False, "error": "代理地址为空或无法解析"}

    # 直连作为对照（用于识别「代理没生效但能上网」的情况）
    def _direct_ip() -> str:
        try:
            with urllib.request.urlopen("https://api.ipify.org", timeout=8) as r:
                return r.read().decode().strip()
        except Exception:  # noqa: BLE001
            return ""

    direct = _direct_ip()

    handlers = []
    if px.scheme.startswith("socks"):
        try:
            import socks  # noqa: F401
        except Exception:  # noqa: BLE001
            return {"ok": False, "error": "SOCKS 代理需要 PySocks（已内置，请重启程序）",
                    "direct_ip": direct}
        # PySocks 的 urllib 支持走全局 socket 包装，不在本函数内使用
        return {"ok": False,
                "error": "SOCKS 代理的连通性测试暂只支持在聊天请求中验证；"
                         "HTTP/SOCKS 均已在客户端生效，可直接发一次对话验证",
                "direct_ip": direct}

    auth = px.auth_header()
    proxy_handler = urllib.request.ProxyHandler({
        "http": f"http://{px.host}:{px.port}",
        "https": f"http://{px.host}:{px.port}",
    })
    handlers.append(proxy_handler)
    if auth:
        class _AuthProxyHandler(urllib.request.ProxyHandler):
            def proxy_open(self, req, proxy, type):  # noqa: A003
                req.add_header("Proxy-Authorization", auth)
                return super().proxy_open(req, proxy, type)

        handlers[0] = _AuthProxyHandler({
            "http": f"http://{px.host}:{px.port}",
            "https": f"http://{px.host}:{px.port}",
        })

    opener = urllib.request.build_opener(*handlers)
    exit_ip = ""
    err = ""
    try:
        with opener.open("https://api.ipify.org", timeout=12) as r:
            exit_ip = r.read().decode().strip()
    except Exception as exc:  # noqa: BLE001
        err = str(exc)
        try:
            with opener.open("http://httpbin.org/ip", timeout=10) as r:
                exit_ip = (_json.loads(r.read().decode()).get("origin") or "").split(",")[0].strip()
                err = ""
        except Exception as exc2:  # noqa: BLE001
            err = f"{exc} / {exc2}"

    elapsed = int((_time.perf_counter() - started) * 1000)
    if not exit_ip:
        return {"ok": False, "error": f"代理不可用：{err}", "direct_ip": direct,
                "elapsed_ms": elapsed}

    same = bool(direct) and exit_ip == direct
    return {
        "ok": True,
        "exit_ip": exit_ip,
        "direct_ip": direct,
        "changed": not same,
        "warning": "出口 IP 与直连相同，代理可能未生效" if same else "",
        "proxy": px.describe(),
        "elapsed_ms": elapsed,
    }


def _clash_mixed_port() -> str:
    """探测本机是否有 Clash/Mihomo 混合端口在监听（7890/7891/7897 常见）。"""
    import socket

    for port in (7897, 7890, 7891):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.4):
                return f"http://127.0.0.1:{port}"
        except OSError:
            continue
    return ""


def _clash_global_mode():
    """通过 Clash 外部控制 API 判断是否全局模式（None=探测不到）。"""
    import json as _json
    import urllib.request

    class _NoProxy(urllib.request.ProxyHandler):
        def proxy_open(self, req, proxy, type):  # noqa: A003
            return None

    opener = urllib.request.build_opener(_NoProxy({}))
    for port in (9090, 9097):
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/proxies",
                headers={"User-Agent": "webai2api"})
            with opener.open(req, timeout=0.8) as r:
                mode = str((_json.load(r) or {}).get("mode") or "").lower()
            if mode:
                return mode == "global"
        except Exception:
            continue
    return None


def _network_status() -> dict:
    """网络环境诊断：系统代理、TUN 网卡、Clash 混合端口、全局/规则模式。

    TUN 开启 ≠ 全部走节点：规则模式下国内站点命中 DIRECT 规则，
    「直连」出口仍是家宽 IP；只有切到全局模式才会失效。
    """
    from .egress import detect_tun_interface, system_proxy_enabled

    tun = detect_tun_interface()
    clash = _clash_mixed_port()
    glob = _clash_global_mode() if tun or clash else None
    return {
        "system_proxy": system_proxy_enabled(),
        "tun": tun,
        "clash_proxy": clash,
        "global_mode": bool(glob) if glob is not None else None,
        "tun_note": (
            f"检测到 TUN 网卡「{tun}」：Clash 规则模式下国内站（DeepSeek/Kimi/千问）"
            "命中 DIRECT 规则，「直连」出口仍是真实家宽 IP；若切到「全局模式」，"
            "直连账号也会走节点。"
        ) if tun else "",
    }


def _safe_account(a: dict) -> dict:
    """返回账号（凭证脱敏）。"""
    import json as _json

    try:
        profile = _json.loads(a.get("profile") or "{}")
    except Exception:
        profile = {}
    return {
        "id": a["id"], "provider": a["provider"], "name": a["name"],
        "enabled": a["enabled"], "note": a["note"],
        "has_token": bool(a.get("auth_token")),
        "has_device": bool(a.get("device_id")),
        "has_settings": bool(a.get("settings_token")),
        "has_cookie": bool(a.get("cookie")),
        "health": a.get("health", ""),
        "checked_at": a.get("checked_at", 0),
        "profile": profile,
        "proxy": a.get("proxy", ""),
        "egress": a.get("egress", "system"),
        "created_at": a["created_at"], "updated_at": a["updated_at"],
    }


def _license_info() -> dict:
    """授权信息。客户版会校验；专业版只决定是否显示签发卡。"""
    issuer_ok = _issuer_available()
    try:
        from .edition import is_lite
        if not is_lite():
            return {"enabled": False, "issuer": issuer_ok}
        from .license import status
        from .paths import user_data_dir
        st = status(user_data_dir(), store=None)
        return {"enabled": True, "issuer": issuer_ok, **st}
    except Exception:
        return {"enabled": False, "issuer": issuer_ok}


def _safe_accounts(items: list[dict]) -> list[dict]:
    return [_safe_account(a) for a in items]

# ── opencode 配置辅助 ──────────────────────────────
def _opencode_config_path() -> Path:
    env = os.environ.get("WEBAI2API_OPENCODE_CONFIG", "")
    if env:
        return Path(env)
    return Path.home() / ".config" / "opencode" / "opencode.jsonc"


def _read_current_port() -> int | None:
    """尝试读取 run.py 写下的 current_port.txt，返回端口或 None。"""
    try:
        from .paths import user_data_dir
        p = user_data_dir().parent / "current_port.txt"
        if not p.exists():
            return None
        first = p.read_text(encoding="utf-8").strip().splitlines()[0]
        return int(first)
    except Exception:
        return None


def _opencode_status_io() -> dict:
    """阻塞 IO 全部放在线程池里跑：是否存在/是否安装/是否含 webai2api provider。"""
    path = _opencode_config_path()
    try:
        exists = path.exists()
    except Exception:
        exists = False
    try:
        installed = bool(shutil.which("opencode")) or exists
    except Exception:
        installed = exists
    has_provider = False
    if exists:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            has_provider = _provider_has_entry(text, "webai2api")
        except Exception:
            has_provider = False
    return {"installed": installed, "config_path": str(path), "has_provider": has_provider}


def _build_opencode_snippet(host: str, key: str, models: list[str]) -> str:
    lines = []
    for m in models:
        mid = str(m)
        low = mid.lower()
        jmid = json.dumps(mid, ensure_ascii=False)
        prov = infer_provider(mid)
        # 除 deepseek-chat 这类明确的非思考模型外，其余都按可输出思考处理
        reasoning = "false" if low in ("deepseek-chat", "deepseek-v4-flash") else "true"
        lim = DEFAULT_LIMITS.get(prov, DEFAULT_LIMITS["deepseek"])
        ctx, out = lim["context"], lim["output"]
        variants = ""
        if prov == "chatglm":
            variants = (',\n      "variants": { "快速": { "reasoningEffort": "low" }, '
                        '"深度": { "reasoningEffort": "medium" }, '
                        '"极致": { "reasoningEffort": "max" } }')
        elif prov == "kimi":
            # Kimi 三档：标准 / 进阶 / 极致；K2.6（快速模型）无极致档
            if "k2.6" in low:
                variants = (',\n      "variants": { "标准": { "reasoningEffort": "low" }, '
                            '"进阶": { "reasoningEffort": "high" } }')
            else:
                variants = (',\n      "variants": { "标准": { "reasoningEffort": "low" }, '
                            '"进阶": { "reasoningEffort": "high" }, '
                            '"极致": { "reasoningEffort": "max" } }')
        elif prov == "qwen":
            # Plus / Max：自动 / 思考 / 快速；Omni-Flash：只有 思考 / 快速
            if "flash" in low:
                variants = (',\n      "variants": { "思考": { "reasoningEffort": "high" }, '
                            '"快速": { "reasoningEffort": "low" } }')
            else:
                variants = (',\n      "variants": { "自动": { "reasoningEffort": "medium" }, '
                            '"思考": { "reasoningEffort": "high" }, '
                            '"快速": { "reasoningEffort": "low" } }')
        lines.append(
            f'    {jmid}: {{ "name": {jmid}, "attachment": true, "reasoning": {reasoning},\n'
            f'      "modalities": {{ "input": ["image","text"], "output": ["text"] }},\n'
            f'      "limit": {{ "context": {ctx}, "output": {out} }}{variants} }}'
        )
    return (
        '"webai2api": {\n'
        '  "npm": "@ai-sdk/openai-compatible",\n'
        '  "name": "webai2api",\n'
        '  "options": {\n'
        f'    "baseURL": {json.dumps(host + "/v1", ensure_ascii=False)},\n'
        f'    "apiKey": {json.dumps(str(key), ensure_ascii=False)},\n'
        '    "maxRetries": 0\n'
        '  },\n'
        '  "models": {\n'
        + ",\n".join(lines) +
        '\n  }\n}'
    )


def _strip_jsonc_comments(text: str) -> str:
    """逐字符移除 // 行注释与 /* */ 块注释；字符串字面量（含转义）内部不动。

    幂等：对已无注释的文本再次调用结果不变。
    """
    out: list[str] = []
    i = 0
    n = len(text)
    in_str = False
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if text.startswith("//", i):
            end = text.find("\n", i)
            if end < 0:
                break
            i = end  # 保留换行符
            continue
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end < 0 else end + 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _skip_ws(text: str, i: int) -> int:
    while i < len(text) and text[i] in " \t\r\n":
        i += 1
    return i


def _match_close(text: str, open_idx: int) -> int:
    """从 open_idx 处的 { 或 [ 开始做配对，返回配对闭合符下标；字符串/注释内的括号不计。

    找不到闭合返回 -1。
    """
    depth = 0
    i = open_idx
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            i += 1
            while i < n:
                c = text[i]
                if c == "\\":
                    i += 2
                    continue
                if c == '"':
                    break
                i += 1
            i += 1
            continue
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _find_provider_span(text: str) -> tuple[int, int] | None:
    """定位顶层 "provider" 对象的区间 (起 {, 止 } 含)。找不到返回 None。

    逐字符扫描：注释被跳过；字符串字面量内部的 "provider" 因其后不跟 : 而跳过。
    返回值下标是**原始 text** 中的下标，可直接切片。
    """
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            j = i + 1
            while j < n:
                c = text[j]
                if c == "\\":
                    j += 2
                    continue
                if c == '"':
                    break
                j += 1
            literal = text[i:j + 1]
            k = _skip_ws_comment(text, j + 1)
            if literal == '"provider"' and k < n and text[k] == ":":
                k = _skip_ws_comment(text, k + 1)
                if k < n and text[k] == "{":
                    close = _match_close(text, k)
                    if close < 0:
                        return None
                    return (k, close)
            i = j + 1
            continue
        if text.startswith("//", i):
            nl = text.find("\n", i)
            i = n if nl < 0 else nl + 1
            continue
        if text.startswith("/*", i):
            ce = text.find("*/", i + 2)
            i = n if ce < 0 else ce + 2
            continue
        i += 1
    return None


def _skip_ws_comment(text: str, i: int) -> int:
    """跳过空白与 //、/* */ 注释。"""
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in " \t\r\n":
            i += 1
            continue
        if text.startswith("//", i):
            nl = text.find("\n", i)
            i = n if nl < 0 else nl + 1
            continue
        if text.startswith("/*", i):
            ce = text.find("*/", i + 2)
            i = n if ce < 0 else ce + 2
            continue
        break
    return i


def _find_top_level_key(text: str, start: int, end: int, name: str) -> int:
    """在 provider 区间 (start, end) 内查找第一层键 "name": 的绝对下标，找不到返回 -1。

    start/end 为顶层 '{' '}' 的下标。只认深度 0 的键，避免命中兄弟 provider
    内层嵌套的同名键（如 real.models.webai2api）。跳过注释与字符串字面量。
    """
    i = start + 1
    depth = 0
    target = '"' + name + '"'
    while i < end:
        ch = text[i]
        if ch == '"':
            j = i + 1
            while j < end:
                c = text[j]
                if c == "\\":
                    j += 2
                    continue
                if c == '"':
                    break
                j += 1
            literal = text[i:j + 1]
            k = _skip_ws_comment(text, j + 1)
            if depth == 0 and literal == target and k < len(text) and text[k] == ":":
                return i
            i = j + 1
            continue
        if text.startswith("//", i):
            nl = text.find("\n", i)
            i = end if nl < 0 else nl + 1
            continue
        if text.startswith("/*", i):
            ce = text.find("*/", i + 2)
            i = end if ce < 0 else ce + 2
            continue
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        i += 1
    return -1


def _provider_has_entry(text: str, name: str) -> bool:
    """provider 对象**第一层**是否存在键 "name":（嵌套同名不算）。"""
    span = _find_provider_span(text)
    if not span:
        return False
    start, end = span
    return _find_top_level_key(text, start, end, name) >= 0


def _remove_provider_block(text: str, name: str) -> str:
    """删掉 provider 对象**第一层**的 "name": { ... }, 块（只在该区间内操作）。"""
    span = _find_provider_span(text)
    if not span:
        return text
    start, end = span
    key_idx = _find_top_level_key(text, start, end, name)
    if key_idx < 0:
        return text
    brace = text.find("{", key_idx)
    if brace < 0 or brace > end:
        return text
    close = _match_close(text, brace)
    if close < 0:
        return text
    # 连同其后的逗号一起删（若有）
    tail = close + 1
    while tail < len(text) and text[tail] in " \t\r\n":
        tail += 1
    if tail < len(text) and text[tail] == ",":
        tail += 1
        while tail < len(text) and text[tail] in " \t\r\n":
            tail += 1
    # 去掉 key 之前紧邻的缩进和残留逗号
    head = key_idx
    while head > start + 1 and text[head - 1] in " \t":
        head -= 1
    if head > start + 1 and text[head - 1] == ",":
        head -= 1
        while head > start + 1 and text[head - 1] in " \t":
            head -= 1
    out = text[:head] + text[tail:]
    # 重新配对清理区间内的尾逗号（, } / , ]），只作用于该 provider 区间
    new_span = _find_provider_span(out)
    if new_span:
        s2, e2 = new_span
        seg = out[s2:e2 + 1]
        cleaned = re.sub(r",(\s*[}\]])", r"\1", seg)
        out = out[:s2] + cleaned + out[e2 + 1:]
    return out


def _write_opencode_config(snippet: str) -> dict:
    path = _opencode_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text('{\n  "$schema": "https://opencode.ai/config.json",\n'
                        '  "provider": {\n  }\n}\n', encoding="utf-8")

    text = path.read_text(encoding="utf-8")
    backup = path.with_name(f"{path.name}.bak_{datetime.now().strftime('%Y%m%d%H%M%S_%f')}")
    backup.write_text(text, encoding="utf-8")

    stripped = _strip_jsonc_comments(text)
    span = _find_provider_span(text)
    if not span:
        raise RuntimeError("配置里找不到 provider 段，请手动粘贴")
    _brace, close_pos = span
    # provider 块闭合后下一个非空白字符必须是 , 或 }（顶层对象结尾）
    tail = _strip_jsonc_comments(text[close_pos + 1:]).lstrip(" \t\n\r")
    if tail and tail[0] not in ",}":
        raise RuntimeError("配置格式异常，请手动粘贴")

    # 移除旧的 webai2api 块
    text = _remove_provider_block(text, "webai2api")

    # 重新定位 provider 的 {（同样限定在 provider 对象内，避免注释/字符串干扰）
    span2 = _find_provider_span(text)
    if not span2:
        raise RuntimeError("配置格式异常，请手动粘贴")
    orig_brace = span2[0]
    if orig_brace < 0 or orig_brace >= len(text) or text[orig_brace] != "{":
        raise RuntimeError("配置格式异常，请手动粘贴")

    indent = "    "
    block = snippet.replace("\n", "\n" + indent)
    after = text[orig_brace + 1:]
    # 去掉 after 开头空白和已有的尾部逗号，避免双逗号
    after_stripped = after.lstrip(" \t\n\r")
    if after_stripped.startswith(","):
        after_stripped = after_stripped[1:].lstrip(" \t\n\r")
    # 把 block 插入到 { 之后，保留原来内容；确保后面有换行
    if after_stripped and not after_stripped.endswith("\n"):
        after_stripped += "\n"
    # provider 段原本为空（直接以 } 收尾）时不能加逗号，否则产生悬空 ",}"
    sep = "" if after_stripped.startswith("}") else ","
    out = text[:orig_brace + 1] + "\n" + indent + block + sep + "\n" + after_stripped

    # 同时清理 disabled_providers 里的 webai2api
    out = _remove_disabled_provider(out, "webai2api")

    path.write_text(out, encoding="utf-8")
    return {"ok": True, "path": str(path), "backup": str(backup)}


def _remove_disabled_provider(text: str, name: str) -> str:
    """从 disabled_providers 数组中移除指定项。"""
    key = '"disabled_providers"'
    idx = text.find(key)
    if idx < 0:
        return text
    arr_start = text.find("[", idx)
    if arr_start < 0:
        return text
    depth = 0
    arr_end = -1
    for i in range(arr_start, len(text)):
        ch = text[i]
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
            if depth == 0:
                arr_end = i
                break
    if arr_end < 0:
        return text
    arr_text = text[arr_start + 1:arr_end]
    # 简单正则替换掉包含 name 的项
    pattern = re.compile(r'\s*"' + re.escape(name) + r'"\s*,?')
    new_arr = pattern.sub("", arr_text)
    # 清理末尾逗号
    new_arr = re.sub(r',\s*]', "]", "[" + new_arr + "]")[1:-1]
    return text[:arr_start + 1] + new_arr + text[arr_end:]


def _opencode_structure_ok(text: str) -> bool:
    """括号/字符串是否配平（注释内的括号忽略），用于判断能否安全改写。"""
    depth = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            i += 1
            while i < n:
                c = text[i]
                if c == "\\":
                    i += 2
                    continue
                if c == '"':
                    break
                i += 1
            i += 1
            continue
        if text.startswith("//", i):
            nl = text.find("\n", i)
            i = n if nl < 0 else nl + 1
            continue
        if text.startswith("/*", i):
            ce = text.find("*/", i + 2)
            i = n if ce < 0 else ce + 2
            continue
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth < 0:
                return False
        i += 1
    return depth == 0


def _remove_opencode_provider() -> dict:
    path = _opencode_config_path()
    if not path.exists():
        # 本来就没接入：幂等成功，不建文件
        return {"ok": True, "not_present": True, "path": str(path)}
    text = path.read_text(encoding="utf-8")

    has_provider = _find_provider_span(text) is not None
    in_provider = has_provider and _provider_has_entry(text, "webai2api")
    in_disabled = '"webai2api"' in _strip_jsonc_comments(text) and (
        _remove_disabled_provider(text, "webai2api") != text)
    if not in_provider and not in_disabled:
        # provider 段不存在，或存在但没有 webai2api：幂等成功，不写文件不出备份
        return {"ok": True, "not_present": True, "path": str(path)}

    # 有残留需要删，但 JSONC 本身已损坏：改写可能把用户配置越改越坏，直接报错让用户手改
    if not _opencode_structure_ok(text):
        raise RuntimeError("配置文件 JSONC 格式异常（括号/引号不配对），无法安全移除")

    backup = path.with_name(f"{path.name}.bak_{datetime.now().strftime('%Y%m%d%H%M%S_%f')}")
    backup.write_text(text, encoding="utf-8")

    new_text = _remove_provider_block(text, "webai2api") if has_provider else text
    new_text = _remove_disabled_provider(new_text, "webai2api")
    if new_text == text:
        return {"ok": True, "not_present": True, "path": str(path), "backup": str(backup)}
    path.write_text(new_text, encoding="utf-8")
    return {"ok": True, "path": str(path), "backup": str(backup)}


def _local_license_status() -> dict:
    # Lite 已有授权实现（与 lite_api._license_status 同一取法），避免两份实现
    try:
        from .license import status as _license_status_fn
        from .paths import user_data_dir
        data_dir = user_data_dir()
        try:
            st = _license_status_fn(data_dir, Store(data_dir / "webai2api.db"))
        except Exception:
            st = _license_status_fn(data_dir, None)
        return {"valid": bool(st.get("valid"))}
    except Exception:
        return {"valid": False}
