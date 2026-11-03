"""管理面板后端 API（/admin/api/*），中文界面用。

鉴权：密码登录后发放随机 token，放在 X-Admin-Token 头里。
token 有有效期（默认 12 小时），并且登录失败有简单限速。
"""

from __future__ import annotations

import hmac
import secrets
import time
from logging import Logger
from typing import Any

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from . import __version__
from .providers import MAX_LIMITS, ProviderRegistry, infer_provider
from .store import Store

router = APIRouter(prefix="/admin/api")

TOKEN_TTL = 12 * 3600  # token 有效期
_MAX_FAILS = 8  # 连续失败上限
_FAIL_WINDOW = 300  # 失败统计窗口（秒）


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


def _to_bool(v: Any) -> bool:
    """面板/前端传来的布尔容忍归一：字符串 "false" 不能当真值。

    此前 `1 if v else 0`——JSON 里 "false"/"0" 是非空字符串 → 真值，
    想「禁用」反而变成「启用」（2026-11-02 审查）。
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def _range_window(
    range_key: str, start: str = "", end: str = ""
) -> tuple[float | None, float | None]:
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


def create_admin_router(
    store: Store, registry: ProviderRegistry, login_mgr, logger: Logger
) -> APIRouter:
    # 函数内创建 router（遮蔽模块级单例）：模块级 router 在多个 app 实例
    # 共存时（测试/热重载）闭包捕获旧 store，先注册先匹配 → 泄漏
    router = APIRouter(prefix="/admin/api")
    toks = _TOKENS

    def _guard(request: Request):
        if not toks.valid(request.headers.get("x-admin-token", "")):
            return JSONResponse(
                status_code=401, content={"ok": False, "error": "未登录"}
            )
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
        bad = [m for m in models if infer_provider(m) not in provs]
        if not bad:
            return ""
        return (
            "模型 "
            + "、".join(bad)
            + " 需要对应平台的账号；请先勾选该平台的账号，或取消这些模型。"
        )

    @router.post("/login")
    async def login(request: Request):
        if toks.rate_limited():
            return {"ok": False, "error": "尝试过于频繁，请稍后再试"}
        body = await _json(request)
        if body is None:
            return {"ok": False, "error": "请求格式错误"}
        pw = str(body.get("password", ""))
        expect = await run_in_threadpool(store.get_setting, "admin_password", "admin")
        if not hmac.compare_digest(
            pw.encode("utf-8"), str(expect).encode("utf-8")
        ):
            toks.note_fail()
            return {"ok": False, "error": "密码错误"}
        toks.note_ok()
        return {
            "ok": True,
            "token": toks.issue(),
            # 必须 encode：str 对 str 的 compare_digest 遇非 ASCII 密码
            # 抛 TypeError → 正确密码也 500，面板永久锁死（2026-11-01 审计）
            "default_password": hmac.compare_digest(
                str(expect).encode("utf-8"), b"admin"
            ),
        }

    @router.post("/logout")
    async def logout(request: Request):
        toks.revoke(request.headers.get("x-admin-token", ""))
        return {"ok": True}

    @router.get("/state")
    async def state(
        request: Request,
        range: str = "all",
        start: str = "",
        end: str = "",
        limit: int = 50,
        offset: int = 0,
        key_id: str = "",
        account_id: str = "",
        status: str = "",
    ):
        g = _guard(request)
        if g:
            return g
        accounts = await run_in_threadpool(store.list_accounts)
        since_ts, until_ts = _range_window(range, start, end)
        limit = max(1, min(int(limit or 50), 500))
        offset = max(0, int(offset or 0))
        catalog = await run_in_threadpool(registry.model_catalog)
        return {
            "ok": True,
            "range": range,
            "version": __version__,
            "accounts": _safe_accounts(accounts),
            "login": {a["id"]: login_mgr.state(a["id"]) for a in accounts},
            "keys": await run_in_threadpool(store.list_keys),
            "models": [m["id"] for m in catalog],
            "model_providers": {m["id"]: infer_provider(m["id"]) for m in catalog},
            "usage": await run_in_threadpool(store.usage_summary, since_ts, until_ts),
            "today": await run_in_threadpool(store.today_summary),
            "account_recent_errors": await run_in_threadpool(
                store.recent_account_errors
            ),
            "cooldowns": await run_in_threadpool(registry.cooldowns),
            "network": await run_in_threadpool(_network_status),
            "key_usage": await run_in_threadpool(store.key_usage_map),
            "recent": await run_in_threadpool(
                store.recent_usage, limit, offset, status, "", key_id, account_id
            ),
            "recent_total": await run_in_threadpool(
                store.count_usage, "", status, key_id, account_id
            ),
            "recent_limit": limit,
            "recent_offset": offset,
            "settings": {
                # 37 个设置键一次批量读取（原先逐个 await = 37 次线程池往返）
                **await run_in_threadpool(store.get_many, STATE_SETTING_DEFAULTS),
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
        from .providers import PUBLIC_PROVIDERS

        if str(b.get("provider", "deepseek")).strip().lower() not in PUBLIC_PROVIDERS:
            return {"ok": False, "error": "该平台暂未开放"}
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
        await run_in_threadpool(registry.refresh, acc.get("provider"))
        return {"ok": True, "account": _safe_account(acc)}

    @router.put("/accounts/{account_id}")
    async def update_account(account_id: str, request: Request):
        g = _guard(request)
        if g:
            return g
        b = await _json(request)
        if b is None:
            return {"ok": False, "error": "请求格式错误"}
        fields = {
            k: b[k]
            for k in (
                "name",
                "auth_token",
                "device_id",
                "settings_token",
                "cookie",
                "enabled",
                "note",
                "proxy",
                "egress",
            )
            if k in b
        }
        acc0 = await run_in_threadpool(store.get_account, account_id)
        if "egress" in fields:
            from .egress import normalize_egress

            fields["egress"] = normalize_egress(str(fields["egress"]))
        # 凭证有变化 → 重新检查
        if any(
            k in fields for k in ("auth_token", "device_id", "settings_token", "cookie")
        ):
            fields["health"] = ""
            fields["checked_at"] = 0.0
        if "enabled" in fields:
            fields["enabled"] = 1 if _to_bool(fields["enabled"]) else 0
        await run_in_threadpool(store.update_account, account_id, **fields)
        await run_in_threadpool(
            registry.refresh, acc0.get("provider") if acc0 else None
        )
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
            status, profile, new_token, new_refresh = await run_in_threadpool(
                _check, acc
            )
            await run_in_threadpool(
                store.set_account_health,
                account_id,
                status,
                profile,
                new_token or None,
                new_refresh or None,
            )
            return {"ok": True, "status": status, "profile": profile}
        except Exception as exc:  # noqa: BLE001
            from .gateway import _classify_error

            # 手动检测的失败≠凭证过期：网络层失败（DNS/代理/超时/TLS）此前
            # 一律标 expired，两级选号会把好号集体打成兜底号（2026-11-01
            # 审计）。与网关运行时同口径：只有 auth 类才动 health，其余
            # 保持原状（账号照常参与轮询，等下次检查/真实请求再验证）。
            if _classify_error(exc) == "auth":
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
        acc_del = await run_in_threadpool(store.get_account, account_id)
        # qwen 账号删除前停掉它的隐藏中继：否则孤儿 WebView2 常驻、relay
        # worker 继续对已删 id 空转（2026-11-01 审计）
        if acc_del and acc_del.get("provider") == "qwen":
            try:
                from .qwen_relay import stop_relay

                await run_in_threadpool(stop_relay, account_id)
            except Exception:  # noqa: BLE001
                pass
        await run_in_threadpool(store.delete_account, account_id)
        await run_in_threadpool(
            registry.refresh, acc_del.get("provider") if acc_del else None
        )
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
            # 风险指标：走 AccountClient 的差量持久化（当日累计跨重启保留）
            if hasattr(ac, "persist_and_display_risk"):
                try:
                    stats = await run_in_threadpool(ac.persist_and_display_risk)
                except Exception:  # noqa: BLE001
                    stats = {}
            elif client is not None and hasattr(client, "risk_stats"):
                try:
                    stats = client.risk_stats()
                except Exception:  # noqa: BLE001
                    stats = {}
            else:
                stats = {}
            if hasattr(ac, "daily_used"):
                try:
                    stats["daily_count"] = int(await run_in_threadpool(ac.daily_used))
                except Exception:  # noqa: BLE001
                    stats["daily_count"] = int(getattr(ac, "_daily_count", 0) or 0)
            else:
                stats["daily_count"] = int(getattr(ac, "_daily_count", 0) or 0)
            stats["daily_limit"] = int(getattr(ac, "daily_limit", 0) or 0)
            stats["cooldown_left"] = (
                int(ac.cooldown_left()) if hasattr(ac, "cooldown_left") else 0
            )
            items.append(
                {
                    "id": ac.account.get("id", ""),
                    "name": ac.account.get("name", ""),
                    "provider": ac.account.get("provider", ""),
                    "proxy": bool(ac.account.get("proxy")),
                    "egress": ac.account.get("egress", "system"),
                    "stats": stats,
                }
            )
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

    @router.post("/browser/cleanup")
    async def browser_cleanup(request: Request):
        """清除本机残留的浏览器登录痕迹（一次性登录目录 + 旧版默认 WebView2 存储）。"""
        g = _guard(request)
        if g:
            return g
        from .login_webview import cleanup_browser_residue

        return await run_in_threadpool(cleanup_browser_residue)

    # 会话管理支持的平台 → (注册表模型名, 平台 provider)
    _SESSION_PROVIDERS = {
        "deepseek": "deepseek-chat",
        "glm": "GLM-5.3-Flash",
        "chatglm": "GLM-5.3-Flash",
        "kimi": "kimi-k2.6",
        # qwen：列表接口未探明（被阿里云 WAF 拦截），暂不支持
    }

    @router.get("/sessions")
    async def list_sessions(request: Request, provider: str = "", account_id: str = ""):
        """列出各账号的上游会话数量与标题（只读，用于预览要清理什么）。"""
        g = _guard(request)
        if g:
            return g
        prov = (provider or "deepseek").strip().lower()
        model_key = _SESSION_PROVIDERS.get(prov)
        if not model_key:
            return {
                "ok": False,
                "error": f"暂不支持 {prov} 的会话清理（支持：deepseek / glm / kimi）",
            }

        pool = registry.pool_for_model(model_key)

        def _collect():
            out = []
            for ac in pool.available():
                if account_id and ac.account["id"] != account_id:
                    continue
                row = {
                    "account_id": ac.account["id"],
                    "account": ac.account.get("name", ""),
                }
                try:
                    sessions = ac.client.list_upstream_sessions()
                    row["total"] = len(sessions)
                    row["sessions"] = [
                        {"id": s.get("id"), "title": s.get("title") or ""}
                        for s in sessions[:200]
                    ]
                except Exception as exc:  # noqa: BLE001
                    row["total"] = -1
                    row["error"] = str(exc)[:200]
                out.append(row)
            return out

        rows = await run_in_threadpool(_collect)
        return {"ok": True, "accounts": rows}

    @router.post("/sessions/purge")
    async def purge_sessions(
        request: Request, provider: str = "", account_id: str = ""
    ):
        """删除上游历史会话。

        参数：provider=deepseek、account_id（可空=全部账号）、
        dry_run（true 只统计）、keep_titles（逗号分隔，命中标题的保留）。
        """
        g = _guard(request)
        if g:
            return g
        prov = (provider or "deepseek").strip().lower()
        model_key = _SESSION_PROVIDERS.get(prov)
        if not model_key:
            return {
                "ok": False,
                "error": f"暂不支持 {prov} 的会话清理（支持：deepseek / glm / kimi）",
            }

        body = await _json(request) or {}
        dry_run = bool(body.get("dry_run"))
        keep = [
            str(x).strip()
            for x in str(body.get("keep_titles") or "").split(",")
            if str(x).strip()
        ]

        pool = registry.pool_for_model(model_key)

        def _purge():
            out = []
            for ac in pool.available():
                if account_id and ac.account["id"] != account_id:
                    continue
                row = {
                    "account_id": ac.account["id"],
                    "account": ac.account.get("name", ""),
                }
                try:
                    row.update(
                        ac.client.purge_upstream_sessions(
                            keep_titles=tuple(keep), dry_run=dry_run
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    row["error"] = str(exc)[:200]
                out.append(row)
            return out

        rows = await run_in_threadpool(_purge)
        return {"ok": True, "dry_run": dry_run, "accounts": rows}

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
            store.add_key,
            b.get("name", "未命名 Key"),
            None,
            accounts,
            models,
        )
        note = str(b.get("note") or "").strip()
        # 归属人/到期/额度四字段已退役：不再读取请求值，也不再写入
        if note:
            await run_in_threadpool(store.update_key, k["id"], note=note)
        return {"ok": True, "key": k}

    @router.put("/keys/{key_id}")
    async def update_key(key_id: str, request: Request):
        g = _guard(request)
        if g:
            return g
        b = await _json(request) or {}
        if "enabled" in b:
            await run_in_threadpool(store.set_key_enabled, key_id, _to_bool(b["enabled"]))
        name = b.get("name")
        accounts = b.get("accounts")
        models = b.get("models")
        note = b.get("note")
        # 归属人/到期时间/每日与每月 Token 上限四项已退役（2026-10-28 拍板，
        # 2026-11-02 连同 DB 列与网关执行侧一并物理删除）：请求里的这些字段
        # 直接忽略。
        if accounts is not None or models is not None:
            cur = next(
                (
                    k
                    for k in await run_in_threadpool(store.list_keys)
                    if k["id"] == key_id
                ),
                None,
            )
            nacc = (
                [str(x) for x in accounts if str(x).strip()]
                if accounts is not None
                else (cur or {}).get("accounts", [])
            )
            nmod = (
                [str(x).strip() for x in models if str(x).strip()]
                if models is not None
                else (cur or {}).get("models", [])
            )
            bad = await _incompatible_models(nacc, nmod)
            if bad:
                return {"ok": False, "error": bad}
        if (
            name is not None
            or accounts is not None
            or models is not None
            or note is not None
        ):
            await run_in_threadpool(
                store.update_key,
                key_id,
                name=(str(name).strip() or "未命名 Key") if name is not None else None,
                accounts=(
                    [str(x) for x in accounts if str(x).strip()]
                    if accounts is not None
                    else None
                ),
                models=(
                    [str(x).strip() for x in models if str(x).strip()]
                    if models is not None
                    else None
                ),
                note=str(note or "").strip() if note is not None else None,
            )
        return {"ok": True}

    @router.get("/keys/{key_id}/detail")
    async def key_detail(
        key_id: str, request: Request, range: str = "all",
        start: str = "", end: str = "",
    ):
        g = _guard(request)
        if g:
            return g
        since, _ = _range_window(range, start, end)
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
    async def export_usage(
        request: Request, range: str = "all", start: str = "", end: str = "",
        status: str = "", key_id: str = "", account_id: str = "",
    ):
        g = _guard(request)
        if g:
            return g
        since, until = _range_window(range, start, end)
        rows = await run_in_threadpool(
            store.export_usage, since, until, status, key_id, account_id
        )
        import csv
        import io

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            [
                "时间",
                "Key",
                "Key ID",
                "账号",
                "账号 ID",
                "平台",
                "模型",
                "输入",
                "输出",
                "合计",
                "状态",
                "总耗时ms",
                "首Token ms",
                "错误码",
            ]
        )
        def _csv_safe(v):
            # 公式注入防护：= + - @ 开头的单元格在 Excel 里会当公式执行
            sv = str(v)
            return "'" + sv if sv[:1] in ("=", "+", "-", "@") else sv

        for r in rows:
            writer.writerow(
                [
                    _csv_safe(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["ts"]))),
                    _csv_safe(r.get("api_key_name", "")),
                    _csv_safe(r.get("api_key_id", "")),
                    _csv_safe(r.get("account_name", "")),
                    _csv_safe(r.get("account_id", "")),
                    _csv_safe(r.get("provider", "")),
                    _csv_safe(r.get("model", "")),
                    r.get("prompt_tokens", 0),
                    r.get("completion_tokens", 0),
                    r.get("total_tokens", 0),
                    _csv_safe(r.get("status", "")),
                    r.get("latency_ms", 0),
                    r.get("first_token_ms", 0),
                    _csv_safe(r.get("error_code", "")),
                ]
            )
        from fastapi.responses import Response

        data = "\ufeff" + buf.getvalue()
        return Response(
            content=data.encode("utf-8"),
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": 'attachment; filename="webai2api-usage.csv"'
            },
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
        for k in (
            "deepseek_wasm_path",
            "admin_password",
            "deepseek_anti_min",
            "deepseek_anti_max",
            "glm_base_url",
            "glm_default_model",
            "glm_default_effort",
            "glm_anti_min",
            "glm_anti_max",
            "usage_retain_days",
            "deepseek_default_model",
            "kimi_base_url",
            "kimi_default_model",
            "kimi_default_effort",
            "kimi_anti_min",
            "kimi_anti_max",
            "qwen_base_url",
            "qwen_default_model",
            "qwen_default_effort",
            "qwen_anti_min",
            "qwen_anti_max",
        ):
            if k in b and (
                str(b[k]).strip()
                or k not in (
                    "admin_password",
                    "deepseek_base_url", "glm_base_url",
                    "kimi_base_url", "qwen_base_url",
                    # 防风控延迟存空串会让 float('') 炸掉整个平台的池构建
                    # （2026-10-30 审计），空值一律保留旧值
                    "deepseek_anti_min", "deepseek_anti_max",
                    "glm_anti_min", "glm_anti_max",
                    "kimi_anti_min", "kimi_anti_max",
                    "qwen_anti_min", "qwen_anti_max",
                )
            ):
                await run_in_threadpool(store.set_setting, k, str(b[k]).strip())
        # 额度类：夹到该模型允许的上限内，防止填出荒谬值
        for k in (
            "deepseek_context_limit",
            "deepseek_output_limit",
            "glm_context_limit",
            "glm_output_limit",
            "kimi_context_limit",
            "kimi_output_limit",
            "qwen_context_limit",
            "qwen_output_limit",
        ):
            if k not in b:
                continue
            prov = (
                "chatglm"
                if k.startswith("glm")
                else "kimi"
                if k.startswith("kimi")
                else "qwen"
                if k.startswith("qwen")
                else "deepseek"
            )
            kind = "context" if "context" in k else "output"
            cap = MAX_LIMITS[prov][kind]
            try:
                val = int(float(str(b[k]).strip()))
            except Exception:
                continue
            val = max(1, min(val, cap))
            await run_in_threadpool(store.set_setting, k, str(val))
        for bk in (
            "deepseek_session_cache",
            "deepseek_search",
            "glm_search",
            "glm_session_cache",
            "kimi_search",
            "kimi_session_cache",
            "qwen_search",
            "qwen_session_cache",
            "tray_enabled",
        ):
            if bk in b:
                await run_in_threadpool(
                    store.set_setting, bk, "true" if _to_bool(b[bk]) else "false"
                )
        try:
            await run_in_threadpool(registry.refresh)
        except Exception:
            pass
        return {"ok": True}

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
        models = (
            [model_override]
            if model_override
            else [str(m) for m in (k.get("models") or [])]
        )
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
            return {
                "ok": False,
                "error": f"移除失败：{exc}",
                "hint": '请手动删除 opencode.jsonc 中 provider 下的 "webai2api" 块',
            }
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
        return {
            "ok": True,
            "installed": info["installed"],
            "config_path": info["config_path"],
            "has_provider": info["has_provider"],
        }

    return router


# ── 签发记录（仅作者本机） ──────────────────────────
STATE_SETTING_DEFAULTS = {
    "deepseek_wasm_path": "",
    "deepseek_session_cache": "true",
    "deepseek_default_model": "deepseek-chat",
    "tray_enabled": "true",
    "deepseek_search": "false",
    "deepseek_anti_min": "1.5",
    "deepseek_anti_max": "3.0",
    "glm_base_url": "https://chatglm.cn",
    "glm_default_model": "GLM-5.3",
    "glm_default_effort": "high",
    "glm_anti_min": "1.5",
    "glm_anti_max": "3.0",
    "glm_search": "false",
    "glm_session_cache": "true",
    "usage_retain_days": "0",
    "deepseek_context_limit": "1000000",
    "deepseek_output_limit": "384000",
    "glm_context_limit": "1000000",
    "glm_output_limit": "128000",
    "kimi_base_url": "https://www.kimi.com",
    "kimi_default_model": "kimi-k3",
    "kimi_default_effort": "",
    "kimi_anti_min": "1.5",
    "kimi_anti_max": "3.0",
    "kimi_search": "false",
    "kimi_session_cache": "true",
    "kimi_context_limit": "262144",
    "kimi_output_limit": "32768",
    "qwen_base_url": "https://chat.qwen.ai",
    "qwen_default_model": "qwen3.8-max",
    "qwen_default_effort": "",
    "qwen_anti_min": "1.5",
    "qwen_anti_max": "3.0",
    "qwen_search": "false",
    "qwen_session_cache": "true",
    "qwen_context_limit": "262144",
    "qwen_output_limit": "65536",
}


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
            return {
                "ok": False,
                "error": "SOCKS 代理需要 PySocks（已内置，请重启程序）",
                "direct_ip": direct,
            }
        # PySocks 的 urllib 支持走全局 socket 包装，不在本函数内使用
        return {
            "ok": False,
            "error": "SOCKS 代理的连通性测试暂只支持在聊天请求中验证；"
            "HTTP/SOCKS 均已在客户端生效，可直接发一次对话验证",
            "direct_ip": direct,
        }

    auth = px.auth_header()
    proxy_handler = urllib.request.ProxyHandler(
        {
            "http": f"http://{px.host}:{px.port}",
            "https": f"http://{px.host}:{px.port}",
        }
    )
    handlers.append(proxy_handler)
    if auth:

        class _AuthProxyHandler(urllib.request.ProxyHandler):
            def proxy_open(self, req, proxy, type):  # noqa: A003
                req.add_header("Proxy-Authorization", auth)
                return super().proxy_open(req, proxy, type)

        handlers[0] = _AuthProxyHandler(
            {
                "http": f"http://{px.host}:{px.port}",
                "https": f"http://{px.host}:{px.port}",
            }
        )

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
                exit_ip = (
                    (_json.loads(r.read().decode()).get("origin") or "")
                    .split(",")[0]
                    .strip()
                )
                err = ""
        except Exception as exc2:  # noqa: BLE001
            err = f"{exc} / {exc2}"

    elapsed = int((_time.perf_counter() - started) * 1000)
    if not exit_ip:
        return {
            "ok": False,
            "error": f"代理不可用：{err}",
            "direct_ip": direct,
            "elapsed_ms": elapsed,
        }

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
                f"http://127.0.0.1:{port}/proxies", headers={"User-Agent": "webai2api"}
            )
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
        )
        if tun
        else "",
    }


def _safe_account(a: dict) -> dict:
    """返回账号（凭证脱敏）。"""
    import json as _json

    try:
        profile = _json.loads(a.get("profile") or "{}")
    except Exception:
        profile = {}
    return {
        "id": a["id"],
        "provider": a["provider"],
        "name": a["name"],
        "enabled": a["enabled"],
        "note": a["note"],
        "has_token": bool(a.get("auth_token")),
        "has_device": bool(a.get("device_id")),
        "has_settings": bool(a.get("settings_token")),
        "has_cookie": bool(a.get("cookie")),
        "health": a.get("health", ""),
        "checked_at": a.get("checked_at", 0),
        "profile": profile,
        "proxy": a.get("proxy", ""),
        "egress": a.get("egress", "system"),
        "created_at": a["created_at"],
        "updated_at": a["updated_at"],
    }


def _safe_accounts(items: list[dict]) -> list[dict]:
    return [_safe_account(a) for a in items]


# ── opencode 配置辅助 ──────────────────────────────
# 已拆分到 opencode_config.py（纯函数/纯 IO，可独立测试）；这里只
# re-export 端点真用的 5 个名字，其余由测试直连新模块
from .opencode_config import (  # noqa: E402
    _build_opencode_snippet,
    _opencode_status_io,
    _read_current_port,
    _remove_opencode_provider,
    _write_opencode_config,
)
