"""客户端版（lite）后端接口。

设计目标：普通用户只需要「登录账号 → 一键接入 → 看今天用了多少」。

与专业版的区别：
- 不暴露 Key 管理：内部自动维护一个名为 internal 的 Key
- 不暴露额度、归属、备注、CSV、按 Key 统计
- 只提供：状态、登录、模型开关、用量（今日 + 近 7 天）、接入配置、授权
"""
from __future__ import annotations

import json
import os
import shutil
import time
from logging import Logger
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from . import __version__
from .edition import is_lite
from .providers import ProviderRegistry
from .store import Store

router = APIRouter(prefix="/admin/api")

INTERNAL_KEY_NAME = "internal"
PROVIDER_LABEL = {"deepseek": "DeepSeek", "chatglm": "智谱清言"}

# 面向普通用户的模型说明（不暴露技术细节）
MODEL_INFO = {
    "deepseek-chat": ("DeepSeek 标准", "日常问答、写作、代码"),
    "deepseek-reasoner": ("DeepSeek 深度思考", "复杂推理、数学、长文分析"),
    "GLM-5.3": ("智谱 GLM-5.3", "旗舰模型，支持三档思考强度"),
    "GLM-5.3-Flash": ("智谱 GLM-5.3 Flash", "更快、更省"),
}


def _ok(**kw) -> dict:
    return {"ok": True, **kw}


def _err(msg: str, code: int = 200) -> JSONResponse:
    return JSONResponse(status_code=code, content={"ok": False, "error": msg})


def create_lite_router(store: Store, registry: ProviderRegistry,
                       login_mgr, logger: Logger) -> APIRouter:
    from .admin_api import _TOKENS

    def _guard(request: Request):
        if not _TOKENS.valid(request.headers.get("x-admin-token", "")):
            return JSONResponse(status_code=401, content={"ok": False, "error": "未登录"})
        return None

    # ── 内部 Key ─────────────────────────────────────
    def _internal_key() -> dict:
        """确保存在内部 Key，返回它。普通用户永远看不到它的值。"""
        for k in store.list_keys():
            if k.get("name") == INTERNAL_KEY_NAME:
                return k
        k = store.add_key(INTERNAL_KEY_NAME)
        logger.info("已创建内部 Key: %s", k["id"])
        return k

    # ── 登录（面板鉴权）─────────────────────────────
    @router.post("/login")
    async def login(request: Request):
        import hmac
        if _TOKENS.rate_limited():
            return {"ok": False, "error": "尝试过于频繁，请稍后再试"}
        try:
            body = await request.json()
        except Exception:
            body = {}
        pw = str((body or {}).get("password", ""))
        expect = await run_in_threadpool(store.get_setting, "admin_password", "admin")
        if not hmac.compare_digest(pw, expect):
            _TOKENS.note_fail()
            return {"ok": False, "error": "密码错误"}
        _TOKENS.note_ok()
        return {"ok": True, "token": _TOKENS.issue(),
                "default_password": hmac.compare_digest(expect, "admin")}

    @router.post("/logout")
    async def logout(request: Request):
        _TOKENS.revoke(request.headers.get("x-admin-token", ""))
        return {"ok": True}

    # ── 状态 ─────────────────────────────────────────
    @router.get("/lite/state")
    async def lite_state(request: Request):
        g = _guard(request)
        if g:
            return g
        accounts = await run_in_threadpool(store.list_accounts)
        keys = await run_in_threadpool(store.list_keys)
        models = [m["id"] for m in await run_in_threadpool(registry.model_catalog)]

        acc_list = []
        for a in accounts:
            acc_list.append({
                "id": a["id"],
                "provider": a["provider"],
                "provider_label": PROVIDER_LABEL.get(a["provider"], a["provider"]),
                "name": a["name"],
                "enabled": bool(a.get("enabled")),
                "logged_in": bool((a.get("auth_token") or "").strip()
                                  or (a.get("cookie") or "").strip()),
                "health": a.get("health", ""),
                "login": login_mgr.state(a["id"]),
            })

        today = await run_in_threadpool(store.today_summary)
        usage = await run_in_threadpool(store.usage_summary, time.time() - 7 * 86400, None)
        daily = (usage or {}).get("daily") or []

        enabled_models = []
        for mid in models:
            title, desc = MODEL_INFO.get(mid, (mid, ""))
            enabled_models.append({"id": mid, "title": title, "desc": desc,
                                   "provider": "chatglm" if mid.lower().startswith("glm") else "deepseek"})

        lic = _license_status()
        return _ok(
            version=__version__,
            edition="lite",
            accounts=acc_list,
            models=enabled_models,
            ready=any(a["logged_in"] for a in acc_list),
            today=today.get("today", {}),
            yesterday=today.get("yesterday", {}),
            daily=daily[-7:],
            license=lic,
            internal_key_ready=bool(keys),
        )

    # ── 登录 ─────────────────────────────────────────
    @router.post("/lite/login")
    async def lite_login(request: Request):
        """首次使用时：按平台创建（或复用）账号并启动内置浏览器登录。"""
        g = _guard(request)
        if g:
            return g
        try:
            body = await request.json()
        except Exception:
            body = {}
        provider = str(body.get("provider") or "deepseek").strip().lower()
        if provider not in ("deepseek", "chatglm"):
            return _err("不支持的平台")

        # 复用同平台第一个账号；没有则新建
        acc = next((a for a in store.list_accounts(provider)), None)
        if acc is None:
            name = PROVIDER_LABEL.get(provider, provider)
            acc = await run_in_threadpool(
                store.add_account, provider, name, {}, "")
            await run_in_threadpool(registry.refresh)
            logger.info("已创建账号：%s (%s)", name, acc["id"])
        res = login_mgr.start(acc["id"])
        if not res.get("ok"):
            return _err(res.get("error", "启动登录失败"))
        return _ok(account_id=acc["id"])

    @router.get("/lite/login/{account_id}")
    async def lite_login_state(account_id: str, request: Request):
        g = _guard(request)
        if g:
            return g
        st = login_mgr.state(account_id)
        if st.get("status") == "success":
            await run_in_threadpool(registry.refresh)
        return _ok(state=st)

    @router.post("/lite/accounts/{account_id}/remove")
    async def lite_remove_account(account_id: str, request: Request):
        g = _guard(request)
        if g:
            return g
        await run_in_threadpool(store.delete_account, account_id)
        await run_in_threadpool(registry.refresh)
        return _ok()

    # ── 接入配置 ─────────────────────────────────────
    @router.get("/lite/connect")
    async def lite_connect(request: Request):
        """返回 opencode 配置片段 + 内部 Key + 端口。"""
        g = _guard(request)
        if g:
            return g
        k = await run_in_threadpool(_internal_key)
        port = request.url.port or 8800
        host = f"http://127.0.0.1:{port}"
        models = [m["id"] for m in await run_in_threadpool(registry.model_catalog)]
        snippet = _build_snippet(host, k["key"], models)
        return _ok(key=k["key"], base_url=f"{host}/v1", snippet=snippet,
                   config_path=str(_opencode_config_path()))

    @router.post("/lite/connect/apply")
    async def lite_connect_apply(request: Request):
        """把配置写进 opencode.jsonc（自动备份；失败返回片段让用户手动粘贴）。"""
        g = _guard(request)
        if g:
            return g
        k = await run_in_threadpool(_internal_key)
        port = request.url.port or 8800
        host = f"http://127.0.0.1:{port}"
        models = [m["id"] for m in await run_in_threadpool(registry.model_catalog)]
        snippet = _build_snippet(host, k["key"], models)
        try:
            res = await run_in_threadpool(_write_opencode_config, snippet)
        except Exception as exc:  # noqa: BLE001
            logger.warning("写入 opencode 配置失败: %s", exc)
            return _ok(applied=False, snippet=snippet,
                       error=f"自动写入失败：{exc}")
        return _ok(applied=bool(res.get("ok")), snippet=snippet, **res)

    @router.get("/lite/connect/check")
    async def lite_connect_check(request: Request):
        g = _guard(request)
        if g:
            return g
        path = _opencode_config_path()
        exists = path.exists()
        installed = bool(shutil.which("opencode")) or exists
        has_block = False
        if exists:
            try:
                has_block = "webai2api" in path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                pass
        return _ok(installed=installed, config_exists=exists,
                   configured=has_block, config_path=str(path))

    # ── 设置 ─────────────────────────────────────────
    @router.post("/lite/settings")
    async def lite_settings(request: Request):
        g = _guard(request)
        if g:
            return g
        try:
            body = await request.json()
        except Exception:
            body = {}
        if "autostart" in body:
            try:
                from .system_api import set_autostart
                await run_in_threadpool(set_autostart, bool(body["autostart"]))
            except Exception as exc:  # noqa: BLE001
                logger.warning("设置开机自启失败: %s", exc)
        if "tray" in body:
            await run_in_threadpool(
                store.set_setting, "tray_enabled",
                "true" if body["tray"] else "false")
        return _ok()

    # ── 授权 ─────────────────────────────────────────
    @router.get("/lite/license")
    async def lite_license(request: Request):
        g = _guard(request)
        if g:
            return g
        return _ok(**_license_status())

    @router.post("/lite/license/install")
    async def lite_license_install(request: Request):
        """粘贴授权文件内容 → 校验 → 写入 data/license.key。"""
        g = _guard(request)
        if g:
            return g
        try:
            body = await request.json()
        except Exception:
            return _err("请求格式错误")
        text = str(body.get("content") or "").strip()
        if not text:
            return _err("请粘贴授权内容")
        try:
            doc = json.loads(text)
        except Exception:
            return _err("授权内容不是合法 JSON")
        from .license import license_path
        path = license_path(store.db_path.parent)
        try:
            path.write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            return _err(f"写入失败：{exc}")
        st = _license_status()
        if not st.get("valid"):
            try:
                path.unlink()
            except Exception:
                pass
            return _err(st.get("reason") or "授权无效")
        return _ok(**st)

    @router.get("/lite/machine")
    async def lite_machine(request: Request):
        g = _guard(request)
        if g:
            return g
        from .license import machine_code
        return _ok(machine=machine_code())

    return router


# ── 辅助 ────────────────────────────────────────────
def _license_status() -> dict:
    from .license import status
    from .paths import user_data_dir
    from .store import Store as _Store
    data_dir = user_data_dir()
    try:
        st = _Store(data_dir / "webai2api.db")
        return status(data_dir, st)
    except Exception:
        return status(data_dir, None)


def _opencode_config_path() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
        # opencode 在 Windows 上实际使用 ~/.config/opencode
        return Path.home() / ".config" / "opencode" / "opencode.jsonc"
    return Path.home() / ".config" / "opencode" / "opencode.jsonc"


def _build_snippet(host: str, key: str, models: list[str]) -> str:
    lines = []
    for m in models:
        mid = str(m)
        reasoning = "true" if ("reasoner" in mid.lower() or mid.lower().startswith("glm")) else "false"
        ctx, out = (1000000, 128000) if mid.lower().startswith("glm") else (1000000, 384000)
        variants = ""
        if mid.lower().startswith("glm"):
            variants = (',\n      "variants": { "快速": { "reasoningEffort": "low" }, '
                        '"深度": { "reasoningEffort": "medium" }, '
                        '"极致": { "reasoningEffort": "max" } }')
        lines.append(
            f'    "{mid}": {{ "name": "{mid}", "attachment": true, "reasoning": {reasoning},\n'
            f'      "modalities": {{ "input": ["image","text"], "output": ["text"] }},\n'
            f'      "limit": {{ "context": {ctx}, "output": {out} }}{variants} }}'
        )
    return (
        '"webai2api": {\n'
        '  "npm": "@ai-sdk/openai-compatible",\n'
        '  "name": "webai2api",\n'
        '  "options": {\n'
        f'    "baseURL": "{host}/v1",\n'
        f'    "apiKey": "{key}",\n'
        '    "maxRetries": 0\n'
        '  },\n'
        '  "models": {\n'
        + ",\n".join(lines) +
        '\n  }\n}'
    )


def _write_opencode_config(snippet: str) -> dict:
    """把 provider 块合并进 opencode.jsonc。

    采用最小侵入：定位 "provider": { 之后插入；无法定位则返回失败。
    写入前备份为 opencode.jsonc.bak_<时间戳>。
    """
    path = _opencode_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text('{\n  "$schema": "https://opencode.ai/config.json",\n'
                        '  "provider": {\n  }\n}\n', encoding="utf-8")

    text = path.read_text(encoding="utf-8")
    backup = path.with_name(f"{path.name}.bak_{time.strftime('%Y%m%d%H%M%S')}")
    backup.write_text(text, encoding="utf-8")

    stripped = _remove_provider_block(text, "webai2api")
    idx = stripped.find('"provider"')
    if idx < 0:
        raise RuntimeError("配置里找不到 provider 段，请手动粘贴")
    brace = stripped.find("{", idx)
    if brace < 0:
        raise RuntimeError("配置格式异常，请手动粘贴")

    indent = "    "
    block = snippet.replace("\n", "\n" + indent)
    out = stripped[:brace + 1] + "\n" + indent + block + ",\n" + stripped[brace + 1:]
    path.write_text(out, encoding="utf-8")
    return {"ok": True, "path": str(path), "backup": str(backup)}


def _remove_provider_block(text: str, name: str) -> str:
    """删掉已存在的 "webai2api": { ... }, 块（用于重复接入时覆盖）。"""
    key = f'"{name}"'
    idx = text.find(key)
    if idx < 0:
        return text
    brace = text.find("{", idx)
    if brace < 0:
        return text
    depth = 0
    for i in range(brace, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                while end < len(text) and text[end] in ", \t":
                    end += 1
                return text[:idx] + text[end:]
    return text
