"""本机系统集成接口：托盘/开机自启/备份/日志/错误。

这些操作只对本机有意义，因此沿用管理端 token 鉴权。
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from logging import Logger

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool

from .store import Store

router = APIRouter(prefix="/admin/api")


def _hidden_kwargs() -> dict:
    if os.name != "nt":
        return {}
    return {"creationflags": 0x08000000}


def _startup_dir() -> str:
    return os.path.join(
        os.environ.get("APPDATA", ""),
        r"Microsoft\Windows\Start Menu\Programs\Startup",
    )


def _startup_lnk() -> str:
    return os.path.join(_startup_dir(), "webai2api.lnk")


def _exe_path() -> str:
    if getattr(sys, "frozen", False):
        return sys.executable
    return os.path.abspath(sys.argv[0])


def autostart_enabled() -> bool:
    return os.name == "nt" and os.path.exists(_startup_lnk())


def set_autostart(enabled: bool) -> bool:
    """写入/删除启动目录快捷方式。"""
    if os.name != "nt":
        return False
    lnk = _startup_lnk()
    if not enabled:
        try:
            os.remove(lnk)
        except FileNotFoundError:
            pass
        return False
    os.makedirs(_startup_dir(), exist_ok=True)
    exe = _exe_path()
    workdir = os.path.dirname(exe)
    ps = (
        "$ws=New-Object -ComObject WScript.Shell;"
        f"$s=$ws.CreateShortcut('{lnk}');"
        f"$s.TargetPath='{exe}';"
        f"$s.WorkingDirectory='{workdir}';"
        f"$s.IconLocation='{exe},0';"
        "$s.Save()"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
        capture_output=True, timeout=20, **_hidden_kwargs(),
    )
    return os.path.exists(lnk)


def create_system_router(store: Store, logger: Logger) -> APIRouter:
    # 复用管理端 token：延迟导入避免循环依赖
    from .admin_api import _TOKENS

    def _guard(request: Request):
        from fastapi.responses import JSONResponse

        if not _TOKENS.valid(request.headers.get("x-admin-token", "")):
            return JSONResponse(status_code=401, content={"ok": False, "error": "未登录"})
        return None

    @router.get("/system")
    async def system_state(request: Request):
        g = _guard(request)
        if g:
            return g
        bdir = store.db_path.parent / "backups"
        backups = []
        if bdir.exists():
            for f in sorted(bdir.glob("webai2api-*.db"), reverse=True)[:10]:
                backups.append({"name": f.name, "size": f.stat().st_size})
        return {
            "ok": True,
            "autostart": autostart_enabled(),
            "backups": backups,
            "data_dir": str(store.db_path.parent),
            "usage_db": store.usage_db_size(),
        }

    @router.post("/system/autostart")
    async def set_auto(request: Request):
        g = _guard(request)
        if g:
            return g
        body = await request.json()
        enabled = bool(body.get("enabled"))
        ok = await run_in_threadpool(set_autostart, enabled)
        return {"ok": True, "autostart": autostart_enabled() if ok or not enabled else False}

    @router.post("/system/backup")
    async def make_backup(request: Request):
        g = _guard(request)
        if g:
            return g
        path = await run_in_threadpool(store.auto_backup)
        return {"ok": True, "file": path.name if path else ""}

    @router.post("/system/usage/clean")
    async def clean_usage(request: Request):
        """立即按当前保留天数清理用量，并回收空间。"""
        g = _guard(request)
        if g:
            return g
        try:
            body = await request.json()
        except Exception:
            body = {}
        retain = int(body.get("days", 0) or 0)
        if retain <= 0:
            return {"ok": False, "error": "请先把保留天数设为大于 0 的值"}
        removed = await run_in_threadpool(store.prune_usage, retain)
        await run_in_threadpool(store.checkpoint)
        if removed:
            await run_in_threadpool(store.vacuum)
        return {"ok": True, "removed": removed, "usage_db": await run_in_threadpool(store.usage_db_size)}

    @router.get("/system/logs")
    async def logs(request: Request, lines: int = 200):
        g = _guard(request)
        if g:
            return g
        base = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, "webai2api.log")
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                tail = f.readlines()[-max(1, min(int(lines or 200), 2000)):]
        except FileNotFoundError:
            tail = []
        return {"ok": True, "lines": [x.rstrip("\n") for x in tail]}

    @router.get("/system/errors")
    async def errors(request: Request):
        g = _guard(request)
        if g:
            return g
        return {"ok": True, "errors": list(reversed(getattr(
            request.app.state, "recent_errors", [])[-20:]))}

    @router.post("/system/shutdown")
    async def shutdown(request: Request):
        g = _guard(request)
        if g:
            return g
        logger.info("收到面板退出请求，正在退出")
        import threading

        def _exit():
            time.sleep(0.3)
            os._exit(0)

        threading.Thread(target=_exit, daemon=True).start()
        return {"ok": True}

    return router
