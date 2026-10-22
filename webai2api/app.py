"""webai2api 主应用。

把多家 AI 网页版反向代理成统一的 OpenAI 兼容 API，带中文管理面板。

启动：
    python -m webai2api
"""
from __future__ import annotations

import logging
import threading
import time

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.staticfiles import StaticFiles

from . import __version__
from .admin_api import create_admin_router
from .gateway import create_gateway_router
from .health import HealthMonitor
from .login import LoginManager
from .paths import default_wasm_path, resource_path, user_data_dir
from .providers import ProviderRegistry
from .store import Store

WEB_DIR = resource_path("webai2api", "web")
if not WEB_DIR.exists():
    WEB_DIR = resource_path("web")

USAGE_CLEAN_INTERVAL = 6 * 3600


def start_usage_cleaner(store: Store, logger: logging.Logger) -> threading.Thread:
    """后台线程：每 6 小时按 usage_retain_days 清理一次历史用量。

    保留天数改成 0 时线程继续活着但不删数据，所以改设置无需重启。
    """
    def _run() -> None:
        while True:
            time.sleep(USAGE_CLEAN_INTERVAL)
            try:
                retain = int(store.get_setting("usage_retain_days", "0") or 0)
                if retain > 0:
                    removed = store.prune_usage(retain)
                    if removed:
                        logger.info("用量清理：删除 %s 条超过 %s 天的记录", removed, retain)
            except Exception as exc:  # noqa: BLE001
                logger.warning("用量清理失败: %s", exc)

    th = threading.Thread(target=_run, name="usage-cleaner", daemon=True)
    th.start()
    return th


SESSION_CLEAN_INTERVAL = 20 * 60


def start_session_cleaner(registry: ProviderRegistry, logger: logging.Logger,
                          interval: float = SESSION_CLEAN_INTERVAL) -> threading.Thread:
    """后台线程：定期让各账号客户端清理上游已废弃的会话。

    只调用实现了 cleanup_stale_sessions 的客户端（目前是 DeepSeek），
    其它 provider 直接跳过；单个账号失败只记 warning，不影响其它账号。
    """
    def _run() -> None:
        while True:
            time.sleep(interval)
            try:
                for ac in registry.all_accounts():
                    client = getattr(ac, "client", None)
                    cleanup = getattr(client, "cleanup_stale_sessions", None)
                    if not callable(cleanup):
                        continue
                    try:
                        cleanup()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("会话清理失败（账号 %s）: %s",
                                       ac.account.get("name") or ac.account.get("id"), exc)
            except Exception as exc:  # noqa: BLE001
                logger.warning("会话清理线程异常: %s", exc)

    th = threading.Thread(target=_run, name="session-cleaner", daemon=True)
    th.start()
    return th


def build_app(store: Store, registry: ProviderRegistry, logger: logging.Logger) -> FastAPI:
    app = FastAPI(title="webai2api", version=__version__)
    login_mgr = LoginManager(store, registry, logger)
    recent_errors: list[dict] = []

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        logger.error("未处理异常 %s %s: %s", request.method, request.url.path, exc, exc_info=True)
        recent_errors.append({
            "ts": time.time(),
            "method": request.method,
            "path": request.url.path,
            "error": f"{type(exc).__name__}: {exc}",
        })
        del recent_errors[:-20]
        return JSONResponse(
            status_code=500,
            content={"error": {
                "message": f"服务内部错误：{type(exc).__name__}: {exc}",
                "type": "internal_error",
                "code": "internal_error",
            }},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        if request.url.path.startswith("/admin/api"):
            return JSONResponse(
                status_code=exc.status_code,
                content={"ok": False, "error": str(exc.detail)},
            )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"message": str(exc.detail), "type": "http_error"}},
        )

    app.state.recent_errors = recent_errors

    @app.middleware("http")
    async def _guard_admin(request: Request, call_next):
        # 面板只允许本机访问：经由隧道 / 反代进来的请求一律拒绝，
        # 避免把管理界面暴露到公网（API Key 网关本身仍可访问）。
        path = request.url.path
        if path == "/admin" or path.startswith("/admin/"):
            via_proxy = bool(
                request.headers.get("cf-connecting-ip")
                or request.headers.get("cf-ray")
                or request.headers.get("x-forwarded-for")
            )
            client = request.client.host if request.client else ""
            if via_proxy or client not in ("127.0.0.1", "::1", "localhost"):
                return JSONResponse(
                    status_code=403,
                    content={"error": {"message": "管理面板仅允许本机访问",
                                       "type": "permission_error",
                                       "code": "permission_error"}},
                )
        return await call_next(request)

    app.include_router(create_gateway_router(store, registry, logger))
    app.include_router(create_admin_router(store, registry, login_mgr, logger))
    from .system_api import create_system_router
    app.include_router(create_system_router(store, logger))

    @app.get("/health")
    async def health():
        acs = registry.available()
        return {
            "status": "ok",
            "version": __version__,
            "accounts_available": len(acs),
        }

    # 面板静态页
    @app.get("/admin")
    async def admin_index():
        idx = WEB_DIR / "index.html"
        if not idx.exists():
            return JSONResponse(status_code=404, content={"error": "面板文件缺失"})
        return FileResponse(
            idx,
            headers={"Cache-Control": "no-store, must-revalidate"},
        )

    if WEB_DIR.exists():
        class _NoCacheStatic(StaticFiles):
            async def get_response(self, path, scope):
                resp = await super().get_response(path, scope)
                resp.headers["Cache-Control"] = "no-cache, must-revalidate"
                return resp
        app.mount("/admin/static", _NoCacheStatic(directory=str(WEB_DIR)), name="static")

    return app


def create_default_app() -> FastAPI:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    logger = logging.getLogger("webai2api")
    data_dir = user_data_dir()
    store = Store(data_dir / "webai2api.db")
    # 首次运行时把默认 wasm 路径写入设置（打包后指向 exe 同级或包内）
    if not store.get_setting("deepseek_wasm_path", ""):
        store.set_setting("deepseek_wasm_path", default_wasm_path())
    registry = ProviderRegistry(store, logging.getLogger("webai2api.providers"))

    # 启动时按保留期清理历史用量（0/空 = 永久保留）
    try:
        retain = int(store.get_setting("usage_retain_days", "0") or 0)
        if retain > 0:
            removed = store.prune_usage(retain)
            if removed:
                logger.info("用量清理：删除 %s 条超过 %s 天的记录", removed, retain)
    except Exception as exc:  # noqa: BLE001
        logger.warning("用量清理失败: %s", exc)

    # 常驻清理：长期运行也不会让用量表无限增长
    start_usage_cleaner(store, logger)
    # 常驻清理：定期回收上游 DeepSeek 的过期会话（避免会话堆积）
    try:
        start_session_cleaner(registry, logger)
    except Exception as exc:  # noqa: BLE001
        logger.warning("会话清理线程启动失败: %s", exc)

    logger.info("webai2api %s 初始化完成，数据库: %s", __version__, store.db_path)

    app = build_app(store, registry, logger)
    try:
        monitor = HealthMonitor(store, logger)
        monitor.start()
        app.state.health_monitor = monitor
    except Exception as exc:  # noqa: BLE001
        logger.warning("健康检查启动失败: %s", exc)
    return app
