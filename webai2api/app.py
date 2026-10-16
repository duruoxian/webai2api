"""webai2api 主应用。

把多家 AI 网页版反向代理成统一的 OpenAI 兼容 API，带中文管理面板。

启动：
    python -m webai2api
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .admin_api import create_admin_router
from .gateway import create_gateway_router
from .login import LoginManager
from .paths import default_wasm_path, resource_path, user_data_dir
from .providers import ProviderRegistry
from .store import Store

WEB_DIR = resource_path("webai2api", "web")
if not WEB_DIR.exists():
    WEB_DIR = resource_path("web")


def build_app(store: Store, registry: ProviderRegistry, logger: logging.Logger) -> FastAPI:
    app = FastAPI(title="webai2api", version=__version__)
    login_mgr = LoginManager(store, registry, logger)

    app.include_router(create_gateway_router(store, registry, logger))
    app.include_router(create_admin_router(store, registry, login_mgr, logger))

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
        return FileResponse(idx)

    if WEB_DIR.exists():
        app.mount("/admin/static", StaticFiles(directory=str(WEB_DIR)), name="static")

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
    logger.info("webai2api %s 初始化完成，数据库: %s", __version__, store.db_path)
    return build_app(store, registry, logger)
