"""系统托盘：显示/打开面板、退出。

托盘是否启用由设置项 tray_enabled 决定（默认开启）。
"""
from __future__ import annotations

import os
import threading
import webbrowser
from logging import Logger


def start_tray(admin_url: str, logger: Logger) -> None:
    if os.name != "nt":
        return
    try:
        from webai2api.paths import resource_path, user_data_dir
        from webai2api.store import Store
    except Exception:
        return
    try:
        store = Store(user_data_dir() / "webai2api.db")
        if str(store.get_setting("tray_enabled", "true")).strip().lower() in (
                "0", "false", "no", "off"):
            return
    except Exception:
        pass

    import pystray
    from PIL import Image

    try:
        from webai2api.edition import is_lite
        icon_name = "webai2api-lite.ico" if is_lite() else "webai2api.ico"
    except Exception:
        icon_name = "webai2api.ico"
    icon_path = resource_path("assets", icon_name)
    image = Image.open(icon_path) if icon_path.exists() else Image.new("RGBA", (64, 64), (10, 124, 255, 255))

    def _open(icon=None, item=None):
        try:
            webbrowser.open(admin_url)
        except Exception:
            pass

    def _quit(icon=None, item=None):
        try:
            from webai2api.paths import user_data_dir as _ud
            from webai2api.store import Store as _Store
            _Store(_ud() / "webai2api.db").checkpoint()
        except Exception:
            pass
        os._exit(0)

    menu = pystray.Menu(
        pystray.MenuItem("打开面板", _open, default=True),
        pystray.MenuItem("退出", _quit),
    )
    try:
        from webai2api.edition import is_lite
        tray_title = "webai2api 客户版" if is_lite() else "webai2api"
    except Exception:
        tray_title = "webai2api"
    icon = pystray.Icon("webai2api", image, tray_title, menu)
    threading.Thread(target=icon.run, name="tray", daemon=True).start()
    logger.info("系统托盘已启动")
