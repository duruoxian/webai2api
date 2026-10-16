"""exe 入口：启动 webai2api 服务，并用原生窗口打开管理面板。

- 服务（uvicorn）跑在后台线程；
- 主线程用 pywebview 打开原生窗口（Windows 走 WebView2）；
- 若原生窗口不可用，回退到系统默认浏览器。
"""
from __future__ import annotations

import logging
import os
import socket
import sys
import threading
import time

HOST = "127.0.0.1"
DEFAULT_PORT = int(os.environ.get("WEBAI2API_PORT", "8800"))
WINDOW_TITLE = "webai2api 控制台"


def _find_free_port(preferred: int) -> int:
    """优先用 preferred，被占用则往上找一个空闲端口。"""
    for port in [preferred] + list(range(preferred + 1, preferred + 20)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((HOST, port))
                return port
            except OSError:
                continue
    # 极端情况：让系统分配
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def _wait_port(port: int, timeout: float = 20.0) -> bool:
    """轮询等待服务端口可连接。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex((HOST, port)) == 0:
                return True
        time.sleep(0.2)
    return False


def main() -> None:
    log_dir = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(__file__)
    log_path = os.path.join(log_dir, "webai2api.log")

    # windowed exe 下 sys.stdout/stderr 为 None，会让 uvicorn / logging 报错。
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = log_file
    if sys.stderr is None:
        sys.stderr = log_file

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = [logging.StreamHandler(log_file)]

    # 内置浏览器登录辅助模式：<exe> --login <account_id>
    if len(sys.argv) >= 3 and sys.argv[1] == "--login":
        from webai2api.login_webview import run_webview_login

        run_webview_login(sys.argv[2])
        return

    logger = logging.getLogger("webai2api")
    port = _find_free_port(DEFAULT_PORT)
    logger.info("启动 webai2api (frozen=%s) 端口=%s", getattr(sys, "frozen", False), port)

    from webai2api.app import create_default_app
    app = create_default_app()

    base_url = f"http://{HOST}:{port}/v1"
    admin_url = f"http://{HOST}:{port}/admin"

    # 把实际端口写到 exe 同级文件，方便用户/脚本查看
    try:
        with open(os.path.join(log_dir, "current_port.txt"), "w", encoding="utf-8") as f:
            f.write(f"{port}\n{base_url}\n")
    except Exception:
        pass

    import asyncio

    import uvicorn

    config = uvicorn.Config(app, host=HOST, port=port, log_level="info", loop="asyncio")
    server = uvicorn.Server(config)

    def _run_server() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(server.serve())
        except Exception as exc:  # noqa: BLE001
            logger.error("服务启动异常: %s", exc, exc_info=True)

    threading.Thread(target=_run_server, name="uvicorn", daemon=True).start()
    logger.info("服务地址 %s", base_url)

    if not _wait_port(port):
        logger.error("服务在超时时间内未就绪")

    # 主线程打开原生窗口
    try:
        import webview

        from webai2api.paths import resource_path

        webview.create_window(WINDOW_TITLE, admin_url, width=1180, height=800, min_size=(900, 600))
        icon = resource_path("assets", "webai2api.ico")
        webview.start(icon=str(icon) if icon.exists() else None)
        logger.info("窗口已关闭，退出")
        os._exit(0)
    except Exception as exc:  # noqa: BLE001
        logger.error("原生窗口启动失败，回退浏览器: %s", exc, exc_info=True)

    # 回退：系统默认浏览器 + 主线程常驻
    import webbrowser

    try:
        webbrowser.open(admin_url)
    except Exception:
        pass
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
