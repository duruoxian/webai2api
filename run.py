"""exe 入口：启动 webai2api 服务，并自动打开管理面板。"""
from __future__ import annotations

import logging
import os
import socket
import sys
import threading
import time
import webbrowser

HOST = "127.0.0.1"
DEFAULT_PORT = int(os.environ.get("WEBAI2API_PORT", "8800"))


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

    logger = logging.getLogger("webai2api")
    port = _find_free_port(DEFAULT_PORT)
    logger.info("启动 webai2api (frozen=%s) 端口=%s", getattr(sys, "frozen", False), port)

    from webai2api.app import create_default_app
    app = create_default_app()

    base_url = f"http://{HOST}:{port}/v1"

    # 把实际端口写到 exe 同级文件，方便用户/脚本查看
    try:
        with open(os.path.join(log_dir, "current_port.txt"), "w", encoding="utf-8") as f:
            f.write(f"{port}\n{base_url}\n")
    except Exception:
        pass

    def _open_browser() -> None:
        time.sleep(2.5)
        try:
            webbrowser.open(f"http://{HOST}:{port}/admin")
        except Exception:
            pass

    threading.Thread(target=_open_browser, daemon=True).start()

    import asyncio

    import uvicorn

    try:
        config = uvicorn.Config(app, host=HOST, port=port, log_level="info", loop="asyncio")
        server = uvicorn.Server(config)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        logger.info("服务地址 %s", base_url)
        loop.run_until_complete(server.serve())
    except Exception as exc:  # noqa: BLE001
        logger.error("服务启动异常: %s", exc, exc_info=True)
        raise


if __name__ == "__main__":
    main()
