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


def _default_port() -> int:
    if os.environ.get("WEBAI2API_PORT"):
        return int(os.environ["WEBAI2API_PORT"])
    return 8800


DEFAULT_PORT = _default_port()
WINDOW_TITLE = "webai2api 控制台"
APP_ICON_NAME = "webai2api.ico"


def _graceful_exit() -> None:
    """退出前把 WAL 落盘，停掉千问中继，再结束进程。"""
    try:
        from webai2api.qwen_relay import stop_all_relays
        stop_all_relays()
    except Exception:
        pass
    try:
        from webai2api.paths import user_data_dir
        from webai2api.store import Store
        Store(user_data_dir() / "webai2api.db").checkpoint()
    except Exception:
        pass
    os._exit(0)


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


def _open_log(log_dir: str):
    """尝试在 log_dir 写日志；无权限时回退到用户数据目录。

    exe 被装进 Program Files 等普通权限写不了的目录时，必须回退到
    数据目录，否则启动第一步就 PermissionError。
    """
    candidates = [log_dir]
    try:
        from webai2api.paths import user_data_dir as _ud
        candidates.append(str(_ud()))
    except Exception:
        pass
    candidates.append(os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "webai2api"))
    from logging.handlers import RotatingFileHandler

    last_exc = None
    for d in candidates:
        try:
            os.makedirs(d, exist_ok=True)
            # 轮转：10MB × 3 份备份。此前用裸 open 无上限，一周就 45MB。
            return RotatingFileHandler(
                os.path.join(d, "webai2api.log"), maxBytes=10 * 1024 * 1024,
                backupCount=3, encoding="utf-8")
        except OSError as exc:
            last_exc = exc
    raise last_exc  # type: ignore[misc]


def main() -> None:
    log_dir = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(__file__)
    # windowed exe 下 sys.stdout/stderr 为 None，会让 uvicorn / logging 报错。
    log_handler = _open_log(log_dir)
    log_path = log_handler.baseFilename
    log_dir = os.path.dirname(log_path)
    log_sink = open(log_path, "a", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = log_sink
    if sys.stderr is None:
        sys.stderr = log_sink

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = [log_handler]

    # 内置浏览器登录辅助模式：<exe> --login <account_id>
    if len(sys.argv) >= 3 and sys.argv[1] == "--login":
        from webai2api.login_webview import run_webview_login

        run_webview_login(sys.argv[2])
        return

    # 千问浏览器会话中继：<exe> --relay-qwen <account_id> [--quiet]
    if len(sys.argv) >= 3 and sys.argv[1] == "--relay-qwen":
        from webai2api.qwen_relay import run_qwen_relay

        run_qwen_relay(sys.argv[2], quiet="--quiet" in sys.argv[3:])
        return

    logger = logging.getLogger("webai2api")

    # 单实例锁：同一个 data 目录只允许一个进程写库
    # （exe 装在 Program Files 时 log_dir 可能回退成数据目录，
    #   锁必须永远落在 user_data_dir，不能跟着 log_dir 走）
    from webai2api.paths import user_data_dir as _udd
    data_dir = str(_udd())
    os.makedirs(data_dir, exist_ok=True)
    lock_path = os.path.join(data_dir, "webai2api.lock")
    lock_file = open(lock_path, "a+", encoding="utf-8")
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        logger.warning("检测到已有实例在运行，本进程退出")
        try:
            import ctypes
            # 5 秒自动消失，避免阻塞第二个进程
            ctypes.windll.user32.MessageBoxTimeoutW(
                0, "webai2api 已经在运行了。\n请从任务栏/托盘打开，或先退出旧实例。",
                "webai2api", 0x40, 0, 5000,
            )
        except Exception:
            pass
        return

    port = _find_free_port(DEFAULT_PORT)
    logger.info("启动 webai2api (frozen=%s) 端口=%s", getattr(sys, "frozen", False), port)

    from webai2api.app import create_default_app
    app = create_default_app()

    # 每日自动备份 + 启动时 WAL checkpoint
    try:
        from webai2api.store import Store
        from webai2api.paths import user_data_dir
        _st = Store(user_data_dir() / "webai2api.db")
        _st.checkpoint()
        _st.auto_backup()
    except Exception as exc:  # noqa: BLE001
        logger.warning("启动备份失败: %s", exc)

    # 千问中继预热：启动即拉起已登录账号的隐藏中继，消掉首次请求约 15 秒冷启动。
    # 只预热「启用且已有凭证」的账号；quiet 模式下登录态失效也不会弹窗，
    # 用户在面板里重新登录后下次启动自然恢复预热。
    try:
        import threading as _threading

        from webai2api.paths import user_data_dir
        from webai2api.qwen_relay import RelayManager
        from webai2api.store import Store

        def _prewarm_qwen_relays() -> None:
            try:
                st = Store(user_data_dir() / "webai2api.db")
                for acc in st.list_accounts("qwen"):
                    if not (acc.get("enabled")
                            and (acc.get("auth_token") or "").strip()):
                        continue
                    try:
                        RelayManager(str(acc["id"]), logger).ensure(
                            timeout=90, quiet=True)
                        logger.info("千问中继预热完成: %s", acc.get("name") or acc["id"])
                    except Exception as exc:  # noqa: BLE001
                        logger.info("千问中继预热失败(%s): %s",
                                    acc.get("name") or acc.get("id"), exc)
            except Exception as exc:  # noqa: BLE001
                logger.info("千问中继预热跳过: %s", exc)

        _threading.Thread(target=_prewarm_qwen_relays,
                          name="qwen-prewarm", daemon=True).start()
    except Exception as exc:  # noqa: BLE001
        logger.warning("千问中继预热初始化失败: %s", exc)

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
    # 托盘（可选，设置里开关）
    try:
        from webai2api.tray import start_tray
        start_tray(admin_url, logger)
    except Exception as exc:  # noqa: BLE001
        logger.warning("托盘启动失败: %s", exc)

    try:
        import webview

        from webai2api.paths import resource_path, user_data_dir

        webview.create_window(
            WINDOW_TITLE, admin_url,
            width=1180, height=800, min_size=(900, 600),
            background_color="#eceef1",
        )
        icon = resource_path("assets", APP_ICON_NAME)
        # private_mode=False：pywebview 默认隐私模式会丢弃 localStorage，
        # 面板主题（wapi_theme）/登录态每次重开都被清空。存储指向数据目录
        # 的 panel_profile，与登录窗口（子进程、独立一次性目录）互不影响。
        webview.start(
            private_mode=False,
            storage_path=str(user_data_dir() / "panel_profile"),
            icon=str(icon) if icon.exists() else None,
        )
        logger.info("窗口已关闭，退出")
        _graceful_exit()
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

