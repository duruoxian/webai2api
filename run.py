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
    try:
        from webai2api.edition import is_lite
        return 8802 if is_lite() else 8800
    except Exception:
        return 8800


def _is_lite_edition() -> bool:
    try:
        from webai2api.edition import is_lite
        return is_lite()
    except Exception:
        return False


DEFAULT_PORT = _default_port()
WINDOW_TITLE = "webai2api 客户版" if _is_lite_edition() else "webai2api 控制台"
APP_ICON_NAME = "webai2api-lite.ico" if _is_lite_edition() else "webai2api.ico"


def _graceful_exit() -> None:
    """退出前把 WAL 落盘，再结束进程。"""
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


_RESIZE_PROC = None


def _install_frameless_resize(win) -> None:
    """给无边框窗口补回「拖边缘缩放」（Windows 原生 hit-test）。"""
    global _RESIZE_PROC
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.SetWindowLongPtrW.restype = ctypes.c_void_p
    user32.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_void_p]
    user32.GetWindowLongPtrW.restype = ctypes.c_void_p
    user32.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.CallWindowProcW.restype = ctypes.c_ssize_t
    user32.CallWindowProcW.argtypes = [ctypes.c_void_p, wintypes.HWND, ctypes.c_uint,
                                       ctypes.c_size_t, ctypes.c_ssize_t]
    CALLBACK = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, wintypes.HWND, ctypes.c_uint,
                                  ctypes.c_size_t, ctypes.c_ssize_t)

    try:
        hwnd = int(win.native.Handle.ToInt64())
    except Exception:
        return

    GWL_STYLE = -16
    WS_THICKFRAME = 0x00040000
    try:
        style = int(user32.GetWindowLongPtrW(hwnd, GWL_STYLE) or 0)
        user32.SetWindowLongPtrW(hwnd, GWL_STYLE, style | WS_THICKFRAME)
    except Exception:
        pass

    border = 6
    old_proc = [0]

    def proc(h, msg, wp, lp):
        if msg == 0x0084:  # WM_NCHITTEST
            try:
                rect = wintypes.RECT()
                user32.GetWindowRect(h, ctypes.byref(rect))
                x = ctypes.c_short(lp & 0xFFFF).value
                y = ctypes.c_short((lp >> 16) & 0xFFFF).value
                left, right = x - rect.left, rect.right - x
                top, bottom = y - rect.top, rect.bottom - y
                if left < border and top < border: return 13
                if right < border and top < border: return 14
                if left < border and bottom < border: return 16
                if right < border and bottom < border: return 17
                if left < border: return 10
                if right < border: return 11
                if top < border: return 12
                if bottom < border: return 15
            except Exception:
                pass
        return user32.CallWindowProcW(old_proc[0], h, msg, wp, lp)

    new_proc = CALLBACK(proc)
    old_proc[0] = user32.SetWindowLongPtrW(hwnd, -4, new_proc)
    _RESIZE_PROC = (new_proc, old_proc)


class WindowApi:
    """暴露给页面的无边框窗口控制。"""

    def __init__(self) -> None:
        self._win = None
        self._maximized = False

    def bind(self, win) -> None:
        self._win = win

    def minimize(self) -> None:
        if self._win:
            self._win.minimize()

    def toggle_max(self) -> bool:
        if not self._win:
            return False
        if self._maximized:
            self._win.restore()
            self._maximized = False
        else:
            self._win.maximize()
            self._maximized = True
        return self._maximized

    def close(self) -> None:
        if self._win:
            self._win.destroy()


def _open_log(log_dir: str):
    """尝试在 log_dir 写日志；无权限时回退到用户数据目录。"""
    candidates = [log_dir]
    try:
        from webai2api.paths import user_data_dir as _ud
        candidates.append(str(_ud()))
    except Exception:
        pass
    candidates.append(os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "webai2api"))
    last_exc = None
    for d in candidates:
        try:
            os.makedirs(d, exist_ok=True)
            return open(os.path.join(d, "webai2api.log"), "a", encoding="utf-8", buffering=1)
        except OSError as exc:
            last_exc = exc
    raise last_exc  # type: ignore[misc]


def main() -> None:
    log_dir = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(__file__)
    # windowed exe 下 sys.stdout/stderr 为 None，会让 uvicorn / logging 报错。
    log_file = _open_log(log_dir)
    log_path = log_file.name
    log_dir = os.path.dirname(log_path)
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

    # 单实例锁：与数据库同目录（程序目录不可写时 paths 会回退到用户数据目录）
    try:
        from webai2api.paths import user_data_dir as _udd
        data_dir = str(_udd())
    except Exception:
        data_dir = os.path.join(log_dir, "data")
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

        from webai2api.paths import resource_path

        win = webview.create_window(
            WINDOW_TITLE, admin_url,
            width=1180, height=800, min_size=(900, 600),
            background_color="#eceef1",
        )
        icon = resource_path("assets", APP_ICON_NAME)
        webview.start(icon=str(icon) if icon.exists() else None)
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

