"""Cloudflare 快速隧道：把本机服务临时暴露到公网，方便异地朋友使用。

只用 `cloudflared tunnel --url ...` 的临时隧道（trycloudflare.com），
不需要 Cloudflare 账号，也不需要域名；每次启动地址都会变。
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from logging import Logger

URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")

_hidden = {"creationflags": 0x08000000} if os.name == "nt" else {}


def cloudflared_path() -> str:
    """找 cloudflared：优先 exe 同级，其次 _internal / 源码根目录，最后 PATH。"""
    names = ["cloudflared.exe"] if os.name == "nt" else ["cloudflared"]
    roots: list[str] = []
    if getattr(sys, "frozen", False):
        roots.append(os.path.dirname(sys.executable))
        roots.append(getattr(sys, "_MEIPASS", ""))
    else:
        roots.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        roots.append(os.path.dirname(os.path.abspath(__file__)))
    for root in roots:
        if not root:
            continue
        for name in names:
            p = os.path.join(root, name)
            if os.path.isfile(p):
                return p
    from shutil import which
    return which("cloudflared") or ""


class Tunnel:
    """单条快速隧道的生命周期管理。"""

    def __init__(self, port: int, logger: Logger) -> None:
        self.port = port
        self.logger = logger
        self._proc: subprocess.Popen | None = None
        self._url = ""
        self._error = ""
        self._started_at = 0.0
        self._lock = threading.Lock()

    # ── 状态 ──
    def state(self) -> dict:
        with self._lock:
            running = bool(self._proc and self._proc.poll() is None)
            if not running and self._proc is not None:
                # 进程已经退出，回收一次
                try:
                    self._proc.poll()
                except Exception:
                    pass
            return {
                "running": running,
                "url": self._url if running else "",
                "error": self._error,
                "since": self._started_at if running else 0,
                "binary": bool(cloudflared_path()),
            }

    # ── 启动 ──
    def start(self, timeout: float = 45.0) -> dict:
        with self._lock:
            if self._proc and self._proc.poll() is None and self._url:
                return {"ok": True, "url": self._url, "already": True}
            exe = cloudflared_path()
            if not exe:
                self._error = "未找到 cloudflared.exe，请放到程序目录后重试"
                return {"ok": False, "error": self._error}
            self._url = ""
            self._error = ""
            try:
                self._proc = subprocess.Popen(
                    [exe, "tunnel", "--no-autoupdate",
                     "--url", f"http://127.0.0.1:{self.port}"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    **_hidden,
                )
            except Exception as exc:  # noqa: BLE001
                self._error = f"启动失败：{exc}"
                return {"ok": False, "error": self._error}
            proc = self._proc

        deadline = time.time() + timeout
        buf: list[str] = []
        while time.time() < deadline:
            if proc.poll() is not None:
                tail = "".join(buf)[-600:]
                with self._lock:
                    self._error = f"cloudflared 已退出：{tail.strip() or '无输出'}"
                return {"ok": False, "error": self._error}
            line = proc.stdout.readline() if proc.stdout else ""
            if not line:
                time.sleep(0.15)
                continue
            buf.append(line)
            m = URL_RE.search(line)
            if m:
                with self._lock:
                    self._url = m.group(0)
                    self._started_at = time.time()
                self.logger.info("隧道已建立: %s", self._url)
                threading.Thread(target=self._drain, args=(proc,), daemon=True).start()
                return {"ok": True, "url": self._url}
        with self._lock:
            self._error = "等待隧道地址超时（网络或代理可能挡住了 Cloudflare）"
        self.stop()
        return {"ok": False, "error": self._error}

    def _drain(self, proc: subprocess.Popen) -> None:
        """持续读走输出，避免管道写满导致 cloudflared 卡住。"""
        try:
            for _ in proc.stdout:  # type: ignore[union-attr]
                pass
        except Exception:
            pass

    def stop(self) -> dict:
        with self._lock:
            proc = self._proc
            self._proc = None
            self._url = ""
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=6)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        return {"ok": True}


_INSTANCE: Tunnel | None = None
_INSTANCE_LOCK = threading.Lock()


def get_tunnel(port: int, logger: Logger) -> Tunnel:
    global _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is None or _INSTANCE.port != port:
            _INSTANCE = Tunnel(port, logger)
        return _INSTANCE


def stop_all() -> None:
    """进程退出时调用，确保子进程不会残留。"""
    global _INSTANCE
    with _INSTANCE_LOCK:
        inst = _INSTANCE
    if inst is not None:
        inst.stop()
