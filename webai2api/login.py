"""内置浏览器登录（管理端）。

登录动作由一个**独立子进程**完成：子进程用内嵌 WebView2（pywebview）
打开 DeepSeek，完全在 app 内运行、每账号独立 profile，
**不调用系统 Edge / Chrome**，因此与用户日常浏览器零冲突。

主进程只负责：启动子进程 + 读取子进程写入的状态文件（data/login_state/<id>.json）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from logging import Logger
from pathlib import Path

from .paths import resource_path, user_data_dir
from .store import Store


def _state_dir() -> Path:
    d = user_data_dir() / "login_state"
    d.mkdir(parents=True, exist_ok=True)
    return d


def state_path(acc_id: str) -> Path:
    return _state_dir() / f"{acc_id}.json"


def read_state(acc_id: str) -> dict:
    try:
        return json.loads(state_path(acc_id).read_text(encoding="utf-8"))
    except Exception:
        return {"status": "idle", "message": ""}


def write_state(acc_id: str, status: str, message: str = "") -> None:
    p = state_path(acc_id)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"status": status, "message": message}, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(p)


class LoginManager:
    def __init__(self, store: Store, registry, logger: Logger) -> None:
        self.store = store
        self.registry = registry
        self.logger = logger
        self._lock = threading.Lock()
        self._procs: dict[str, subprocess.Popen] = {}

    def state(self, acc_id: str) -> dict:
        st = read_state(acc_id)
        if st.get("status") == "success":
            try:
                self.registry.refresh()
            except Exception:
                pass
        return st

    def _helper_cmd(self, acc_id: str) -> list[str]:
        if getattr(sys, "frozen", False):
            return [sys.executable, "--login", acc_id]
        return [sys.executable, str(resource_path("run.py")), "--login", acc_id]

    def _kill_leftover_browsers(self, acc_id: str) -> None:
        """杀掉仍占用该账号 profile 的 WebView2 进程。

        WebView2 的浏览器进程会常驻；如果上一个登录窗口没完全退出，
        新窗口会挂到同一个进程上，导致直接复用了旧的 cookie / 登录态。
        """
        if sys.platform != "win32":
            return
        marker = f"browser_profiles{os.sep}{acc_id}"
        marker_alt = marker.replace("\\", "/")
        try:
            out = subprocess.run(
                ["wmic", "process", "where", "name='msedgewebview2.exe'",
                 "get", "ProcessId,CommandLine", "/format:csv"],
                capture_output=True, text=True, timeout=15,
                encoding="utf-8", errors="ignore",
            ).stdout
        except Exception:
            return
        for line in out.splitlines():
            if marker not in line and marker_alt not in line:
                continue
            parts = [p.strip() for p in line.split(",")]
            pid = parts[-1] if parts else ""
            if pid.isdigit():
                try:
                    subprocess.run(["taskkill", "/F", "/PID", pid],
                                   capture_output=True, timeout=10)
                    self.logger.info("已清理残留浏览器进程 pid=%s", pid)
                except Exception:
                    pass

    def start(self, acc_id: str) -> dict:
        acc = self.store.get_account(acc_id)
        if not acc:
            return {"ok": False, "error": "账号不存在"}

        cur = read_state(acc_id)
        with self._lock:
            proc = self._procs.get(acc_id)
            alive = proc is not None and proc.poll() is None
        if alive and cur.get("status") in ("opening", "waiting"):
            return {"ok": False, "error": "该账号正在登录中"}

        # 清掉该账号残留的 WebView2 进程，否则新开的窗口会复用旧会话
        self._kill_leftover_browsers(acc_id)

        write_state(acc_id, "opening", "正在打开内置浏览器…")
        try:
            proc = subprocess.Popen(
                self._helper_cmd(acc_id),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:  # noqa: BLE001
            write_state(acc_id, "error", f"启动内置浏览器失败: {exc}")
            return {"ok": False, "error": f"启动内置浏览器失败: {exc}"}

        with self._lock:
            self._procs[acc_id] = proc
        self.logger.info("账号 %s 已启动内置浏览器登录（pid=%s）", acc.get("name"), proc.pid)
        return {"ok": True}

