"""内置浏览器登录（轻量版）。

不依赖 Playwright：直接调用系统自带的 Microsoft Edge（Windows）或 Chrome，
以独立 profile 启动并开启远程调试端口，通过 CDP 协议在页面里执行 JS 抓取凭证。

每个账号一个独立 user-data-dir，互不干扰，也不影响系统浏览器的登录态。
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import urllib.request
from logging import Logger
from pathlib import Path

from .adapters.deepseek.pool import ClientPool
from .store import Store

_EXTRACT_JS = r"""
(function(){
  const raw = {
    t: localStorage.getItem('userToken'),
    d: localStorage.getItem('deepseek-device-id:chat'),
    s: localStorage.getItem('settingsJwt'),
    c: document.cookie,
  };
  const unwrap = (v) => { if(!v) return ''; try{ const o=JSON.parse(v);
    if(o && typeof o==='object' && 'value' in o) return o.value; }catch(e){} return v; };
  const token = unwrap(raw.t);
  let settings = '';
  if(raw.s){ try{ const o=JSON.parse(raw.s); const val=(o&&o.value)?o.value:o;
    settings = val&&val.jwt ? val.jwt : (typeof val==='string'?val:''); }catch(e){ settings=raw.s; } }
  return JSON.stringify({
    auth_token: typeof token==='string'?token:'',
    device_id: raw.d||'', settings_token: settings||'', cookie: raw.c||''
  });
})()
"""


def _find_browser() -> str | None:
    """找系统上的 Edge / Chrome。"""
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def profiles_dir(db_path: Path) -> Path:
    d = db_path.parent / "profiles"
    d.mkdir(parents=True, exist_ok=True)
    return d


class LoginManager:
    def __init__(self, store: Store, pool: ClientPool, logger: Logger) -> None:
        self.store = store
        self.pool = pool
        self.logger = logger
        self._lock = threading.Lock()
        self._sessions: dict[str, dict] = {}

    def state(self, acc_id: str) -> dict:
        with self._lock:
            return self._sessions.get(acc_id, {"status": "idle", "message": ""})

    def _set(self, acc_id: str, status: str, message: str = "") -> None:
        with self._lock:
            self._sessions[acc_id] = {"status": status, "message": message}

    def start(self, acc_id: str) -> dict:
        acc = self.store.get_account(acc_id)
        if not acc:
            return {"ok": False, "error": "账号不存在"}
        with self._lock:
            cur = self._sessions.get(acc_id)
            if cur and cur["status"] in ("waiting", "opening"):
                return {"ok": False, "error": "该账号正在登录中"}
        threading.Thread(target=self._run_login, args=(acc,), daemon=True).start()
        return {"ok": True}

    def _run_login(self, acc: dict) -> None:
        acc_id = acc["id"]
        self._set(acc_id, "opening", "正在打开浏览器…")

        browser = _find_browser()
        if not browser:
            self._set(acc_id, "error", "未找到 Edge/Chrome 浏览器，请用「手动粘贴凭证」方式")
            return

        profile = profiles_dir(self.store.db_path) / acc_id
        port = 9500 + (abs(hash(acc_id)) % 400)
        proc = None
        try:
            proc = subprocess.Popen(
                [browser, f"--remote-debugging-port={port}",
                 f"--user-data-dir={profile}",
                 "--no-first-run", "--no-default-browser-check",
                 "https://chat.deepseek.com/"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            self._set(acc_id, "waiting", "请在浏览器窗口里完成登录…")

            ws_url = None
            for _ in range(30):
                time.sleep(1)
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=3) as r:
                        targets = json.loads(r.read().decode())
                    for t in targets:
                        if t.get("type") == "page" and "deepseek" in t.get("url", ""):
                            ws_url = t.get("webSocketDebuggerUrl")
                            break
                    if ws_url:
                        break
                except Exception:
                    continue

            if not ws_url:
                self._set(acc_id, "error", "无法连接到浏览器调试端口，请重试")
                return

            import asyncio
            import websockets

            async def _grab():
                async with websockets.connect(ws_url, max_size=None) as ws:
                    deadline = time.time() + 300
                    while time.time() < deadline:
                        await ws.send(json.dumps({
                            "id": 1, "method": "Runtime.evaluate",
                            "params": {"expression": _EXTRACT_JS, "returnByValue": True},
                        }))
                        for _ in range(10):
                            msg = await asyncio.wait_for(ws.recv(), timeout=5)
                            data = json.loads(msg)
                            if data.get("id") == 1:
                                val = (((data.get("result") or {}).get("result") or {})
                                       .get("value"))
                                if val:
                                    try:
                                        creds = json.loads(val)
                                    except Exception:
                                        creds = None
                                    if creds and creds.get("auth_token"):
                                        return creds
                                break
                        await asyncio.sleep(2)
                return None

            creds = asyncio.run(_grab())
            if not creds:
                self._set(acc_id, "error", "等待登录超时或未检测到登录，请重试")
                return

            self.store.update_account(
                acc_id,
                auth_token=creds["auth_token"],
                device_id=creds.get("device_id", ""),
                settings_token=creds.get("settings_token", ""),
                cookie=creds.get("cookie", ""),
            )
            self.pool.refresh()
            self._set(acc_id, "success", "登录成功，凭证已保存")
            self.logger.info("账号 %s 登录成功并已保存凭证", acc.get("name"))
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("账号 %s 登录失败: %s", acc.get("name"), exc)
            self._set(acc_id, "error", f"登录失败: {exc}")
        finally:
            if proc:
                self._kill_browser_tree(proc)

    @staticmethod
    def _kill_browser_tree(proc) -> None:
        """杀掉整个浏览器进程树（Edge/Chrome 会派生很多子进程）。"""
        try:
            import subprocess as _sp
            _sp.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    stdout=_sp.DEVNULL, stderr=_sp.DEVNULL, timeout=10)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
