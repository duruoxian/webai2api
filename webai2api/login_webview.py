"""内置浏览器登录（WebView2 版）。

用一个**内嵌的 WebView2 窗口**（pywebview）打开 DeepSeek 登录页，
完全在 app 内部运行，独立 storage_path（每账号一套），
**不调用系统 Edge / Chrome**，因此与用户日常浏览器零冲突。

运行方式：主程序以子进程调用
    <exe> --login <account_id>
登录成功后抓取凭证写入 SQLite，并把状态写到 data/login_state/<id>.json。
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from .paths import user_data_dir
from .store import Store

EXTRACT_JS = r"""
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


def _login_state_dir() -> Path:
    d = user_data_dir() / "login_state"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_state(acc_id: str, status: str, message: str = "") -> None:
    p = _login_state_dir() / f"{acc_id}.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"status": status, "message": message}, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(p)


def run_webview_login(acc_id: str) -> None:
    """在子进程里跑：打开内嵌浏览器，抓凭证，写状态。"""
    import webview

    data_dir = user_data_dir()
    store = Store(data_dir / "webai2api.db")
    acc = store.get_account(acc_id)
    if not acc:
        write_state(acc_id, "error", "账号不存在")
        return

    write_state(acc_id, "opening", "正在打开内置浏览器…")

    # 每个账号一套独立 storage，互不干扰，也不碰系统浏览器
    storage = data_dir / "browser_profiles" / acc_id
    storage.mkdir(parents=True, exist_ok=True)

    window = webview.create_window(
        f"登录 DeepSeek · {acc.get('name', '')}",
        "https://chat.deepseek.com/",
        width=1000,
        height=760,
        text_select=True,
    )

    done = {"v": False}

    def worker() -> None:
        deadline = time.time() + 600
        announced = False
        while time.time() < deadline and not done["v"]:
            time.sleep(2)
            try:
                val = window.evaluate_js(EXTRACT_JS)
            except Exception:
                continue
            if not announced:
                announced = True
                write_state(acc_id, "waiting", "请在内置浏览器里完成 DeepSeek 登录…")
            if not val:
                continue
            try:
                creds = json.loads(val)
            except Exception:
                continue
            if creds.get("auth_token"):
                store.update_account(
                    acc_id,
                    auth_token=creds["auth_token"],
                    device_id=creds.get("device_id", ""),
                    settings_token=creds.get("settings_token", ""),
                    cookie=creds.get("cookie", ""),
                )
                done["v"] = True
                write_state(acc_id, "success", "登录成功，凭证已保存")
                try:
                    window.destroy()
                except Exception:
                    pass
                return
        if not done["v"]:
            write_state(acc_id, "error", "等待登录超时，请重试")
            try:
                window.destroy()
            except Exception:
                pass

    threading.Thread(target=worker, daemon=True).start()
    try:
        webview.start(private_mode=False, storage_path=str(storage))
    except Exception as exc:  # noqa: BLE001
        write_state(acc_id, "error", f"内置浏览器启动失败: {exc}")
