"""内置浏览器登录（WebView2 版）。

用一个**内嵌的 WebView2 窗口**（pywebview）打开对应平台的登录页，
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

DEEPSEEK_URL = "https://chat.deepseek.com/"
CHATGLM_URL = "https://chatglm.cn/main/alltoolsdetail?lang=zh"
DEEPSEEK_EXTRACT_JS = r"""
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
    device_id: raw.d||'', settings_token: settings||'', cookie: raw.c||'',
    u: navigator.userAgent||'', p: navigator.platform||''
  });
})()
"""

# 智谱清言：refresh_token 在 cookie 里（chatglm_refresh_token）
# 注意：未登录时站点也会下发一个游客 refresh_token（JWT 里 is_guest=true），
# 必须忽略，否则会瞬间“登录成功”并关闭窗口。
CHATGLM_EXTRACT_JS = r"""
(function(){
  function getCookie(name){
    const m = document.cookie.match(new RegExp('(?:^|; )' + name + '=([^;]*)'));
    return m ? decodeURIComponent(m[1]) : '';
  }
  let refresh = getCookie('chatglm_refresh_token');
  if(!refresh){
    try{
      const v = localStorage.getItem('chatglm_refresh_token');
      if(v){ refresh = JSON.parse(v); if(refresh && refresh.value) refresh = refresh.value; }
    }catch(e){}
  }
  let guest = false;
  if(refresh){
    try{
      let p = refresh.split('.')[1] || '';
      p = p.replace(/-/g,'+').replace(/_/g,'/');
      while(p.length % 4) p += '=';
      const payload = JSON.parse(atob(p));
      guest = payload && payload.is_guest === true;
    }catch(e){}
  }
  return JSON.stringify({
    auth_token: refresh||'', device_id: '', settings_token: '', cookie: document.cookie||'',
    guest: guest
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

    provider = acc.get("provider", "deepseek")
    if provider == "chatglm":
        start_url, extract_js, title = CHATGLM_URL, CHATGLM_EXTRACT_JS, "登录 智谱清言"
    else:
        start_url, extract_js, title = DEEPSEEK_URL, DEEPSEEK_EXTRACT_JS, "登录 DeepSeek"

    write_state(acc_id, "opening", "正在打开内置浏览器…")

    # 每个账号一套独立 storage，互不干扰，也不碰系统浏览器
    storage = data_dir / "browser_profiles" / acc_id
    # 关键：每次登录都用全新的 profile，避免复用上一次的 cookie/会话，
    # 否则新账号会直接显示成上一个账号已登录的状态。
    if storage.exists():
        import shutil
        for _ in range(3):
            try:
                shutil.rmtree(storage, ignore_errors=False)
                break
            except Exception:
                time.sleep(0.4)
    storage.mkdir(parents=True, exist_ok=True)

    window = webview.create_window(
        f"{title} · {acc.get('name', '')}",
        start_url,
        width=1000,
        height=760,
        text_select=True,
    )

    done = {"v": False}
    # 新建账号（还没有凭证）时必须确保是全新会话；
    # 已有凭证的账号允许复用 profile（方便刷新 token）。
    fresh = not (acc.get("auth_token") or "").strip()
    # 同平台其它账号已用的凭证，抓到就说明是旧会话残留
    other_tokens = {
        (a.get("auth_token") or "").strip()
        for a in store.list_accounts(provider)
        if a.get("id") != acc_id and (a.get("auth_token") or "").strip()
    }

    def worker() -> None:
        deadline = time.time() + 600
        announced = False
        guest_noted = False
        while time.time() < deadline and not done["v"]:
            time.sleep(1.5)
            try:
                val = window.evaluate_js(extract_js)
            except Exception:
                continue
            if not announced:
                announced = True
                write_state(acc_id, "waiting", f"请在内置浏览器里完成 {provider} 登录…")
            if not val:
                continue
            try:
                creds = json.loads(val)
            except Exception:
                continue
            token = (creds.get("auth_token") or "").strip()
            if not token:
                continue
            # 智谱未登录时会下发游客 token，不能当作登录成功
            if creds.get("guest"):
                if not guest_noted:
                    guest_noted = True
                    write_state(acc_id, "waiting", "当前是游客会话，请在页面里扫码 / 手机号登录新账号…")
                continue
            # 抓到的凭证如果已经属于别的账号，说明是旧会话残留，清掉重来
            if token in other_tokens:
                try:
                    window.evaluate_js(
                        "document.cookie.split(';').forEach(c=>{"
                        "const n=c.split('=')[0].trim();"
                        "document.cookie=n+'=;expires=Thu, 01 Jan 1970 00:00:00 GMT;path=/';});"
                        "try{localStorage.clear()}catch(e){}"
                        "try{sessionStorage.clear()}catch(e){}"
                        "setTimeout(()=>location.reload(),300);"
                    )
                except Exception:
                    pass
                if not guest_noted:
                    guest_noted = True
                    write_state(acc_id, "waiting", "检测到旧账号会话，已清除，请登录新账号…")
                time.sleep(1.5)
                continue
            # 全新账号：如果抓到的还是旧的残留 cookie，先清掉再等用户重新登录
            if fresh and token == (acc.get("auth_token") or ""):
                try:
                    window.evaluate_js(
                        "document.cookie.split(';').forEach(c=>{"
                        "const n=c.split('=')[0].trim();"
                        "document.cookie=n+'=;expires=Thu, 01 Jan 1970 00:00:00 GMT;path=/';});"
                        "try{localStorage.clear()}catch(e){}"
                    )
                except Exception:
                    pass
                time.sleep(1.5)
                continue
            # 给 WebView2 一点时间把 cookie 落盘
            time.sleep(1.2)
            fields: dict = {
                "auth_token": token,
                "device_id": creds.get("device_id", ""),
                "settings_token": creds.get("settings_token", ""),
                "cookie": creds.get("cookie", ""),
            }
            # 记录登录时真实浏览器画像（WebView2/Edge 的 UA 与 platform），
            # 供后续 API 请求复用，避免同一 token 出现两套 UA。
            ua = (creds.get("ua") or "").strip()
            platform = (creds.get("p") or "").strip()
            if ua or platform:
                try:
                    cur = store.get_account(acc_id) or {}
                    try:
                        profile = json.loads(cur.get("profile") or "{}")
                    except (TypeError, ValueError):
                        profile = {}
                    if not isinstance(profile, dict):
                        profile = {}
                    if ua:
                        profile["ua"] = ua
                    if platform:
                        profile["platform"] = platform
                    fields["profile"] = json.dumps(profile, ensure_ascii=False)
                except Exception:
                    pass
            store.update_account(acc_id, **fields)
            done["v"] = True
            write_state(acc_id, "success", "登录成功，凭证已保存")
            time.sleep(0.6)
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
