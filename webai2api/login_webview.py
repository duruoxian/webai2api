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
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

from .paths import user_data_dir
from .store import Store

# 同一时刻只允许一个登录窗口（WebView2 代理参数是进程级环境变量）
_WEBVIEW_PROXY_LOCK = threading.Lock()

DEEPSEEK_URL = "https://chat.deepseek.com/"
CHATGLM_URL = "https://chatglm.cn/main/alltoolsdetail?lang=zh"
KIMI_URL = "https://www.kimi.com/"
QWEN_URL = "https://chat.qwen.ai/"
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

# ── Kimi / 通义千问：登录凭证在 localStorage（zustand persist / OpenWebUI token）──
# 通用策略：遍历 localStorage（含嵌套 JSON 的 state.accessToken / token / access_token），
# 找出「像 JWT」且**不属于游客态**的字符串作为 auth_token。
# Kimi 未登录也会下发 anonymous_access_token / anonymous_refresh_token，必须排除，
# 否则窗口会瞬间"登录成功"。guest 判定：JWT payload 里带 is_anonymous/anonymous 标记，
# 或键名本身含 anonymous。
def _jwt_find_js(guest_markers: str) -> str:
    return r"""
(function(){
  const JWT = /^[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{4,}$/;
  const SKIP = /(anonymous|guest|demo)/i;
  function decodePayload(t){
    try{
      let p = String(t).split('.')[1] || '';
      p = p.replace(/-/g,'+').replace(/_/g,'/');
      while(p.length % 4) p += '=';
      return JSON.parse(atob(p));
    }catch(e){ return {}; }
  }
  const isGuestToken = (t, key) => {
    if(key && SKIP.test(key)) return true;
    const pl = decodePayload(t);
    if(pl && (pl.is_anonymous === true || pl.anonymous === true || pl.is_guest === true)) return true;
    return !!(pl && pl.uid === 'anonymous');
  };
  // 只接受「访问令牌」：明确排除 refresh（Kimi 会同时下发 access/refresh，
  // 用 refresh 调 /api/user 会 401 auth.token.invalid）
  const isRefreshToken = (t, key) => {
    if(key && /refresh/i.test(key)) return true;
    const pl = decodePayload(t);
    return !!(pl && (pl.typ === 'refresh' || pl.type === 'refresh'));
  };
  let auth = '', key = '', refreshTok = '';
  const consider = (val, k) => {
    if(typeof val !== 'string' || auth) return;
    if(!JWT.test(val)) return;
    if(isGuestToken(val, k)) return;
    if(isRefreshToken(val, k)){
      // 顺手记住 refresh token，供 Kimi 自动续期（access 只有 15 分钟）
      if(!refreshTok) refreshTok = val;
      return;
    }
    auth = val; key = k || '';
  };
  const walk = (obj, path, depth) => {
    if(!obj || depth > 3) return;
    if(typeof obj === 'string'){ consider(obj, path); return; }
    if(typeof obj !== 'object') return;
    for(const k of Object.keys(obj)){
      const v = obj[k];
      const p = path ? (path + '.' + k) : k;
      if(typeof v === 'string') consider(v, p);
      else if(v && typeof v === 'object') walk(v, p, depth + 1);
      if(auth) return;
    }
  };
  for(let i = 0; i < localStorage.length && !auth; i++){
    const lk = localStorage.key(i);
    const raw = localStorage.getItem(lk);
    if(!raw) continue;
    // 键名优先：access_token / accessToken 一定要排在 refresh 前面处理
    if(JWT.test(raw) && !isGuestToken(raw, lk) && !isRefreshToken(raw, lk)){
      auth = raw; key = lk; break;
    }
    try{ walk(JSON.parse(raw), lk, 0); }catch(e){}
  }
  // 兜底：先扫一遍所有 access 键，再扫 JSON 嵌套（避免随机遍历顺序拿到 refresh）
  if(!auth){
    for(let i = 0; i < localStorage.length && !auth; i++){
      const lk = localStorage.key(i);
      if(!/access/i.test(lk)) continue;
      const raw = localStorage.getItem(lk);
      if(raw && JWT.test(raw) && !isGuestToken(raw, lk) && !isRefreshToken(raw, lk)){
        auth = raw; key = lk;
      }
    }
  }
  if(!auth){ try{ walk(window.__INITIAL_STATE__ || {}, 'state', 0); }catch(e){} }
  let guest = false;
  if(!auth){
    // 没找到用户级 JWT，但存在游客 token → 明确告诉调用方"还在游客态"
    for(let i = 0; i < localStorage.length; i++){
      const lk = localStorage.key(i);
      const raw = localStorage.getItem(lk);
      if(!raw) continue;
      try{
        const o = JSON.parse(raw);
        const s = (o && o.state) ? o.state : o;
        for(const k of Object.keys(s || {})){
          const v = s[k];
          if(typeof v === 'string' && JWT.test(v) && SKIP.test(k)){ guest = true; break; }
        }
      }catch(e){
        if(JWT.test(raw) && SKIP.test(lk)) guest = true;
      }
      if(guest) break;
    }
  }
  return JSON.stringify({
    auth_token: auth, device_id: '', settings_token: refreshTok,
    cookie: document.cookie||'', guest: guest,
    token_key: key,
    u: navigator.userAgent||'', p: navigator.platform||''
  });
})()
"""


KIMI_EXTRACT_JS = _jwt_find_js("anonymous")
QWEN_EXTRACT_JS = _jwt_find_js("guest")


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


def _kill_webview_processes_for(storage: Path) -> None:
    r"""杀掉命令行里带本账号 storage 路径的 msedgewebview2 进程。

    残留进程会锁住 Cookies / Local Storage，导致 rmtree 失败、旧登录态被复用。
    只杀 cmdline 命中给定路径的进程：主面板（data/panel_profile）、
    千问中继（browser_profiles/<acc>）等其它目录不受影响。
    """
    try:
        needle = str(storage).lower().replace("\\", "/")
        ps = (
            "Get-CimInstance Win32_Process -Filter \"name='msedgewebview2.exe'\" "
            "| Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
        )
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True, text=True, timeout=8,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout or "[]"
        data = json.loads(out) if out.strip() else []
        if isinstance(data, dict):
            data = [data]
        killed = False
        for p in data:
            cl = str(p.get("CommandLine") or "").lower().replace("\\", "/")
            pid = p.get("ProcessId")
            if pid and needle in cl:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    capture_output=True, timeout=8,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                killed = True
        if killed:
            time.sleep(0.6)  # 等进程真正退出、释放文件锁
    except Exception:
        pass


def _purge_login_state(storage: Path) -> None:
    """rmtree 整体失败时的兜底：只删保留登录态的子文件/目录。

    注意 WebView2 的 cookie 实际落在 EBWebView/Default/Network/Cookies，
    老版本 pywebview/Chrome 才在 Default/Cookies，两处都要清。
    """
    default = storage / "EBWebView" / "Default"
    if not default.exists():
        default = storage / "Default"
    if not default.exists():
        return
    for rel in ("Network/Cookies", "Network/Cookies-journal", "Cookies", "Cookies-journal",
                "Login Data", "Login Data-journal", "Web Data", "Web Data-journal",
                "History", "History-journal"):
        try:
            f = default / rel
            if f.exists():
                f.unlink()
        except Exception:
            pass
    for rel in ("Local Storage", "Session Storage", "IndexedDB", "Service Worker",
                "Cache", "Code Cache", "GPUCache"):
        try:
            d = default / rel
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass


def _egress_badge_js(mode: str) -> str:
    """注入到登录页右下角的只读出口标签（每轮轮询重新确保存在）。"""
    label = {"direct": "直连", "custom": "自定义代理",
             "system": "跟随系统代理"}.get(mode, mode)
    text = "webai2api 登录窗口 · 当前出口：" + label
    safe = json.dumps(text, ensure_ascii=False)
    return (
        "(function(){try{"
        "var b=document.getElementById('webai-egress-badge');"
        "if(!b){if(!document.body)return;"
        "b=document.createElement('div');b.id='webai-egress-badge';"
        "b.style.cssText='position:fixed;right:10px;bottom:10px;z-index:2147483647;"
        "background:#1c1c1e;color:#fff;padding:6px 10px;border-radius:8px;"
        "font:12px/1.4 system-ui;box-shadow:0 2px 8px rgba(0,0,0,.3);opacity:.92';"
        "document.body.appendChild(b);}b.textContent=" + safe + ";}catch(e){}})()"
    )


def cleanup_browser_residue(logger=None) -> dict:
    """清除本机残留的浏览器登录痕迹（面板「清浏览器痕迹」按钮的后端）。

    范围：
    - browser_profiles/_session/*：一次性登录目录的残留（正常在每次登录开头
      被清扫，pywebview 关窗即杀进程时会漏掉）；
    - %APPDATA%\\pywebview：旧版本（<0.1.21）登录窗口共用的默认 WebView2
      存储，旧 cookie（如微信授权残留）会一直留在机器上；新版面板已改用
      data/panel_profile，此目录成为纯残留，整目录删除。
    不碰 browser_profiles/<acc_id>（千问中继的常驻登录态，删了要重新扫码）。
    """
    import logging

    log = logger or logging.getLogger("webai2api.login")
    data_dir = user_data_dir()
    removed: list[str] = []

    # 1) 一次性登录目录残留
    session_root = data_dir / "browser_profiles" / "_session"
    if session_root.exists():
        for child in session_root.iterdir():
            if child.is_dir():
                _kill_webview_processes_for(child)
                shutil.rmtree(child, ignore_errors=True)
            if not child.exists():
                removed.append(str(child))
        try:
            session_root.rmdir()
            removed.append(str(session_root))
        except OSError:
            pass

    # 2) 旧版默认 WebView2 存储。只杀 cmdline 里带该路径的进程：
    #    新版主面板用 data/panel_profile、登录窗口用一次性目录、
    #    千问中继用 browser_profiles/<acc>，都不会被误伤。
    legacy = Path(
        os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    ) / "pywebview"
    if legacy.exists():
        _kill_webview_processes_for(legacy)
        shutil.rmtree(legacy, ignore_errors=True)
        if not legacy.exists():
            removed.append(str(legacy))

    log.info("浏览器痕迹清理完成，移除 %d 项", len(removed))
    return {"ok": True, "removed": removed}


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
    elif provider == "kimi":
        start_url, extract_js, title = KIMI_URL, KIMI_EXTRACT_JS, "登录 Kimi"
    elif provider == "qwen":
        start_url, extract_js, title = QWEN_URL, QWEN_EXTRACT_JS, "登录 通义千问"
    else:
        start_url, extract_js, title = DEEPSEEK_URL, DEEPSEEK_EXTRACT_JS, "登录 DeepSeek"

    write_state(acc_id, "opening", "正在打开内置浏览器…")

    # 内置浏览器登录窗口的 profile 策略：
    # - 其它平台：每次登录用一个**全新的一次性目录**，绝不复用。页面永远是
    #   未登录状态，不会带出上一个账号的 cookie / 微信授权信息。
    #   （pywebview 关窗时可能直接结束进程、跳过清理，所以历史残留在
    #   每次登录开头统一清扫。）
    # - 千问例外：它的聊天请求靠一个常驻的隐藏中继窗口在页面里发 fetch，
    #   登录态必须留在 browser_profiles/<acc_id> 里给中继用，所以固定用该
    #   目录；干净度靠"打开前彻底删除重建 + 精准兜底清理"保证。
    base = data_dir / "browser_profiles"
    session_root = base / "_session"
    if session_root.exists():
        for stale in session_root.glob(f"{acc_id}-*"):
            _kill_webview_processes_for(stale)
            shutil.rmtree(stale, ignore_errors=True)
    if provider == "qwen":
        storage = base / acc_id
    else:
        storage = session_root / f"{acc_id}-{uuid.uuid4().hex[:12]}"
    if storage.exists():
        _kill_webview_processes_for(storage)
        cleared = False
        for _ in range(10):
            shutil.rmtree(storage, ignore_errors=True)
            if not storage.exists():
                cleared = True
                break
            time.sleep(0.3)
        if not cleared:
            # 极端情况删不掉：精准清登录态文件（含 WebView2 真实 cookie 路径），
            # 绝不允许旧 cookie/localStorage 残留导致登录页带出旧账号。
            _purge_login_state(storage)
    storage.mkdir(parents=True, exist_ok=True)
    from .egress import normalize_egress
    egress_mode = normalize_egress(str(acc.get("egress") or "system"))

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
        badge_injected = 0.0
        while time.time() < deadline and not done["v"]:
            time.sleep(1.5)
            # 右下角只读角标：只在需要时注入（页面导航后重新补），
            # 不再每秒反复操作 DOM——Kimi 这类重页面会被持续重排拖到卡死。
            now = time.time()
            if now - badge_injected > 5.0:
                badge_injected = now
                try:
                    window.evaluate_js(_egress_badge_js(egress_mode))
                except Exception:
                    pass
            try:
                val = window.evaluate_js(extract_js)
            except Exception:
                # 页面正在加载/重绘时 evaluate_js 会抛错；不要紧贴着重试
                time.sleep(0.5)
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
                    # 账号级固定设备指纹：首次登录生成一次，之后永不改变
                    if not profile.get("device_id"):
                        import hashlib
                        _pfx = {"chatglm": "glm", "kimi": "kimi",
                                "qwen": "qwen"}.get(provider, provider)
                        seed = f"{acc_id}:{ua or platform or token[:16]}"
                        profile["device_id"] = hashlib.sha256(
                            f"{_pfx}-device:{seed}".encode()).hexdigest()[:32]
                    fields["profile"] = json.dumps(profile, ensure_ascii=False)
                except Exception:
                    pass
            store.update_account(acc_id, **fields)
            done["v"] = True
            write_state(acc_id, "success", "登录成功，凭证已保存")
            # 千问要把整个 profile 搬给常驻中继，必须先让 WebView2 把
            # localStorage/cookie 刷盘（IndexedDB/LevelDB 有延迟写），
            # 否则搬过去的是空壳，中继会一直认为未登录。
            if provider == "qwen":
                try:
                    # 主动触发一次存储落盘，再多等几秒
                    window.evaluate_js(
                        "try{localStorage.setItem('__flush__',Date.now());"
                        "localStorage.removeItem('__flush__');}catch(e){}"
                    )
                except Exception:
                    pass
                time.sleep(4.0)
            else:
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
    # 内置浏览器（WebView2）的出口要与账号设置一致：
    #   direct → 强制忽略系统代理；custom → 用账号代理；system → 跟随系统
    # WebView2 通过 WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS 接受额外命令行参数。
    with _WEBVIEW_PROXY_LOCK:
        prev_args = os.environ.get("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS", "")
        extra = _webview_proxy_args(acc)
        try:
            if extra:
                os.environ["WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS"] = (
                    (prev_args + " " + extra).strip())
            try:
                # private_mode=False：站点把 cookie/localStorage 正常写进这个
                # 一次性目录（InPrivate 下部分站点会拒绝登录），目录本身每次
                # 都是新建的，用完即删，所以同样保证绝对干净。
                webview.start(private_mode=False, storage_path=str(storage))
            except Exception as exc:  # noqa: BLE001
                write_state(acc_id, "error", f"内置浏览器启动失败: {exc}")
            # webview.start 返回后 WebView2 可能仍有残留子进程拖着窗口不放，
            # 导致本进程永不退出、profile 也没机会落盘。这里主动收尾。
            if done["v"]:
                _kill_webview_processes_for(storage)
                time.sleep(1.5)
        finally:
            if extra:
                if prev_args:
                    os.environ["WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS"] = prev_args
                else:
                    os.environ.pop("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS", None)
            # 先确保 WebView2 把 localStorage / cookie 落盘，再释放文件锁
            _kill_webview_processes_for(storage)
            time.sleep(1.0)
            # 一次性 profile（非千问）：窗口关闭后整目录丢弃，不留在磁盘上；
            # 千问的 profile 是常驻中继要复用的，必须保留，绝不能删。
            if provider != "qwen":
                for _ in range(10):
                    if not storage.exists():
                        break
                    shutil.rmtree(storage, ignore_errors=True)
                    if storage.exists():
                        time.sleep(0.3)


def _webview_proxy_args(acc: dict) -> str:
    """把账号 egress 翻译成 WebView2 的代理参数（空串=不干预）。"""
    from .egress import EGRESS_CUSTOM, EGRESS_DIRECT, normalize_egress
    from .proxy import parse_proxy

    mode = normalize_egress(str(acc.get("egress") or "system"))
    if mode == EGRESS_DIRECT:
        # 强制绕过系统代理，和聊天请求保持一致
        return '--proxy-server="direct://" --proxy-bypass-list=*'
    if mode == EGRESS_CUSTOM:
        px = parse_proxy(str(acc.get("proxy") or ""))
        if px is None:
            return ""
        scheme = "socks5" if px.scheme.startswith("socks") else "http"
        # 注意：Chromium 命令行不支持代理认证，带账号密码的自定义代理
        # 在内置浏览器里可能弹认证框；聊天请求不受影响。
        return f'--proxy-server="{scheme}://{px.host}:{px.port}"'
    return ""
