"""账号健康检查：定期验证 token 是否仍然有效，并缓存账号资料。

- DeepSeek：GET /api/v0/users/current（带 Bearer token）
- 智谱清言：先用 refresh_token 换 access_token，再 GET /chatglm/user-api/user/info
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from logging import Logger

from .store import Store


DEFAULT_DEEPSEEK_UA = (
    "Mozilla/5.0 (X11; Linux x86_64; rv:156.0) Gecko/20100101 Firefox/156.0"
)


def _parse_profile(account: dict) -> dict:
    """把 account['profile']（JSON 字符串）容错解析成 dict。"""
    raw = account.get("profile") or "{}"
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def deepseek_headers(account: dict) -> dict:
    """构造 DeepSeek 健康检查请求头。

    UA / x-device-model 优先取登录时记录的真实浏览器画像（profile.ua /
    profile.platform），缺失时回退默认 Firefox UA 与空 device-model。
    """
    base = "https://chat.deepseek.com"
    profile = _parse_profile(account)
    ua = str(profile.get("ua") or "").strip() or DEFAULT_DEEPSEEK_UA
    platform = str(profile.get("platform") or "").strip()
    headers = {
        "Authorization": f"Bearer {account.get('auth_token', '')}",
        "x-device-id": account.get("device_id", ""),
        "x-device-model": platform,
        "x-client-bundle-id": "com.deepseek.chat",
        "x-client-platform": "web",
        "x-client-version": "2.5.0",
        "x-client-locale": "zh_CN",
        "x-client-timezone-offset": "28800",
        "User-Agent": ua,
        "Origin": base,
        "Referer": base + "/",
        "Accept": "*/*",
    }
    if account.get("cookie"):
        headers["Cookie"] = account["cookie"]
    return headers


def _open_account(account: dict, req: urllib.request.Request, timeout: int):
    """按账号 egress 发起请求（与聊天链路保持同源出口）。

    direct → 强制忽略系统代理；system → 跟随系统；custom → 账号代理。
    """
    from urllib.parse import urlsplit

    from .egress import egress_opener

    host = urlsplit(req.full_url).hostname or ""
    opener = egress_opener(
        str(account.get("egress") or "system"),
        str(account.get("proxy") or ""),
        host,
    )
    return opener.open(req, timeout=timeout)


def _deepseek_check(account: dict, timeout: int = 15) -> tuple[str, dict, str]:
    base = "https://chat.deepseek.com"
    headers = deepseek_headers(account)
    req = urllib.request.Request(base + "/api/v0/users/current", headers=headers)
    with _open_account(account, req, timeout) as r:
        data = json.load(r)
    biz = (data.get("data") or {}).get("biz_data") or {}
    if not biz.get("id"):
        raise RuntimeError(data.get("msg") or "token 无效")
    profile = {
        "uid": biz.get("id", ""),
        "email": biz.get("email", ""),
        "phone": biz.get("mobile_number", ""),
    }
    return "ok", profile, ""


def _chatglm_check(account: dict, timeout: int = 15) -> tuple[str, dict, str]:
    from .adapters.chatglm.client import _build_headers
    from .adapters.chatglm.config import DEFAULT_BASE_URL, SIGN_SECRET

    # 与聊天请求使用同一套账号级固定指纹，避免「同 token 换设备」的强风控信号
    def _prof() -> dict:
        raw = account.get("profile")
        if isinstance(raw, dict):
            return raw
        try:
            data = json.loads(raw or "{}")
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    prof = _prof()
    dev_id = str(prof.get("device_id") or "") or hashlib.sha256(
        f"glm-device:{account.get('id') or account.get('name') or ''}".encode()
    ).hexdigest()[:32]
    ua = str(prof.get("ua") or "")
    dev_model = str(prof.get("platform") or "")

    def _hdr(token: str) -> dict:
        return _build_headers(SIGN_SECRET, token, device_id=dev_id,
                              user_agent=ua, device_model=dev_model)

    refresh = account.get("auth_token", "")
    if not refresh:
        raise RuntimeError("缺少 refresh_token")
    req = urllib.request.Request(
        DEFAULT_BASE_URL + "/chatglm/user-api/user/refresh",
        data=b"{}",
        headers={**_hdr(refresh), "Content-Type": "application/json"},
        method="POST",
    )
    with _open_account(account, req, timeout) as r:
        data = json.load(r)
    result = data.get("result") or {}
    access = result.get("access_token") or ""
    if not access:
        raise RuntimeError(data.get("message") or "refresh_token 已失效")
    req2 = urllib.request.Request(
        DEFAULT_BASE_URL + "/chatglm/user-api/user/info",
        headers={**_hdr(access), "Accept": "application/json"},
    )
    with _open_account(account, req2, timeout) as r2:
        info = json.load(r2)
    res = info.get("result") or {}
    profile = {
        "uid": res.get("_id", result.get("user_id", "")),
        "nickname": res.get("nickname", ""),
        "phone": res.get("phone", ""),
        "avatar": res.get("avatar", ""),
    }
    # 顺带把轮换后的 refresh_token 存回去
    new_refresh = result.get("refresh_token") or ""
    return "ok", profile, new_refresh


def _kimi_check(account: dict, timeout: int = 15) -> tuple[str, dict, str]:
    """Kimi：GET /api/user（Bearer auth_token）。401 → 由外层记 expired。"""
    token = str(account.get("auth_token") or "").strip()
    if not token:
        raise RuntimeError("缺少 accessToken，请先在面板登录 Kimi")
    profile = _parse_profile(account)
    ua = str(profile.get("ua") or "").strip()
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "User-Agent": ua or DEFAULT_DEEPSEEK_UA,
        "Origin": "https://www.kimi.com",
        "Referer": "https://www.kimi.com/",
    }
    if account.get("cookie"):
        headers["Cookie"] = account["cookie"]
    req = urllib.request.Request("https://www.kimi.com/api/user", headers=headers)
    with _open_account(account, req, timeout) as r:
        data = json.load(r)
    # 兼容 {"data":{...}} 与直接返回用户对象两种形态
    user = data.get("data") if isinstance(data.get("data"), dict) else data
    uid = str(user.get("id") or user.get("user_id") or user.get("uid") or "")
    if not uid:
        raise RuntimeError(str(data.get("message") or "token 无效"))
    return "ok", {
        "uid": uid,
        "nickname": str(user.get("name") or user.get("nickname") or ""),
    }, ""


def _qwen_check(account: dict, timeout: int = 15) -> tuple[str, dict, str]:
    """通义千问：GET /api/v2/auths/（OpenWebUI 用户接口）。

    注意 Qwen 的信封：HTTP 200 也可能 {"success":false,...,"code":"Unauthorized"}，
    必须按 body 判断而不是状态码。
    """
    token = str(account.get("auth_token") or "").strip()
    if not token:
        raise RuntimeError("缺少 access_token，请先在面板登录通义千问")
    profile = _parse_profile(account)
    ua = str(profile.get("ua") or "").strip()
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "User-Agent": ua or DEFAULT_DEEPSEEK_UA,
        "Origin": "https://chat.qwen.ai",
        "Referer": "https://chat.qwen.ai/",
    }
    if account.get("cookie"):
        headers["Cookie"] = account["cookie"]
    req = urllib.request.Request("https://chat.qwen.ai/api/v2/auths/", headers=headers)
    with _open_account(account, req, timeout) as r:
        data = json.load(r)
    if data.get("success") is False:
        err = (data.get("data") or {})
        msg = str(err.get("code") or err.get("details") or "token 无效")
        if "unauth" in msg.lower():
            raise RuntimeError(f"access_token 已过期: {msg}")
        raise RuntimeError(msg)
    user = data.get("data") or data
    uid = str(user.get("id") or user.get("user_id") or "")
    if not uid:
        raise RuntimeError("token 无效（响应缺少用户 id）")
    return "ok", {
        "uid": uid,
        "email": str(user.get("email") or ""),
        "nickname": str(user.get("name") or user.get("nickname") or ""),
    }, ""


def check_account(account: dict) -> tuple[str, dict, str]:
    provider = account.get("provider", "")
    if provider == "chatglm":
        return _chatglm_check(account)
    if provider == "kimi":
        return _kimi_check(account)
    if provider == "qwen":
        return _qwen_check(account)
    return _deepseek_check(account)


class HealthMonitor:
    """后台线程：每 N 分钟检查一次启用的账号。"""

    def __init__(self, store: Store, logger: Logger, interval: int = 1800) -> None:
        self.store = store
        self.logger = logger
        self.interval = max(300, int(interval))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="health", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def check_all(self, only_stale: bool = False) -> list[dict]:
        results = []
        for account in self.store.list_accounts():
            if not account.get("enabled"):
                continue
            if only_stale and time.time() - float(account.get("checked_at") or 0) < self.interval:
                continue
            try:
                status, profile, new_token = check_account(account)
                self.store.set_account_health(
                    account["id"], status, profile, new_token or None)
                results.append({"id": account["id"], "status": status})
            except urllib.error.HTTPError as exc:
                status = "expired" if exc.code in (401, 403) else f"http_{exc.code}"
                self.store.set_account_health(account["id"], status)
                results.append({"id": account["id"], "status": status})
            except Exception as exc:  # noqa: BLE001
                self.store.set_account_health(account["id"], "error")
                results.append({"id": account["id"], "status": "error",
                                "error": str(exc)[:200]})
                self.logger.warning("健康检查失败 %s: %s", account.get("name"), exc)
        return results

    def _run(self) -> None:
        # 启动后稍等，避免和首次登录/初始化抢资源
        time.sleep(20)
        while not self._stop.is_set():
            try:
                self.check_all(only_stale=True)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("健康检查循环异常: %s", exc)
            self._stop.wait(self.interval)
