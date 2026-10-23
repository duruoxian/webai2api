"""账号出口（egress）决策：直连 / 跟随系统代理 / 自定义代理。

为什么需要单独一层：
- 平台风控会把「登录 IP」和「API 请求 IP」做一致性校验，两者必须同源。
- 国内用户常开 Clash/VPN，但 Clash 的 TUN 模式会在网卡层接管全部流量，
  代码里「不设代理」并不等于真直连。因此：
    * direct  = 强制绕过系统代理（并给出 TUN 警告）
    * system  = 跟随系统代理（Clash 系统代理模式）
    * custom  = 账号自带 proxy 字段

代理 URL 解析统一走 proxy.py。
"""
from __future__ import annotations

import os
import urllib.request

from .proxy import parse_proxy

# 出口模式
EGRESS_DIRECT = "direct"    # 强制直连（忽略系统代理）
EGRESS_SYSTEM = "system"    # 跟随系统代理
EGRESS_CUSTOM = "custom"    # 账号自定义代理

VALID_EGRESS = (EGRESS_DIRECT, EGRESS_SYSTEM, EGRESS_CUSTOM)


def normalize_egress(value: str | None) -> str:
    v = (value or "").strip().lower()
    return v if v in VALID_EGRESS else EGRESS_SYSTEM


def resolve_proxy_url(egress: str, proxy: str) -> str:
    """把账号的 egress 设置翻译成「实际要用的代理 URL」。

    返回空串表示：不主动设置代理。
    注意：system 模式返回空串，由调用方（urllib）自动读系统代理；
    direct 模式必须由调用方显式禁用代理。
    """
    mode = normalize_egress(egress)
    if mode == EGRESS_CUSTOM:
        return (proxy or "").strip()
    if mode == EGRESS_SYSTEM:
        return ""
    return ""


def direct_opener() -> urllib.request.OpenerDirector:
    """构造一个完全不使用代理的 opener（direct 模式专用）。"""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def proxy_map_for(egress: str, proxy: str) -> dict[str, str]:
    """返回 urllib ProxyHandler 的代理映射。

    direct / system → 空字典（system 由 urllib 默认 opener 读取系统设置）。
    custom → http/https 映射到账号代理。
    """
    mode = normalize_egress(egress)
    if mode != EGRESS_CUSTOM:
        return {}
    px = parse_proxy((proxy or "").strip())
    if px is None:
        return {}
    url = px.url_for_urllib()
    return {"http": url, "https": url}


def egress_opener(
    egress: str, proxy: str, host: str = ""
) -> urllib.request.OpenerDirector:
    """按 egress 返回合适的 opener。

    - direct → 无代理（显式 ProxyHandler({})，阻止 urllib 读系统代理）
    - system → 读取系统代理（优先环境变量，Windows 再读注册表）
    - custom → 指定代理（含认证）
    """
    mode = normalize_egress(egress)
    if mode == EGRESS_SYSTEM:
        sys_url = system_proxy_for(host) if host else ""
        if not sys_url:
            return urllib.request.build_opener(urllib.request.ProxyHandler({}))
        px = parse_proxy(sys_url)
        if px is None:
            return urllib.request.build_opener(urllib.request.ProxyHandler({}))
        proxy_map = {"http": px.url_for_urllib(), "https": px.url_for_urllib()}
        return urllib.request.build_opener(urllib.request.ProxyHandler(proxy_map))
    proxy_map = proxy_map_for(mode, proxy)
    if mode == EGRESS_CUSTOM and proxy_map:
        px = parse_proxy((proxy or "").strip())
        assert px is not None
        if px.username:
            auth = px.auth_header()

            class _AuthProxy(urllib.request.ProxyHandler):
                def proxy_open(self, req, proxy_, type_):  # noqa: A003
                    req.add_header("Proxy-Authorization", auth)
                    return super().proxy_open(req, proxy_, type_)

            return urllib.request.build_opener(_AuthProxy(proxy_map))
        return urllib.request.build_opener(urllib.request.ProxyHandler(proxy_map))
    # direct 或 custom 但代理无效：强制无代理
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _windows_registry_proxy() -> tuple[str, str]:
    """读 Windows 注册表系统代理，返回 (代理URL, 绕过列表)。"""
    if os.name != "nt":
        return "", ""
    try:
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
        )
        try:
            try:
                enabled, _ = winreg.QueryValueEx(key, "ProxyEnable")
            except OSError:
                enabled = 0
            try:
                server, _ = winreg.QueryValueEx(key, "ProxyServer")
            except OSError:
                server = ""
            try:
                override, _ = winreg.QueryValueEx(key, "ProxyOverride")
            except OSError:
                override = ""
        finally:
            winreg.CloseKey(key)
    except Exception:
        return "", ""
    if not enabled:
        return "", override
    text = str(server or "").strip()
    if not text:
        return "", override
    if "=" in text:
        parts = {}
        for item in text.split(";"):
            if "=" in item:
                k, v = item.split("=", 1)
                parts[k.strip().lower()] = v.strip()
        text = parts.get("https") or parts.get("http") or ""
    if not text:
        return "", override
    if "://" not in text:
        text = "http://" + text
    return text, override


def system_proxy_for(host: str) -> str:
    """读取系统代理设置，返回适用于该 host 的代理 URL（空=直连）。

    优先环境变量；Windows 上再读注册表（Clash 写入的就是注册表）。
    """
    proxy_url = ""
    bypass = ""
    try:
        proxies = urllib.request.getproxies()
        proxy_url = str(proxies.get("https") or proxies.get("http") or "").strip()
        bypass = str(proxies.get("no") or "")
    except Exception:
        pass
    if not proxy_url:
        reg_url, reg_bypass = _windows_registry_proxy()
        proxy_url = reg_url
        if reg_bypass:
            bypass = (bypass + "," + reg_bypass) if bypass else reg_bypass
    if not proxy_url:
        return ""
    if "://" not in proxy_url:
        proxy_url = "http://" + proxy_url

    host_l = (host or "").lower()
    for item in bypass.replace(";", ",").split(","):
        item = item.strip().lower().lstrip(".")
        if not item or item == "<local>":
            continue
        if host_l == item or host_l.endswith("." + item):
            return ""
    return proxy_url


def apply_system_proxy(conn, host: str) -> bool:
    """给 http.client 连接套上系统代理隧道；返回是否实际使用了代理。"""
    url = system_proxy_for(host)
    if not url:
        return False
    px = parse_proxy(url)
    if px is None:
        return False
    px.apply(conn, host)
    return True


def describe_egress(egress: str, proxy: str) -> str:
    mode = normalize_egress(egress)
    if mode == EGRESS_DIRECT:
        return "直连"
    if mode == EGRESS_CUSTOM:
        px = parse_proxy(proxy)
        return f"代理 {px.describe()}" if px else "自定义代理（未填写/无效）"
    return "跟随系统代理"


def system_proxy_enabled() -> bool:
    """Windows 系统代理是否开启（仅诊断用）。"""
    if os.name != "nt":
        return bool(urllib.request.getproxies())
    try:
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
        )
        try:
            enabled, _ = winreg.QueryValueEx(key, "ProxyEnable")
            return bool(enabled)
        finally:
            winreg.CloseKey(key)
    except Exception:
        return False


def detect_tun_interface() -> str:
    """检测是否存在 Clash/Mihomo 之类的 TUN 网卡（Windows）。

    返回网卡名，未检测到返回空串。仅用于面板提示：
    开了 TUN 时，direct 模式也会被网卡层接管，不是真直连。
    """
    if os.name != "nt":
        return ""
    try:
        import subprocess

        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-NetAdapter | Where-Object {$_.Status -eq 'Up'} | "
             "Select-Object -ExpandProperty InterfaceDescription"],
            capture_output=True, text=True, timeout=6,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        text = (out.stdout or "").lower()
        for needle in ("mihomo", "clash", "tun", "wintun", "tap-windows"):
            if needle in text:
                return needle
    except Exception:
        pass
    return ""
