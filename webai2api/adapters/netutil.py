"""适配器共用网络助手：URL 拆分 + 按出口模式建连。

此前 chatglm / kimi / qwen 各自复制了一份 `_split` / `_conn`（约 40 行
× 3），行为一致、默认 host 不同——统一到这里。出口决策（direct/system/
custom）与聊天链路保持同源；代理隧道的目标 host 用**裸主机名**（CONNECT
语义），带自定义端口的 base_url 只影响连接目标本身。
"""
from __future__ import annotations

import http.client
from logging import Logger
from urllib.parse import urlsplit

from ..egress import EGRESS_CUSTOM, EGRESS_SYSTEM, apply_system_proxy, normalize_egress


def split_url(url: str, default_host: str = "") -> tuple[str, str]:
    """拆成 (host[:port], path[?query])。host 带非默认端口（自建/测试环境）。"""
    parts = urlsplit(url)
    host = parts.hostname or default_host
    if parts.port:
        host = f"{host}:{parts.port}"
    path = parts.path or "/"
    if parts.query:
        path = f"{path}?{parts.query}"
    return host, path


def open_conn(
    url: str,
    timeout: int,
    *,
    egress: str = "system",
    proxy: str = "",
    default_host: str = "",
    logger: Logger | None = None,
) -> http.client.HTTPConnection:
    """按 URL scheme 建连（默认 https），并按出口模式套代理。

    - custom → 账号代理（无效回退直连）
    - system → 跟随系统代理（命中时套 CONNECT 隧道）
    - direct → 不主动设代理
    """
    parts = urlsplit(url)
    scheme = (parts.scheme or "https").lower()
    bare_host = parts.hostname or default_host
    host_for_conn = f"{bare_host}:{parts.port}" if parts.port else bare_host
    if scheme == "http":
        conn = http.client.HTTPConnection(
            host_for_conn, timeout=max(int(timeout), 30))
    else:
        conn = http.client.HTTPSConnection(
            host_for_conn, timeout=max(int(timeout), 30))

    mode = normalize_egress(egress)
    if mode == EGRESS_CUSTOM:
        from ..proxy import ProxyConfigError, parse_proxy

        try:
            px = parse_proxy(proxy or "")
            if px is not None:
                px.apply(conn, bare_host)
                if logger:
                    logger.info("出口: 自定义代理 %s", px.describe())
            else:
                if logger:
                    logger.warning("自定义代理无效，回退直连: %r", proxy)
        except ProxyConfigError as exc:
            if logger:
                logger.warning("自定义代理无效，回退直连: %s", exc)
    elif mode == EGRESS_SYSTEM:
        if apply_system_proxy(conn, bare_host):
            if logger:
                logger.info("出口: 跟随系统代理")
    return conn
