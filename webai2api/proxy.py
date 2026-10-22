"""HTTP/HTTPS 代理支持（每账号可配置独立出口 IP）。

支持的写法：
    http://host:port          普通 HTTP 代理（HTTPS 走 CONNECT 隧道）
    http://user:pass@host:port
    socks5://host:port        （需要 PySocks；未安装时抛 ProxyConfigError）
    host:port                 等价 http://host:port

设计原则：解析失败只记警告并回退直连，不让代理配置错误拖垮整个账号。
"""
from __future__ import annotations

import base64
import http.client
from urllib.parse import unquote, urlsplit


class ProxyConfigError(Exception):
    pass


class Proxy:
    __slots__ = ("scheme", "host", "port", "username", "password", "raw")

    def __init__(self, scheme: str, host: str, port: int,
                 username: str = "", password: str = "", raw: str = "") -> None:
        self.scheme = scheme
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.raw = raw

    # ── 构造 / 解析 ────────────────────────────────────
    @classmethod
    def parse(cls, value: str) -> "Proxy | None":
        text = (value or "").strip()
        if not text:
            return None
        if "://" not in text:
            text = "http://" + text
        try:
            parts = urlsplit(text)
            scheme = (parts.scheme or "http").lower()
            host = parts.hostname or ""
            port = int(parts.port or (1080 if scheme.startswith("socks") else 8080))
        except Exception as exc:  # noqa: BLE001
            raise ProxyConfigError(f"代理地址无法解析: {value!r} ({exc})") from exc
        if not host:
            raise ProxyConfigError(f"代理地址缺少主机名: {value!r}")
        if scheme not in ("http", "https", "socks5", "socks5h", "socks4"):
            raise ProxyConfigError(f"不支持的代理协议: {scheme}（支持 http/https/socks5）")
        return cls(
            scheme=scheme, host=host, port=port,
            username=unquote(parts.username or ""),
            password=unquote(parts.password or ""),
            raw=text,
        )

    # ── 描述（日志/面板用，隐藏密码）────────────────────
    def describe(self) -> str:
        auth = f"{self.username}:***@" if self.username else ""
        return f"{self.scheme}://{auth}{self.host}:{self.port}"

    def auth_header(self) -> str:
        if not self.username:
            return ""
        raw = f"{self.username}:{self.password}".encode()
        return "Basic " + base64.b64encode(raw).decode()

    # ── 应用到连接 ─────────────────────────────────────
    def apply(self, conn: http.client.HTTPConnection, target_host: str,
              target_port: int = 443, timeout: int | None = None) -> None:
        """让现有连接走代理。

        http 代理 + HTTPS 目标 → set_tunnel（CONNECT）。
        https 代理 → 需要 TLS 套壳，本实现按 http 处理（多数供应商兼容）。
        socks → 需要 PySocks；用其 monkeypatch 方式包装 socket。
        """
        if self.scheme.startswith("socks"):
            self._apply_socks(conn, target_host, target_port)
            return
        headers = {}
        auth = self.auth_header()
        if auth:
            headers["Proxy-Authorization"] = auth
        conn.set_tunnel(target_host, target_port, headers=headers)

    def _apply_socks(self, conn: http.client.HTTPConnection,
                     target_host: str, target_port: int) -> None:
        try:
            import socks  # PySocks
        except Exception as exc:  # noqa: BLE001
            raise ProxyConfigError(
                "SOCKS 代理需要安装 PySocks（pip install PySocks）"
            ) from exc
        stype = socks.SOCKS4 if self.scheme == "socks4" else socks.SOCKS5
        conn.sock = socks.socksocket()
        conn.sock.set_proxy(
            stype, self.host, self.port, rdns=True,
            username=self.username or None, password=self.password or None,
        )
        conn.sock.settimeout(conn.timeout)
        conn.sock.connect((target_host, target_port))


def parse_proxy(value: str) -> Proxy | None:
    """安全解析：非法配置返回 None（调用方回退直连）。"""
    try:
        return Proxy.parse(value)
    except ProxyConfigError:
        return None
