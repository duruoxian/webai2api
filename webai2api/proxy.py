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
from urllib.parse import quote, unquote, urlsplit


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

    def url_for_urllib(self) -> str:
        """urllib ProxyHandler 使用的代理 URL（含认证）。"""
        auth = ""
        if self.username:
            user = quote(self.username, safe="")
            pwd = quote(self.password, safe="")
            auth = f"{user}:{pwd}@"
        return f"{self.scheme}://{auth}{self.host}:{self.port}"

    # ── 应用到连接 ─────────────────────────────────────
    def apply(self, conn: http.client.HTTPConnection, target_host: str,
              target_port: int = 443, timeout: int | None = None) -> None:
        """让现有连接走代理（必须在首次 request 之前调用）。

        http.client 是「延迟连接」的：构造时并不建连，直到 request() 才按
        conn.host/port 去连。所以正确姿势是——把连接本身改指到代理服务器，
        再用 set_tunnel 声明要隧道到的真实目标；二者缺一，CONNECT 会被发到
        真实主机（被 TUN/Clash 以 400 拒绝），或直接明文打到代理。

        http 代理 + HTTPS 目标 → set_tunnel（CONNECT）。
        https 代理 → 手工三步：与代理建 TLS → 在 TLS 里发 CONNECT →
        （https 目标）再对目标包一层 TLS。stdlib 的 set_tunnel 流程是
        「明文 CONNECT 再 wrap」，套不到 https 代理上。
        socks → 需要 PySocks；socks 建连后对 https 目标当场 wrap TLS。
        """
        if self.scheme.startswith("socks"):
            self._apply_socks(conn, target_host, target_port)
            return
        if self.scheme == "https":
            self._apply_https(conn, target_host, target_port, timeout)
            return
        # 关键：连接指向代理，而非目标主机
        conn.host = self.host
        conn.port = self.port
        if timeout is not None:
            conn.timeout = max(int(timeout), 1)
        headers = {}
        auth = self.auth_header()
        if auth:
            headers["Proxy-Authorization"] = auth
        conn.set_tunnel(target_host, target_port, headers=headers)

    def _apply_https(self, conn: http.client.HTTPConnection,
                     target_host: str, target_port: int,
                     timeout: int | None = None) -> None:
        """https 代理：先与代理建立 TLS，再在 TLS 内发 CONNECT 隧道。

        sock 预先赋给 conn 后 http.client 的 send() 会跳过 connect()，
        与 socks 路径同一机制；https 目标在此对隧道再包一层目标 TLS
        （SNI=目标主机名）。
        """
        import socket as _socket
        import ssl as _ssl

        to = timeout if timeout is not None else getattr(conn, "timeout", None)
        to = to if isinstance(to, (int, float)) else None
        raw = _socket.create_connection((self.host, self.port), timeout=to)
        if to is not None:
            raw.settimeout(to)
        sock = _ssl.create_default_context().wrap_socket(
            raw, server_hostname=self.host)
        headers = {"Host": f"{self.host}:{self.port}",
                   "Proxy-Connection": "keep-alive"}
        auth = self.auth_header()
        if auth:
            headers["Proxy-Authorization"] = auth
        req = (
            f"CONNECT {target_host}:{target_port} HTTP/1.1\r\n"
            + "".join(f"{k}: {v}\r\n" for k, v in headers.items())
            + "\r\n"
        )
        sock.sendall(req.encode("latin-1"))
        resp = b""
        while b"\r\n\r\n" not in resp:
            chunk = sock.recv(4096)
            if not chunk:
                sock.close()
                raise ProxyConfigError("https 代理 CONNECT 无响应")
            resp += chunk
            if len(resp) > 65536:
                sock.close()
                raise ProxyConfigError("https 代理 CONNECT 响应异常")
        status_line = resp.split(b"\r\n", 1)[0].decode("latin-1", "replace")
        parts = status_line.split()
        if len(parts) < 2 or not parts[1].startswith("2"):
            sock.close()
            raise ProxyConfigError(f"https 代理 CONNECT 被拒: {status_line[:120]}")
        if target_port == 443:
            conn.sock = _ssl.create_default_context().wrap_socket(
                sock, server_hostname=target_host)
        else:
            # http 目标：直接用明文隧道
            conn.sock = sock

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
        # 裸构造的连接 timeout 是 stdlib 哨兵对象（socket._GLOBAL_DEFAULT_TIMEOUT），
        # 直接 settimeout 会 TypeError；生产路径恒为数值
        to = getattr(conn, "timeout", None)
        if isinstance(to, (int, float)):
            conn.sock.settimeout(to)
        conn.sock.connect((target_host, target_port))
        # https 目标必须在这里立刻套 TLS：sock 预先赋值后，http.client 的
        # send() 会因 self.sock 非空跳过 connect()，而 HTTPSConnection 的
        # TLS 握手恰恰只在 connect() 里做——不 wrap 的话请求明文打 443
        # （2026-11-02 审查，stdlib 源码实锤；SNI 用真实目标主机名）。
        if isinstance(conn, http.client.HTTPSConnection):
            server_hostname = (target_host or conn.host or "").split(":")[0]
            conn.sock = conn._context.wrap_socket(
                conn.sock, server_hostname=server_hostname)


def parse_proxy(value: str) -> Proxy | None:
    """安全解析：非法配置返回 None（调用方回退直连）。"""
    try:
        return Proxy.parse(value)
    except ProxyConfigError:
        return None
