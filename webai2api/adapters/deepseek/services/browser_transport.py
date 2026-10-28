"""deepseek 上游的浏览器指纹传输层（curl_cffi）。

## 为什么需要

`_urlopen` 走的是 Python `http.client`，TLS 握手指纹是 CPython 的（JA3/JA4
与任何真实浏览器都对不上）。请求头伪装再全，传输层也露馅：风控把「头部正常
但 TLS 不是浏览器」当成脚本流量，是账号被批量标记的常见佐证。

curl_cffi 用 BoringSSL 复刻 Chrome 的 ClientHello，握手特征与真实 Chrome
一致，配合一致的请求头，行为签名才自洽。

## 设计约束

1. **绝不引入新的失败面**：curl_cffi 缺失/导入失败/请求异常时，调用方
   `_urlopen` 自动回退到原 `http.client` 路径，行为与改造前完全一致。
2. **只换传输，不换语义**：返回对象的 `status / headers / read() / getcode()`
   与 urllib 响应兼容，上层代码零改动。
3. **响应体一次性读尽**：与原有 `_BufferedResponse` 一致，连接归还池。
4. **错误兼容**：HTTP >= 400 抛 `urllib.error.HTTPError`，传输异常抛
   `urllib.error.URLError`，上层重试逻辑原样生效。
"""

from __future__ import annotations

import io
import logging
import threading
import urllib.error

logger = logging.getLogger("deepseek2api.browser_transport")

# 目标浏览器指纹版本。**必须与 deepseek_auth.DEFAULT_USER_AGENT 同家族**，
# 否则「Firefox UA + Chrome TLS」是比纯脚本更强的识别特征。
# 当前 UA 伪装 Firefox，故拟 Firefox 指纹。
DEFAULT_IMPERSONATE = "firefox133"

_CURL_AVAILABLE = False
try:  # pragma: no cover - 环境相关
    from curl_cffi import requests as _curl_requests

    _CURL_AVAILABLE = True
except Exception:  # noqa: BLE001
    _CURL_AVAILABLE = False


def available() -> bool:
    """curl_cffi 是否可用（不可用则调用方回退 http.client）。"""
    return _CURL_AVAILABLE


_SYSTEM = ""
try:  # pragma: no cover
    import sys as _sys

    _SYSTEM = _sys.platform
except Exception:  # noqa: BLE001
    pass

# libcurl CURLSSLOPT_NATIVE_CA == 0x10
_SSL_NATIVE_CA = 0x10


def _is_windows() -> bool:
    return _SYSTEM.startswith("win")


_CA_PATH_CACHE: str | None = None
_CA_LOCK = threading.Lock()


def _system_ca_bundle() -> str:
    """导出 Windows 系统根证书为 PEM，返回路径（失败返回空串）。

    背景：本机 libcurl 加载 certifi 的 cacert.pem 会报 curl(77)
    "error adding trust anchors"，但读系统证书库导出的 PEM 正常。
    导出结果缓存到临时目录，进程内只做一次。
    """
    global _CA_PATH_CACHE
    if _CA_PATH_CACHE is not None:
        return _CA_PATH_CACHE
    with _CA_LOCK:
        if _CA_PATH_CACHE is not None:
            return _CA_PATH_CACHE
        _CA_PATH_CACHE = ""
        if not _is_windows():
            return _CA_PATH_CACHE
        try:
            import base64
            import ssl as _ssl
            import tempfile
            from pathlib import Path

            chunks: list[str] = []
            for store in ("ROOT", "CA"):
                try:
                    certs = _ssl.enum_certificates(store)
                except Exception:  # noqa: BLE001
                    continue
                for der, enc, _trust in certs:
                    if enc != "x509_asn":
                        continue
                    b = base64.b64encode(der).decode("ascii")
                    body = "\n".join(b[i : i + 64] for i in range(0, len(b), 64))
                    chunks.append(
                        "-----BEGIN CERTIFICATE-----\n"
                        f"{body}\n"
                        "-----END CERTIFICATE-----\n"
                    )
            if not chunks:
                return _CA_PATH_CACHE
            path = Path(tempfile.gettempdir()) / "webai2api_sys_ca.pem"
            path.write_text("".join(chunks), encoding="ascii")
            _CA_PATH_CACHE = str(path)
        except Exception:  # noqa: BLE001
            _CA_PATH_CACHE = ""
    return _CA_PATH_CACHE


class _BufferedResponse:
    """与 urllib 响应兼容的已缓冲响应（只读）。"""

    def __init__(self, url: str, status: int, reason: str, headers, body: bytes):
        self.url = url
        self.status = status
        self.reason = reason
        self.headers = headers
        self._body = body

    def read(self, *args):
        return self._body

    def getcode(self):
        return self.status

    def geturl(self):
        return self.url

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class BrowserSession:
    """按代理配置缓存 curl_cffi Session（复用连接池）。

    线程安全：Session 构造受锁保护；curl_cffi Session 本身可在多线程
    并发使用（内部连接池）。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, object] = {}

    def _session_for(self, proxy: str):
        key = proxy or ""
        with self._lock:
            sess = self._sessions.get(key)
            if sess is None:
                kwargs: dict = {"impersonate": DEFAULT_IMPERSONATE}
                if proxy:
                    kwargs["proxies"] = {"http": proxy, "https": proxy}
                sess = _curl_requests.Session(**kwargs)
                # 本机 libcurl 读不了 certifi 的 PEM（curl 77），改用系统
                # 证书库导出的 PEM；导出失败则保持默认（会触发回退）。
                ca = _system_ca_bundle()
                if ca:
                    sess.verify = ca
                self._sessions[key] = sess
            return sess

    def request(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: dict | None = None,
        data: bytes | None = None,
        timeout: int = 15,
        proxy: str = "",
    ):
        """发出请求并返回 urllib 兼容的缓冲响应。

        HTTP >= 400 抛 HTTPError（与原 _urlopen 一致），传输层异常抛
        URLError。
        """
        try:
            sess = self._session_for(proxy)
            resp = sess.request(
                method=method,
                url=url,
                headers=headers or {},
                data=data,
                timeout=timeout or 15,
            )
            status = int(resp.status_code)
            raw = resp.content or b""
            resp_headers = resp.headers
        except urllib.error.HTTPError:
            raise
        except Exception as exc:  # noqa: BLE001  统一成 URLError，上层重试逻辑不变
            raise urllib.error.URLError(str(exc)) from exc

        reason = ""
        try:
            reason = resp.reason or ""
        except Exception:  # noqa: BLE001
            pass
        if status >= 400:
            raise urllib.error.HTTPError(
                url, status, reason, resp_headers, io.BytesIO(raw)
            )
        return _BufferedResponse(url, status, reason, resp_headers, raw)

    def close(self) -> None:
        with self._lock:
            for sess in self._sessions.values():
                try:
                    sess.close()
                except Exception:  # noqa: BLE001
                    pass
            self._sessions.clear()


def resolve_proxy(egress: str, proxy_url: str) -> tuple[str, bool]:
    """把账号 egress 配置翻译成 curl_cffi 的代理参数。

    返回 (proxy, fallback)。fallback=True 表示情况复杂（跟随系统代理等），
    交由调用方回退到原有 http.client 路径，避免两套代理逻辑打架。

    - direct      → ("", False)      显式直连
    - custom      → (proxy_url, False) 自定义代理
    - system      → 若系统未配代理则直连；配了则 fallback=True
    """
    mode = (egress or "system").strip().lower()
    if mode == "direct":
        return "", False
    if mode == "custom":
        url = (proxy_url or "").strip()
        if url:
            return url, False
        return "", False
    # system：只在系统确实没有代理时接管，否则交回原逻辑
    try:
        from urllib.request import getproxies

        proxies = getproxies() or {}
        https_proxy = proxies.get("https") or proxies.get("http") or ""
    except Exception:  # noqa: BLE001
        https_proxy = ""
    if https_proxy:
        return "", True
    return "", False
