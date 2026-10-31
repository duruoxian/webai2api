from __future__ import annotations

import gzip
import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from logging import Logger

from ..config import AppConfig
from ..logging_utils import debug_dump
from .cookie_jar import CookieJar


@dataclass(slots=True)
class HIFTokens:
    leim: str
    dliq: str
    fetched_at: float
    ttl_seconds: int = 600

    @property
    def is_expired(self) -> bool:
        return time.time() > self.fetched_at + self.ttl_seconds - 30


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64; rv:156.0) Gecko/20100101 Firefox/156.0"
)


class DeepSeekAuthManager:
    def __init__(self, config: AppConfig, logger: Logger) -> None:
        self.config = config
        self.logger = logger
        self._lock = threading.RLock()
        self._hif_tokens: HIFTokens | None = None
        # 动态 cookie jar（承载 HWWAFSESID / ds_session_id 等的自动更新）
        self.cookie_jar = CookieJar(seed_cookie=config.cookie, logger=logger)

    def get_browser_headers(self) -> dict[str, str]:
        return {
            "Accept": "*/*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
            "User-Agent": (
                getattr(self.config, "user_agent", "") or DEFAULT_USER_AGENT
            ),
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Origin": "https://chat.deepseek.com",
            "Referer": "https://chat.deepseek.com/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
        }

    def get_auth_headers(self) -> dict[str, str]:
        headers = {
            **self.get_browser_headers(),
            "Authorization": f"Bearer {self.config.auth_token}",
            "x-device-id": self.config.device_id,
            "x-device-model": getattr(self.config, "device_model", "") or "",
            "x-client-bundle-id": "com.deepseek.chat",
            "x-client-platform": "web",
            "x-client-version": "2.5.0",
            "x-client-locale": "zh_CN",
            "x-client-timezone-offset": "28800",
        }
        if self.config.settings_token:
            headers["x-settings-token"] = self.config.settings_token
        # 关键：动态构造 Cookie 头
        cookie_header = self.cookie_jar.build_cookie_header()
        if cookie_header:
            headers["Cookie"] = cookie_header
        return headers

    # ── 每次请求后调用，将 Set-Cookie 合并回 jar ──
    def absorb_response_cookies(self, response) -> None:
        try:
            headers = response.headers
            # http.client 的 HTTPMessage 用 get_all；curl_cffi 的 Headers
            # 只有 get_list——主路径装了 curl_cffi 时这里此前直接 AttributeError
            # 被吞掉，WAF 轮换 cookie 一条都收不回（2026-11-01 审计）
            getter = getattr(headers, "get_all", None) or getattr(
                headers, "get_list", None
            )
            set_cookies = (getter("Set-Cookie") if getter else None) or []
            if set_cookies:
                self.cookie_jar.update_from_set_cookie(set_cookies)
        except Exception as exc:
            self.logger.debug("吸收 Set-Cookie 失败: %s", exc)

    # ── HIF 获取：正确解析 value 并吸收 Set-Cookie ──
    def _open_hif_url(self, url: str, timeout: int = 10):
        """HIF 请求按账号 egress 发出：裸 urlopen 会从本机直连 IP 发请求，
        与走代理的 chat 请求制造「登录 IP 与请求 IP 分裂」——egress.py 注释
        明确说平台校验一致性；egress=direct 时语义也被破坏（2026-11-01 审计）。
        与 file_uploader 2026-10-30 的修复同族。"""
        from urllib.parse import urlsplit

        from ....egress import egress_opener

        egress = str(getattr(self.config, "egress", "system") or "system")
        proxy_url = str(getattr(self.config, "proxy", "") or "")
        host = urlsplit(url).hostname or ""
        req = urllib.request.Request(
            url, method="GET", headers=self.get_auth_headers()
        )
        opener = egress_opener(egress, proxy_url, host)
        return opener.open(req, timeout=timeout)

    def _fetch_hif_tokens(self) -> HIFTokens:
        self.logger.info("正在获取 HIF tokens...")

        leim_value = ""
        dliq_value = ""

        # x-hif-leim
        try:
            with self._open_hif_url(self.config.hif_leim_url) as response:
                self.absorb_response_cookies(response)
                raw = response.read().decode("utf-8", errors="replace")
                payload = json.loads(raw)
                leim_value = payload.get("data", {}).get("biz_data", {}).get("value", "")
                if not leim_value:
                    raise RuntimeError(f"响应中未找到 value 字段: {raw[:200]}")
                self.logger.debug("获取 x-hif-leim 成功: %s", leim_value[:50])
        except Exception as exc:
            self.logger.warning("获取 x-hif-leim 失败: %s", exc)
            leim_value = ""

        # x-hif-dliq
        try:
            with self._open_hif_url(self.config.hif_dliq_url) as response:
                self.absorb_response_cookies(response)
                raw = response.read().decode("utf-8", errors="replace")
                payload = json.loads(raw)
                dliq_value = payload.get("data", {}).get("biz_data", {}).get("value", "")
                if not dliq_value:
                    raise RuntimeError(f"响应中未找到 value 字段: {raw[:200]}")
                self.logger.debug("获取 x-hif-dliq 成功: %s", dliq_value[:50])
        except Exception as exc:
            self.logger.warning("获取 x-hif-dliq 失败: %s", exc)
            dliq_value = ""

        tokens = HIFTokens(leim=leim_value, dliq=dliq_value, fetched_at=time.time())
        # 完整结果才进缓存。部分失败（只取到一个）此前被缓存 600s，期间所有
        # 请求带空 HIF 头；且失败线程的赋值会覆盖另一线程刚取到的有效值
        # （快路径竞态，2026-11-01 审计）。部分/全空：只返回给本次请求用，
        # 缓存保持原状（空/过期 → 下次调用立即重试）。
        if leim_value and dliq_value:
            self._hif_tokens = tokens
        self.logger.info(
            "HIF tokens 获取完成 (leim=%s dliq=%s)", bool(leim_value), bool(dliq_value)
        )
        return tokens

    def get_hif_tokens(self) -> HIFTokens:
        # 快路径无锁读；两次网络调用（各 10s 超时）不持锁——此前持 RLock
        # 取 token，并发请求最长被串行阻塞 ~20s（2026-10-30 审计 DS-P3）
        cached = self._hif_tokens
        if cached and not cached.is_expired:
            return cached
        with self._lock:
            # 双重检查：另一个线程可能已取到新 token
            cached = self._hif_tokens
            if cached and not cached.is_expired:
                return cached
        tokens = self._fetch_hif_tokens()
        if tokens.leim or tokens.dliq:
            return tokens
        return HIFTokens(leim="", dliq="", fetched_at=0.0)

    def build_full_headers(
        self,
        pow_response: str,              # 已经是 Base64 编码的完整 header 值
        content_type: str = "application/json",
    ) -> dict[str, str]:
        """构建包含所有必要信息的完整请求头。"""
        hif = self.get_hif_tokens()

        headers = {
            **self.get_auth_headers(),
            "Content-Type": content_type,
            "x-hif-leim": hif.leim,
            "x-hif-dliq": hif.dliq,
            "x-ds-pow-response": pow_response,   # 直接使用，不再处理
        }
        return headers

    def _read_json_response(self, response) -> dict[str, object]:
        try:
            raw_body = response.read()
            content_encoding = response.headers.get("Content-Encoding", "").lower()

            if content_encoding == "gzip":
                raw_body = gzip.decompress(raw_body)
            elif content_encoding == "deflate":
                import zlib
                try:
                    raw_body = zlib.decompress(raw_body)
                except zlib.error:
                    raw_body = zlib.decompress(raw_body, -zlib.MAX_WBITS)
            elif content_encoding == "br":
                try:
                    import brotli
                    raw_body = brotli.decompress(raw_body)
                except ImportError:
                    self.logger.warning("收到 Brotli 压缩响应但未安装 brotli 库")

            debug_dump(self.logger, self.config.debug_dump_all, "DeepSeek 原始 JSON 响应体", raw_body)

            try:
                text = raw_body.decode("utf-8")
            except UnicodeDecodeError:
                text = raw_body.decode("utf-8", errors="replace")
                self.logger.warning("DeepSeek 响应包含非 UTF-8 字节，已使用 replace 模式解码")

            payload = json.loads(text)

        except gzip.BadGzipFile as exc:
            raise RuntimeError("DeepSeek 响应 gzip 解压失败") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"DeepSeek 响应不是合法 JSON: {exc}") from exc

        if not isinstance(payload, dict):
            raise RuntimeError(f"DeepSeek 响应格式异常，期望 JSON 对象，实际是: {type(payload).__name__}")
        return payload