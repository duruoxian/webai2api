"""
DeepSeek Web API 客户端。

修复要点：
- 上游错误 → 正确的 HTTP 状态码
- usage 使用 DeepSeek 返回的 accumulated_token_usage
- cache key 包含 system + model + tool_calls 完整结构
- 每个请求独立 StreamState
- reasoning_content 输出
- context_length_exceeded → 清理缓存+会话，使用完整历史重开新对话
- rate_limit_reached → 冷却 15 秒后自动重试（最多 3 次）
- 工具调用：哨兵协议 + 流式检测器 + OpenAI 格式输出
"""

from __future__ import annotations

import codecs
import gzip
import http.client
import json
import math
import socket
import threading
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
import random
import zlib
from dataclasses import dataclass
from logging import Logger
from typing import Callable

from ..config import AppConfig
from ..logging_utils import debug_dump
from .deepseek_auth import DeepSeekAuthManager
from . import browser_transport
from .pow_solver import PoWSolver
from .session_cache import SessionCache, SessionState, conv_fingerprint, prefix_hash
from .tool_calling import (
    ParsedToolCall,
    StreamingToolDetector,
    TOOL_CLOSE_PLURAL,
    TOOL_CLOSE_SINGULAR,
    TOOL_OPEN_PLURAL,
    TOOL_OPEN_SINGULAR,
    configure_tool_parsing,
    parse_tool_calls,
)
from .file_uploader import FileUploader
from .translator import (
    StreamState,
    convert_openai_to_deepseek,
    convert_openai_to_deepseek_incremental,
    convert_deepseek_to_openai_chunk,
    extract_images_from_messages,
)
from .citation_utils import CitationStripper, strip_citation_marks

# ── 重试策略 ────────────────────────────────────────────────────

RATE_LIMIT_COOLDOWN_SECONDS = 60
MAX_RATE_LIMIT_RETRIES = 3
MAX_CONTEXT_LENGTH_RETRIES = 2
MAX_FILE_PARSE_RETRIES = 1

# ── 断流自动续写 ────────────────────────────────────
# 上游偶尔在长思考/长回答中途掐流且不发 finish 帧。检测到断流后，向
# 同一会话发一条「继续」（parent_message_id 指向截断消息，网页版
# 「继续生成」同款机制），续写增量拼进同一个 SSE 流。最多续写 2 次；
# 每次续写都是一轮新的 completion PoW + 请求（有日志可查）。
MAX_AUTO_CONTINUATIONS = 2
CONTINUE_PROMPT = "继续"


def _drop_trailing_partial_markup(text: str) -> str:
    """非流式续写前清场：截掉尾部未闭合的 DSML/哨兵 opener 残块。

    续写轮会重发完整调用块；残块留在文本里会让闭合块的正则匹配吞进
    垃圾，导致整个调用解析失败。"""
    if not text:
        return text
    cut = len(text)
    for opener, closer in (
        (TOOL_OPEN_PLURAL, TOOL_CLOSE_PLURAL),
        (TOOL_OPEN_SINGULAR, TOOL_CLOSE_SINGULAR),
    ):
        pos = text.rfind(opener)
        if pos >= 0 and closer not in text[pos:]:
            cut = min(cut, pos)
    return text[:cut] if cut < len(text) else text


class UpstreamAPIError(RuntimeError):
    def __init__(
        self, status_code: int, message: str, payload: dict[str, object] | None = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload or {}


# 账号封禁/限制类 hint 关键词。命中即熔断：该账号长冷却，绝不重试、绝不续写
# （对着已封账号继续打请求只会加重风控记录）。
_BANNED_HINT_KEYWORDS = (
    "封禁",
    "被封",
    "封号",
    "封停",
    "账号已被限制",
    "账号异常",
    "永久限制",
    "banned",
    "suspended",
)


def _hint_is_banned(finish_reason: object, content: object) -> bool:
    low = f"{finish_reason or ''} {content or ''}".lower()
    return any(k in low for k in _BANNED_HINT_KEYWORDS)


class AccountBannedError(UpstreamAPIError):
    """上游明确提示账号被限制/封禁（区别于限流与内容审核）。"""

    def __init__(self, message: str) -> None:
        super().__init__(status_code=403, message=message)


class QueueTimeoutError(RuntimeError):
    pass


class ContextLengthExceededError(RuntimeError):
    """DeepSeek 返回 context_length_exceeded（对话长度上限）。"""


class RateLimitError(RuntimeError):
    """DeepSeek 返回 rate_limit_reached（消息发送过于频繁）。"""


class FileParseError(RuntimeError):
    """DeepSeek 返回 file_parse_error（上传的文件解析失败，消息被拒）。"""


@dataclass(slots=True)
class QueueLease:
    ticket: int
    release_callback: Callable[[int], None]
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.release_callback(self.ticket)


class ConcurrentRequestQueue:
    def __init__(self, logger: Logger, wait_timeout: int, max_concurrency: int) -> None:
        self.logger = logger
        self.wait_timeout = wait_timeout
        self.max_concurrency = max(1, max_concurrency)
        self._condition = threading.Condition()
        self._next_ticket = 0
        self._serving_ticket = 0
        self._released_tickets: set[int] = set()

    def acquire(self, request_name: str) -> QueueLease:
        with self._condition:
            ticket = self._next_ticket
            self._next_ticket += 1
            queue_ahead = max(
                0, ticket - (self._serving_ticket + self.max_concurrency) + 1
            )
            start = time.monotonic()

            if queue_ahead > 0:
                self.logger.info(
                    "请求进入队列 ticket=%s ahead=%s request=%s",
                    ticket,
                    queue_ahead,
                    request_name,
                )

            while ticket >= self._serving_ticket + self.max_concurrency:
                remaining = self.wait_timeout - (time.monotonic() - start)
                if remaining <= 0:
                    ahead = ticket - (self._serving_ticket + self.max_concurrency) + 1
                    # 关键修复：超时必须把 ticket 标记为已跳过，否则
                    # _serving_ticket 永远卡在这里，后续所有请求永久饿死。
                    self._skip(ticket)
                    raise QueueTimeoutError(f"队列等待超时，前方仍有 {ahead} 个请求。")
                self._condition.wait(timeout=remaining)

            active_slots = ticket - self._serving_ticket + 1
            self.logger.info(
                "请求获得执行槽位 ticket=%s active=%s/%s request=%s",
                ticket,
                active_slots,
                self.max_concurrency,
                request_name,
            )
            return QueueLease(ticket=ticket, release_callback=self._release)

    def pending(self) -> int:
        """已取号未释放的请求数（排队+在服务），作为选号负载均衡水位。"""
        with self._condition:
            return max(0, self._next_ticket - self._serving_ticket)

    def _release(self, ticket: int) -> None:
        with self._condition:
            self._released_tickets.add(ticket)
            self._advance_locked()
            self.logger.info("请求离开执行槽位 ticket=%s", ticket)

    def _skip(self, ticket: int) -> None:
        """acquire 等待超时时调用：把无人认领的 ticket 视为已释放。

        若不跳过，_serving_ticket 将永远停在这个 ticket 上，
        后续所有请求都无法获得槽位（队列永久饿死）。
        """
        with self._condition:
            self._released_tickets.add(ticket)
            self._advance_locked()
            self.logger.warning("队列 ticket=%s 等待超时，已跳过该序号", ticket)

    def _advance_locked(self) -> None:
        """推进 _serving_ticket 越过所有已释放的 ticket。调用方须已持有 _condition。"""
        while self._serving_ticket in self._released_tickets:
            self._released_tickets.remove(self._serving_ticket)
            self._serving_ticket += 1
        self._condition.notify_all()


# ═══════════════════════════════════════════════════════════
# Conversation helpers
# ═══════════════════════════════════════════════════════════


def _strip_image_messages(payload: dict) -> dict:
    """返回去掉所有 image_url 的多模态内容后的 payload 副本（纯文本降级用）。"""
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return payload
    new_messages = []
    for m in messages:
        if not isinstance(m, dict):
            new_messages.append(m)
            continue
        content = m.get("content")
        if not isinstance(content, list):
            new_messages.append(m)
            continue
        kept = [
            p
            for p in content
            if not (isinstance(p, dict) and p.get("type") == "image_url")
        ]
        if not kept:
            kept = [{"type": "text", "text": "[图片已忽略：上游解析失败]"}]
        new_m = dict(m)
        new_m["content"] = kept
        new_messages.append(new_m)
    new_payload = dict(payload)
    new_payload["messages"] = new_messages
    return new_payload


def _split_messages(messages: list[dict]) -> tuple[list[dict], dict]:
    """
    返回 (prefix_messages, last_message)。

    - prefix_messages：除最后一条外的所有消息（保留 system / tool_calls / tool_call_id
      等完整结构，用于 cache key 计算）
    - last_message：最后一条消息的完整 dict

    采用「除最后一条外全部为前缀」的语义，可保证：
      Turn 1 请求 messages=[user1] → prefix=[]，完成后存 hash([user1, assistant1])
      Turn 2 请求 messages=[user1, assistant1, user2] → prefix=[user1, assistant1]，命中
    """
    raw: list[dict] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        entry: dict = {"role": m.get("role", "user"), "content": ""}

        content = m.get("content")
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict) and p.get("type") == "text":
                    parts.append(str(p.get("text", "")))
                elif isinstance(p, str):
                    parts.append(p)
            entry["content"] = "\n".join(parts)
        elif content is not None:
            entry["content"] = str(content)

        if m.get("role") == "assistant" and isinstance(m.get("tool_calls"), list):
            entry["tool_calls"] = m["tool_calls"]
        if m.get("role") == "tool" and m.get("tool_call_id"):
            entry["tool_call_id"] = m["tool_call_id"]

        raw.append(entry)

    if not raw:
        return [], {"role": "user", "content": ""}
    if len(raw) == 1:
        return [], raw[0]
    return raw[:-1], raw[-1]


def _payload_has_tools(payload: dict) -> bool:
    tools = payload.get("tools")
    if not isinstance(tools, list) or not tools:
        return False
    return payload.get("tool_choice", "auto") != "none"


def _estimate_tokens(text: str) -> int:
    """估算 token：中文约 1.5 字/token，其余约 4 字符/token。

    DeepSeek 网页版上游不返回真实 token 数，只能估算。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return max(1, int(cjk / 1.5 + other / 4))


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _make_content_chunk(
    text: str, model: str, completion_id: str, role: str | None = None
) -> str:
    delta: dict = {"content": text}
    if role:
        delta["role"] = role
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "system_fingerprint": "",
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": None,
                "logprobs": None,
            }
        ],
    }
    return _sse(chunk)


def _make_tool_call_chunks(
    calls: list[ParsedToolCall],
    model: str,
    completion_id: str,
    base_index: int = 0,
) -> list[str]:
    """按 OpenAI 流式规范：id+name 一个 chunk，arguments 一个 chunk。

    base_index：本轮之前**已经发出**的工具调用条数。流式检测器按
    `</invoke>` 逐条切块（一次 action 只含一条调用），若每条 action 都从
    index=0 起编，客户端会把后续调用的 arguments 全部追加到第 0 个槽位，
    表现为「N 个调用的 JSON 首尾相接塞进一个调用」。必须全局递增。
    """
    chunks: list[str] = []
    for i, call in enumerate(calls):
        idx = base_index + i
        cid = completion_id
        created = int(time.time())

        head = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "system_fingerprint": "",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": idx,
                                "id": call.id,
                                "type": "function",
                                "function": {"name": call.name, "arguments": ""},
                            }
                        ],
                    },
                    "finish_reason": None,
                    "logprobs": None,
                }
            ],
        }
        chunks.append(_sse(head))

        args = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "system_fingerprint": "",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": idx,
                                "function": {"arguments": call.arguments},
                            }
                        ],
                    },
                    "finish_reason": None,
                    "logprobs": None,
                }
            ],
        }
        chunks.append(_sse(args))
    return chunks


def _make_finish_chunk(finish_reason: str, model: str, completion_id: str) -> str:
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "system_fingerprint": "",
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
    }
    return _sse(chunk)


class DeepSeekWebClient:
    """DeepSeek Web API 客户端。"""

    def __init__(self, config: AppConfig, logger: Logger) -> None:
        self.config = config
        self.logger = logger
        # 可复用的上游 HTTPS 连接（keep-alive，省去重复 TLS 握手）
        self._http_conn = None
        self._http_host = None
        self._http_proxy = ""
        # 浏览器指纹传输会话（curl_cffi，惰性创建；不可用时为 None）
        self._browser = None
        # 风险指标计数器（进程内，面板展示）
        self._stats: dict[str, int] = {}
        self._stats_lock = threading.Lock()
        # 工具调用解析行为：严格工具名匹配默认开，JSON 碰运气解析默认关。
        configure_tool_parsing(
            strict_names=config.tool_name_strict,
            json_fallback=config.tool_json_fallback,
        )
        self.auth = DeepSeekAuthManager(config=config, logger=logger)
        self.pow_solver = PoWSolver(wasm_path=config.wasm_path, logger=logger)
        self.request_queue = ConcurrentRequestQueue(
            logger=logger,
            wait_timeout=config.queue_wait_timeout,
            max_concurrency=config.max_concurrency,
        )
        self.session_cache = SessionCache(
            max_size=config.session_cache_max_size,
            ttl_seconds=config.session_cache_ttl,
            persist_path=getattr(config, "session_store_path", "") or None,
        )
        self.cache_enabled = config.session_cache_enabled

        self._proxy_sessions: set[str] = set()
        self._proxy_sessions_lock = threading.Lock()
        # 从持久化缓存恢复的 session 也要纳入追踪，否则重启后这些上游会话
        # 永远不会被 cleanup_stale_sessions 回收（内存集合重启即空）。
        try:
            self._proxy_sessions.update(self.session_cache.active_session_ids())
        except Exception:  # noqa: BLE001
            pass
        # 同账号上游请求最小间隔
        self._last_upstream_at: float = 0.0
        self._last_upstream_lock = threading.Lock()

        self.file_uploader = FileUploader(
            config=config,
            auth=self.auth,
            pow_solver=self.pow_solver,
            logger=logger,
        )
        self.file_upload_enabled = getattr(config, "file_upload_enabled", True)

        if self.cache_enabled:
            self.logger.info(
                "会话缓存已启用 max_size=%d ttl=%ds",
                config.session_cache_max_size,
                config.session_cache_ttl,
            )
        else:
            self.logger.info("会话缓存已禁用（delete_conversation=true，每次请求独立）")

        if config.anti_rate_limit_enabled:
            self.logger.info(
                "AntiRateLimit | enabled | delay %.2f~%.2fs",
                config.anti_rate_limit_min_delay,
                config.anti_rate_limit_max_delay,
            )

    # ═══════════════════════════════════════════════════════
    # 可复用的 HTTPS 连接（keep-alive，省去重复 TLS 握手）
    # ═══════════════════════════════════════════════════════

    def _open_with_egress(self, request, timeout: int = 15):
        """按账号 egress 设置发起请求（用于非 keep-alive 的场景）。

        direct 模式强制忽略系统代理，system 跟随系统，custom 用账号代理。
        """
        from urllib.parse import urlsplit

        from ....egress import egress_opener

        egress = str(getattr(self.config, "egress", "system") or "system")
        proxy_url = str(getattr(self.config, "proxy", "") or "")
        host = urlsplit(request.full_url).hostname or ""
        opener = egress_opener(egress, proxy_url, host)
        return opener.open(request, timeout=timeout)

    def _get_http_conn(self, host: str, timeout: int):
        conn = self._http_conn
        egress = str(getattr(self.config, "egress", "system") or "system")
        proxy_url = str(getattr(self.config, "proxy", "") or "")
        if (
            conn is None
            or self._http_host != host
            or self._http_proxy != f"{egress}|{proxy_url}"
        ):
            conn = http.client.HTTPSConnection(host, timeout=max(int(timeout), 30))
            from ....egress import (
                EGRESS_CUSTOM,
                EGRESS_SYSTEM,
                apply_system_proxy,
                normalize_egress,
            )

            mode = normalize_egress(egress)
            if mode == EGRESS_CUSTOM:
                from ....proxy import ProxyConfigError, parse_proxy

                try:
                    px = parse_proxy(proxy_url)
                    if px is not None:
                        px.apply(conn, host)
                        self.logger.info("出口: 自定义代理 %s", px.describe())
                    else:
                        self.logger.warning("自定义代理无效，回退直连: %r", proxy_url)
                except ProxyConfigError as exc:
                    self.logger.warning("自定义代理无效，回退直连: %s", exc)
            elif mode == EGRESS_SYSTEM:
                if apply_system_proxy(conn, host):
                    self.logger.info("出口: 跟随系统代理")
                else:
                    self.logger.info("出口: 系统代理未启用 → 直连")
            else:
                self.logger.info("出口: 直连（已显式忽略系统代理）")
            self._http_conn = conn
            self._http_host = host
            self._http_proxy = f"{egress}|{proxy_url}"
        return conn

    def _browser_request(
        self, url: str, *, method: str, headers: dict, data, timeout: int
    ):
        """浏览器指纹传输（curl_cffi）。返回 None 表示应回退 http.client。

        **任何**异常都返回 None 交给原路径重试：换传输不能引入新的失败面。
        """
        if not browser_transport.available():
            return None
        egress = str(getattr(self.config, "egress", "system") or "system")
        proxy_url = str(getattr(self.config, "proxy", "") or "")
        proxy, complex_mode = browser_transport.resolve_proxy(egress, proxy_url)
        if complex_mode:
            return None
        try:
            if getattr(self, "_browser", None) is None:
                self._browser = browser_transport.BrowserSession()
            return self._browser.request(
                url,
                method=method,
                headers=headers,
                data=data,
                timeout=timeout,
                proxy=proxy,
            )
        except urllib.error.HTTPError:
            # HTTP 层错误是**业务结果**（401/403/429 等），必须原样抛给
            # 上层分类逻辑，不能被当成传输失败回退重发。
            raise
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("浏览器指纹传输失败，回退 http.client: %s", exc)
            return None

    def _drop_http_conn(self) -> None:
        conn = self._http_conn
        self._http_conn = None
        self._http_host = None
        self._http_proxy = ""
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    def _urlopen(self, request: urllib.request.Request, timeout: int = 15):
        """用可复用连接发起请求，读完响应体后返回与 urllib 兼容的响应对象。

        立刻把 body 读干净，连接随即回到空闲态可复用；返回的对象与连接解耦，
        with 退出时不会关闭连接。出错时抛 urllib.error.HTTPError / URLError。

        **传输层优先走 curl_cffi 浏览器指纹**（browser_transport），避免
        「头部像浏览器、TLS 像脚本」的指纹矛盾；不可用/代理模式不确定时
        静默回退到原 http.client 路径，行为与改造前一致。
        """
        import io
        from urllib.parse import urlsplit

        class _BufferedResponse:
            def __init__(self, url, status, reason, headers, body):
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

        url = request.full_url
        parts = urlsplit(url)
        host = parts.hostname or "chat.deepseek.com"
        path = parts.path or "/"
        if parts.query:
            path = f"{path}?{parts.query}"
        method = request.get_method()
        headers = {k: v for k, v in request.header_items()}
        data = request.data

        # ── 传输层：浏览器指纹优先，失败即回退 ──
        resp = self._browser_request(
            url, method=method, headers=headers, data=data, timeout=timeout
        )
        if resp is not None:
            return resp

        last_exc: Exception | None = None
        for _ in range(2):
            conn = self._get_http_conn(host, timeout)
            try:
                conn.request(method, path, body=data, headers=headers)
                resp = conn.getresponse()
                status = resp.status
                reason = resp.reason
                resp_headers = resp.headers
                raw = resp.read()
            except Exception as exc:  # noqa: BLE001  连接层故障 → 丢弃后重连一次
                last_exc = exc
                self._drop_http_conn()
                continue
            if status >= 400:
                self._drop_http_conn()
                raise urllib.error.HTTPError(
                    url, status, reason, resp_headers, io.BytesIO(raw)
                )
            return _BufferedResponse(url, status, reason, resp_headers, raw)
        raise urllib.error.URLError(str(last_exc))

    # ═══════════════════════════════════════════════════════
    # SSE 事件公共处理（流式 / 非流式两条路径共用，避免逻辑分叉）
    # ═══════════════════════════════════════════════════════

    def _handle_hint_event(self, event: dict) -> bool:
        """处理 DeepSeek hint 事件。

        返回 True 表示该事件已消费完毕，调用方应 continue；
        错误类 hint 会直接抛 ContextLengthExceededError / RateLimitError。
        """
        if not (isinstance(event, dict) and event.get("_event_type") == "hint"):
            return False
        if event.get("type") == "error":
            finish_reason = event.get("finish_reason")
            if finish_reason not in (
                "context_length_exceeded",
                "rate_limit_reached",
                "file_parse_error",
            ) and _hint_is_banned(finish_reason, event.get("content")):
                raise AccountBannedError(
                    str(event.get("content") or "DeepSeek: 账号已被限制")
                )
            if finish_reason == "context_length_exceeded":
                raise ContextLengthExceededError("DeepSeek: 达到对话长度上限")
            if finish_reason == "rate_limit_reached":
                raise RateLimitError(
                    event.get("content") or "DeepSeek: 消息发送过于频繁"
                )
            if finish_reason == "file_parse_error":
                raise FileParseError(
                    event.get("content") or "DeepSeek: 文件解析失败，消息未能发送"
                )
        return True

    @staticmethod
    def _extract_ready_message_id(event: dict) -> str | None:
        """ready 事件 → response_message_id；非 ready 事件返回 None。"""
        if isinstance(event, dict) and event.get("_event_type") == "ready":
            resp_id = event.get("response_message_id")
            if resp_id is not None:
                return str(resp_id)
        return None

    @staticmethod
    def _extract_batch_token_usage(event: dict) -> int | None:
        """BATCH 事件 → accumulated_token_usage；非 BATCH 事件返回 None。"""
        if isinstance(event, dict) and event.get("o") == "BATCH":
            batch_v = event.get("v")
            if isinstance(batch_v, list):
                for item in batch_v:
                    if (
                        isinstance(item, dict)
                        and item.get("p") == "accumulated_token_usage"
                    ):
                        try:
                            return int(item.get("v", 0))
                        except (TypeError, ValueError):
                            return None
        return None

    # ═══════════════════════════════════════════════════════
    # Conversation resolution
    # ═══════════════════════════════════════════════════════

    def _resolve_conversation(self, payload: dict[str, object]) -> dict:
        messages = payload.get("messages", [])
        if not isinstance(messages, list):
            messages = []

        # ★ 先上传图片（缓存命中路径和 fresh 路径都需要它）
        ref_file_ids = self._upload_images_from_messages(messages)

        full_prefix, last_message = _split_messages(messages)
        last_content = (
            last_message.get("content", "") if isinstance(last_message, dict) else ""
        )
        model = str(payload.get("model", self.config.default_model))
        p_hash = prefix_hash(full_prefix, model=model)

        # 会话 key 用「对话指纹」：只取 system 首段 + 首条 user 首段 + model + key_id，
        # 不随轮次/工具轮变化，避免每轮都新建上游会话。
        key_id = str(payload.get("_wapi_key_id") or "")
        conv = conv_fingerprint(messages, model, key_id)
        msg_count = len(messages)

        cached = self.session_cache.get(conv) if self.cache_enabled else None

        # 校验：历史被压缩/改写/非延续 → 丢弃缓存条目，走 fresh（不删上游会话）
        # 注意用 <（不是 <=）：网络重试会原样重发同一 payload，msg_count 相等属正常延续。
        if cached and cached.msg_count and msg_count < cached.msg_count:
            self.logger.info(
                "Cache STALE | conv=%s... | history rewritten (msgs=%d < %d) → fresh",
                conv[:12],
                msg_count,
                cached.msg_count,
            )
            self._bump("cache_stale")
            self.session_cache.delete(conv)
            cached = None

        # 校验：轮数超上限 → 丢弃缓存条目，走 fresh（交给清理线程删上游会话）
        if cached:
            max_turns = int(getattr(self.config, "max_turns_per_session", 40) or 40)
            if cached.turn_count >= max_turns:
                self.logger.info(
                    "Cache STALE | conv=%s... | max turns reached (%d >= %d) → fresh",
                    conv[:12],
                    cached.turn_count,
                    max_turns,
                )
                self.session_cache.delete(conv)
                self._bump("cache_stale")
                cached = None

        if cached:
            ds_payload = convert_openai_to_deepseek_incremental(payload, self.config)
            ds_payload["chat_session_id"] = cached.session_id
            if ref_file_ids:  # ← 注入
                ds_payload["ref_file_ids"] = ref_file_ids

            self.logger.info(
                "Cache HIT  | conv=%s... | session=%s... | turn=%d | "
                "prompt_len=%d | files=%d",
                conv[:12],
                cached.session_id[:12],
                cached.turn_count + 1,
                len(ds_payload.get("prompt", "")),
                len(ref_file_ids),
            )
            self._bump("cache_hit")
            return {
                "mode": "cached",
                "ds_payload": ds_payload,
                "session_id": cached.session_id,
                "cache_key": conv,
                "full_prefix": full_prefix,
                "last_message": last_message,
                "last_content": last_content,
                "model": model,
                "turn_count": cached.turn_count,
                "msg_count": msg_count,
            }

        ds_payload = convert_openai_to_deepseek(payload, self.config)
        if ref_file_ids:  # ← 注入
            ds_payload["ref_file_ids"] = ref_file_ids

        self.logger.info(
            "Cache %s | conv=%s... | prefix=%s... | prompt_len=%d (full history packed) | files=%d",
            "MISS" if self.cache_enabled else "OFF",
            conv[:12],
            p_hash[:12],
            len(ds_payload.get("prompt", "")),
            len(ref_file_ids),
        )
        if self.cache_enabled:
            self._bump("cache_miss")
        return {
            "mode": "fresh",
            "ds_payload": ds_payload,
            "session_id": "",
            "cache_key": conv,
            "full_prefix": full_prefix,
            "last_message": last_message,
            "last_content": last_content,
            "model": model,
            "turn_count": 0,
            "msg_count": msg_count,
        }

    def _store_after_response(
        self,
        resolved: dict,
        session_id: str,
        assistant_content: str,
        request_message_id: str | None = None,
        tool_calls: list[ParsedToolCall] | None = None,
    ) -> None:
        if not self.cache_enabled:
            if session_id:
                with self._proxy_sessions_lock:
                    self._proxy_sessions.add(session_id)
            return

        # 下一轮的 cache key = 本轮完整前缀 + 本轮"当前输入消息" + assistant 输出
        new_prefix = list(resolved["full_prefix"])
        last_msg = resolved.get("last_message")
        if isinstance(last_msg, dict):
            new_prefix.append(dict(last_msg))
        else:
            new_prefix.append(
                {"role": "user", "content": resolved.get("last_content", "")}
            )

        assistant_entry: dict = {
            "role": "assistant",
            "content": assistant_content or "",
        }
        if tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments},
                }
                for c in tool_calls
            ]
        new_prefix.append(assistant_entry)

        new_hash = prefix_hash(new_prefix, model=resolved["model"])

        next_turn = (resolved.get("turn_count") or 0) + 1
        self.session_cache.put(
            resolved["cache_key"],
            SessionState(
                session_id=session_id,
                turn_count=next_turn,
                last_message_id=request_message_id,
                msg_count=int(resolved.get("msg_count", 0) or 0),
                history_hash=new_hash,
            ),
        )

        with self._proxy_sessions_lock:
            self._proxy_sessions.add(session_id)

        self.logger.info(
            "Cached     | %s → conv=%s... | history=%s... | session=%s... | turn=%d | tools=%d | msg_id=%s",
            resolved["mode"],
            str(resolved.get("cache_key", ""))[:12],
            new_hash[:12],
            session_id[:12],
            next_turn,
            len(tool_calls) if tool_calls else 0,
            (request_message_id or "N/A")[:12] if request_message_id else "N/A",
        )

    # ═══════════════════════════════════════════════════════
    # Cache / session cleanup
    # ═══════════════════════════════════════════════════════

    def _invalidate_cache_for_resolved(self, resolved: dict) -> None:
        if not resolved:
            return
        key = resolved.get("cache_key")
        if key:
            self.session_cache.delete(key)

    def _invalidate_and_delete_session(self, resolved: dict) -> None:
        """context_length_exceeded 时：从缓存移除 + 删除上游会话。"""
        session_id = resolved.get("session_id") or ""

        # 缓存中所有指向该 session 的条目都清掉
        if session_id and self.cache_enabled:
            removed = self.session_cache.delete_by_session_id(session_id)
            if removed:
                self.logger.info(
                    "Cache purge | session=%s... | removed=%d entries",
                    session_id[:12],
                    removed,
                )
        self._invalidate_cache_for_resolved(resolved)

        if session_id:
            self._delete_session(session_id)

        # 清空 resolved 里的 session，避免后续误用
        resolved["session_id"] = ""

    def cleanup_stale_sessions(self) -> None:
        active = self.session_cache.active_session_ids()
        with self._proxy_sessions_lock:
            stale = self._proxy_sessions - active
            if not stale:
                return
            stale_list = list(stale)
            self._proxy_sessions.difference_update(stale_list)

        for sid in stale_list:
            self._delete_session(sid)

        if stale_list:
            self.logger.info(
                "Cleanup    | %d stale session(s) deleted | %d tracked | %d active",
                len(stale_list),
                len(self._proxy_sessions),
                len(active),
            )

    def chat_completion(self, payload: dict[str, object]) -> dict[str, object]:
        resolved = self._resolve_conversation(payload)
        # 一次 completion 内所有 chunk/结果共享同一个 id（OpenAI 规范）。
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        return self._chat_completion_impl(payload, resolved, completion_id)

    def _chat_completion_impl(
        self, payload: dict, resolved: dict, completion_id: str
    ) -> dict:
        # 槽位获取/释放与限流 sleep 的协调全部在循环内部处理，
        # 保证 sleep 时不占槽位、finally 释放的永远是当前持有的 lease。
        return self._chat_completion_loop(payload, resolved, completion_id)

    def _chat_completion_loop(
        self, payload: dict, resolved: dict, completion_id: str
    ) -> dict:
        context_retries = 0
        rate_limit_retries = 0
        file_retries = 0
        has_tools = _payload_has_tools(payload)
        slot_name = f"chat:{payload.get('model', 'unknown')}"

        # 注意：限流 sleep 前必须先释放槽位再睡醒重拿，
        # 否则一次限流（60s×3次）会连带堵住整个队列数分钟。
        lease = self.request_queue.acquire(slot_name)
        try:
            while True:
                response = None
                session_id = resolved.get("session_id") or ""
                try:
                    response, session_id = self._open_chat_stream(resolved)

                    full_text = ""
                    full_reasoning = ""
                    last_response_message_id: str | None = None
                    accumulated_tokens = 0
                    state = StreamState()
                    saw_upstream_finish = False

                    # 非流式全程在内部消费，断流时可以透明续写（客户端无感）：
                    # 同会话发「继续」，parent 指向截断消息，增量接着累计。
                    _cont_rounds = 0
                    while True:
                        for event in self._iter_sse_events(response):
                            if not event:
                                continue

                            if self._handle_hint_event(event):
                                continue

                            if self._is_finish_event(event):
                                saw_upstream_finish = True

                            msg_id = self._extract_ready_message_id(event)
                            if msg_id is not None:
                                last_response_message_id = msg_id

                            usage = self._extract_batch_token_usage(event)
                            if usage is not None:
                                accumulated_tokens = usage

                            openai_chunk = convert_deepseek_to_openai_chunk(
                                event,
                                resolved["model"],
                                state,
                                completion_id,
                            )
                            if openai_chunk:
                                choices = openai_chunk.get("choices", [])
                                if choices and isinstance(choices[0], dict):
                                    delta = choices[0].get("delta", {})
                                    c = delta.get("content", "")
                                    if c:
                                        full_text += c
                                    r = delta.get("reasoning_content", "")
                                    if r:
                                        full_reasoning += r

                        if (
                            saw_upstream_finish
                            or _cont_rounds >= MAX_AUTO_CONTINUATIONS
                        ):
                            break
                        if not (
                            last_response_message_id
                            and str(last_response_message_id).isdigit()
                            and session_id
                        ):
                            break
                        # 工具标记被截断：截掉残块，续写轮会重发完整调用块
                        trimmed = _drop_trailing_partial_markup(full_text)
                        if trimmed != full_text:
                            self.logger.warning(
                                "上游断流且工具标记被截断，丢弃残块（续写轮会重发完整调用）"
                            )
                            full_text = trimmed
                        self.logger.warning(
                            "上游流提前中断（未收到 finish 帧），自动续写 %d/%d",
                            _cont_rounds,
                            MAX_AUTO_CONTINUATIONS,
                        )
                        self._bump("auto_continued")
                        _cont_rounds += 1
                        try:
                            try:
                                response.close()
                            except Exception:
                                pass
                            cont_resolved = self._continuation_resolved(
                                resolved, session_id, last_response_message_id
                            )
                            response, session_id = self._open_chat_stream(cont_resolved)
                        except Exception as exc:  # noqa: BLE001
                            # 续写失败按截断收场
                            self.logger.warning("自动续写失败（按截断返回）: %s", exc)
                            break

                    # ── 工具调用检测（无条件尝试；缺 tools 时只认 DSML / 哨兵标记） ──
                    parsed = parse_tool_calls(
                        full_text,
                        payload.get("tools"),
                        require_marker=not has_tools,
                    )
                    if parsed.calls:
                        self.logger.info(
                            "ToolCall   | blocking | %s",
                            ", ".join(c.name for c in parsed.calls),
                        )
                        self._store_after_response(
                            resolved,
                            session_id,
                            parsed.clean_text,
                            last_response_message_id,
                            tool_calls=parsed.calls,
                        )
                        if (
                            self.config.delete_conversation
                            and resolved["mode"] == "fresh"
                            and session_id
                        ):
                            self._delete_session(session_id)
                        return self._build_tool_call_result(
                            payload,
                            resolved,
                            parsed.calls,
                            accumulated_tokens,
                            completion_id,
                        )

                    # 空正文兜底（对齐 Kimi/GLM 的 finalize_reasoning 语义）：
                    # 正文空、思考有货 → 思考顶成正文并清空 reasoning_content，
                    # 否则非流式返回空 content，且客户端会在思考与正文里
                    # 看到重复的同一份答案
                    if not full_text and full_reasoning:
                        full_text = full_reasoning
                        full_reasoning = ""

                    if not saw_upstream_finish:
                        self.logger.warning(
                            "上游流提前中断（未收到 finish 帧）text=%d reasoning=%d，按截断返回 finish=length",
                            len(full_text),
                            len(full_reasoning),
                        )

                    # JSON 模式校验（截断时跳过：半截 JSON 必然校验失败，
                    # 不能把已经消费完的内部流变成 502 报错）
                    if self.config.strip_citations:
                        full_text = strip_citation_marks(full_text)
                    if saw_upstream_finish:
                        self._validate_response_format(payload, full_text)

                    self._store_after_response(
                        resolved, session_id, full_text, last_response_message_id
                    )

                    if (
                        self.config.delete_conversation
                        and resolved["mode"] == "fresh"
                        and session_id
                    ):
                        self._delete_session(session_id)

                    return self._build_non_stream_result(
                        payload,
                        resolved,
                        full_text,
                        full_reasoning,
                        accumulated_tokens,
                        completion_id,
                        finish_reason="length" if not saw_upstream_finish else "stop",
                    )

                except ContextLengthExceededError:
                    context_retries += 1
                    self.logger.warning(
                        "上下文超限 (attempt=%d/%d)，从缓存移除并删除会话，使用完整历史重开新对话",
                        context_retries,
                        MAX_CONTEXT_LENGTH_RETRIES,
                    )
                    self._invalidate_and_delete_session(resolved)
                    if context_retries >= MAX_CONTEXT_LENGTH_RETRIES:
                        raise
                    # 重新解析 → 由于缓存被清空 + 会话已删除，将走 fresh 路径
                    resolved = self._resolve_conversation(payload)

                except FileParseError as exc:
                    file_retries += 1
                    self.logger.warning(
                        "上游文件解析失败 (attempt=%d/%d): %s",
                        file_retries,
                        MAX_FILE_PARSE_RETRIES,
                        exc,
                    )
                    self._invalidate_and_delete_session(resolved)
                    if file_retries >= MAX_FILE_PARSE_RETRIES:
                        raise UpstreamAPIError(
                            status_code=502,
                            message=f"上游文件解析失败: {exc}",
                        ) from exc
                    cleared = self.file_uploader.clear_cache()
                    self.logger.warning("已清空 %d 条文件缓存，去掉附件后重试", cleared)
                    payload = _strip_image_messages(payload)
                    resolved = self._resolve_conversation(payload)

                except RateLimitError as exc:
                    rate_limit_retries += 1
                    self._bump("rate_limit")
                    if rate_limit_retries >= MAX_RATE_LIMIT_RETRIES:
                        self.logger.error(
                            "上游限流重试次数耗尽 (%d/%d)",
                            rate_limit_retries,
                            MAX_RATE_LIMIT_RETRIES,
                        )
                        raise UpstreamAPIError(
                            status_code=429,
                            message=f"上游限流: {exc}",
                        ) from exc

                    self.logger.warning(
                        "上游限流 (%d/%d)，%d 秒后重试（已释放队列槽位）... 原因: %s",
                        rate_limit_retries,
                        MAX_RATE_LIMIT_RETRIES,
                        RATE_LIMIT_COOLDOWN_SECONDS,
                        exc,
                    )
                    # 先释放槽位再 sleep：60 秒的空等不应占着并发槽位，
                    # 睡醒后重新排队拿槽位。
                    lease.release()
                    try:
                        time.sleep(RATE_LIMIT_COOLDOWN_SECONDS)
                    finally:
                        lease = self.request_queue.acquire(slot_name)
                    # 继续 while True，resolved 保持原样（session 已在 _open_chat_stream 里回写）

                finally:
                    if response is not None:
                        try:
                            response.close()
                        except Exception:
                            pass
        finally:
            lease.release()

    def _upload_images_from_messages(
        self,
        messages: list[dict],
    ) -> list[str]:
        """把消息里所有 image_url 上传成 DeepSeek file_id。"""
        if not self.file_upload_enabled:
            return []

        images = extract_images_from_messages(messages)
        if not images:
            return []

        self.logger.info("Vision | 检测到 %d 张图片，开始上传", len(images))
        file_ids: list[str] = []
        for i, img in enumerate(images):
            name_hint = f"image_{i}.png"
            file_id = self.file_uploader.upload_from_openai_image(
                img["url"],
                name_hint=name_hint,
            )
            if file_id:
                file_ids.append(file_id)
            else:
                self.logger.warning("Vision | 第 %d 张图片上传失败，已跳过", i)

        return file_ids

    # ═══════════════════════════════════════════════════════
    # Public API - stream
    # ═══════════════════════════════════════════════════════

    def stream_chat_completion(self, payload: dict[str, object]):
        resolved = self._resolve_conversation(payload)
        lease = self.request_queue.acquire(f"stream:{payload.get('model', 'unknown')}")
        # 一次 completion 内所有 chunk 共享同一个 id（OpenAI 规范）。
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        has_tools = _payload_has_tools(payload)

        def generate():
            nonlocal lease, payload
            current_resolved = resolved
            current_session_id = current_resolved.get("session_id") or ""
            anything_yielded = False
            context_retries = 0
            rate_limit_retries = 0
            file_retries = 0

            try:
                while True:
                    response = None
                    full_text = ""
                    full_reasoning = ""
                    try:
                        response, current_session_id = self._open_chat_stream(
                            current_resolved
                        )

                        stripper = (
                            CitationStripper() if self.config.strip_citations else None
                        )
                        last_response_message_id: str | None = None
                        accumulated_tokens = 0

                        # ★ 检测器永远启动。
                        #   has_tools=True  → require_marker=False：解析 DSML / 哨兵标记
                        #                     （代码块/裸 JSON 碰运气解析默认关闭，
                        #                      可用 TOOL_JSON_FALLBACK=true 恢复）
                        #   has_tools=False → require_marker=True ：只认 DSML / 哨兵标记，
                        #                     防止模型无视 tool_choice=none 或缺失 tools 时
                        #                     把 DSML 块原样漏进 content
                        detector = StreamingToolDetector(
                            payload.get("tools"),
                            require_marker=not has_tools,
                        )
                        tool_calls_emitted = False
                        emitted_calls: list[ParsedToolCall] = []
                        upstream_finish_reason: str | None = None
                        saw_upstream_finish = False
                        content_emitted = False

                        def _is_role_only(chunk: dict) -> bool:
                            try:
                                d = chunk["choices"][0].get("delta") or {}
                                return (
                                    d.get("role") == "assistant"
                                    and not (d.get("content") or "")
                                    and not (d.get("reasoning_content") or "")
                                    and not d.get("tool_calls")
                                )
                            except Exception:  # noqa: BLE001
                                return False

                        def _consume(resp, *, skip_role: bool) -> bool:
                            """消费一个上游 SSE 流直到 finish/断流，返回是否见到 finish 帧。

                            检测器/剥离器/累计计数**跨轮共享**——断流自动续写轮的
                            增量直接拼进同一个 SSE 输出，客户端看到的仍是连续的
                            一条回答。续写轮（skip_role=True）跳过重复的 role chunk。
                            """
                            nonlocal full_text, full_reasoning
                            nonlocal last_response_message_id, accumulated_tokens
                            nonlocal \
                                tool_calls_emitted, \
                                emitted_calls, \
                                upstream_finish_reason
                            nonlocal content_emitted, anything_yielded
                            st = StreamState()
                            for event in self._iter_sse_events(resp):
                                if not event:
                                    continue

                                if self._handle_hint_event(event):
                                    continue

                                msg_id = self._extract_ready_message_id(event)
                                if msg_id is not None:
                                    last_response_message_id = msg_id

                                usage = self._extract_batch_token_usage(event)
                                if usage is not None:
                                    accumulated_tokens = usage

                                openai_chunk = convert_deepseek_to_openai_chunk(
                                    event,
                                    current_resolved["model"],
                                    st,
                                    completion_id,
                                )

                                if not openai_chunk:
                                    if self._is_finish_event(event):
                                        return True
                                    continue

                                choices = openai_chunk.get("choices") or []
                                if not choices or not isinstance(choices[0], dict):
                                    if not (skip_role and _is_role_only(openai_chunk)):
                                        yield _sse(openai_chunk)
                                        anything_yielded = True
                                    if self._is_finish_event(event):
                                        return True
                                    continue

                                delta = choices[0].get("delta") or {}
                                finish_reason = choices[0].get("finish_reason")
                                if finish_reason is not None:
                                    # 嵌套快照可能同时携带 content 与 finish_reason，
                                    # 必须先记录，否则会被 r/c 分支吞掉导致漏判完成
                                    upstream_finish_reason = finish_reason
                                c = delta.get("content", "") or ""
                                r = delta.get("reasoning_content", "") or ""

                                if r:
                                    # reasoning 原样透传，不参与检测
                                    full_reasoning += r
                                    yield _sse(openai_chunk)
                                    anything_yielded = True
                                elif c:
                                    # ★ 内容统一喂给检测器
                                    full_text += c
                                    for action in detector.feed(c):
                                        if action.kind == "text" and action.text:
                                            out = (
                                                stripper.feed(action.text)
                                                if stripper
                                                else action.text
                                            )
                                            if out:
                                                yield _make_content_chunk(
                                                    out,
                                                    current_resolved["model"],
                                                    completion_id,
                                                )
                                                content_emitted = True
                                                anything_yielded = True
                                        elif (
                                            action.kind == "tool_calls"
                                            and action.tool_calls
                                        ):
                                            for ch in _make_tool_call_chunks(
                                                action.tool_calls,
                                                current_resolved["model"],
                                                completion_id,
                                                base_index=len(emitted_calls),
                                            ):
                                                yield ch
                                            tool_calls_emitted = True
                                            emitted_calls.extend(action.tool_calls)
                                            anything_yielded = True
                                elif finish_reason is None:
                                    # 纯 role chunk / 空 delta chunk：原样透传
                                    # （续写轮客户端已收到过 role，跳过重复的）
                                    if skip_role and _is_role_only(openai_chunk):
                                        continue
                                    yield _sse(openai_chunk)
                                    anything_yielded = True
                                # 带 finish_reason 的块不透传，finish 我们最后自己发

                                if self._is_finish_event(event):
                                    return True
                            return False

                        saw = yield from _consume(response, skip_role=False)
                        if saw or upstream_finish_reason is not None:
                            saw_upstream_finish = True

                        # ── 断流自动续写 ──
                        # 上游没发 finish 帧就断流（长思考/长回答/大上下文被掐）：
                        # 向同一会话发一条「继续」（parent_message_id 指向截断的
                        # 消息——网页版「继续生成」的同款机制），续写增量经同一个
                        # 检测器/剥离器拼进当前 SSE 流，客户端看到的仍是连续的
                        # 一条回答。工具标记被截断时丢弃残块，续写轮会重发完整调用。
                        # 注意：检测器 finalize 必须等续写全部结束后再做。
                        _cont_rounds = 0
                        while (
                            not saw_upstream_finish
                            and _cont_rounds < MAX_AUTO_CONTINUATIONS
                            and last_response_message_id
                            and str(last_response_message_id).isdigit()
                            and current_session_id
                        ):
                            _cont_rounds += 1
                            self._bump("auto_continued")
                            if detector.pending_markup:
                                self.logger.warning(
                                    "上游断流且工具标记被截断，丢弃残块（续写轮会重发完整调用）"
                                )
                                detector = StreamingToolDetector(
                                    payload.get("tools"),
                                    require_marker=not has_tools,
                                )
                            self.logger.warning(
                                "上游流提前中断（未收到 finish 帧），自动续写 %d/%d",
                                _cont_rounds,
                                MAX_AUTO_CONTINUATIONS,
                            )
                            try:
                                cont_resolved = self._continuation_resolved(
                                    current_resolved,
                                    current_session_id,
                                    last_response_message_id,
                                )
                                cont_resp, cont_session = self._open_chat_stream(
                                    cont_resolved
                                )
                                if cont_session:
                                    current_session_id = cont_session
                                try:
                                    saw = yield from _consume(cont_resp, skip_role=True)
                                finally:
                                    try:
                                        cont_resp.read()
                                    except Exception:
                                        pass
                                    try:
                                        cont_resp.close()
                                    except Exception:
                                        pass
                            except Exception as exc:  # noqa: BLE001
                                # 续写失败按截断收场，绝不能把已输出的流变成报错
                                self.logger.warning(
                                    "自动续写失败（按截断返回）: %s", exc
                                )
                                break
                            if saw or upstream_finish_reason is not None:
                                saw_upstream_finish = True

                        # ── 流结束收尾（finalize 必须在续写全部结束后）──
                        # 1. 检测器残余
                        for action in detector.finalize():
                            if action.kind == "text" and action.text:
                                out = (
                                    stripper.feed(action.text)
                                    if stripper
                                    else action.text
                                )
                                if out:
                                    yield _make_content_chunk(
                                        out, current_resolved["model"], completion_id
                                    )
                                    content_emitted = True
                                    anything_yielded = True
                            elif action.kind == "tool_calls" and action.tool_calls:
                                for ch in _make_tool_call_chunks(
                                    action.tool_calls,
                                    current_resolved["model"],
                                    completion_id,
                                    base_index=len(emitted_calls),
                                ):
                                    yield ch
                                tool_calls_emitted = True
                                emitted_calls.extend(action.tool_calls)
                                anything_yielded = True

                        # 2. stripper 尾部残留
                        if stripper is not None:
                            tail = stripper.finalize()
                            if tail:
                                yield _make_content_chunk(
                                    tail, current_resolved["model"], completion_id
                                )
                                content_emitted = True
                                anything_yielded = True

                        # 2.5 空正文兜底（对齐 Kimi/GLM 的 finalize_reasoning）：
                        #     正文一个字都没发、思考里有货——上游把答案整个写进
                        #     思考，或思考中途被上游掐流时，把思考顶成正文，
                        #     否则客户端只看得到空消息或「全部藏在思考里」。
                        if (
                            not tool_calls_emitted
                            and not content_emitted
                            and full_reasoning
                        ):
                            yield _make_content_chunk(
                                full_reasoning, current_resolved["model"], completion_id
                            )
                            content_emitted = True
                            anything_yielded = True
                            if not full_text:
                                full_text = full_reasoning

                        # 3. JSON 模式校验（仅在没有工具调用发出且正常结束时：
                        #    截断的半截 JSON 必然校验失败，不能让已输出的流变报错）
                        if not tool_calls_emitted and saw_upstream_finish:
                            self._validate_response_format(payload, full_text)

                        # 4. usage chunk（无条件发送：客户端即使没传 include_usage，
                        #    opencode 等也需要它才能统计用量）
                        if True:
                            prompt_text = current_resolved["ds_payload"].get(
                                "prompt", ""
                            )
                            prompt_tokens = (
                                max(1, _estimate_tokens(prompt_text))
                                if prompt_text
                                else 0
                            )
                            completion_tokens = accumulated_tokens or max(
                                1, _estimate_tokens(full_text)
                            )
                            usage_chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": current_resolved["model"],
                                "system_fingerprint": "",
                                "choices": [],
                                "usage": {
                                    "prompt_tokens": prompt_tokens,
                                    "completion_tokens": completion_tokens,
                                    "total_tokens": prompt_tokens + completion_tokens,
                                },
                            }
                            yield _sse(usage_chunk)

                        # 5. ★ 统一在末尾发 finish chunk
                        #    发出过工具调用 → "tool_calls"
                        #    上游没发 finish 帧就断流 → "length"（静默截断：
                        #    长思考/大上下文时上游会中途掐流，要检测出来，
                        #    不能装作正常结束）
                        #    否则沿用上游 finish_reason（一般 "stop"），缺失时兜底 "stop"
                        if tool_calls_emitted:
                            final_finish = "tool_calls"
                        elif not saw_upstream_finish:
                            self.logger.warning(
                                "上游流提前中断（未收到 finish 帧）text=%d reasoning=%d，按截断返回 finish=length",
                                len(full_text),
                                len(full_reasoning),
                            )
                            final_finish = "length"
                        else:
                            final_finish = upstream_finish_reason or "stop"
                        yield _make_finish_chunk(
                            final_finish, current_resolved["model"], completion_id
                        )

                        yield "data: [DONE]\n\n"

                        # 6. 缓存 / 会话记录
                        if full_text or full_reasoning or tool_calls_emitted:
                            stored_text = (
                                strip_citation_marks(full_text)
                                if self.config.strip_citations
                                else full_text
                            )
                            self._store_after_response(
                                current_resolved,
                                current_session_id,
                                stored_text,
                                last_response_message_id,
                                tool_calls=emitted_calls or None,
                            )

                        if (
                            self.config.delete_conversation
                            and current_resolved["mode"] == "fresh"
                            and current_session_id
                        ):
                            self._delete_session(current_session_id)

                        return  # success

                    except ContextLengthExceededError:
                        context_retries += 1
                        self.logger.warning(
                            "流式上下文超限 (attempt=%d/%d, yielded=%s)",
                            context_retries,
                            MAX_CONTEXT_LENGTH_RETRIES,
                            anything_yielded,
                        )
                        self._invalidate_and_delete_session(current_resolved)
                        if (
                            anything_yielded
                            or context_retries >= MAX_CONTEXT_LENGTH_RETRIES
                        ):
                            raise
                        current_resolved = self._resolve_conversation(payload)
                        current_session_id = current_resolved.get("session_id") or ""

                    except FileParseError as exc:
                        file_retries += 1
                        self.logger.warning(
                            "上游文件解析失败 (attempt=%d/%d, yielded=%s): %s",
                            file_retries,
                            MAX_FILE_PARSE_RETRIES,
                            anything_yielded,
                            exc,
                        )
                        self._invalidate_and_delete_session(current_resolved)
                        if anything_yielded or file_retries >= MAX_FILE_PARSE_RETRIES:
                            raise UpstreamAPIError(
                                status_code=502,
                                message=f"上游文件解析失败: {exc}",
                            ) from exc
                        # 清掉可能已损坏的 file_id 缓存，去掉图片重发（纯文本降级）
                        cleared = self.file_uploader.clear_cache()
                        self.logger.warning(
                            "已清空 %d 条文件缓存，去掉附件后重试",
                            cleared,
                        )
                        payload = _strip_image_messages(payload)
                        current_resolved = self._resolve_conversation(payload)
                        current_session_id = current_resolved.get("session_id") or ""

                    except RateLimitError as exc:
                        rate_limit_retries += 1
                        if anything_yielded:
                            self.logger.error(
                                "流式已 yield %d 字节内容，无法重试限流: %s",
                                len(full_text),
                                exc,
                            )
                            raise UpstreamAPIError(
                                status_code=429,
                                message=f"上游限流（已在流式响应中）: {exc}",
                            ) from exc

                        if rate_limit_retries >= MAX_RATE_LIMIT_RETRIES:
                            self.logger.error(
                                "上游限流重试次数耗尽 (%d/%d)",
                                rate_limit_retries,
                                MAX_RATE_LIMIT_RETRIES,
                            )
                            raise UpstreamAPIError(
                                status_code=429,
                                message=f"上游限流: {exc}",
                            ) from exc

                        self.logger.warning(
                            "上游限流 (%d/%d)，%d 秒后重试（已释放队列槽位）... 原因: %s",
                            rate_limit_retries,
                            MAX_RATE_LIMIT_RETRIES,
                            RATE_LIMIT_COOLDOWN_SECONDS,
                            exc,
                        )
                        # 先释放槽位再 sleep：60 秒的空等不应占着并发槽位，
                        # 睡醒后重新排队拿槽位。
                        lease.release()
                        try:
                            time.sleep(RATE_LIMIT_COOLDOWN_SECONDS)
                        finally:
                            lease = self.request_queue.acquire(
                                f"stream:{payload.get('model', 'unknown')}"
                            )
                        # 继续 while True；current_session_id 已由 _open_chat_stream 回写

                    finally:
                        if response is not None:
                            try:
                                response.close()
                            except Exception:
                                pass
            finally:
                lease.release()

        return generate()

    # ═══════════════════════════════════════════════════════
    # 风险指标（面板展示用，进程内累计，重启归零）
    # ═══════════════════════════════════════════════════════

    def _bump(self, name: str, delta: int = 1) -> None:
        with self._stats_lock:
            self._stats[name] = self._stats.get(name, 0) + delta

    def risk_stats(self) -> dict:
        with self._stats_lock:
            data = dict(self._stats)
        cache_size = (
            self.session_cache.size if hasattr(self.session_cache, "size") else 0
        )
        active = 0
        try:
            active = len(self.session_cache.active_session_ids())
        except Exception:  # noqa: BLE001
            pass
        total = data.get("cache_hit", 0) + data.get("cache_miss", 0)
        hit_rate = round(data.get("cache_hit", 0) / total, 3) if total else 0.0
        return {
            "sessions_created": data.get("session_created", 0),
            "sessions_deleted": data.get("session_deleted", 0),
            "cache_hit": data.get("cache_hit", 0),
            "cache_miss": data.get("cache_miss", 0),
            "cache_hit_rate": hit_rate,
            "cache_stale": data.get("cache_stale", 0),
            "cache_size": cache_size,
            "active_sessions": active,
            "upstream_requests": data.get("upstream", 0),
            "rate_limit_hits": data.get("rate_limit", 0),
            "daily_count": 0,  # 由池层填充
            "daily_limit": int(getattr(self.config, "daily_request_limit", 0) or 0),
        }

    # ═══════════════════════════════════════════════════════
    # Anti rate-limit
    # ═══════════════════════════════════════════════════════

    def _anti_rate_limit_pause(self, context: str = "") -> None:
        """
        防风控：在每次真正打到上游的聊天请求前随机延迟。

        触发点：_open_chat_stream() 的入口。这样天然覆盖：
          - 非流式 / 流式
          - 缓存命中 (mode=cached) / 全量重放 (mode=fresh)
          - rate limit 重试、context length 重试
          - 工具链的每一轮（工具调用走的是同一条 chat 通道）
        """
        if not self.config.anti_rate_limit_enabled:
            return

        min_d = max(0.0, float(self.config.anti_rate_limit_min_delay))
        max_d = float(self.config.anti_rate_limit_max_delay)
        if max_d <= 0:
            return

        # 同账号两次上游请求之间的硬性最小间隔（防突发）
        # 锁内只做时间戳预约，sleep 放在锁外，避免串行化其它请求。
        min_interval = float(getattr(self.config, "min_request_interval", 0.0) or 0.0)
        if min_interval > 0:
            with self._last_upstream_lock:
                now = time.time()
                slot = max(now, self._last_upstream_at + min_interval)
                self._last_upstream_at = slot
            wait = slot - time.time()
            if wait > 0:
                self.logger.info(
                    "AntiRateLimit | min-interval sleep=%.2fs%s",
                    wait,
                    f" ({context})" if context else "",
                )
                time.sleep(wait)

        # 长尾分布：绝大多数请求短延迟，少量请求出现 20~60s 的长停顿
        chance = float(getattr(self.config, "long_pause_chance", 0.02) or 0.0)
        if chance > 0 and random.random() < chance:
            delay = random.uniform(20, 60)
        else:
            mid = max(0.1, (min_d + max_d) / 2)
            delay = random.lognormvariate(math.log(mid), 0.6)
        upper = max(1.0, max_d * 3)
        delay = min(max(delay, min_d), upper)

        self.logger.info(
            "AntiRateLimit | sleep=%.2fs before upstream request%s",
            delay,
            f" ({context})" if context else "",
        )
        time.sleep(delay)

    # ═══════════════════════════════════════════════════════
    # Internal: request pipeline
    # ═══════════════════════════════════════════════════════

    def _continuation_resolved(
        self, current_resolved: dict, session_id: str, parent_message_id: str
    ) -> dict:
        """构造断流续写轮的 resolved：同会话、parent 指向截断消息、只发「继续」。

        cache_key 用一次性哨兵——绝不读/写会话缓存（续写轮对客户端不可见，
        后续轮仍按「用户 + 完整回答」的前缀哈希匹配，见 _store_after_response）。
        """
        synthetic = {
            "model": current_resolved["model"],
            "messages": [{"role": "user", "content": CONTINUE_PROMPT}],
        }
        ds_payload = convert_openai_to_deepseek_incremental(synthetic, self.config)
        ds_payload["chat_session_id"] = session_id
        try:
            ds_payload["parent_message_id"] = int(parent_message_id)
        except (TypeError, ValueError):
            pass
        return {
            "mode": "cached",
            "ds_payload": ds_payload,
            "session_id": session_id,
            "cache_key": f"__continue__{uuid.uuid4().hex}",
            "full_prefix": list(current_resolved.get("full_prefix") or []),
            "last_message": None,
            "last_content": "",
            "model": current_resolved["model"],
            "turn_count": 0,
            "msg_count": 0,
        }

    def _open_chat_stream(self, resolved: dict):
        self._anti_rate_limit_pause(context=f"mode={resolved['mode']}")
        self._bump("upstream")
        ds_payload = resolved["ds_payload"]
        mode = resolved["mode"]
        session_id = resolved["session_id"]

        if mode == "fresh":
            if session_id:
                # rate limit 重试路径：上次已经创建了 session，直接复用
                self.logger.info(
                    "Step 3/4: 复用已创建的 session=%s...（rate limit 重试）",
                    session_id[:12],
                )
                ds_payload["chat_session_id"] = session_id
            else:
                self.logger.info("Step 1/4: 获取 PoW challenge (session)...")
                pow_challenge = self._get_pow_challenge()

                self.logger.info(
                    "Step 2/4: 求解 PoW (difficulty=%s)...",
                    pow_challenge.get("difficulty"),
                )
                session_pow_header = self.pow_solver.solve(
                    challenge=pow_challenge["challenge"],
                    salt=pow_challenge["salt"],
                    difficulty=pow_challenge["difficulty"],
                    algorithm=pow_challenge.get("algorithm", "DeepSeekHashV1"),
                    expire_at=pow_challenge.get("expire_at", 0),
                    signature=pow_challenge.get("signature", ""),
                )
                self.logger.info("PoW 求解完成")

                self.logger.info("Step 3/4: 创建新 chat session...")
                session_id = self._create_session(session_pow_header)
                ds_payload["chat_session_id"] = session_id
                # 关键：回写 resolved，rate limit 重试时能复用
                resolved["session_id"] = session_id
                self.logger.info("Session 创建成功: %s", session_id)
                self._bump("session_created")
        else:
            self.logger.info("Step 3/4: 复用缓存 session=%s...", session_id[:12])
            cached_state = self.session_cache.get(resolved["cache_key"])
            if cached_state and cached_state.last_message_id is not None:
                # 上游要求 parent_message_id 必须是 u32
                try:
                    parent_id = int(cached_state.last_message_id)
                except (TypeError, ValueError):
                    self.logger.warning(
                        "无效的 last_message_id，忽略 parent_message_id: %r",
                        cached_state.last_message_id,
                    )
                else:
                    ds_payload["parent_message_id"] = parent_id
                    self.logger.info("使用 parent_message_id=%d", parent_id)

        model = ds_payload.get("model_type", self.config.default_model)
        self.logger.info(
            "Step 4/4: 发送 completion 请求 model=%s mode=%s...", model, mode
        )

        self.logger.info("获取 PoW challenge (completion)...")
        completion_pow_challenge = self._get_pow_challenge()

        self.logger.info(
            "求解 PoW (completion, difficulty=%s)...",
            completion_pow_challenge.get("difficulty"),
        )
        completion_pow_header = self.pow_solver.solve(
            challenge=completion_pow_challenge["challenge"],
            salt=completion_pow_challenge["salt"],
            difficulty=completion_pow_challenge["difficulty"],
            algorithm=completion_pow_challenge.get("algorithm", "DeepSeekHashV1"),
            expire_at=completion_pow_challenge.get("expire_at", 0),
            signature=completion_pow_challenge.get("signature", ""),
            target_path=completion_pow_challenge.get(
                "target_path",
                "/api/v0/chat/completion",
            ),  # ← 新增
        )
        self.logger.info("Completion PoW 求解完成")

        body = json.dumps(ds_payload, ensure_ascii=False).encode("utf-8")

        headers = self.auth.build_full_headers(
            pow_response=completion_pow_header,
            content_type="application/json",
        )

        debug_dump(
            self.logger, self.config.debug_dump_all, "DeepSeek completion 请求体", body
        )
        debug_dump(
            self.logger,
            self.config.debug_dump_all,
            "DeepSeek completion 请求头",
            headers,
        )

        try:
            request = urllib.request.Request(
                self.config.chat_completion_url,
                data=body,
                method="POST",
                headers=headers,
            )
            response = self._open_with_egress(
                request, timeout=self.config.request_timeout
            )
            self.auth.absorb_response_cookies(response)
        except urllib.error.HTTPError as exc:
            error_payload = self._read_error_payload(exc)
            message = self._build_error_message(exc.code, error_payload)

            if mode == "cached":
                self.session_cache.delete(resolved["cache_key"])
                self.logger.warning(
                    "缓存 session 失效，已从缓存移除: %s", resolved["cache_key"][:12]
                )

            raise UpstreamAPIError(
                status_code=exc.code, message=message, payload=error_payload
            ) from exc
        except urllib.error.URLError as exc:
            raise UpstreamAPIError(
                status_code=502, message=f"网络连接失败: {exc}"
            ) from exc

        # 非 SSE 响应处理
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/event-stream" not in content_type:
            try:
                raw_body = response.read()
                content_encoding = response.headers.get("Content-Encoding", "").lower()
                if content_encoding == "gzip":
                    raw_body = gzip.decompress(raw_body)
                elif content_encoding == "deflate":
                    try:
                        raw_body = zlib.decompress(raw_body)
                    except zlib.error:
                        raw_body = zlib.decompress(raw_body, -zlib.MAX_WBITS)

                debug_dump(self.logger, True, "⚠️ DeepSeek 返回非 SSE 响应", raw_body)
                text = raw_body.decode("utf-8", errors="replace")

                if mode == "cached":
                    self.session_cache.delete(resolved["cache_key"])

                try:
                    error_payload = json.loads(text)
                    if isinstance(error_payload, dict):
                        status_code = self._map_upstream_error(error_payload)
                        biz_err = (
                            error_payload.get("biz_data", {})
                            .get("biz_data", {})
                            .get("error", {})
                        )
                        error_msg = (
                            biz_err.get("msg")
                            or error_payload.get("msg")
                            or error_payload.get("message")
                            or text
                        )
                        raise UpstreamAPIError(
                            status_code=status_code,
                            message=f"DeepSeek 返回错误: {error_msg}",
                            payload=error_payload,
                        )
                except json.JSONDecodeError:
                    pass

                raise UpstreamAPIError(
                    status_code=502,
                    message=f"DeepSeek 返回非 SSE 响应: {text[:200]}",
                    payload={"raw": text},
                )
            except Exception:
                response.close()
                raise

        return response, session_id

    # ═══════════════════════════════════════════════════════
    # Upstream error mapping
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _map_upstream_error(payload: dict) -> int:
        code = payload.get("code")
        msg = str(payload.get("msg", "")).upper()

        if code == 40300 or "MISSING_HEADER" in msg:
            return 502
        if "INVALID_POW" in msg:
            return 502
        if code == 429 or "RATE" in msg or "LIMIT" in msg:
            return 429
        if code == 400 or "INVALID" in msg or "PARAM" in msg:
            return 400
        if "TIMEOUT" in msg:
            return 504
        return 502

    # ═══════════════════════════════════════════════════════
    # Response helpers
    # ═══════════════════════════════════════════════════════

    def _validate_response_format(self, payload: dict, full_text: str) -> None:
        response_format = payload.get("response_format") or {}
        if (
            isinstance(response_format, dict)
            and response_format.get("type") == "json_object"
        ):
            try:
                json.loads(full_text)
            except json.JSONDecodeError as exc:
                # 这是上游模型的问题，不是客户端请求的问题 → 502 而非 400
                raise UpstreamAPIError(
                    status_code=502,
                    message=f"上游模型未返回合法 JSON: {exc}",
                ) from exc

    def _build_non_stream_result(
        self,
        payload: dict,
        resolved: dict,
        full_text: str,
        full_reasoning: str,
        accumulated_tokens: int,
        completion_id: str,
        finish_reason: str = "stop",
    ) -> dict:
        prompt_text = resolved["ds_payload"].get("prompt", "")
        prompt_tokens = max(1, _estimate_tokens(prompt_text)) if prompt_text else 0
        completion_tokens = accumulated_tokens or max(1, _estimate_tokens(full_text))

        message: dict = {
            "role": "assistant",
            "content": full_text,
        }
        if full_reasoning:
            message["reasoning_content"] = full_reasoning

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": str(payload.get("model", self.config.default_model)),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def _build_tool_call_result(
        self,
        payload: dict,
        resolved: dict,
        calls: list[ParsedToolCall],
        accumulated_tokens: int,
        completion_id: str,
    ) -> dict:
        prompt_text = resolved["ds_payload"].get("prompt", "")
        prompt_tokens = max(1, _estimate_tokens(prompt_text)) if prompt_text else 0
        completion_tokens = accumulated_tokens or max(
            1, sum(len(c.arguments) for c in calls) // 4
        )

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": str(payload.get("model", self.config.default_model)),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [c.to_openai() for c in calls],
                    },
                    "finish_reason": "tool_calls",
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    # ═══════════════════════════════════════════════════════
    # DeepSeek API calls
    # ═══════════════════════════════════════════════════════

    def _get_pow_challenge(self) -> dict[str, object]:
        body = json.dumps(
            {
                "target_path": "/api/v0/chat/completion",
            }
        ).encode("utf-8")

        headers = {
            **self.auth.get_auth_headers(),
            "Content-Type": "application/json",
        }

        try:
            request = urllib.request.Request(
                self.config.pow_challenge_url,
                data=body,
                method="POST",
                headers=headers,
            )
            with self._urlopen(request, timeout=15) as response:
                self.auth.absorb_response_cookies(response)
                payload = self.auth._read_json_response(response)
        except urllib.error.HTTPError as exc:
            error_payload = self._read_error_payload(exc)
            raise UpstreamAPIError(
                status_code=exc.code,
                message=f"获取 PoW challenge 失败: {self._build_error_message(exc.code, error_payload)}",
                payload=error_payload,
            ) from exc
        except urllib.error.URLError as exc:
            raise UpstreamAPIError(
                status_code=502, message=f"获取 PoW challenge 网络失败: {exc}"
            ) from exc

        biz_data = payload.get("data", {}).get("biz_data", {}).get("challenge", {})
        if not biz_data:
            biz_data = payload.get("data", {})
            if not biz_data.get("challenge"):
                raise UpstreamAPIError(
                    status_code=500,
                    message=f"PoW challenge 响应格式异常: {json.dumps(payload, ensure_ascii=False)[:300]}",
                    payload=payload,
                )

        challenge = {
            "algorithm": biz_data.get("algorithm", "DeepSeekHashV1"),
            "challenge": biz_data.get("challenge", ""),
            "salt": biz_data.get("salt", ""),
            "signature": biz_data.get("signature", ""),
            "difficulty": int(biz_data.get("difficulty", 144000)),
            "expire_at": int(biz_data.get("expire_at", 0)),
            "target_path": biz_data.get("target_path", "/api/v0/chat/completion"),
        }

        debug_dump(
            self.logger, self.config.debug_dump_all, "PoW challenge 响应", challenge
        )
        return challenge

    def _create_session(self, pow_header: str) -> str:
        body = json.dumps({}).encode("utf-8")

        headers = self.auth.build_full_headers(
            pow_response=pow_header,
            content_type="application/json",
        )

        try:
            request = urllib.request.Request(
                self.config.session_create_url,
                data=body,
                method="POST",
                headers=headers,
            )
            with self._urlopen(request, timeout=15) as response:
                self.auth.absorb_response_cookies(response)
                payload = self.auth._read_json_response(response)
        except urllib.error.HTTPError as exc:
            error_payload = self._read_error_payload(exc)
            raise UpstreamAPIError(
                status_code=exc.code,
                message=f"创建 session 失败: {self._build_error_message(exc.code, error_payload)}",
                payload=error_payload,
            ) from exc
        except urllib.error.URLError as exc:
            raise UpstreamAPIError(
                status_code=502, message=f"创建 session 网络失败: {exc}"
            ) from exc

        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        biz_data = data.get("biz_data", {}) if isinstance(data, dict) else {}

        session_id = ""
        chat_session = biz_data.get("chat_session")
        if isinstance(chat_session, dict):
            session_id = str(chat_session.get("id", "") or "")
        if not session_id:
            session_id = str(biz_data.get("id", "") or "")
        if not session_id:
            session_id = str(biz_data.get("chat_session_id", "") or "")
        if not session_id:
            session_id = str(data.get("id", "") or "")

        if not session_id:
            raise UpstreamAPIError(
                status_code=500,
                message=f"创建 session 响应中未找到 id: {json.dumps(payload, ensure_ascii=False)[:300]}",
                payload=payload,
            )

        debug_dump(self.logger, self.config.debug_dump_all, "Session 创建响应", payload)
        self.logger.info("Session ID 提取成功: %s", session_id)
        return session_id

    def _delete_session(self, session_id: str) -> None:
        if not session_id:
            return
        try:
            body = json.dumps({"chat_session_ids": [session_id]}).encode("utf-8")
            headers = {
                **self.auth.get_auth_headers(),
                "Content-Type": "application/json",
            }
            request = urllib.request.Request(
                self.config.session_delete_url,
                data=body,
                method="POST",
                headers=headers,
            )
            with self._urlopen(request, timeout=10) as response:
                self.auth.absorb_response_cookies(response)
                response.read()  # 读空响应体，保证连接可复用

            with self._proxy_sessions_lock:
                self._proxy_sessions.discard(session_id)

            self.logger.info("已删除 DeepSeek 会话 session_id=%s", session_id)
            self._bump("session_deleted")
        except Exception as exc:
            self.logger.warning(
                "删除 DeepSeek 会话失败 session_id=%s error=%s", session_id, exc
            )

    def _is_finish_event(self, event: dict) -> bool:
        data = event.get("data")
        if isinstance(data, dict):
            if data.get("p") == "response/status" and data.get("v") == "FINISHED":
                return True
            if data.get("p") == "response" and data.get("o") == "BATCH":
                batch_v = data.get("v")
                if isinstance(batch_v, list):
                    for item in batch_v:
                        if (
                            isinstance(item, dict)
                            and item.get("p") == "quasi_status"
                            and item.get("v") == "FINISHED"
                        ):
                            return True
        if event.get("_event_type") == "close":
            return True
        # 裸 v 快照深处嵌 FINISHED（完整响应快照形态，可能没有 close 帧）——
        # 认不出它会把已完成回答误判为断流并追加续写
        for v in (
            event.get("v"),
            (event.get("data") or {}).get("v")
            if isinstance(event.get("data"), dict)
            else None,
        ):
            if isinstance(v, dict):
                resp = v.get("response")
                if isinstance(resp, dict) and resp.get("status") == "FINISHED":
                    return True
        return False

    def _iter_sse_events(self, response):
        pending = ""
        decoder = codecs.getincrementaldecoder("utf-8")("ignore")

        def emit_block(block: str):
            lines = block.split("\n")
            event_type = None
            data_lines = []

            for line in lines:
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].strip())
                elif line.startswith(":"):
                    continue

            if not data_lines:
                return None

            payload_str = "\n".join(data_lines)
            debug_dump(
                self.logger,
                self.config.debug_dump_all,
                "DeepSeek 原始 SSE block",
                block,
            )

            if event_type in ("ready", "title", "close", "hint"):
                try:
                    parsed = json.loads(payload_str)
                except json.JSONDecodeError:
                    parsed = {"raw": payload_str}
                if isinstance(parsed, dict):
                    parsed["_event_type"] = event_type
                return parsed

            if event_type == "update_session":
                return None

            try:
                parsed = json.loads(payload_str)
                if event_type:
                    parsed["_event_type"] = event_type
                debug_dump(
                    self.logger,
                    self.config.debug_dump_all,
                    "DeepSeek 解析后的 SSE payload",
                    parsed,
                )
                return parsed
            except json.JSONDecodeError:
                self.logger.debug("忽略无法解析的 SSE 片段: %s", payload_str[:200])
                return None

        while True:
            try:
                # 用 read1：有多少读多少，立刻返回。
                # 用 read(4096) 会一直阻塞到攒满 4096 字节，首字延迟会明显变高。
                raw_chunk = response.read1(4096)
            except http.client.IncompleteRead as exc:
                raw_chunk = exc.partial or b""
                self.logger.warning(
                    "上游 SSE 连接提前断开(IncompleteRead) bytes=%s", len(raw_chunk)
                )
                if raw_chunk:
                    pending += decoder.decode(raw_chunk, False).replace("\r\n", "\n")
                    while "\n\n" in pending:
                        blk, pending = pending.split("\n\n", 1)
                        event = emit_block(blk.strip())
                        if event:
                            yield event
                break
            except (
                ConnectionResetError,
                ConnectionAbortedError,
                BrokenPipeError,
            ) as exc:
                self.logger.warning("上游 SSE 连接被重置(%s)", type(exc).__name__)
                break
            except (socket.timeout, TimeoutError) as exc:
                self.logger.warning("上游 SSE 读取超时(%s)", type(exc).__name__)
                break
            except OSError as exc:
                self.logger.warning(
                    "上游 SSE 网络 IO 异常(%s: %s)", type(exc).__name__, exc
                )
                break

            if not raw_chunk:
                break

            pending += decoder.decode(raw_chunk, False).replace("\r\n", "\n")

            while "\n\n" in pending:
                blk, pending = pending.split("\n\n", 1)
                event = emit_block(blk.strip())
                if event:
                    yield event

        remaining = decoder.decode(b"", True)
        if remaining:
            pending += remaining

        if pending.strip():
            for blk in pending.split("\n\n"):
                if blk.strip():
                    event = emit_block(blk.strip())
                    if event:
                        yield event

    def _read_error_payload(self, error: urllib.error.HTTPError) -> dict[str, object]:
        try:
            raw_body = error.read()
            content_encoding = error.headers.get("Content-Encoding", "").lower()
            if content_encoding == "gzip":
                raw_body = gzip.decompress(raw_body)
            text = raw_body.decode("utf-8", errors="ignore")
        except Exception as exc:
            return {"message": f"读取上游错误响应失败: {exc}"}

        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
        return {"message": text}

    def _build_error_message(self, status_code: int, payload: dict[str, object]) -> str:
        biz_data = payload.get("biz_data", {})
        if isinstance(biz_data, dict):
            inner = biz_data.get("biz_data", {})
            if isinstance(inner, dict):
                error = inner.get("error", {})
                if isinstance(error, dict) and error.get("msg"):
                    return error["msg"]
            if biz_data.get("msg"):
                return str(biz_data["msg"])

        message = str(payload.get("msg", "")).strip()
        if not message:
            message = str(payload.get("message", "")).strip()
        if not message:
            message = f"DeepSeek 请求失败 HTTP {status_code}"
        return message
