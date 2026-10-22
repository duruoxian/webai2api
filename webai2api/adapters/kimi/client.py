"""Kimi（www.kimi.com）网页版客户端。

把网页版接口封装成 OpenAI 兼容的 chat_completion / stream_chat_completion。
协议（2026-10 实测）：
  1. Authorization: Bearer <accessToken>（登录时抓取）
  2. POST /api/chat  body 必须含 name，返回 SSE 流
  3. 会话缓存：同一段对话复用 chat 会话 id，多轮只发增量；超限轮换
"""
from __future__ import annotations

import hashlib
import http.client
import json
import random
import threading
import time
import uuid
from logging import Logger
from urllib.parse import urlsplit

from .config import KimiConfig

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
]

BASE_HEADERS = {
    "Accept": "text/event-stream",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Cache-Control": "no-cache",
    "Content-Type": "application/json",
    "Origin": "https://www.kimi.com",
    "Pragma": "no-cache",
    "Referer": "https://www.kimi.com/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
}


class KimiError(Exception):
    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def _build_headers(
    config: KimiConfig,
    token: str | None = None,
    stream: bool = True,
) -> dict[str, str]:
    """构造请求头。同一账号 device/UA 稳定（登录时记录），绝不每请求随机。"""
    ua = config.user_agent or USER_AGENTS[0]
    headers = {
        **BASE_HEADERS,
        "User-Agent": ua,
        "Accept": "text/event-stream" if stream else "application/json",
        "X-Request-Id": uuid.uuid4().hex,
    }
    if config.device_model:
        headers["X-Device-Model"] = config.device_model
    tok = (token if token is not None else config.auth_token) or ""
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    if config.cookie:
        headers["Cookie"] = config.cookie
    return headers


def _text_of(msg: dict) -> str:
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        out = ""
        for item in c:
            if isinstance(item, dict) and item.get("type") == "text":
                out += item.get("text") or ""
        return out
    return ""


def _estimate_tokens(text: str) -> int:
    """估算 token 数：中文约 1.5 字/token，其他约 4 字符/token。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return max(1, int(cjk / 1.5 + other / 4))


def _usage(payload: dict, text: str) -> dict:
    msgs = payload.get("messages") or []
    p = max(1, _estimate_tokens("".join(_text_of(m) for m in msgs if isinstance(m, dict))))
    c = max(1, _estimate_tokens(text or ""))
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}


def _tier_for(effort: str, model: str) -> str:
    """客户端 reasoning_effort → Kimi 三档思考：标准 / 进阶 / 极致。

    K2.6（网页标注"快速"）没有极致档，请求极致自动降为进阶。
    """
    e = (effort or "").strip().lower()
    m = (model or "").lower()
    if e in ("max", "highest", "extreme"):
        return "进阶" if "k2.6" in m else "极致"
    if e in ("medium", "high"):
        return "进阶"
    return "标准"


def _effort_to_payload(effort: str, model: str = "") -> dict:
    """把客户端 reasoning_effort 收敛成 Kimi 网页版的思考开关。

    Kimi 网页版思考模式三档：标准（不思考）/ 进阶 / 极致（K2.6 无极致档）。
    上游具体字段以真实抓包为准；当前按 thinking.type 开关映射：
      标准 → disabled；进阶/极致 → enabled。
    """
    tier = _tier_for(effort, model)
    return {"thinking": {"type": "disabled" if tier == "标准" else "enabled"}}


class KimiWebClient:
    def __init__(self, config: KimiConfig, logger: Logger) -> None:
        self.config = config
        self.logger = logger
        # 会话缓存：conv_key -> (chat_id, last_used, turns)
        self._conv_lock = threading.Lock()
        self._conv_cache: dict[str, tuple[str, float, int]] = {}
        self._persist_path = str(getattr(config, "session_store_path", "") or "")
        self._dirty = False
        self._last_persist = 0.0
        self._persist_min_interval = 5.0
        if self._persist_path:
            self._load_conv_cache()
        self._last_call_at: float = 0.0
        self._last_call_lock = threading.Lock()

    # ── 会话缓存持久化 ──────────────────────────────────
    def _load_conv_cache(self) -> None:
        path = self._persist_path
        if not path:
            return
        try:
            import os

            if not os.path.exists(path):
                return
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as exc:  # noqa: BLE001
            self.logger.debug("Kimi 会话缓存载入失败: %s", exc)
            return
        if not isinstance(raw, dict):
            return
        now = time.time()
        loaded = 0
        for key, item in raw.items():
            if not isinstance(key, str) or not isinstance(item, dict):
                continue
            chat_id = item.get("chat_id")
            if not isinstance(chat_id, str) or not chat_id:
                continue
            try:
                ts = float(item.get("last_used") or now)
                turns = int(item.get("turns") or 0)
            except (TypeError, ValueError):
                continue
            if now - ts > self.config.session_cache_ttl:
                continue
            self._conv_cache[key] = (chat_id, ts, turns)
            loaded += 1
        while len(self._conv_cache) > self.config.session_cache_max_size:
            oldest = min(self._conv_cache.items(), key=lambda kv: kv[1][1])[0]
            self._conv_cache.pop(oldest, None)
        if loaded:
            self.logger.info("Kimi 会话缓存恢复 %d 条（%s）", loaded, path)

    def _persist_conv(self, force: bool = False) -> None:
        """调用方需持有 self._conv_lock。节流写盘，异常只记 debug。"""
        path = self._persist_path
        if not path:
            return
        now = time.time()
        if not force and now - self._last_persist < self._persist_min_interval:
            self._dirty = True
            return
        try:
            import os

            data = {
                k: {"chat_id": v[0], "last_used": v[1], "turns": v[2]}
                for k, v in self._conv_cache.items()
            }
            tmp = f"{path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, path)
            self._dirty = False
            self._last_persist = now
        except Exception as exc:  # noqa: BLE001
            self.logger.debug("Kimi 会话缓存落盘失败: %s", exc)

    def flush_cache(self) -> None:
        with self._conv_lock:
            if self._persist_path and self._dirty:
                self._persist_conv(force=True)

    def _conv_key(self, payload: dict) -> str:
        """稳定会话指纹：system + 首条 user + model + key_id。带 key_id 做租户隔离。"""
        msgs = payload.get("messages") or []
        first = ""
        system = ""
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            text = _text_of(m)
            if role == "system" and not system:
                system = text
            elif role == "user" and not first:
                first = text
        raw = (f"{system[:200]}\x00{first[:500]}\x00"
               f"{payload.get('model') or ''}\x00{payload.get('_wapi_key_id') or ''}")
        return hashlib.sha1(raw.encode("utf-8", "ignore")).hexdigest()

    def _cached_conv(self, key: str) -> str:
        if not self.config.session_cache_enabled:
            return ""
        with self._conv_lock:
            item = self._conv_cache.get(key)
            if not item:
                return ""
            chat_id, ts, turns = item
            if time.time() - ts > self.config.session_cache_ttl:
                self._conv_cache.pop(key, None)
                self._persist_conv()
                self._bump("stale")
                return ""
            max_turns = int(getattr(self.config, "max_turns_per_session", 40) or 40)
            if turns >= max_turns:
                self._conv_cache.pop(key, None)
                self._persist_conv()
                self._bump("stale")
                self.logger.info("会话轮次已达上限(%d) → 新建会话", max_turns)
                return ""
            self._conv_cache[key] = (chat_id, time.time(), turns)
            self._persist_conv()
            self._bump("hit")
            return chat_id

    def risk_stats(self) -> dict:
        """风控指标（面板展示，进程内累计）。"""
        with self._conv_lock:
            total = len(self._conv_cache)
            now = time.time()
            active = sum(1 for (_c, ts, _t) in self._conv_cache.values()
                         if now - ts <= self.config.session_cache_ttl)
            hit = getattr(self, "_stat_hit", 0)
            miss = getattr(self, "_stat_miss", 0)
            return {
                "sessions_created": getattr(self, "_stat_created", 0),
                "sessions_deleted": getattr(self, "_stat_deleted", 0),
                "cache_hit": hit,
                "cache_miss": miss,
                "cache_hit_rate": round(hit / (hit + miss), 3) if (hit + miss) else 0.0,
                "cache_stale": getattr(self, "_stat_stale", 0),
                "cache_size": total,
                "active_sessions": active,
                "upstream_requests": getattr(self, "_stat_upstream", 0),
                "rate_limit_hits": getattr(self, "_stat_rate_limit", 0),
                "daily_count": 0,
                "daily_limit": int(getattr(self.config, "daily_request_limit", 0) or 0),
            }

    def _bump(self, name: str) -> None:
        setattr(self, f"_stat_{name}", getattr(self, f"_stat_{name}", 0) + 1)

    def _store_conv(self, key: str, chat_id: str) -> None:
        if not self.config.session_cache_enabled or not chat_id:
            return
        with self._conv_lock:
            if len(self._conv_cache) >= self.config.session_cache_max_size:
                oldest = min(self._conv_cache.items(), key=lambda kv: kv[1][1])[0]
                self._conv_cache.pop(oldest, None)
            prev = self._conv_cache.get(key)
            turns = (prev[2] + 1) if prev and prev[0] == chat_id else 1
            first_time = prev is None or prev[0] != chat_id
            self._conv_cache[key] = (chat_id, time.time(), turns)
            self._persist_conv()
            if first_time:
                self._bump("created")

    # ── HTTP 基础 ──────────────────────────────────────
    @staticmethod
    def _split(url: str) -> tuple[str, str]:
        parts = urlsplit(url)
        host = parts.hostname or "www.kimi.com"
        path = parts.path or "/"
        if parts.query:
            path = f"{path}?{parts.query}"
        return host, path

    def _conn(self, url: str, timeout: int):
        parts = urlsplit(url)
        host = parts.hostname or "www.kimi.com"
        port = parts.port
        if (parts.scheme or "https") == "http":
            conn = http.client.HTTPConnection(host, port or 80, timeout=max(int(timeout), 30))
        else:
            conn = http.client.HTTPSConnection(host, port or 443, timeout=max(int(timeout), 30))
        from ...egress import (
            EGRESS_CUSTOM,
            EGRESS_SYSTEM,
            apply_system_proxy,
            normalize_egress,
        )

        mode = normalize_egress(getattr(self.config, "egress", "system"))
        if mode == EGRESS_CUSTOM:
            proxy_url = str(getattr(self.config, "proxy", "") or "")
            from ...proxy import ProxyConfigError, parse_proxy
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
        return conn

    def _post_json(self, url: str, body: dict, headers: dict, timeout: int) -> dict:
        conn = self._conn(url, timeout)
        try:
            _, path = self._split(url)
            conn.request("POST", path, body=json.dumps(body).encode("utf-8"), headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
            if resp.status >= 400:
                raise KimiError(f"HTTP {resp.status}: {raw[:200].decode('utf-8', 'replace')}")
            try:
                return json.loads(raw.decode("utf-8"))
            except Exception as exc:  # noqa: BLE001
                raise KimiError(f"响应不是合法 JSON: {raw[:200]!r}") from exc
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _open_stream(self, url: str, body: dict, headers: dict, timeout: int):
        conn = self._conn(url, timeout)
        _, path = self._split(url)
        conn.request("POST", path, body=json.dumps(body).encode("utf-8"), headers=headers)
        resp = conn.getresponse()
        if resp.status >= 400:
            raw = resp.read()
            conn.close()
            raise KimiError(f"HTTP {resp.status}: {raw[:200].decode('utf-8', 'replace')}")
        return conn, resp

    # ── 请求体 ─────────────────────────────────────────
    def _body(self, payload: dict, chat_id: str = "") -> dict:
        """构造 /api/chat 请求体。

        注意：网页版是多轮服务端会话制（传 chat_session_id 续聊），
        这里按 OpenAI 全量消息发送，仅在有缓存会话时带上续聊字段。
        """
        messages = payload.get("messages") or []
        model = str(payload.get("model") or self.config.default_model)
        body: dict = {
            "name": model,
            "model_name": model,
            "messages": messages,
            "stream": True,
        }
        body.update(_effort_to_payload(
            str(payload.get("reasoning_effort") or self.config.default_effort), model))
        if self.config.search_enabled:
            body["search"] = {"type": "auto"}
        if chat_id:
            body["chat_id"] = chat_id
        return body

    def _throttle(self) -> None:
        """防风控节奏：硬性最小间隔 + 长尾延迟（大多数短、少量长停顿）。"""
        if not self.config.anti_rate_limit_enabled:
            return
        min_interval = float(getattr(self.config, "min_request_interval", 0.0) or 0.0)
        if min_interval > 0:
            with self._last_call_lock:
                now = time.time()
                slot = max(now, self._last_call_at + min_interval)
                self._last_call_at = slot
            wait = slot - time.time()
            if wait > 0:
                time.sleep(wait)

        lo = max(0.0, float(self.config.anti_rate_limit_min_delay))
        hi = max(lo, float(self.config.anti_rate_limit_max_delay))
        if hi <= 0:
            return

        chance = float(getattr(self.config, "long_pause_chance", 0.02) or 0.0)
        if chance > 0 and random.random() < chance:
            delay = random.uniform(20, 60)
        else:
            import math
            mid = max(0.1, (lo + hi) / 2)
            delay = random.lognormvariate(math.log(mid), 0.6)
        delay = min(max(delay, lo), max(1.0, hi * 3))
        self.logger.info("AntiRateLimit | sleep=%.2fs before upstream request", delay)
        time.sleep(delay)

    # ── 会话清理（供后台清理线程调用） ─────────────────
    def cleanup_stale_sessions(self) -> None:
        """删除已过期/超轮次的 Kimi 会话，并从缓存移除。"""
        if not self.config.delete_conversation:
            return
        now = time.time()
        ttl = self.config.session_cache_ttl
        max_turns = int(getattr(self.config, "max_turns_per_session", 40) or 40)
        stale: list[tuple[str, str]] = []
        with self._conv_lock:
            for key, (chat_id, ts, turns) in list(self._conv_cache.items()):
                if now - ts > ttl or turns >= max_turns:
                    stale.append((key, chat_id))
                    self._conv_cache.pop(key, None)
            if stale:
                self._persist_conv(force=True)
        if not stale:
            return
        removed = 0
        for _key, chat_id in stale:
            try:
                self._post_json(
                    self.config.chat_delete_url(chat_id), {},
                    _build_headers(self.config, stream=False),
                    min(self.config.request_timeout, 15),
                )
                removed += 1
            except Exception:  # noqa: BLE001
                pass
        if removed:
            self.logger.info("Kimi 会话清理：删除 %d 个过期会话", removed)
            for _ in range(removed):
                self._bump("deleted")

    # ── SSE 解析 ───────────────────────────────────────
    @staticmethod
    def _parse_event(raw: dict) -> tuple[str, str]:
        """从 Kimi SSE 事件提取 (text_delta, reasoning_delta)。

        兼容两种常见形态：
          {"choices":[{"delta":{"content":..,"reasoning_content":..}}]}
          {"type":"content|think","content":..}
        """
        if not isinstance(raw, dict):
            return "", ""
        choices = raw.get("choices")
        if isinstance(choices, list) and choices:
            delta = choices[0].get("delta") or {}
            if isinstance(delta, dict):
                return (str(delta.get("content") or ""),
                        str(delta.get("reasoning_content") or ""))
        t = str(raw.get("type") or "")
        c = raw.get("content")
        if isinstance(c, str):
            if t in ("think", "reasoning", "thinking"):
                return "", c
            return c, ""
        return "", ""

    def _iter_events(self, resp):
        buffer = ""
        while True:
            line = resp.readline()
            if not line:
                break
            text = line.decode("utf-8", "replace").rstrip("\r\n")
            if text.startswith("data:"):
                buffer += text[5:].strip()
            elif text == "":
                if buffer:
                    if buffer != "[DONE]":
                        try:
                            yield json.loads(buffer)
                        except Exception:
                            pass
                    buffer = ""
        if buffer and buffer != "[DONE]":
            try:
                yield json.loads(buffer)
            except Exception:
                pass

    def _stream_events(self, payload: dict):
        model = payload.get("model") or self.config.default_model
        conv_key = self._conv_key(payload)
        cached = self._cached_conv(conv_key)
        if not cached:
            self._bump("miss")

        self._throttle()
        self._bump("upstream")
        conn, resp = self._open_stream(
            self.config.chat_url,
            self._body(payload, chat_id=cached),
            _build_headers(self.config),
            self.config.request_timeout,
        )
        chat_id = ""
        text = ""
        reasoning = ""
        try:
            for event in self._iter_events(resp):
                cid = event.get("chat_id") or event.get("id")
                if isinstance(cid, str) and cid and cid != chat_id:
                    chat_id = cid
                d_text, d_reason = self._parse_event(event)
                text += d_text
                reasoning += d_reason
                yield {"text": text, "reasoning": reasoning, "chat_id": chat_id}
        finally:
            try:
                resp.read()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass

        if chat_id and self.config.session_cache_enabled:
            self._store_conv(conv_key, chat_id)
        yield {"__done__": True, "text": text, "reasoning": reasoning, "chat_id": chat_id}

    # ── 对外接口 ───────────────────────────────────────
    def chat_completion(self, payload: dict) -> dict:
        model = payload.get("model") or self.config.default_model
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        text = reasoning = ""
        for ev in self._stream_events(payload):
            text, reasoning = ev["text"], ev["reasoning"]

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text,
                            "reasoning_content": reasoning or None},
                "finish_reason": "stop",
            }],
            "usage": _usage(payload, text),
        }

    def stream_chat_completion(self, payload: dict):
        model = payload.get("model") or self.config.default_model
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        def chunk(delta: dict, finish=None, usage=None) -> str:
            obj = {
                "id": completion_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            if usage is not None:
                obj["usage"] = usage
            return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

        yield chunk({"role": "assistant", "content": ""})

        sent_text = ""
        sent_reasoning = ""
        try:
            for ev in self._stream_events(payload):
                dr = ev["reasoning"][len(sent_reasoning):]
                if dr:
                    sent_reasoning = ev["reasoning"]
                    yield chunk({"reasoning_content": dr})
                dt = ev["text"][len(sent_text):]
                if dt:
                    sent_text = ev["text"]
                    yield chunk({"content": dt})
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Kimi 流式请求失败: %s", exc)
            err = {"error": {"message": f"上游错误: {exc}", "type": "upstream_error"}}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

        yield chunk({}, usage=_usage(payload, sent_text))
        yield chunk({}, finish="stop")
        yield "data: [DONE]\n\n"
