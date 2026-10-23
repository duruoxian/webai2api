"""通义千问网页版（chat.qwen.ai，OpenWebUI 架构）客户端。

把网页版接口封装成 OpenAI 兼容的 chat_completion / stream_chat_completion。
协议（2026-10 实测）：
  1. Authorization: Bearer <access_token>（登录时从浏览器抓取）
  2. POST /api/v2/chat/completions?chat_id=<会话id>，返回 SSE（OpenAI chunk 形态）
  3. 会话缓存：同一对话复用 chat_id（OpenWebUI 服务端保存历史）
"""
from __future__ import annotations

import hashlib
import http.client
import json
import random
import threading
import time
import urllib.error
import urllib.request
import uuid
from logging import Logger
from urllib.parse import quote, urlsplit

from .config import QwenConfig

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
    "Origin": "https://chat.qwen.ai",
    "Pragma": "no-cache",
    "Referer": "https://chat.qwen.ai/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
}


class QwenError(Exception):
    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def _build_headers(
    config: QwenConfig,
    token: str | None = None,
    stream: bool = True,
) -> dict[str, str]:
    """构造请求头。UA 用登录时记录的真实画像，绝不每请求随机。"""
    ua = config.user_agent or USER_AGENTS[0]
    headers = {
        **BASE_HEADERS,
        "User-Agent": ua,
        "Accept": "text/event-stream" if stream else "application/json",
        "X-Request-Id": uuid.uuid4().hex,
    }
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


def _supports_auto(model: str) -> bool:
    """Qwen3.7-Plus / Qwen3.8-Max 有「自动」档；Omni-Flash 没有。"""
    m = (model or "").lower()
    return "flash" not in m


def _qwen_tier(effort: str, model: str) -> str:
    """客户端 reasoning_effort → 千问思考档：自动 / 思考 / 快速。

    - 自动（auto）：由模型自行决定是否思考（auto_thinking）
    - 思考（thinking）：强制思考
    - 快速（fast）：不思考
    Omni-Flash 只有 思考 / 快速 两档，请求"自动"降为"思考"。
    """
    e = (effort or "").strip().lower()
    if e in ("minimal", "auto", "medium"):
        return "自动" if _supports_auto(model) else "思考"
    if e in ("high", "max", "highest", "extreme"):
        return "思考"
    if e in ("none", "low", "fast"):
        return "快速"
    # 未指定 → 自动（Flash 无自动档 → 思考）
    return "自动" if _supports_auto(model) else "思考"


def _effort_to_payload(effort: str, model: str = "") -> dict:
    """客户端 reasoning_effort → OpenWebUI 的思考开关。

    自动 → thinking_enabled=False + auto_thinking=True
    思考 → thinking_enabled=True
    快速 → thinking_enabled=False
    """
    tier = _qwen_tier(effort, model)
    if tier == "自动":
        return {"thinking_enabled": False, "auto_thinking": True}
    if tier == "思考":
        return {"thinking_enabled": True, "auto_thinking": False}
    return {"thinking_enabled": False, "auto_thinking": False}


class QwenWebClient:
    def __init__(self, config: QwenConfig, logger: Logger) -> None:
        self.config = config
        self.logger = logger
        # access_token 只有 15 分钟：用登录 cookie 调 /auths/refresh 自动续期
        self._token_lock = threading.Lock()
        self._access_token = str(getattr(config, "auth_token", "") or "")
        self._token_checked_at = 0.0
        self._token_ttl = 300.0
        self._conv_lock = threading.Lock()
        self._conv_cache: dict[str, tuple[str, float, int, str]] = {}
        self._persist_path = str(getattr(config, "session_store_path", "") or "")
        self._relay_port: int = 0
        self._relay_lock = threading.Lock()
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
            self.logger.debug("Qwen 会话缓存载入失败: %s", exc)
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
            response_id = str(item.get("response_id") or "")
            self._conv_cache[key] = (chat_id, ts, turns, response_id)
            loaded += 1
        while len(self._conv_cache) > self.config.session_cache_max_size:
            oldest = min(self._conv_cache.items(), key=lambda kv: kv[1][1])[0]
            self._conv_cache.pop(oldest, None)
        if loaded:
            self.logger.info("Qwen 会话缓存恢复 %d 条（%s）", loaded, path)

    def _persist_conv(self, force: bool = False) -> None:
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
                k: {
                    "chat_id": v[0],
                    "last_used": v[1],
                    "turns": v[2],
                    "response_id": v[3],
                }
                for k, v in self._conv_cache.items()
            }
            tmp = f"{path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, path)
            self._dirty = False
            self._last_persist = now
        except Exception as exc:  # noqa: BLE001
            self.logger.debug("Qwen 会话缓存落盘失败: %s", exc)

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
            chat_id, ts, turns, response_id = item
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
            self._conv_cache[key] = (chat_id, time.time(), turns, response_id)
            self._persist_conv()
            self._bump("hit")
            return chat_id

    def _cached_response_id(self, key: str) -> str:
        with self._conv_lock:
            item = self._conv_cache.get(key)
            return item[3] if item else ""

    def risk_stats(self) -> dict:
        with self._conv_lock:
            total = len(self._conv_cache)
            now = time.time()
            active = sum(1 for (_c, ts, _t, _r) in self._conv_cache.values()
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

    def _store_conv(self, key: str, chat_id: str, response_id: str = "") -> None:
        if not self.config.session_cache_enabled or not chat_id:
            return
        with self._conv_lock:
            if len(self._conv_cache) >= self.config.session_cache_max_size:
                oldest = min(self._conv_cache.items(), key=lambda kv: kv[1][1])[0]
                self._conv_cache.pop(oldest, None)
            prev = self._conv_cache.get(key)
            turns = (prev[2] + 1) if prev and prev[0] == chat_id else 1
            first_time = prev is None or prev[0] != chat_id
            resp = response_id or (prev[3] if prev and prev[0] == chat_id else "")
            self._conv_cache[key] = (chat_id, time.time(), turns, resp)
            self._persist_conv()
            if first_time:
                self._bump("created")

    # ── HTTP 基础 ──────────────────────────────────────
    @staticmethod
    def _split(url: str) -> tuple[str, str]:
        parts = urlsplit(url)
        host = parts.hostname or "chat.qwen.ai"
        path = parts.path or "/"
        if parts.query:
            path = f"{path}?{parts.query}"
        return host, path

    def _conn(self, url: str, timeout: int):
        parts = urlsplit(url)
        host = parts.hostname or "chat.qwen.ai"
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
                if resp.status == 429:
                    self._bump("rate_limit")
                raise QwenError(f"HTTP {resp.status}: {raw[:200].decode('utf-8', 'replace')}")
            try:
                return json.loads(raw.decode("utf-8"))
            except Exception as exc:  # noqa: BLE001
                raise QwenError(f"响应不是合法 JSON: {raw[:200]!r}") from exc
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
            if resp.status == 429:
                self._bump("rate_limit")
            raw = resp.read()
            conn.close()
            raise QwenError(f"HTTP {resp.status}: {raw[:200].decode('utf-8', 'replace')}")
        return conn, resp

    # ── access_token 自动续期 ───────────────────────────
    def _valid_token(self, force: bool = False) -> str:
        """返回可用的 access_token；过期时用登录 cookie 换新的。

        Qwen 的 JWT 只有 15 分钟，续期接口：
          GET /api/v2/auths/refresh（withCredentials，靠 cookie 鉴权）
        返回 {"success":true,"data":{"access_token":...,"expires_at":...}}
        """
        with self._token_lock:
            now = time.time()
            if not force and self._access_token and now - self._token_checked_at < self._token_ttl:
                return self._access_token
            if not self.config.cookie:
                self._token_checked_at = now
                return self._access_token
            url = self.config.refresh_url
            headers = {
                "Accept": "application/json",
                "User-Agent": self.config.user_agent or USER_AGENTS[0],
                "Origin": self.config.base_url,
                "Referer": self.config.base_url + "/",
                "Cookie": self.config.cookie,
            }
            try:
                conn = self._conn(url, 20)
                _, path = self._split(url)
                conn.request("GET", path, headers=headers)
                resp = conn.getresponse()
                raw = resp.read()
                conn.close()
                if resp.status >= 400:
                    if resp.status == 429:
                        self._bump("rate_limit")
                    self.logger.warning("Qwen 续期失败 HTTP %s", resp.status)
                    self._token_checked_at = now
                    return self._access_token
                data = json.loads(raw.decode("utf-8"))
                inner = data.get("data") if isinstance(data.get("data"), dict) else {}
                acc = str(inner.get("access_token") or "")
                if acc:
                    self._access_token = acc
                    self.config.auth_token = acc
                    self.logger.info("Qwen access_token 已自动续期")
                self._token_checked_at = now
                return self._access_token
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Qwen 续期异常: %s", exc)
                self._token_checked_at = now
                return self._access_token

    # ── 请求体 ─────────────────────────────────────────
    def _body(self, payload: dict, chat_id: str = "") -> dict:
        model = str(payload.get("model") or self.config.default_model)
        body: dict = {
            "model": model,
            "messages": payload.get("messages") or [],
            "stream": True,
            "chat_id": chat_id,
            "incremental_output": True,
        }
        body.update(_effort_to_payload(
            str(payload.get("reasoning_effort") or self.config.default_effort), model))
        if self.config.search_enabled:
            body["web_search"] = "auto"
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
        """删除已过期/超轮次的 Qwen 会话，并从缓存移除。"""
        if not self.config.delete_conversation:
            return
        now = time.time()
        ttl = self.config.session_cache_ttl
        max_turns = int(getattr(self.config, "max_turns_per_session", 40) or 40)
        stale: list[tuple[str, str]] = []
        with self._conv_lock:
            for key, (chat_id, ts, turns, _resp) in list(self._conv_cache.items()):
                if now - ts > ttl or turns >= max_turns:
                    stale.append((key, chat_id))
                    self._conv_cache.pop(key, None)
            if stale:
                self._persist_conv(force=True)
        if not stale:
            return
        removed = 0
        port = int(self._relay_port or 0)
        for _key, chat_id in stale:
            if port and self._relay_delete(port, chat_id):
                removed += 1
                continue
            try:
                conn = self._conn(self.config.chat_delete_url(chat_id), 15)
                _, path = self._split(self.config.chat_delete_url(chat_id))
                conn.request("DELETE", path, headers=_build_headers(self.config, stream=False))
                resp = conn.getresponse()
                resp.read()
                conn.close()
                removed += 1
            except Exception:  # noqa: BLE001
                pass
        if removed:
            self.logger.info("Qwen 会话清理：删除 %d 个过期会话", removed)
            for _ in range(removed):
                self._bump("deleted")

    def _relay_delete(self, port: int, chat_id: str) -> bool:
        """优先用中继删除会话（绕风控）；失败返回 False 让调用方走直连兜底。"""
        try:
            raw = json.dumps({"chat_id": chat_id}, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/delete", data=raw, method="POST",
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=35) as resp:
                data = json.loads(resp.read().decode("utf-8", "replace") or "{}")
            return bool(data.get("ok"))
        except Exception:  # noqa: BLE001
            return False

    # ── SSE 解析 ───────────────────────────────────────
    @staticmethod
    def _parse_event(raw: dict) -> tuple[str, str]:
        """从 OpenWebUI SSE 事件提取 (text_delta, reasoning_delta)。"""
        if not isinstance(raw, dict):
            return "", ""
        choices = raw.get("choices")
        if isinstance(choices, list) and choices:
            delta = choices[0].get("delta") or {}
            if isinstance(delta, dict):
                content = delta.get("content") or ""
                if not isinstance(content, str):
                    content = str(content)
                reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if not isinstance(reasoning, str):
                    reasoning = str(reasoning)
                return content, reasoning
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

    def _ensure_relay(self) -> int:
        with self._relay_lock:
            if self._relay_port:
                return self._relay_port
            from ...qwen_relay import RelayManager

            manager = RelayManager(self.config.account_id, self.logger)
            self._relay_port = manager.ensure()
            return self._relay_port

    def _stream_via_relay(self, payload: dict):
        conv_key = self._conv_key(payload)
        cached = self._cached_conv(conv_key)
        parent = self._cached_response_id(conv_key) if cached else ""
        if not cached:
            self._bump("miss")

        messages = payload.get("messages") or []
        user_text = ""
        system_text = ""
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            text = _text_of(m)
            if role == "system" and not system_text:
                system_text = text
            elif role == "user":
                user_text = text

        model = str(payload.get("model") or self.config.default_model)
        effort = str(payload.get("reasoning_effort") or self.config.default_effort)
        opts = _effort_to_payload(effort, model)
        tier = _qwen_tier(effort, model)
        mode = {"自动": "Auto", "思考": "Thinking", "快速": "Fast"}.get(tier, "Auto")

        req_body = {
            "prompt": user_text,
            "system": system_text,
            "model": model,
            "chat_id": cached or "",
            "parent_id": parent or "",
            "thinking_enabled": bool(opts.get("thinking_enabled")),
            "auto_thinking": bool(opts.get("auto_thinking")),
            "thinking_mode": mode,
            "auto_search": bool(self.config.search_enabled),
            "base_url": self.config.base_url,
        }
        self._throttle()
        self._bump("upstream")
        try:
            port = self._ensure_relay()
        except Exception as exc:  # noqa: BLE001
            self._relay_port = 0
            raise QwenError(f"通义千问中继启动失败: {exc}") from exc

        url = f"http://127.0.0.1:{port}/chat"
        raw = json.dumps(req_body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url, data=raw, method="POST",
            headers={"Content-Type": "application/json",
                     "Accept": "text/event-stream"})
        try:
            resp = urllib.request.urlopen(req, timeout=self.config.request_timeout)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
            self._relay_port = 0
            raise QwenError(f"通义千问中继不可用: HTTP {exc.code} {detail}") from exc
        except Exception as exc:  # noqa: BLE001
            self._relay_port = 0
            raise QwenError(f"通义千问中继不可用: {exc}") from exc

        chat_id = cached
        response_id = parent
        text = ""
        reasoning = ""
        try:
            for line in resp:
                try:
                    line = line.decode("utf-8", "replace").strip()
                except Exception:
                    continue
                if not line.startswith("data:"):
                    continue
                payload_txt = line[5:].strip()
                if not payload_txt:
                    continue
                try:
                    event = json.loads(payload_txt)
                except Exception:
                    continue
                typ = event.get("type")
                if typ == "meta":
                    cid = event.get("chat_id")
                    if isinstance(cid, str) and cid:
                        chat_id = cid
                    rid = event.get("response_id")
                    if isinstance(rid, str) and rid:
                        response_id = rid
                    yield {"text": text, "reasoning": reasoning,
                           "chat_id": chat_id or response_id}
                elif typ == "reasoning":
                    reasoning += str(event.get("content") or "")
                    yield {"text": text, "reasoning": reasoning,
                           "chat_id": chat_id or response_id}
                elif typ == "delta":
                    text += str(event.get("content") or "")
                    yield {"text": text, "reasoning": reasoning,
                           "chat_id": chat_id or response_id}
                elif typ == "done":
                    cid = event.get("chat_id")
                    if isinstance(cid, str) and cid:
                        chat_id = cid
                    rid = event.get("response_id")
                    if isinstance(rid, str) and rid:
                        response_id = rid
                    break
                elif typ == "error":
                    self.logger.warning("Qwen 中继错误: %s", event)
                    msg = str(event.get("message") or "中继请求失败")
                    body = event.get("body")
                    if body:
                        msg = f"{msg} | {str(body)[:400]}"
                    raise QwenError(msg, int(event.get("status") or 502))
        finally:
            try:
                resp.close()
            except Exception:
                pass

        if chat_id and self.config.session_cache_enabled:
            self._store_conv(conv_key, chat_id, response_id)
        yield {"__done__": True, "text": text, "reasoning": reasoning,
               "chat_id": chat_id or response_id}

    def _stream_events(self, payload: dict):
        if self.config.relay_enabled and self.config.account_id:
            yield from self._stream_via_relay(payload)
            return

        conv_key = self._conv_key(payload)
        cached = self._cached_conv(conv_key)
        if not cached:
            self._bump("miss")

        self._throttle()
        self._bump("upstream")
        url = self.config.chat_url
        if cached:
            url = f"{url}?chat_id={quote(cached)}"
        body = self._body(payload, chat_id=cached)
        try:
            conn, resp = self._open_stream(
                url, body, _build_headers(self.config, self._valid_token()),
                self.config.request_timeout)
        except QwenError as exc:
            if "401" not in str(exc):
                raise
            self.logger.info("Qwen 返回 401，尝试续期后重试")
            conn, resp = self._open_stream(
                url, body, _build_headers(self.config, self._valid_token(force=True)),
                self.config.request_timeout)
        chat_id = cached
        text = ""
        reasoning = ""
        try:
            for event in self._iter_events(resp):
                cid = event.get("chat_id")
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
            self.logger.warning("Qwen 流式请求失败: %s", exc)
            err = {"error": {"message": f"上游错误: {exc}", "type": "upstream_error"}}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

        yield chunk({}, usage=_usage(payload, sent_text))
        yield chunk({}, finish="stop")
        yield "data: [DONE]\n\n"
