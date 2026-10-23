"""智谱清言（chatglm.cn）客户端。

把网页版接口封装成 OpenAI 兼容的 chat_completion / stream_chat_completion。
协议（参考开源项目 glm-free-api 的思路，MIT）：
  1. 用 refresh_token 换 access_token（有效期约 1 小时，本客户端内缓存）
  2. POST /chatglm/backend-api/assistant/stream 拿 SSE，逐块解析 parts
  3. 结束后（可选）删除会话，避免污染用户对话列表
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

from .config import GlmConfig

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
]

BASE_HEADERS = {
    "Accept": "text/event-stream",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "App-Name": "chatglm",
    "Cache-Control": "no-cache",
    "Content-Type": "application/json",
    "Origin": "https://chatglm.cn",
    "Pragma": "no-cache",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "X-App-Fr": "browser_extension",
    "X-App-Platform": "pc",
    "X-App-Version": "0.0.1",
    "X-Device-Brand": "",
    "X-Device-Model": "",
    "X-Lang": "zh",
}

ACCESS_TOKEN_TTL = 3600


class GlmError(Exception):
    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def _make_sign(secret: str) -> tuple[str, str, str]:
    """官网时间戳签名算法。"""
    a = str(int(time.time() * 1000))
    n = len(a)
    digits = [int(ch) for ch in a]
    r = (sum(digits) - digits[n - 2]) % 10
    timestamp = a[: n - 2] + str(r) + a[n - 1]
    nonce = uuid.uuid4().hex
    sign = hashlib.md5(f"{timestamp}-{nonce}-{secret}".encode()).hexdigest()
    return timestamp, nonce, sign


def _build_headers(
    secret: str,
    token: str | None = None,
    device_id: str = "",
    user_agent: str = "",
    device_model: str = "",
) -> dict[str, str]:
    """构造请求头。

    约定：同一账号的 device_id / UA 必须稳定（否则同 token 反复换设备，是强风控信号）。
    device_id 为空时才退回按 UA 稳定派生，绝不每次随机。
    """
    ts, nonce, sign = _make_sign(secret)
    ua = user_agent or USER_AGENTS[0]
    if device_id:
        did = device_id
    else:
        did = hashlib.sha256(f"glm-device:{ua}".encode()).hexdigest()[:32]
    headers = {
        **BASE_HEADERS,
        "User-Agent": ua,
        "X-Device-Id": did,
        "X-Request-Id": uuid.uuid4().hex,   # 每请求唯一是正常的
        "X-Sign": sign,
        "X-Timestamp": ts,
        "X-Nonce": nonce,
    }
    if device_model:
        headers["X-Device-Model"] = device_model
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _headers_for(config: GlmConfig, token: str | None = None) -> dict[str, str]:
    """按账号配置构造稳定指纹的请求头。"""
    return _build_headers(
        config.sign_secret,
        token,
        device_id=getattr(config, "device_id", "") or "",
        user_agent=getattr(config, "user_agent", "") or "",
        device_model=getattr(config, "device_model", "") or "",
    )


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


def _prepare_content(messages: list[dict]) -> str:
    """把 OpenAI 多轮消息合并成智谱要的单条 user 内容。"""
    if len(messages) < 2:
        return "".join(_text_of(m) + "\n" for m in messages)
    role_map = {"system": "<|sytstem|>", "assistant": "<|assistant|>", "user": "<|user|>"}
    parts = []
    for m in messages:
        role = role_map.get(m.get("role", "user"), m.get("role", "user"))
        parts.append(f"{role}\n{_text_of(m)}\n")
    return "".join(parts) + "<|assistant|>\n"


TOOL_MARKER = "<<<TOOL_CALL>>>"

TOOL_INSTRUCTION = """你可以调用以下工具来完成任务：

{tools}

规则：
1. 如果用户的问题需要调用工具，只输出一行，格式严格为：
{marker}{{"name":"工具名","arguments":{{"参数名":值}}}}
2. 一次只调用一个工具。不要输出任何解释文字。
3. 如果不需要工具，直接用自然语言回答，不要输出 {marker}。
"""


def _tools_prompt(tools: list) -> str:
    """把 OpenAI tools 定义压成给模型看的说明。"""
    lines = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") or t
        name = fn.get("name") or ""
        if not name:
            continue
        desc = fn.get("description") or ""
        params = fn.get("parameters") or {}
        lines.append(f"- {name}: {desc}")
        props = (params.get("properties") or {}) if isinstance(params, dict) else {}
        if props:
            req = set(params.get("required") or [])
            for k, v in props.items():
                typ = (v or {}).get("type", "string")
                mark = "（必填）" if k in req else ""
                lines.append(f"    参数 {k} ({typ}){mark}: {(v or {}).get('description','')}")
    return "\n".join(lines)


def _inject_tools(messages: list, tools: list) -> list:
    """把工具说明塞进 system（没有 system 就新建一条）。"""
    spec = _tools_prompt(tools)
    if not spec:
        return messages
    instruction = TOOL_INSTRUCTION.format(tools=spec, marker=TOOL_MARKER)
    msgs = list(messages or [])
    for i, m in enumerate(msgs):
        if m.get("role") == "system":
            merged = dict(m)
            merged["content"] = (m.get("content") or "") + "\n\n" + instruction
            msgs[i] = merged
            return msgs
    msgs.insert(0, {"role": "system", "content": instruction})
    return msgs


def _strip_tool_block(text: str) -> str:
    """去掉文本里的工具调用标记行。"""
    if TOOL_MARKER not in text:
        return text
    kept = [ln for ln in text.splitlines() if TOOL_MARKER not in ln]
    return "\n".join(kept).strip()


def _parse_tool_calls(text: str, tools: list) -> list:
    """从模型回复里解析工具调用，返回 OpenAI tool_calls 结构。"""
    if not text or not tools:
        return []
    names = set()
    for t in tools:
        if isinstance(t, dict):
            fn = t.get("function") or t
            if fn.get("name"):
                names.add(fn["name"])
    out = []
    for line in text.splitlines():
        idx = line.find(TOOL_MARKER)
        if idx < 0:
            continue
        raw = line[idx + len(TOOL_MARKER):].strip()
        try:
            obj = json.loads(raw)
        except Exception:
            try:
                s, e = raw.find("{"), raw.rfind("}")
                if s < 0 or e <= s:
                    continue
                obj = json.loads(raw[s:e + 1])
            except Exception:
                continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name") or obj.get("tool") or ""
        if not name or (names and name not in names):
            continue
        args = obj.get("arguments", obj.get("args", {}))
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        out.append({
            "id": f"call_{uuid.uuid4().hex[:16]}",
            "type": "function",
            "function": {"name": name, "arguments": args},
        })
        break
    return out


def _estimate_tokens(text: str) -> int:
    """估算 token 数：中文约 1.5 字/token，其他约 4 字符/token。

    智谱上游不返回真实 usage，只能估算；比单纯 len//4 更贴近实际。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return max(1, int(cjk / 1.5 + other / 4))


def _usage(payload: dict, text: str) -> dict:
    """估算用量（智谱上游不返回 token 数）。"""
    p = max(1, _estimate_tokens(_prepare_content(payload.get("messages") or [])))
    c = max(1, _estimate_tokens(text or ""))
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}


def _extract_parts(parts: list, is_silent: bool = False) -> tuple[str, str]:
    text = ""
    reasoning = ""
    for part in parts:
        if not isinstance(part, dict):
            continue
        content = part.get("content")
        if not isinstance(content, list):
            continue
        status = part.get("status")
        for item in content:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            if t == "text":
                text += item.get("text") or ""
            elif t == "think" and not is_silent:
                reasoning += item.get("think") or ""
            elif t == "code":
                code = item.get("code") or ""
                text += "```python\n" + code + ("\n```\n" if status == "finish" else "")
            elif t == "execution_output" and isinstance(item.get("content"), str) and status == "finish":
                text += item["content"] + "\n"
            elif t == "image" and isinstance(item.get("image"), list) and status == "finish":
                for img in item["image"]:
                    u = img.get("image_url") if isinstance(img, dict) else None
                    if isinstance(u, str) and (u.startswith("http://") or u.startswith("https://")):
                        text += f"![图像]({u})"
                text += "\n"
    return text, reasoning


class ChatGlmWebClient:
    def __init__(self, config: GlmConfig, logger: Logger) -> None:
        self.config = config
        self.logger = logger
        self._lock = threading.Lock()
        self._access_token = ""
        self._token_expire = 0.0
        # 会话缓存：同一段对话复用 conversation_id，多轮只发增量（与 DeepSeek 行为一致）
        self._conv_lock = threading.Lock()
        self._conv_cache: dict[str, tuple[str, float, int]] = {}
        # 落盘节流（重启后仍能复用会话，避免重新建会话）
        self._persist_path: str = str(getattr(config, "session_store_path", "") or "")
        self._dirty = False
        self._last_persist = 0.0
        self._persist_min_interval = 5.0
        if self._persist_path:
            self._load_conv_cache()
        # 同账号上游请求最小间隔
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
            self.logger.debug("GLM 会话缓存载入失败: %s", exc)
            return
        if not isinstance(raw, dict):
            return
        now = time.time()
        loaded = 0
        for key, item in raw.items():
            if not isinstance(key, str) or not isinstance(item, dict):
                continue
            conv_id = item.get("conversation_id")
            if not isinstance(conv_id, str) or not conv_id:
                continue
            try:
                ts = float(item.get("last_used") or now)
                turns = int(item.get("turns") or 0)
            except (TypeError, ValueError):
                continue
            if now - ts > self.config.session_cache_ttl:
                continue
            self._conv_cache[key] = (conv_id, ts, turns)
            loaded += 1
        while len(self._conv_cache) > self.config.session_cache_max_size:
            oldest = min(self._conv_cache.items(), key=lambda kv: kv[1][1])[0]
            self._conv_cache.pop(oldest, None)
        if loaded:
            self.logger.info("GLM 会话缓存恢复 %d 条（%s）", loaded, path)

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
                k: {
                    "conversation_id": v[0],
                    "last_used": v[1],
                    "turns": v[2],
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
            self.logger.debug("GLM 会话缓存落盘失败: %s", exc)

    def flush_cache(self) -> None:
        """强制落盘（关机前调用）。"""
        with self._conv_lock:
            if self._persist_path and self._dirty:
                self._persist_conv(force=True)

    def _conv_key(self, payload: dict) -> str:
        """稳定会话指纹：system + 首条 user + model + key_id。

        不随轮次/工具轮变化，避免每轮新建上游会话；带 key_id 做租户隔离。
        """
        msgs = payload.get("messages") or []
        first = ""
        system = ""
        for m in msgs:
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
            conv_id, ts, turns = item
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
            self._conv_cache[key] = (conv_id, time.time(), turns)
            self._persist_conv()
            self._bump("hit")
            return conv_id

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

    def _store_conv(self, key: str, conv_id: str) -> None:
        if not self.config.session_cache_enabled or not conv_id:
            return
        with self._conv_lock:
            if len(self._conv_cache) >= self.config.session_cache_max_size:
                oldest = min(self._conv_cache.items(), key=lambda kv: kv[1][1])[0]
                self._conv_cache.pop(oldest, None)
            prev = self._conv_cache.get(key)
            turns = (prev[2] + 1) if prev and prev[0] == conv_id else 1
            first_time = prev is None or prev[0] != conv_id
            self._conv_cache[key] = (conv_id, time.time(), turns)
            self._persist_conv()
            if first_time:
                self._bump("created")

    @staticmethod
    def _split(url: str) -> tuple[str, str]:
        parts = urlsplit(url)
        host = parts.hostname or "chatglm.cn"
        path = parts.path or "/"
        if parts.query:
            path = f"{path}?{parts.query}"
        return host, path

    def _conn(self, url: str, timeout: int) -> http.client.HTTPSConnection:
        host, _ = self._split(url)
        conn = http.client.HTTPSConnection(host, timeout=max(int(timeout), 30))
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
                raise GlmError(f"HTTP {resp.status}: {raw[:200].decode('utf-8', 'replace')}")
            try:
                return json.loads(raw.decode("utf-8"))
            except Exception as exc:  # noqa: BLE001
                raise GlmError(f"响应不是合法 JSON: {raw[:200]!r}") from exc
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
            raise GlmError(f"HTTP {resp.status}: {raw[:200].decode('utf-8', 'replace')}")
        return conn, resp

    # ── token ──────────────────────────────────────────
    def _access(self) -> str:
        with self._lock:
            if self._access_token and time.time() < self._token_expire:
                return self._access_token
            if not self.config.refresh_token:
                raise GlmError("缺少 refresh_token，请先在面板登录智谱清言账号", 401)
            data = self._post_json(
                self.config.refresh_url, {},
                _headers_for(self.config, self.config.refresh_token),
                min(self.config.request_timeout, 30),
            )
            code = data.get("code", data.get("status"))
            if code not in (0, None):
                msg = str(data.get("message", ""))
                if "40102" in msg:
                    raise GlmError("refresh_token 已过期，请重新登录智谱清言", 401)
                raise GlmError(f"刷新 access_token 失败: {msg}")
            result = data.get("result") or {}
            token = result.get("access_token") or ""
            if not token:
                raise GlmError("刷新 access_token 失败：响应缺少 access_token")
            self._access_token = token
            self._token_expire = time.time() + ACCESS_TOKEN_TTL - 60
            if result.get("refresh_token"):
                self.config.refresh_token = result["refresh_token"]
            return self._access_token

    def _assistant_id(self, model: str) -> str:
        # 24 位以上纯小写字母数字 → 视为智能体 ID
        if len(model) >= 24 and model.isalnum() and model.islower():
            return model
        return self.config.assistant_id

    @staticmethod
    def _resolve_mode(model: str, effort: str) -> tuple[str, str]:
        """把客户端的 reasoning_effort 收敛成智谱官网的三档。

        智谱官网只有三档：快速 / 深度 / 极致。
        客户端可能给五档（Default/Low/Medium/High/Max），映射如下：
          Default / Low        → 快速（不进入思考模式，不带 effort）
          Medium / High        → 深度（chat_mode=zero + effort=high）
          Max                  → 极致（chat_mode=zero + effort=max）
        """
        m = (model or "").lower()
        if "deepresearch" in m:
            return "deep_research", "max"
        e = (effort or "").strip().lower()
        if e in ("", "default", "none", "off", "auto", "low", "minimal"):
            return "", ""
        if e in ("medium", "high"):
            return "zero", "high"
        if e in ("max", "highest", "extreme"):
            return "zero", "max"
        return "zero", "high"

    def _body(self, content: str, model: str, conv_id: str = "", effort: str = "") -> dict:
        chat_mode, effort = self._resolve_mode(model, effort)
        meta = {
            "channel": "",
            "chat_mode": chat_mode or None,
            "draft_id": "",
            "if_plus_model": True,
            "input_question_type": "xxxx",
            "is_networking": bool(self.config.search_enabled),
            "is_test": False,
            "platform": "pc",
            "quote_log_id": "",
            "cogview": {"rm_label_watermark": False},
        }
        if effort:
            meta["reasoning_effort"] = effort
        return {
            "assistant_id": self._assistant_id(model),
            "conversation_id": conv_id,
            "project_id": "",
            "chat_type": "user_chat",
            "messages": [{"role": "user", "content": [{"type": "text", "text": content}]}],
            "meta_data": meta,
        }

    def _throttle(self) -> None:
        """防风控节奏：硬性最小间隔 + 长尾延迟（大多数短、少量长停顿）。"""
        if not self.config.anti_rate_limit_enabled:
            return
        # 同账号两次上游请求的硬性最小间隔
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

        # 长尾：少量请求出现 20~60s 长停顿，其余按对数正态分布
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
        """删除已过期/超轮次的智谱会话，并从缓存移除。

        与 DeepSeek 保持一致：低频批量清理，而不是每请求删除。
        """
        now = time.time()
        ttl = self.config.session_cache_ttl
        max_turns = int(getattr(self.config, "max_turns_per_session", 40) or 40)
        stale: list[tuple[str, str]] = []
        with self._conv_lock:
            for key, (conv_id, ts, turns) in list(self._conv_cache.items()):
                if now - ts > ttl or turns >= max_turns:
                    stale.append((key, conv_id))
                    self._conv_cache.pop(key, None)
            if stale:
                self._persist_conv(force=True)
        if not stale:
            return
        try:
            token = self._access()
        except Exception as exc:  # noqa: BLE001
            self.logger.debug("会话清理跳过（token 获取失败）: %s", exc)
            return
        removed = 0
        for _key, conv_id in stale:
            try:
                self._delete_conversation(conv_id, token)
                removed += 1
            except Exception:  # noqa: BLE001
                pass
        if removed:
            self.logger.info("GLM 会话清理：删除 %d 个过期会话", removed)
            for _ in range(removed):
                self._bump("deleted")

    # ── 事件流 ─────────────────────────────────────────
    def _iter_events(self, resp):
        """逐行读 SSE，产出 data 里的 JSON dict。"""
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
                    try:
                        yield json.loads(buffer)
                    except Exception:
                        pass
                    buffer = ""
        if buffer:
            try:
                yield json.loads(buffer)
            except Exception:
                pass

    def _stream_events(self, payload: dict):
        """执行一次对话，逐事件产出 (full_text, full_reasoning, conv_id)。

        流结束后 yield 一个 dict 形式的收尾对象（键 __done__）。
        """
        model = payload.get("model") or self.config.default_model
        messages = payload.get("messages") or []
        if payload.get("tools"):
            messages = _inject_tools(messages, payload["tools"])
        content = _prepare_content(messages)
        is_silent = "silent" in model

        self._throttle()
        self._bump("upstream")
        token = self._access()
        # 命中会话缓存则复用 conversation_id（与 DeepSeek 的多轮增量行为一致）
        conv_key = self._conv_key(payload)
        cached_conv = self._cached_conv(conv_key) or ""
        if not cached_conv:
            self._bump("miss")
        conn, resp = self._open_stream(
            self.config.stream_url,
            self._body(
                content, model,
                conv_id=cached_conv,
                effort=str(payload.get("reasoning_effort") or self.config.default_effort),
            ),
            _headers_for(self.config, token),
            self.config.request_timeout,
        )
        conv_id = ""
        parts_cache: list = []
        try:
            for event in self._iter_events(resp):
                if not isinstance(event, dict):
                    continue
                if event.get("conversation_id"):
                    conv_id = event["conversation_id"]
                status = event.get("status")
                if isinstance(event.get("parts"), list):
                    parts_cache = event["parts"]
                if status == "finish":
                    break
                text, reasoning = _extract_parts(parts_cache, is_silent)
                yield {"text": text, "reasoning": reasoning, "conv_id": conv_id}
        finally:
            try:
                resp.read()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass

        text, reasoning = _extract_parts(parts_cache, is_silent)
        if conv_id:
            if self.config.session_cache_enabled:
                self._store_conv(conv_key, conv_id)
            elif self.config.delete_conversation:
                self._delete_conversation(conv_id, token)
        yield {"__done__": True, "text": text, "reasoning": reasoning, "conv_id": conv_id}

    def _delete_conversation(self, conv_id: str, token: str) -> None:
        try:
            self._post_json(
                self.config.delete_url,
                {"assistant_id": self.config.assistant_id, "conversation_id": conv_id},
                _headers_for(self.config, token),
                min(self.config.request_timeout, 15),
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.debug("删除智谱会话失败: %s", exc)

    # ── 对外接口 ───────────────────────────────────────
    def chat_completion(self, payload: dict) -> dict:
        model = payload.get("model") or self.config.default_model
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        text = reasoning = ""
        for ev in self._stream_events(payload):
            text, reasoning = ev["text"], ev["reasoning"]

        # 若请求带了 tools，尝试从回复里解析工具调用
        calls = _parse_tool_calls(text, payload.get("tools")) if payload.get("tools") else []
        if calls:
            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": _strip_tool_block(text) or None,
                        "tool_calls": calls,
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": _usage(payload, text),
            }

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
        has_tools = bool(payload.get("tools"))

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
                    # 带工具时先不吐，等流结束统一解析，避免把 JSON 片段漏给客户端
                    if not has_tools:
                        yield chunk({"content": dt})
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("智谱流式请求失败: %s", exc)
            err = {"error": {"message": f"上游错误: {exc}", "type": "upstream_error"}}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

        if has_tools:
            calls = _parse_tool_calls(sent_text, payload.get("tools"))
            if calls:
                for i, c in enumerate(calls):
                    yield chunk({
                        "tool_calls": [{
                            "index": i,
                            "id": c["id"],
                            "type": "function",
                            "function": c["function"],
                        }]
                    })
                yield chunk({}, finish="tool_calls")
                yield "data: [DONE]\n\n"
                return
            # 没解析出工具调用，按普通文本发出
            plain = _strip_tool_block(sent_text)
            if plain:
                yield chunk({"content": plain})

        # 无条件带上 usage（客户端没传 include_usage 也要给，
        # 否则 opencode 等客户端统计不到用量）
        yield chunk({}, usage=_usage(payload, sent_text))
        yield chunk({}, finish="stop")
        yield "data: [DONE]\n\n"



