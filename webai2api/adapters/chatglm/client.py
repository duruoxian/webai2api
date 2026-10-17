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


def _build_headers(secret: str, token: str | None = None) -> dict[str, str]:
    ts, nonce, sign = _make_sign(secret)
    headers = {
        **BASE_HEADERS,
        "User-Agent": random.choice(USER_AGENTS),
        "X-Device-Id": uuid.uuid4().hex,
        "X-Request-Id": uuid.uuid4().hex,
        "X-Sign": sign,
        "X-Timestamp": ts,
        "X-Nonce": nonce,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
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
        self._conv_cache: dict[str, tuple[str, float]] = {}

    def _conv_key(self, payload: dict) -> str:
        """用「首条用户消息 + system」做会话指纹。"""
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
        raw = f"{system}\x00{first}\x00{payload.get('model') or ''}"
        return hashlib.sha1(raw.encode("utf-8", "ignore")).hexdigest()

    def _cached_conv(self, key: str) -> str:
        if not self.config.session_cache_enabled:
            return ""
        with self._conv_lock:
            item = self._conv_cache.get(key)
            if not item:
                return ""
            conv_id, ts = item
            if time.time() - ts > self.config.session_cache_ttl:
                self._conv_cache.pop(key, None)
                return ""
            self._conv_cache[key] = (conv_id, time.time())
            return conv_id

    def _store_conv(self, key: str, conv_id: str) -> None:
        if not self.config.session_cache_enabled or not conv_id:
            return
        with self._conv_lock:
            if len(self._conv_cache) >= self.config.session_cache_max_size:
                oldest = min(self._conv_cache.items(), key=lambda kv: kv[1][1])[0]
                self._conv_cache.pop(oldest, None)
            self._conv_cache[key] = (conv_id, time.time())

    def _drop_conv(self, key: str) -> None:
        with self._conv_lock:
            self._conv_cache.pop(key, None)

    # ── HTTP 基础 ──────────────────────────────────────
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
        return http.client.HTTPSConnection(host, timeout=max(int(timeout), 30))

    def _post_json(self, url: str, body: dict, headers: dict, timeout: int) -> dict:
        conn = self._conn(url, timeout)
        try:
            _, path = self._split(url)
            conn.request("POST", path, body=json.dumps(body).encode("utf-8"), headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
            if resp.status >= 400:
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
                _build_headers(self.config.sign_secret, self.config.refresh_token),
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

    def _invalidate(self) -> None:
        with self._lock:
            self._access_token = ""
            self._token_expire = 0.0

    # ── 请求体 ─────────────────────────────────────────
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
        if not self.config.anti_rate_limit_enabled:
            return
        lo = self.config.anti_rate_limit_min_delay
        hi = max(lo, self.config.anti_rate_limit_max_delay)
        if hi > 0:
            time.sleep(random.uniform(lo, hi))

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
        token = self._access()
        # 命中会话缓存则复用 conversation_id（与 DeepSeek 的多轮增量行为一致）
        conv_key = self._conv_key(payload)
        cached_conv = self._cached_conv(conv_key)
        conn, resp = self._open_stream(
            self.config.stream_url,
            self._body(
                content, model,
                effort=str(payload.get("reasoning_effort") or self.config.default_effort),
            ),
            _build_headers(self.config.sign_secret, token),
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
                _build_headers(self.config.sign_secret, token),
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

