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


def _extract_parts(parts: list, is_silent: bool = False) -> tuple[str, str]:
    """从 parts 快照里重建正文与思考内容。"""
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
        """把思考强度映射成 (chat_mode, reasoning_effort)。

        挡位来自客户端下拉（OpenAI 的 reasoning_effort）：
          Default(空) → 快速；Low → 轻度；Medium/High → 深度；Max → 极致。
        """
        m = (model or "").lower()
        if "deepresearch" in m:
            return "deep_research", (effort or "max")
        e = (effort or "").strip().lower()
        if e in ("", "default", "none", "off", "auto"):
            return "", ""
        if e in ("low", "minimal"):
            return "", "low"
        if e == "medium":
            return "zero", "medium"
        if e == "high":
            return "zero", "high"
        if e == "max":
            return "zero", "max"
        return "zero", e

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
        content = _prepare_content(payload.get("messages") or [])
        is_silent = "silent" in model

        self._throttle()
        token = self._access()
        conn, resp = self._open_stream(
            self.config.stream_url,
            self._body(content, model, effort=str(payload.get("reasoning_effort") or "")),
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
        if self.config.delete_conversation and conv_id:
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
        p = max(1, len(_prepare_content(payload.get("messages") or [])) // 4)
        c = max(1, len(text) // 4)
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
            "usage": {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c},
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
            self.logger.warning("智谱流式请求失败: %s", exc)
            err = {"error": {"message": f"上游错误: {exc}", "type": "upstream_error"}}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return
        yield chunk({}, finish="stop")
        yield "data: [DONE]\n\n"
