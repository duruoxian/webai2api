"""通义千问网页版（chat.qwen.ai，OpenWebUI 架构）客户端。

把网页版接口封装成 OpenAI 兼容的 chat_completion / stream_chat_completion。
协议（2026-10 实测）：
  1. Authorization: Bearer <access_token>（登录时从浏览器抓取）
  2. POST /api/v2/chat/completions?chat_id=<会话id>，返回 SSE（OpenAI chunk 形态）
  3. 会话缓存：同一对话复用 chat_id（OpenWebUI 服务端保存历史）
"""

from __future__ import annotations

import json
import random
import threading
import time
import urllib.error
import urllib.request
import uuid
from logging import Logger
from urllib.parse import quote

from .. import conv_cache, netutil, reasoning_bridge, tool_bridge
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


# ── 工具调用（OpenAI tools 协议）─────────────────────────
# 千问网页版没有 function calling 接口。Qwen 系列在训练时就见过
# Hermes 风格的 <tool_call> 标签（Qwen-Agent 官方模板同款格式），
# 提示词对齐训练格式，模型输出的工具调用比自造标记稳一个量级。
QWEN_TOOL_TAG = "tool_call"
QWEN_RESULT_TAG = "tool_response"
QWEN_TOOL_MARKER = "<<<TOOL_CALL>>>"  # 兜底标记：模型偶尔不按标签输出时仍可解析

# ── 断流自动续写（中继路径）─────────────────────────
# 上游偶尔在长回答中途掐流且中继不发 done 事件。检测到断流后，向同一
# chat 发一条「继续」（parent_id 指向截断的 response——千问会话历史里
# 已有模型的半截回答，它会自然接着写）。续写轮基线拼接，输出对上层
# prefix-diff 保持单调。最多 2 次，失败降级截断。仅中继路径（生产路径）；
# 直连兜底路径无 done 信号可判定，维持原行为。
MAX_AUTO_CONTINUATIONS = 2
CONTINUE_PROMPT = "继续"

QWEN_TOOL_INSTRUCTION = """# Tools

你运行在用户的本地电脑上，可以使用以下工具读写本地文件、执行任务：

{tools}

## 调用规则

- 需要调用工具时，只输出（不要有任何多余文字）：
<tool_call>
{{"name": "工具名", "arguments": {{"参数名": "值"}}}}
</tool_call>
- 一次只调用一个工具。
- 收到 <tool_response> 包裹的工具执行结果后，基于真实结果继续：
  需要更多信息就再调用工具，信息够了就给出最终回答。
- 不需要工具时，直接用自然语言回答，不要输出 <tool_call>。
- 工具就是你操作本地文件的方式，不要说「我不能访问你的系统盘」。
  用户要求创建/写入/生成文件时，必须调用对应的写文件工具完成，
  不要把文件内容整段贴在回答里就当作已经完成。
- 编辑文件时，old_string 必须逐字复制自最近一次读到的文件内容，
  禁止凭记忆改写；不确定或上次编辑失败时，先重新读取文件再编辑。
- 不要把工具调用写进思考：思考里草拟的调用不会被执行，只有回复里的
  <tool_call> 块才算数。
"""


def _tools_prompt(tools: list) -> str:
    """把 OpenAI tools 定义渲染成千问熟悉的 JSON schema 清单，
    每个函数一行原始 JSON——与 Qwen-Agent 官方模板的 <tools> 段一致。"""
    lines = []
    for t in tools or []:
        if isinstance(t, dict):
            fn = t.get("function") or t
            if fn.get("name"):
                lines.append(json.dumps(fn, ensure_ascii=False))
    return "\n".join(lines)


def _inject_tools_system(system_text: str, tools: list) -> str:
    """把工具说明并入 system prompt。"""
    instruction = QWEN_TOOL_INSTRUCTION.format(tools=_tools_prompt(tools))
    return (system_text + "\n\n" + instruction) if system_text else instruction


def _inject_tools_messages(messages: list, tools: list) -> list:
    """直连路径：把工具说明合并进消息数组的第一条 system（或新建）。"""
    msgs = list(messages or [])
    instruction = QWEN_TOOL_INSTRUCTION.format(tools=_tools_prompt(tools))
    for i, m in enumerate(msgs):
        if isinstance(m, dict) and m.get("role") == "system":
            merged = dict(m)
            merged["content"] = (m.get("content") or "") + "\n\n" + instruction
            msgs[i] = merged
            return msgs
    msgs.insert(0, {"role": "system", "content": instruction})
    return msgs


def _active_tools(payload: dict) -> list:
    """本请求生效的工具清单；tool_choice="none" 等价于没有工具。"""
    tools = payload.get("tools") or []
    if payload.get("tool_choice") == "none":
        return []
    return tools


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return max(1, int(cjk / 1.5 + other / 4))


def _usage(payload: dict, text: str, reasoning: str = "") -> dict:
    """本地估算用量。思考与正文都是真实输出，completion 必须都算
    （2026-10-29 与 deepseek/GLM 口径对齐：此前只算正文导致低估）。"""
    msgs = payload.get("messages") or []
    p = max(
        1, _estimate_tokens("".join(_text_of(m) for m in msgs if isinstance(m, dict)))
    )
    c = max(1, _estimate_tokens(text or "") + _estimate_tokens(reasoning or ""))
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


class UpstreamResponseFilter:
    """千问多路候选回答过滤器（移植自 Rfym21/Qwen2API 的同名机制）。

    上游偶尔对同一次请求开启**多路候选回答**：先连续下发多个
    response.created（chat_id/parent_id 相同、response_index 为 0/1、
    response_id 各不同），随后两路增量帧**交错到达**。不区分
    response_id 一律拼接，两路内容就会搅在一起——这正是 qwen3.8-max
    「语无伦次」的根源（官方网页端只展示 index=0 的主回答，所以网页正常）。

    锁定规则：
    - 新协议：锁定 response.created 里 response_index=0 的那一路；
    - 旧协议（没有 created 元数据）：锁定第一路带 response_id 的正文；
    - 上游完全不带 response_id：一律放行。
    """

    def __init__(self) -> None:
        self._accepted: str | None = None
        self._saw_created = False
        self._indexes: dict[str, float] = {}

    def accept(self, ev: dict) -> bool:
        """该帧是否属于已锁定的主回答流。"""
        if not isinstance(ev, dict):
            return True
        created = ev.get("response.created")
        if not isinstance(created, dict):
            resp = ev.get("response")
            created = resp.get("created") if isinstance(resp, dict) else None
        if isinstance(created, dict):
            self._saw_created = True
            rid = created.get("response_id") or created.get("responseId")
            raw_idx = created.get("response_index", created.get("responseIndex"))
            idx: float | None = None
            try:
                if raw_idx is not None:
                    idx = float(raw_idx)
            except (TypeError, ValueError):
                idx = None
            if isinstance(rid, str) and rid:
                if idx is not None:
                    self._indexes[rid] = idx
                if idx == 0:
                    self._accepted = rid
                elif self._accepted is None and idx is None:
                    self._accepted = rid
            return not rid or self._accepted is None or rid == self._accepted
        rid = ev.get("response_id")
        if not isinstance(rid, str) or not rid:
            return True
        if self._accepted is not None:
            return rid == self._accepted
        idx = self._indexes.get(rid)
        if idx == 0:
            self._accepted = rid
            return True
        if idx is not None or self._saw_created:
            return False
        self._accepted = rid
        return True


class QwenWebClient:
    def __init__(self, config: QwenConfig, logger: Logger) -> None:
        self.config = config
        self.logger = logger
        # access_token 只有 15 分钟：用登录 cookie 调 /auths/refresh 自动续期
        self._token_lock = threading.Lock()
        self._access_token = str(getattr(config, "auth_token", "") or "")
        self._token_checked_at = 0.0
        self._token_ttl = 300.0
        self._relay_port: int = 0
        self._relay_lock = threading.Lock()
        self._last_call_at: float = 0.0
        self._last_call_lock = threading.Lock()
        # 会话缓存：conv_key -> (chat_id, last_used, turns, response_id)
        self._conv_cache = conv_cache.ConversationCache(
            config=config,
            logger=logger,
            tag="Qwen",
            id_field="chat_id",
            extra_field="response_id",
            persist_path=str(getattr(config, "session_store_path", "") or ""),
            bump=self._bump,
        )

    # ── 会话缓存持久化（实现见 adapters/conv_cache.py）──
    def flush_cache(self) -> None:
        self._conv_cache.flush()

    def _conv_key(self, payload: dict) -> str:
        """会话指纹（tool_bridge.conv_fingerprint），首条 assistant 防串台。"""
        return tool_bridge.conv_fingerprint(
            payload.get("messages") or [],
            payload.get("model") or "",
            payload.get("_wapi_key_id") or "",
        )

    def _cached_conv(self, key: str) -> str:
        return self._conv_cache.cached(key)[0]

    def _cached_response_id(self, key: str) -> str:
        return self._conv_cache.cached(key)[1]

    def risk_stats(self) -> dict:
        """风控指标（面板展示，进程累计；缓存规模/活跃度来自共享缓存）。"""
        cs = self._conv_cache.stats()
        hit = self._stat("hit")
        miss = self._stat("miss")
        return {
            "sessions_created": self._stat("created"),
            "sessions_deleted": self._stat("deleted"),
            "cache_hit": hit,
            "cache_miss": miss,
            "cache_hit_rate": round(hit / (hit + miss), 3) if (hit + miss) else 0.0,
            "cache_stale": self._stat("stale"),
            "cache_size": cs["cache_size"],
            "active_sessions": cs["active_sessions"],
            "upstream_requests": self._stat("upstream"),
            "rate_limit_hits": self._stat("rate_limit"),
            "daily_count": 0,
            "daily_limit": int(getattr(self.config, "daily_request_limit", 0) or 0),
        }

    def _bump(self, name: str) -> None:
        setattr(self, f"_stat_{name}", getattr(self, f"_stat_{name}", 0) + 1)

    def _stat(self, name: str) -> int:
        return getattr(self, f"_stat_{name}", 0)

    def _store_conv(
        self, key: str, chat_id: str, response_id: str = "", used_tokens: int = 0
    ) -> None:
        self._conv_cache.store(key, chat_id, response_id, used_tokens=used_tokens)

    # ── HTTP 基础 ──────────────────────────────────────
    @staticmethod
    def _split(url: str) -> tuple[str, str]:
        return netutil.split_url(url, "chat.qwen.ai")

    def _conn(self, url: str, timeout: int):
        return netutil.open_conn(
            url,
            timeout,
            egress=getattr(self.config, "egress", "system"),
            proxy=str(getattr(self.config, "proxy", "") or ""),
            default_host="chat.qwen.ai",
            logger=self.logger,
        )

    def _post_json(self, url: str, body: dict, headers: dict, timeout: int) -> dict:
        conn = self._conn(url, timeout)
        try:
            _, path = self._split(url)
            conn.request(
                "POST", path, body=json.dumps(body).encode("utf-8"), headers=headers
            )
            resp = conn.getresponse()
            raw = resp.read()
            if resp.status >= 400:
                if resp.status == 429:
                    self._bump("rate_limit")
                raise QwenError(
                    f"HTTP {resp.status}: {raw[:200].decode('utf-8', 'replace')}"
                )
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
        conn.request(
            "POST", path, body=json.dumps(body).encode("utf-8"), headers=headers
        )
        resp = conn.getresponse()
        if resp.status >= 400:
            if resp.status == 429:
                self._bump("rate_limit")
            raw = resp.read()
            conn.close()
            raise QwenError(
                f"HTTP {resp.status}: {raw[:200].decode('utf-8', 'replace')}"
            )
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
            if (
                not force
                and self._access_token
                and now - self._token_checked_at < self._token_ttl
            ):
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
        messages = payload.get("messages") or []
        tools = _active_tools(payload)
        if tools:
            messages = _inject_tools_messages(messages, tools)
        body: dict = {
            "model": model,
            "messages": messages,
            "stream": True,
            "chat_id": chat_id,
            "incremental_output": True,
        }
        body.update(
            _effort_to_payload(
                str(payload.get("reasoning_effort") or self.config.default_effort),
                model,
            )
        )
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
        stale = self._conv_cache.collect_stale()
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
                conn.request(
                    "DELETE", path, headers=_build_headers(self.config, stream=False)
                )
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
                f"http://127.0.0.1:{port}/delete",
                data=raw,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
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
                reasoning = (
                    delta.get("reasoning_content") or delta.get("reasoning") or ""
                )
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
        # 首轮绝不复用：新客户端会话的首轮指纹常与旧会话相同（同样的
        # system + 开场白），复用会让模型接着旧上下文聊
        cached = (
            ""
            if tool_bridge.is_first_turn(payload.get("messages"))
            else self._cached_conv(conv_key)
        )
        parent = self._cached_response_id(conv_key) if cached else ""
        if not cached:
            self._bump("miss")

        messages = payload.get("messages") or []
        tools = _active_tools(payload)
        system_text = ""
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            if role == "system" and not system_text:
                system_text = _text_of(m)
        if tools:
            system_text = _inject_tools_system(system_text, tools)
        # 多轮关键：最后一条 assistant 之后、上游还没见过的增量。
        # role=tool 的执行结果在这里渲染成 <tool_response> 发给网页会话，
        # 否则工具结果永远到不了模型眼里，多轮工具链是断的。
        user_text = tool_bridge.increment_text(messages, result_tag=QWEN_RESULT_TAG)

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
            url,
            data=raw,
            method="POST",
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
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
        base_text = ""
        base_reasoning = ""
        saw_done = False
        _cont_rounds = 0  # 已发起的续写轮数（不含主轮）
        cont_failed = False
        anchor_seen = False  # 本轮是否拿到过 assistant 锚点（response_id）
        try:
            while True:
                round_text = ""
                round_reasoning = ""
                is_cont = _cont_rounds > 0
                # P1 续写接缝剪除：模型「继续」常把断点前最后一段逐字重说
                # 一遍。增量开头只要「出现在已发尾巴里」就扣住，分叉时提交
                # 最长重叠——剪掉的字节都与客户端已有内容逐字相同。
                seam = (
                    tool_bridge.SeamTrimmer(base_text[-tool_bridge.SEAM_WINDOW :])
                    if is_cont
                    else None
                )
                round_shown = ""  # 本轮已示文本（续写轮=剪除后）
                anchor_base = response_id
                anchor_seen = False
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
                                anchor_seen = True
                            yield {
                                "text": base_text + round_text,
                                "reasoning": base_reasoning + round_reasoning,
                                "chat_id": chat_id or response_id,
                                "finish_seen": saw_done,
                            }
                        elif typ == "reasoning":
                            round_reasoning += str(event.get("content") or "")
                            yield {
                                "text": base_text + round_text,
                                "reasoning": base_reasoning + round_reasoning,
                                "chat_id": chat_id or response_id,
                                "finish_seen": saw_done,
                            }
                        elif typ == "delta":
                            piece = str(event.get("content") or "")
                            if seam is not None:
                                seam.feed(piece)
                                round_shown = seam.committed
                            else:
                                round_text += piece
                                round_shown = round_text
                            yield {
                                "text": base_text + round_shown,
                                "reasoning": base_reasoning + round_reasoning,
                                "chat_id": chat_id or response_id,
                                "finish_seen": saw_done,
                            }
                        elif typ == "done":
                            cid = event.get("chat_id")
                            if isinstance(cid, str) and cid:
                                chat_id = cid
                            rid = event.get("response_id")
                            if isinstance(rid, str) and rid:
                                response_id = rid
                                anchor_seen = True
                            saw_done = True
                            break
                        elif typ == "error":
                            if is_cont:
                                # 续写轮报错：按截断收场，不让已输出的流变报错
                                self.logger.warning(
                                    "续写轮中继错误（按截断返回）: %s",
                                    event.get("message"),
                                )
                                cont_failed = True
                                break
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

                if saw_done or cont_failed or _cont_rounds >= MAX_AUTO_CONTINUATIONS:
                    break
                if not (
                    chat_id
                    and anchor_seen
                    and response_id
                    and response_id != anchor_base
                ):
                    # 无锚点（连会话 id / assistant response id 都没拿到）：
                    # 「继续」会挂到上一轮的 response 上成为孤儿分支，不续写
                    break
                _cont_rounds += 1
                base_text += round_shown  # 只并入已示部分，扣住的复读丢弃
                base_reasoning += round_reasoning
                self._bump("auto_continued")
                self.logger.warning(
                    "上游流提前中断（未见 done 事件），自动续写 %d/%d",
                    _cont_rounds,
                    MAX_AUTO_CONTINUATIONS,
                )
                cont_body = dict(
                    req_body,
                    prompt=CONTINUE_PROMPT,
                    system="",
                    chat_id=chat_id,
                    parent_id=response_id,
                )
                raw = json.dumps(cont_body, ensure_ascii=False).encode("utf-8")
                req = urllib.request.Request(
                    url,
                    data=raw,
                    method="POST",
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream",
                    },
                )
                try:
                    resp = urllib.request.urlopen(
                        req, timeout=self.config.request_timeout
                    )
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning("自动续写失败（按截断返回）: %s", exc)
                    break
        finally:
            try:
                resp.close()
            except Exception:
                pass

        text = base_text + round_shown
        reasoning = base_reasoning + round_reasoning
        if chat_id and self.config.session_cache_enabled:
            store_key = tool_bridge.conv_fingerprint(
                (payload.get("messages") or [])
                + [{"role": "assistant", "content": text}],
                payload.get("model") or "",
                payload.get("_wapi_key_id") or "",
            )
            self._store_conv(
                store_key,
                chat_id,
                response_id if anchor_seen else "",
                used_tokens=conv_cache.estimate_tokens(text, reasoning),
            )
        if not saw_done:
            self.logger.warning(
                "上游流提前中断（续写未救回），按截断返回 finish=length"
            )
        yield {
            "__done__": True,
            "text": text,
            "reasoning": reasoning,
            "chat_id": chat_id or response_id,
            "finish_seen": saw_done,
        }

    def _stream_events(self, payload: dict):
        if self.config.relay_enabled and self.config.account_id:
            yield from self._stream_via_relay(payload)
            return

        conv_key = self._conv_key(payload)
        # 首轮绝不复用（同 _stream_via_relay）
        cached = (
            ""
            if tool_bridge.is_first_turn(payload.get("messages"))
            else self._cached_conv(conv_key)
        )
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
                url,
                body,
                _build_headers(self.config, self._valid_token()),
                self.config.request_timeout,
            )
        except QwenError as exc:
            if "401" not in str(exc):
                raise
            self.logger.info("Qwen 返回 401，尝试续期后重试")
            conn, resp = self._open_stream(
                url,
                body,
                _build_headers(self.config, self._valid_token(force=True)),
                self.config.request_timeout,
            )
        chat_id = cached
        text = ""
        reasoning = ""
        rfilter = UpstreamResponseFilter()
        try:
            for event in self._iter_events(resp):
                if not rfilter.accept(event):
                    continue
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
            store_key = tool_bridge.conv_fingerprint(
                (payload.get("messages") or [])
                + [{"role": "assistant", "content": text}],
                payload.get("model") or "",
                payload.get("_wapi_key_id") or "",
            )
            self._store_conv(store_key, chat_id)
        yield {
            "__done__": True,
            "text": text,
            "reasoning": reasoning,
            "chat_id": chat_id,
        }

    # ── 对外接口 ───────────────────────────────────────
    def _forced_retry(self, payload: dict, tools: list) -> tuple[str, str, list]:
        """tool_choice=required 强校验重试：加强指令再问一次（上限 1 次）。"""
        self.logger.info("Qwen tool_choice=required 首轮未调用工具，加强指令重试一次")
        retry = dict(payload)
        retry["messages"] = list(payload.get("messages") or []) + [
            {
                "role": "user",
                "content": tool_bridge.required_nudge(
                    tag=QWEN_TOOL_TAG, marker=QWEN_TOOL_MARKER
                ),
            }
        ]
        text = reasoning = ""
        for ev in self._stream_events(retry):
            text, reasoning = ev["text"], ev["reasoning"]
        calls = tool_bridge.parse_tool_calls(
            text, tools, tag=QWEN_TOOL_TAG, marker=QWEN_TOOL_MARKER
        )
        if not calls:
            raise QwenError(
                "工具调用强校验失败：重试一次后模型仍未按要求调用工具"
                "（tool_choice=required）"
            )
        return text, reasoning, calls

    def chat_completion(self, payload: dict) -> dict:
        model = payload.get("model") or self.config.default_model
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        tools = _active_tools(payload)
        text = reasoning = ""
        calls: list = []
        finish_seen = True
        chat_id = ""
        # P0 空响应重试：上游正常结束但客户端可见输出全空（标记被剥光也
        # 算空）→ 作废上游会话重跑一次，再空按原样返回
        for attempt in range(2):
            text = reasoning = ""
            calls = []
            finish_seen = True
            chat_id = ""
            for ev in self._stream_events(payload):
                text, reasoning = ev["text"], ev["reasoning"]
                chat_id = str(ev.get("chat_id") or chat_id)
                if ev.get("__done__"):
                    finish_seen = bool(ev.get("finish_seen", True))

            calls = (
                tool_bridge.parse_tool_calls(
                    text, tools, tag=QWEN_TOOL_TAG, marker=QWEN_TOOL_MARKER
                )
                if tools
                else []
            )
            visible = (
                tool_bridge.strip_tool_markup(
                    text, tag=QWEN_TOOL_TAG, marker=QWEN_TOOL_MARKER
                )
                if tools and not calls
                else text
            )
            if calls or visible.strip() or reasoning.strip() or not finish_seen:
                break
            if attempt == 0:
                self.logger.warning(
                    "上游正常结束但输出全空（empty_response），弃会话重试 1/1"
                )
                try:
                    self._conv_cache.delete_by_session_id(chat_id)
                except Exception:
                    pass

        if not calls and tools and tool_bridge.is_forced_tool_choice(payload):
            text, reasoning, calls = self._forced_retry(payload, tools)
        if not calls and tools and not text.strip() and reasoning.strip():
            # P0 思考双通道：正文可见输出为空时才扫 reasoning 里的完整
            # <tool_call> 块——思考里的草稿调用绝不提取（shell 级误执行风险）
            rcalls = tool_bridge.parse_tool_calls(
                reasoning, tools, tag=QWEN_TOOL_TAG, marker=QWEN_TOOL_MARKER
            )
            if rcalls:
                self.logger.warning("思考里发现完整工具调用（正文为空），已提取下发")
                calls = rcalls
                reasoning = tool_bridge.strip_tool_markup(
                    reasoning, tag=QWEN_TOOL_TAG, marker=QWEN_TOOL_MARKER
                )
        if calls:
            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning_content": reasoning or None,
                            "tool_calls": calls,
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": _usage(payload, text, reasoning),
            }

        # 带工具但没解析出调用：剥掉可能残留的标记再下发；
        # 空正文兜底——思考里就是答案时顶上来
        content = (
            tool_bridge.strip_tool_markup(
                text, tag=QWEN_TOOL_TAG, marker=QWEN_TOOL_MARKER
            )
            if tools
            else text
        )
        content, reasoning = reasoning_bridge.finalize_reasoning(
            content, reasoning, legacy=self.config.legacy_reasoning_in_content
        )
        if not finish_seen:
            self.logger.warning(
                "上游流提前中断（续写未救回），按截断返回 finish=length"
            )
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "reasoning_content": reasoning or None,
                    },
                    "finish_reason": "stop" if finish_seen else "length",
                }
            ],
            "usage": _usage(payload, text, reasoning),
        }

    def stream_chat_completion(self, payload: dict):
        model = payload.get("model") or self.config.default_model
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        def chunk(delta: dict, finish=None, usage=None) -> str:
            obj = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            if usage is not None:
                obj["usage"] = usage
            return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

        yield chunk({"role": "assistant", "content": ""})

        tools = _active_tools(payload)
        forced = bool(tools) and tool_bridge.is_forced_tool_choice(payload)

        # tool_choice=required 强校验模式：不直播，解析失败先重试一次，
        # 再失败按明确错误返回（直播一段注定作废的正文没有意义）
        if forced:
            text = reasoning = ""
            try:
                for ev in self._stream_events(payload):
                    text, reasoning = ev["text"], ev["reasoning"]
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Qwen 流式请求失败: %s", exc)
                err = {
                    "error": {"message": f"上游错误: {exc}", "type": "upstream_error"}
                }
                yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            calls = tool_bridge.parse_tool_calls(
                text, tools, tag=QWEN_TOOL_TAG, marker=QWEN_TOOL_MARKER
            )
            if not calls:
                try:
                    text, reasoning, calls = self._forced_retry(payload, tools)
                except QwenError as exc:
                    err = {
                        "error": {"message": str(exc), "type": "tool_required_error"}
                    }
                    yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
            rs = reasoning_bridge.ReasoningStream(
                legacy=self.config.legacy_reasoning_in_content
            )
            out_r, out_c = rs.feed(reasoning, "")
            if out_r:
                yield chunk({"reasoning_content": out_r})
            if out_c:
                yield chunk({"content": out_c})
            tail = rs.close_think()
            if tail:
                yield chunk({"content": tail})
            yield chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": calls[0]["id"],
                            "type": "function",
                            "function": calls[0]["function"],
                        }
                    ]
                }
            )
            yield chunk({}, usage=_usage(payload, text))
            yield chunk({}, finish="tool_calls")
            yield "data: [DONE]\n\n"
            return

        # 有工具时扣住标记文本：<tool_call> 出现前照常流式，出现后扣留等解析，
        # 标记 JSON 不会漏进 content（增量口径扣留器）
        # P0 空响应重试（流式）：正常结束但客户端可见输出全空（纯空白也算
        # 空）→ 弃上游会话重跑一次；重试轮换新扣留器/分发器
        calls: list = []
        buf = None
        emitted_tail = ""
        sent_text = ""
        sent_reasoning = ""
        finish_seen = True
        chat_id = ""
        for attempt in range(2):
            buf = (
                tool_bridge.DeltaToolBuffer([f"<{QWEN_TOOL_TAG}>", QWEN_TOOL_MARKER])
                if tools
                else None
            )
            rs = reasoning_bridge.ReasoningStream(
                legacy=self.config.legacy_reasoning_in_content
            )
            emitted_tail = ""
            sent_text = ""
            sent_reasoning = ""
            finish_seen = True
            chat_id = ""
            try:
                for ev in self._stream_events(payload):
                    if ev.get("__done__"):
                        finish_seen = bool(ev.get("finish_seen", True))
                    chat_id = str(ev.get("chat_id") or chat_id)
                    d_reason = ev["reasoning"][len(sent_reasoning) :]
                    if d_reason:
                        sent_reasoning = ev["reasoning"]
                    d_text = ev["text"][len(sent_text) :]
                    sent_text = ev["text"]
                    d_out = buf.feed_delta(d_text) if buf is not None else d_text
                    out_r, out_c = rs.feed(d_reason, d_out)
                    if out_r:
                        yield chunk({"reasoning_content": out_r})
                    if out_c:
                        emitted_tail = (emitted_tail + out_c)[
                            -tool_bridge.SEAM_WINDOW :
                        ]
                        yield chunk({"content": out_c})
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Qwen 流式请求失败: %s", exc)
                err = {"error": {"message": f"上游错误: {exc}", "type": "upstream_error"}}
                yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return

            full_text = buf.text if buf is not None else sent_text
            calls = (
                tool_bridge.parse_tool_calls(
                    full_text, tools, tag=QWEN_TOOL_TAG, marker=QWEN_TOOL_MARKER
                )
                if tools
                else []
            )
            if (
                calls
                or emitted_tail.strip()
                or sent_reasoning.strip()
                or not finish_seen
            ):
                break
            if attempt == 0:
                self.logger.warning(
                    "上游正常结束但输出全空（empty_response），弃会话重试 1/1"
                )
                try:
                    self._conv_cache.delete_by_session_id(chat_id)
                except Exception:
                    pass

        if calls:
            tail = rs.close_think()
            if tail:
                yield chunk({"content": tail})
            yield chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": calls[0]["id"],
                            "type": "function",
                            "function": calls[0]["function"],
                        }
                    ]
                }
            )
            yield chunk({}, usage=_usage(payload, sent_text, sent_reasoning))
            yield chunk({}, finish="tool_calls")
            yield "data: [DONE]\n\n"
            return

        # P0 思考双通道：正文可见输出为空时才扫 reasoning 里的完整
        # <tool_call> 块——思考里的草稿调用绝不提取（shell 级误执行风险）
        if tools and not calls and not emitted_tail.strip() and sent_reasoning:
            rcalls = tool_bridge.parse_tool_calls(
                sent_reasoning, tools, tag=QWEN_TOOL_TAG, marker=QWEN_TOOL_MARKER
            )
            if rcalls:
                self.logger.warning("思考里发现完整工具调用（正文为空），已提取下发")
                calls = rcalls
                yield chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": calls[0]["id"],
                                "type": "function",
                                "function": calls[0]["function"],
                            }
                        ]
                    }
                )
                yield chunk({}, usage=_usage(payload, sent_text, sent_reasoning))
                yield chunk({}, finish="tool_calls")
                yield "data: [DONE]\n\n"
                return

        # 解析失败可观测性：扣住了标记却没解析成调用（名字不在清单/JSON
        # 残缺），静默剥离会造成「模型以为调了、客户端什么都没收到」的无头案
        if buf is not None and buf.opener_found:
            self.logger.warning(
                "Qwen 输出了工具调用标记但未解析成调用（工具名不在声明清单"
                "或 JSON 残缺），已按普通文本剥离。片段: %r",
                (
                    tool_bridge.strip_tool_markup(
                        buf.text, tag=QWEN_TOOL_TAG, marker=QWEN_TOOL_MARKER
                    )
                    or buf.text
                )[-200:],
            )

        if buf is not None:
            # 没解析出工具调用：扣住的尾巴剥掉标记与调用形态 JSON 再放
            # （白名单外的调用如网页内置 browser，剥掉好过漏给用户）
            tail = tool_bridge.strip_call_json(
                tool_bridge.strip_tool_markup(
                    buf.flush(), tag=QWEN_TOOL_TAG, marker=QWEN_TOOL_MARKER
                )
            )
            if tail:
                out_r, out_c = rs.feed("", tail)
                if out_r:
                    yield chunk({"reasoning_content": out_r})
                if out_c:
                    yield chunk({"content": out_c})
        out_r, out_c = rs.finish()
        if out_r:
            yield chunk({"reasoning_content": out_r})
        if out_c:
            yield chunk({"content": out_c})
        yield chunk({}, usage=_usage(payload, full_text, sent_reasoning))
        if not finish_seen:
            self.logger.warning(
                "上游流提前中断（续写未救回），按截断返回 finish=length"
            )
        yield chunk({}, finish="stop" if finish_seen else "length")
        yield "data: [DONE]\n\n"
