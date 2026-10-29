"""Kimi（www.kimi.com）网页版客户端。

把网页版接口封装成 OpenAI 兼容的 chat_completion / stream_chat_completion。

协议（2026-10 实测）：
  1. 网页版使用 Connect 协议：POST /apiv2/kimi.gateway.chat.v1.ChatService/Chat
  2. Content-Type / Accept = application/connect+json，请求体是 5 字节帧头 + JSON
  3. 响应是 Connect 流：每帧 5 字节帧头 + JSON 事件（op=set/append + mask）
  4. 会话缓存：同一段对话复用 chat_id，多轮只发增量；超限轮换
"""

from __future__ import annotations

import gzip
import json
import random
import struct
import threading
import time
import uuid
from collections import OrderedDict
from logging import Logger

from .. import conv_cache, netutil, reasoning_bridge, tool_bridge
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

CONNECT_SERVICE = "kimi.gateway.chat.v1.ChatService"
DEFAULT_INTERNAL_MODEL = "k2d6-chat"

# 模型选择器 = (内部 model id, scenario)。
# 实测（2026-10）：k3 / k2.8 是 agent 类模型，必须配 SCENARIO_OK_COMPUTER；
# 普通对话模型（k2d6-chat / k28-instant）才用 SCENARIO_CHAT。
# 配错 scenario 会直接 invalid_argument（曾被误当成"账号无权限"）。
_SCENARIO_CHAT = "SCENARIO_CHAT"
_SCENARIO_AGENT = "SCENARIO_OK_COMPUTER"

# 对外模型名 → 网页版选择器（按优先级）。前一项 invalid_argument 会自动降到下一项。
_MODEL_CANDIDATES: dict[str, tuple[tuple[str, str], ...]] = {
    "kimi-k3": (
        ("k3-agent", _SCENARIO_AGENT),
        (DEFAULT_INTERNAL_MODEL, _SCENARIO_CHAT),
    ),
    "kimi-k2.8": (
        ("k28-agent-preview", _SCENARIO_AGENT),
        ("k28-instant", _SCENARIO_CHAT),
        (DEFAULT_INTERNAL_MODEL, _SCENARIO_CHAT),
    ),
    "kimi-k2.6": ((DEFAULT_INTERNAL_MODEL, _SCENARIO_CHAT),),
    "kimi-k2.5": (
        ("k2d5-chat", _SCENARIO_CHAT),
        (DEFAULT_INTERNAL_MODEL, _SCENARIO_CHAT),
    ),
    "kimi-k2": ((DEFAULT_INTERNAL_MODEL, _SCENARIO_CHAT),),
    "kimi": ((DEFAULT_INTERNAL_MODEL, _SCENARIO_CHAT),),
    "moonshot-v1": ((DEFAULT_INTERNAL_MODEL, _SCENARIO_CHAT),),
}
# 内部 id 直接透传时按 id 推断 scenario
_INTERNAL_AGENT_IDS = {"k3", "k3-agent", "k28-agent-preview", "k28-preview"}
_INTERNAL_CHAT_IDS = {"k2d6", "k2d6-chat", "k2d5", "k2d5-chat", "k28-instant"}


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


def _connect_headers(config: KimiConfig, token: str | None = None) -> dict[str, str]:
    headers = _build_headers(config, token, stream=True)
    headers["Content-Type"] = "application/connect+json"
    headers["Accept"] = "application/connect+json"
    headers["Connect-Protocol-Version"] = "1"
    headers["Accept-Encoding"] = "identity"
    headers.pop("Cache-Control", None)
    headers.pop("Pragma", None)
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


def _usage(payload: dict, text: str, reasoning: str = "") -> dict:
    """本地估算用量。思考与正文都是真实输出，completion 必须都算
    （2026-10-29 与 deepseek/GLM 口径对齐：此前只算正文导致低估）。"""
    msgs = payload.get("messages") or []
    p = max(
        1, _estimate_tokens("".join(_text_of(m) for m in msgs if isinstance(m, dict)))
    )
    c = max(1, _estimate_tokens(text or "") + _estimate_tokens(reasoning or ""))
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}


# ── 工具调用（OpenAI tools 协议）─────────────────────────
# 网页版 Kimi 没有 function calling 接口，只能把工具清单写进 system prompt，
# 让模型按 DSML 格式输出调用，再由本模块解析成标准 tool_calls。
# 用 DSML（Kimi 训练里见过的原生工具标记）而不是自造 JSON 标记：字符串走
# CDATA，文件内容里的换行、引号、花括号都不用转义，写文件类工具命中率更高。
# 这样 opencode 之类的客户端才能真正读写本地文件，而不是「假装」写到沙箱路径。
TOOL_MARKER = "<<<TOOL_CALL>>>"  # 旧协议兜底：历史会话/模型偶尔仍按旧格式输出

# ── 断流自动续写 ────────────────────────────────────
# 上游偶尔在长回答中途掐流且不发 done 事件。检测到断流后，向同一
# chat 发一条「继续」（parent 指向截断的 assistant 消息——Kimi 会话
# 历史里已有模型的半截回答，它会自然接着写）。续写轮用全新 _ReplyState
# + 基线拼接，输出对上层 prefix-diff 保持单调。最多 2 次，失败降级截断。
MAX_AUTO_CONTINUATIONS = 2
CONTINUE_PROMPT = "继续"

TOOL_INSTRUCTION = r"""【身份与环境（最重要的前提，先读这条）】
你当前运行在用户 Windows 电脑上的编码代理（如 OpenCode）内部，经 webai2api
网关连到 Kimi。你不是网页版 Kimi Chat 助手，也不在任何云端沙盒。
- 你能调用的唯一工具是下面「可用工具」列表里的本地工具（read / write / edit /
  shell / glob / grep 等），它们直接读写用户电脑本地的文件系统；工作目录就是
  用户的当前目录（如 D:\study）。
- 禁止使用任何不在列表中的工具；禁止输出 /mnt/agents、sandbox:// 之类路径；
  禁止提及沙盒、ipython、web_search、网页版 Kimi Chat——这些与你无关，提到即错。
- 用户问「你是谁」时，回答你是运行在用户电脑上的编码代理（如 OpenCode），
  不要自称 Kimi 或网页聊天助手。
- 用户要求创建、写入、生成文件、执行命令或查看文件时，必须调用对应工具完成，
  文件写到用户指定的本地路径；不要用自然语言说「我做了」或「我无法访问」
  来代替调用工具。

你可以调用以下工具来完成任务：

{tools}

调用协议（DSML，必须严格照格式）：
需要调用工具时，只输出下面这一个可执行块，前后不要附加任何解释文字：
<|DSML|tool_calls>
  <|DSML|invoke name="工具名">
    <|DSML|parameter name="参数名"><![CDATA[参数值]]></|DSML|parameter>
  </|DSML|invoke>
</|DSML|tool_calls>

规则：
1. 参数名必须与工具 schema 完全一致（区分大小写），不要自创或改写参数名。
2. 字符串参数用 <![CDATA[...]]> 包裹；对象/数组/数字/布尔可直接写 JSON。
3. 一次只调用一个工具（一个 invoke）。不要输出 {marker} 这类旧格式。
4. 不需要工具时直接用自然语言回答，禁止输出任何 DSML 标记。
5. 你运行在用户的本地电脑上，工具就是本地工具、路径就是本地路径（见开头【身份与环境】）。
   用户要求创建/写入/生成文件时，必须调用对应的写文件工具完成，
   不要把文件内容整段贴在回答里就当作已经完成。
6. 收到 <|DSML|tool_result ...> 包裹的工具执行结果后，基于真实结果继续：
   需要更多信息就再调用工具，信息够了就给出最终回答（不要再输出标记）。
7. 编辑文件时，old_string 必须逐字复制自最近一次读到的文件内容，
   禁止凭记忆改写；不确定或上次编辑失败时，先重新读取文件再编辑。
8. 不要把工具调用写进思考：思考里草拟的调用不会被执行，只有回复里的
   DSML 块才算数。
"""


def _tools_prompt(tools: list) -> str:
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
                lines.append(
                    f"    参数 {k} ({typ}){mark}: {(v or {}).get('description', '')}"
                )
    return "\n".join(lines)


def _tools_protocol(tools: list) -> str:
    """完整的工具协议说明（清单 + DSML 约定 + 多轮规则）。"""
    return TOOL_INSTRUCTION.format(tools=_tools_prompt(tools), marker=TOOL_MARKER)


def _inject_tools(messages: list, tools: list) -> list:
    spec = _tools_prompt(tools)
    if not spec:
        return messages
    instruction = _tools_protocol(tools)
    msgs = list(messages or [])
    for i, m in enumerate(msgs):
        if isinstance(m, dict) and m.get("role") == "system":
            merged = dict(m)
            merged["content"] = (m.get("content") or "") + "\n\n" + instruction
            msgs[i] = merged
            return msgs
    msgs.insert(0, {"role": "system", "content": instruction})
    return msgs


def _strip_tool_block(text: str) -> str:
    """剥工具标记：归一降级角括号形态，剥完再扫残余碎片（对齐 GLM/deepseek）。"""
    text = tool_bridge.normalize_tool_marker(text or "", TOOL_MARKER)
    out = tool_bridge.strip_tool_markup(text, marker=TOOL_MARKER, dsml=True)
    out = tool_bridge.scrub_marker_fragments(out)
    return tool_bridge.strip_truncated_marker_tail(
        out, [tool_bridge.DSML_TOOL_CALLS_OPEN, TOOL_MARKER]
    ).strip()


def _parse_tool_calls(text: str, tools: list, require_marker: bool = False) -> list:
    return tool_bridge.parse_tool_calls(
        tool_bridge.normalize_tool_marker(text or "", TOOL_MARKER),
        tools,
        marker=TOOL_MARKER,
        dsml=True,
        require_marker=require_marker,
    )


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
    """把 reasoning_effort 收敛成网页版 ChatRequestOptions 字段。"""
    tier = _tier_for(effort, model)
    if tier == "标准":
        return {"thinking": False}
    if tier == "进阶":
        return {"thinking": True, "reasoningEffort": "REASONING_EFFORT_HIGH"}
    return {"thinking": True, "reasoningEffort": "REASONING_EFFORT_MAX"}


def _internal_models(
    model: str | None, *, has_tools: bool = False
) -> list[tuple[str, str]]:
    """对外模型名 → [(内部 model id, scenario), ...]（按优先级）。

    has_tools：请求带工具时跳过 OK_COMPUTER（agent 场景）——抓帧实测该
    场景对注入的工具协议一律回空响应（heartbeat+done，不报错），工具
    调用完全失效；CHAT 场景正常。
    """
    m = (model or "").strip().lower()
    if m in _MODEL_CANDIDATES:
        cands = list(_MODEL_CANDIDATES[m])
    elif m in _INTERNAL_AGENT_IDS:
        cands = [(m, _SCENARIO_AGENT)]
    elif m in _INTERNAL_CHAT_IDS:
        cands = [(m, _SCENARIO_CHAT)]
    else:
        cands = [(DEFAULT_INTERNAL_MODEL, _SCENARIO_CHAT)]
    if has_tools:
        chat = [(mid, sc) for mid, sc in cands if sc != _SCENARIO_AGENT]
        if chat:
            cands = chat
    return cands


def _frame(payload: bytes) -> bytes:
    return struct.pack(">BI", 0, len(payload)) + payload


def _read_exact(resp, size: int) -> bytes:
    buf = b""
    while len(buf) < size:
        chunk = resp.read(size - len(buf))
        if not chunk:
            break
        buf += chunk
    return buf


def _iter_connect_frames(resp):
    """逐帧读取 Connect 流：1 字节 flags + 4 字节大端长度 + payload。"""
    while True:
        header = _read_exact(resp, 5)
        if len(header) < 5:
            return
        flag = header[0]
        length = int.from_bytes(header[1:5], "big")
        payload = _read_exact(resp, length)
        if len(payload) < length:
            return
        if flag & 0x01:
            try:
                payload = gzip.decompress(payload)
            except Exception:
                pass
        yield flag & ~0x01, payload


def _connect_error(payload: bytes) -> tuple[str, str] | None:
    try:
        obj = json.loads(payload.decode("utf-8", "replace"))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    err = obj.get("error")
    if not isinstance(err, dict):
        return None
    return str(err.get("code") or "unknown"), str(err.get("message") or "")


class _ReplyState:
    """把 ChatResponse 事件累积成最终文本。"""

    def __init__(self) -> None:
        self.text: OrderedDict[str, str] = OrderedDict()
        self.think: OrderedDict[str, str] = OrderedDict()
        self.chat_id = ""
        self.last_message_id = ""
        self.assistant_message_id = ""
        self.error = ""
        self.done = False

    def apply(self, obj: dict) -> None:
        if not isinstance(obj, dict):
            return
        if obj.get("error"):
            self.error = str(obj.get("error"))
        chat = obj.get("chat")
        if isinstance(chat, dict):
            cid = chat.get("id")
            if isinstance(cid, str) and cid:
                self.chat_id = cid
        msg = obj.get("message")
        if isinstance(msg, dict):
            mid = msg.get("id")
            if isinstance(mid, str) and mid:
                self.last_message_id = mid
                if msg.get("role") == "assistant":
                    self.assistant_message_id = mid
        block = obj.get("block")
        if isinstance(block, dict):
            bid = str(block.get("id") or "_")
            op = str(obj.get("op") or "")
            mask = str(obj.get("mask") or "")
            for kind, store in (("text", self.text), ("think", self.think)):
                node = block.get(kind)
                if not isinstance(node, dict):
                    continue
                content = node.get("content")
                if not isinstance(content, str):
                    content = ""
                if op == "append" and mask.endswith(".content"):
                    store[bid] = store.get(bid, "") + content
                elif op == "set":
                    # 块快照整段重写（可能变短）：冻结只增不减——回退会让
                    # 流式 prefix-diff 错位咬字（同 GLM parts 重写病）
                    old = store.get(bid, "")
                    store[bid] = content if len(content) >= len(old) else old
                elif content:
                    store[bid] = store.get(bid, "") + content
        if "done" in obj:
            self.done = True

    def text_value(self) -> str:
        return "".join(self.text.values())

    def think_value(self) -> str:
        return "".join(self.think.values())


class KimiWebClient:
    def __init__(self, config: KimiConfig, logger: Logger) -> None:
        self.config = config
        self.logger = logger
        # accessToken 只有 15 分钟：用 refreshToken 自动续期，续期后的 token 缓存在实例上
        self._token_lock = threading.Lock()
        self._access_token = str(getattr(config, "auth_token", "") or "")
        self._refresh_token = str(getattr(config, "refresh_token", "") or "")
        self._token_checked_at = 0.0
        self._token_ttl = 300.0  # 5 分钟内不重复续期
        # 会话缓存：conv_key -> (chat_id, last_used, turns, last_message_id)
        self._conv_cache = conv_cache.ConversationCache(
            config=config,
            logger=logger,
            tag="Kimi",
            id_field="chat_id",
            extra_field="last_message_id",
            persist_path=str(getattr(config, "session_store_path", "") or ""),
            bump=self._bump,
        )
        self._last_call_at: float = 0.0
        self._last_call_lock = threading.Lock()
        # 探测成功的模型映射（避免每轮都先撞一次不可用模型）
        self._resolved_models: dict[str, tuple[str, str]] = {}

    # ── 会话缓存持久化（实现见 adapters/conv_cache.py）──
    def flush_cache(self) -> None:
        self._conv_cache.flush()

    def _cached_conv(self, key: str) -> str:
        return self._conv_cache.cached(key)[0]

    def _cached_last_msg(self, key: str) -> str:
        return self._conv_cache.cached(key)[1]

    def _store_conv(
        self, key: str, chat_id: str, last_msg: str = "", used_tokens: int = 0
    ) -> None:
        self._conv_cache.store(key, chat_id, last_msg, used_tokens=used_tokens)

    def _conv_key(self, payload: dict) -> str:
        """会话指纹（tool_bridge.conv_fingerprint）：system + 首条 user +
        首条 assistant 指纹 + model + key_id。首条 assistant 防跨会话串台
        （不同会话 system/开场白可能相同，模型回复几乎必然不同）。"""
        return tool_bridge.conv_fingerprint(
            payload.get("messages") or [],
            payload.get("model") or "",
            payload.get("_wapi_key_id") or "",
        )

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

    # ── HTTP 基础 ──────────────────────────────────────
    @staticmethod
    def _split(url: str) -> tuple[str, str]:
        return netutil.split_url(url, "www.kimi.com")

    def _conn(self, url: str, timeout: int):
        return netutil.open_conn(
            url,
            timeout,
            egress=getattr(self.config, "egress", "system"),
            proxy=str(getattr(self.config, "proxy", "") or ""),
            default_host="www.kimi.com",
            logger=self.logger,
        )

    def _rpc(self, method: str, body: dict, timeout: int = 20) -> dict:
        """调用 ChatService 的一元方法（DeleteChat 等）。

        一元方法走普通 JSON（application/json），不走 Connect 流式帧。
        """
        url = self.config.rpc_url(method)
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = _build_headers(self.config, self._valid_token(), stream=False)
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "application/json"
        headers.pop("Cache-Control", None)
        headers.pop("Pragma", None)
        conn = self._conn(url, timeout)
        try:
            _, path = self._split(url)
            conn.request("POST", path, body=payload, headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
            if resp.status >= 400:
                if resp.status == 429:
                    self._bump("rate_limit")
                raise KimiError(
                    f"HTTP {resp.status}: {raw[:200].decode('utf-8', 'replace')}"
                )
            try:
                data = json.loads(raw.decode("utf-8")) if raw else {}
            except Exception:
                return {}
            return data if isinstance(data, dict) else {}
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _open_chat(self, body: dict, token: str, timeout: int):
        url = self.config.chat_url
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        conn = self._conn(url, timeout)
        _, path = self._split(url)
        try:
            conn.request(
                "POST",
                path,
                body=_frame(payload),
                headers=_connect_headers(self.config, token),
            )
            resp = conn.getresponse()
        except Exception:
            conn.close()
            raise
        if resp.status >= 400:
            raw = resp.read()
            conn.close()
            if resp.status == 429:
                self._bump("rate_limit")
            raise KimiError(
                f"HTTP {resp.status}: {raw[:200].decode('utf-8', 'replace')}"
            )
        return conn, resp

    # ── accessToken 自动续期 ────────────────────────────
    def _valid_token(self, force: bool = False) -> str:
        """返回可用的 accessToken；过期/401 时用 refreshToken 换新的。

        Kimi 的 accessToken 只有 15 分钟，refreshToken 有 90 天。
        续期接口：GET /api/auth/token/refresh，refresh 作 Bearer。
        返回体同时给新的 access 和 refresh，两个都要记住。
        """
        with self._token_lock:
            now = time.time()
            if (
                not force
                and self._access_token
                and now - self._token_checked_at < self._token_ttl
            ):
                return self._access_token
            if not self._refresh_token:
                self._token_checked_at = now
                return self._access_token
            url = self.config.refresh_url
            headers = {
                "Authorization": f"Bearer {self._refresh_token}",
                "Accept": "application/json",
                "User-Agent": self.config.user_agent or USER_AGENTS[0],
                "Origin": self.config.base_url,
                "Referer": self.config.base_url + "/",
            }
            if self.config.cookie:
                headers["Cookie"] = self.config.cookie
            try:
                conn = self._conn(url, 20)
                _, path = self._split(url)
                conn.request("GET", path, headers=headers)
                resp = conn.getresponse()
                raw = resp.read()
                conn.close()
                if resp.status >= 400:
                    self._bump("rate_limit" if resp.status == 429 else "upstream")
                    self.logger.warning(
                        "Kimi 续期失败 HTTP %s: %s",
                        resp.status,
                        raw[:160].decode("utf-8", "replace"),
                    )
                    self._token_checked_at = now
                    return self._access_token
                data = json.loads(raw.decode("utf-8"))
                acc = str(data.get("access_token") or "")
                ref = str(data.get("refresh_token") or "")
                if acc:
                    self._access_token = acc
                    self.config.auth_token = acc
                if ref:
                    self._refresh_token = ref
                    self.config.refresh_token = ref
                self._token_checked_at = now
                if acc:
                    self.logger.info("Kimi accessToken 已自动续期")
                return self._access_token
            except Exception as exc:  # noqa: BLE001
                try:
                    conn.close()  # 续期中途抛错也必须关连接
                except Exception:
                    pass
                self.logger.warning("Kimi 续期异常: %s", exc)
                self._token_checked_at = now
                return self._access_token

    # ── 请求体 ─────────────────────────────────────────
    def _body(
        self,
        payload: dict,
        chat_id: str = "",
        parent_id: str = "",
        internal_model: str = "",
        scenario: str = _SCENARIO_CHAT,
    ) -> dict:
        messages = payload.get("messages") or []
        # tool_choice="none" 时不注入也不解析，等价于没有工具
        tools = payload.get("tools") or []
        if payload.get("tool_choice") == "none":
            tools = []
        if tools:
            messages = _inject_tools(messages, tools)
        system_text = ""
        for m in messages:
            if not isinstance(m, dict):
                continue
            if m.get("role") == "system" and not system_text:
                system_text = _text_of(m)
        # 多轮关键：最后一条 assistant 之后、上游还没见过的增量。
        # role=tool 的执行结果在这里被渲染成 <tool_result> 发给网页会话，
        # 否则工具结果永远到不了模型眼里，多轮工具链是断的。
        user_text = tool_bridge.increment_text(messages, dsml=True)

        # Kimi 网页版的 options.systemPrompt 实测不生效（CHAT/OK_COMPUTER
        # 场景都无视它），客户端的 system 与工具协议必须随首条用户消息
        # 投递——增量协议会把首轮内容永久留在上游会话里，后续轮无需重发。
        if not chat_id and (system_text or tools):
            delivery = []
            if system_text:
                delivery.append(
                    "<system_instructions>\n" + system_text + "\n</system_instructions>"
                )
            if tools:
                delivery.append(
                    "<tool_protocol>\n" + _tools_protocol(tools) + "\n</tool_protocol>"
                )
            delivery.append(user_text)
            user_text = "\n\n---\n\n".join(delivery)
        model = str(payload.get("model") or self.config.default_model)
        opts = _effort_to_payload(
            str(payload.get("reasoning_effort") or self.config.default_effort), model
        )
        options: dict = {
            "thinking": bool(opts.get("thinking")),
            "model": internal_model or _internal_models(model)[0][0],
        }
        if opts.get("reasoningEffort"):
            options["reasoningEffort"] = opts["reasoningEffort"]
        if system_text:
            options["systemPrompt"] = system_text
        body: dict = {
            "chatId": chat_id,
            "scenario": scenario,
            "message": {
                "role": "user",
                "blocks": [{"text": {"content": user_text}}],
            },
            "options": options,
        }
        if parent_id:
            body["message"]["parentId"] = parent_id
        if self.config.search_enabled:
            body["tools"] = [
                {
                    "type": "TOOL_TYPE_SEARCH",
                    "search": {"force": False},
                }
            ]
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

    # ── 上游会话管理（面板「会话管理」用）──────────────
    def list_upstream_sessions(self, max_pages: int = 20) -> list[dict]:
        """拉取该账号在 Kimi 的上游会话列表。

        接口（实测）：RPC 方法 ``ListChats`` → {"chats": [...]}，
        每项含 id / name / createTime / updateTime / status。
        """
        out: list[dict] = []
        page_token = ""
        for _ in range(max(1, int(max_pages))):
            body: dict = {"pageSize": 50}
            if page_token:
                body["pageToken"] = page_token
            try:
                res = self._rpc("ListChats", body, timeout=20)
            except Exception as exc:  # noqa: BLE001
                raise KimiError(f"拉取 Kimi 会话列表失败: {exc}", 502) from exc
            chats = res.get("chats") or []
            out.extend(chats)
            page_token = str(res.get("nextPageToken") or "")
            if not page_token or not chats:
                break
        return out

    def purge_upstream_sessions(
        self, *, keep_titles: tuple = (), dry_run: bool = False
    ) -> dict:
        """删除该账号上游全部会话（可保留标题命中的）。"""
        sessions = self.list_upstream_sessions()
        doomed: list[str] = []
        kept = 0
        for s in sessions:
            title = str(s.get("name") or s.get("title") or "")
            if any(k and k in title for k in keep_titles):
                kept += 1
                continue
            cid = s.get("id")
            if cid:
                doomed.append(str(cid))

        res = {
            "total": len(sessions),
            "matched": len(doomed),
            "kept": kept,
            "deleted": 0,
            "failed": 0,
            "dry_run": bool(dry_run),
        }
        if dry_run or not doomed:
            return res

        for cid in doomed:
            try:
                self._rpc("DeleteChat", {"chatId": cid}, timeout=15)
                res["deleted"] += 1
            except Exception as exc:  # noqa: BLE001
                res["failed"] += 1
                self.logger.warning("删除 Kimi 会话失败 %s: %s", cid[:12], exc)

        for cid in doomed:
            try:
                self._conv_cache.delete_by_session_id(cid)
            except Exception:  # noqa: BLE001
                pass

        self.logger.info(
            "Kimi 会话清理完成 total=%d deleted=%d failed=%d kept=%d",
            res["total"],
            res["deleted"],
            res["failed"],
            res["kept"],
        )
        return res

    # ── 会话清理（供后台清理线程调用） ─────────────────
    def cleanup_stale_sessions(self) -> None:
        """删除已过期/超轮次的 Kimi 会话，并从缓存移除。"""
        if not self.config.delete_conversation:
            return
        stale = self._conv_cache.collect_stale()
        if not stale:
            return
        removed = 0
        for _key, chat_id in stale:
            try:
                self._rpc(
                    "DeleteChat",
                    {"chatId": chat_id},
                    timeout=min(self.config.request_timeout, 15),
                )
                removed += 1
            except Exception:  # noqa: BLE001
                pass
        if removed:
            self.logger.info("Kimi 会话清理：删除 %d 个过期会话", removed)
            for _ in range(removed):
                self._bump("deleted")

    # ── 流式事件 ───────────────────────────────────────
    def _model_candidates(
        self, model: str, *, has_tools: bool = False
    ) -> list[tuple[str, str]]:
        candidates = _internal_models(model, has_tools=has_tools)
        resolved = self._resolved_models.get(model)
        if resolved and resolved in candidates:
            candidates.remove(resolved)
            candidates.insert(0, resolved)
        return candidates

    def _stream_events(self, payload: dict):
        conv_key = self._conv_key(payload)
        # 首轮绝不复用：新客户端会话的首轮指纹常与旧会话相同（同样的
        # system + 开场白），复用会让模型接着旧上下文聊
        cached = (
            ""
            if tool_bridge.is_first_turn(payload.get("messages"))
            else self._cached_conv(conv_key)
        )
        if not cached:
            self._bump("miss")
        parent = self._cached_last_msg(conv_key) if cached else ""

        self._throttle()
        self._bump("upstream")

        model = str(payload.get("model") or self.config.default_model)
        candidates = self._model_candidates(
            model,
            has_tools=bool(payload.get("tools"))
            and payload.get("tool_choice") != "none",
        )
        last_error: KimiError | None = None

        for idx, (internal, scenario) in enumerate(candidates):
            body = self._body(
                payload,
                chat_id=cached,
                parent_id=parent,
                internal_model=internal,
                scenario=scenario,
            )
            try:
                conn, resp = self._open_chat(
                    body, self._valid_token(), self.config.request_timeout
                )
            except KimiError as exc:
                # accessToken 过期（15 分钟）→ 用 refreshToken 换新后重试一次
                if "401" not in str(exc):
                    raise
                self.logger.info("Kimi 返回 401，尝试续期后重试")
                conn, resp = self._open_chat(
                    body, self._valid_token(force=True), self.config.request_timeout
                )

            state = _ReplyState()
            retry_model = False
            # 正常结束的两种形态：done 事件，或 0x02 结束帧（可能不带 done）。
            # 两者都没出现才算断流——否则正常回答会被误续写。
            finished = False
            try:
                for flag, raw in _iter_connect_frames(resp):
                    if flag == 0x02:
                        finished = True
                        err = _connect_error(raw)
                        if err is not None:
                            code, message = err
                            last_error = KimiError(f"{code}: {message}")
                            if (
                                code == "invalid_argument"
                                and not state.text
                                and not state.think
                                and idx + 1 < len(candidates)
                            ):
                                retry_model = True
                                break
                            raise last_error
                        break
                    try:
                        event = json.loads(raw.decode("utf-8"))
                    except Exception:
                        continue
                    state.apply(event)
                    if state.error:
                        raise KimiError(state.error)
                    yield {
                        "text": state.text_value(),
                        "reasoning": state.think_value(),
                        "chat_id": state.chat_id,
                    }
                    if state.done:
                        finished = True
                        break
            finally:
                try:
                    resp.read()
                except Exception:
                    pass
                try:
                    conn.close()
                except Exception:
                    pass

            if retry_model:
                self.logger.info(
                    "Kimi 模型 %s(%s) 不可用，尝试 %s",
                    internal,
                    scenario,
                    candidates[idx + 1],
                )
                continue

            # ── 断流自动续写 ──
            # 上游没发 done 事件、也没发 0x02 结束帧就断流：向同一 chat 发
            # 「继续」，parent 指向截断的 assistant 消息。续写轮用全新
            # _ReplyState + 基线拼接，输出对上层 prefix-diff 保持单调。
            # 失败降级为截断。锚点必须是**本轮**的 assistant 消息 id——
            # 回退到上一轮的 id 会让「继续」挂在旧上下文上成为孤儿分支。
            cur_chat = state.chat_id or cached
            cur_parent = state.assistant_message_id or state.last_message_id or parent
            base_text = ""
            base_reasoning = ""
            _cont_rounds = 0
            cont_failed = False
            # P1 续写接缝剪除：模型「继续」常把断点前最后一段逐字重说一遍。
            # 增量开头只要「出现在已发尾巴里」就扣住，分叉时提交最长重叠——
            # 剪掉的字节都与客户端已有内容逐字相同，数学上不丢新内容。
            cont_shown = ""  # 本续写轮已放行给客户端的文本（剪除后）
            cont_fed = ""  # 本续写轮上游已给出的原始文本（切增量用）
            seam = None  # 仅续写轮激活；None = 直通
            while (
                not finished
                and not cont_failed
                and _cont_rounds < MAX_AUTO_CONTINUATIONS
                and cur_chat
                and state.assistant_message_id
            ):
                _cont_rounds += 1
                self._bump("auto_continued")
                self.logger.warning(
                    "上游流提前中断（未见结束信号），自动续写 %d/%d",
                    _cont_rounds,
                    MAX_AUTO_CONTINUATIONS,
                )
                if _cont_rounds == 1:
                    base_text += state.text_value()  # 首轮无接缝，全文已示
                else:
                    base_text += cont_shown  # 上一续写轮只并入已示部分
                base_reasoning += state.think_value()
                state = _ReplyState()
                seam = tool_bridge.SeamTrimmer(base_text[-tool_bridge.SEAM_WINDOW :])
                cont_shown = ""
                cont_fed = ""
                cont_finished = False
                try:
                    cont_payload = {
                        "model": model,
                        "messages": [{"role": "user", "content": CONTINUE_PROMPT}],
                    }
                    if payload.get("reasoning_effort"):
                        cont_payload["reasoning_effort"] = payload["reasoning_effort"]
                    cont_body = self._body(
                        cont_payload,
                        chat_id=cur_chat,
                        parent_id=cur_parent,
                        internal_model=internal,
                        scenario=scenario,
                    )
                    conn, resp = self._open_chat(
                        cont_body, self._valid_token(), self.config.request_timeout
                    )
                    try:
                        for flag, raw in _iter_connect_frames(resp):
                            if flag == 0x02:
                                cont_finished = True
                                err = _connect_error(raw)
                                if err is not None:
                                    # 续写被上游拒绝：按截断收场，绝不误报 stop
                                    self.logger.warning(
                                        "续写轮被上游拒绝（按截断返回）: %s: %s",
                                        err[0],
                                        err[1],
                                    )
                                    cont_failed = True
                                break
                            try:
                                event = json.loads(raw.decode("utf-8"))
                            except Exception:
                                continue
                            state.apply(event)
                            if state.error:
                                self.logger.warning(
                                    "续写轮错误（按截断返回）: %s", state.error
                                )
                                cont_failed = True
                                break
                            cur = state.text_value()
                            if seam is not None:
                                piece = cur[len(cont_fed) :]
                                cont_fed = cur
                                seam.feed(piece)
                                cont_shown = seam.committed
                            else:
                                cont_shown = cur
                            yield {
                                "text": base_text + cont_shown,
                                "reasoning": base_reasoning + state.think_value(),
                                "chat_id": state.chat_id or cur_chat,
                                "finish_seen": cont_finished or state.done,
                            }
                            if state.done:
                                cont_finished = True
                                break
                    finally:
                        try:
                            resp.read()
                        except Exception:
                            pass
                        try:
                            conn.close()
                        except Exception:
                            pass
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning("自动续写失败（按截断返回）: %s", exc)
                    break
                finished = (cont_finished or state.done) and not cont_failed
                cur_parent = (
                    state.assistant_message_id or state.last_message_id or cur_parent
                )

            self._resolved_models[model] = (internal, scenario)
            chat_id = state.chat_id or cached or cur_chat
            # 最终可见文本 = 已并基线 + 本轮已示部分（接缝扣住的复读丢弃）；
            # 没进续写轮时本轮全文就是 state.text_value()
            reply = base_text + (cont_shown if _cont_rounds > 0 else state.text_value())
            if chat_id and self.config.session_cache_enabled:
                # store key 必须与客户端多轮回传时的查询 key 一致：
                # 把本轮 assistant（解析后的调用或文本）追加进历史再算指纹
                calls = _parse_tool_calls(reply, payload.get("tools") or [])
                # 思考顶正文对齐：正文为空时交付侧会把思考提升为 content，
                # store 的 assistant 锚点必须用同一文本，否则多轮必 miss
                # （2026-10-30 审计 KIMI-F2；空白/标记差异由指纹归一吸收）
                reply_for_store = reply if reply.strip() else (reasoning or "")
                asst = {
                    "role": "assistant",
                    "content": None if calls else reply_for_store,
                    "tool_calls": calls or None,
                }
                store_key = tool_bridge.conv_fingerprint(
                    (payload.get("messages") or []) + [asst],
                    payload.get("model") or "",
                    payload.get("_wapi_key_id") or "",
                )
                # 最后一轮可能是失败的续写（无消息 id）：回退到 cur_parent，
                # 绝不把缓存里的 parent 倒退回上一轮（会话链会分叉丢上下文）
                self._store_conv(
                    store_key,
                    chat_id,
                    state.assistant_message_id or state.last_message_id or cur_parent,
                    used_tokens=conv_cache.estimate_tokens(
                        state.text_value(), state.think_value()
                    ),
                )
            yield {
                "__done__": True,
                "text": reply,
                "reasoning": base_reasoning + state.think_value(),
                "chat_id": chat_id,
                "finish_seen": finished,
            }
            return

        if last_error is not None:
            raise last_error
        raise KimiError("没有可用的 Kimi 模型")

    # ── 对外接口 ───────────────────────────────────────
    def _forced_retry(
        self, payload: dict, tools: list, first_reply: str = ""
    ) -> tuple[str, str, list]:
        """tool_choice=required 强校验重试：加强指令再问一次（上限 1 次）。

        返回重试后的 (text, reasoning, calls)；仍解析不出调用则抛
        KimiError——按明确错误返回，而不是静默当普通文本。
        """
        self.logger.info("Kimi tool_choice=required 首轮未调用工具，加强指令重试一次")
        retry = dict(payload)
        retry["messages"] = list(payload.get("messages") or [])
        if first_reply:
            # 把首轮真实回复作为 assistant 补进重试上下文：否则首轮场景
            # is_first_turn → 新建上游会话、增量只发 nudge——模型看不到
            # 原始任务，只能凭空编一个工具调用（2026-10-30 审计 KIMI-1/
            # QWEN-2）。补上后指纹命中首轮会话，模型看到 任务+自己的回答+催促。
            retry["messages"].append({"role": "assistant", "content": first_reply})
        retry["messages"].append(
            {
                "role": "user",
                "content": tool_bridge.required_nudge(dsml=True),
            }
        )
        text = reasoning = ""
        for ev in self._stream_events(retry):
            text, reasoning = ev["text"], ev["reasoning"]
        calls = _parse_tool_calls(text, tools)
        if not calls:
            raise KimiError(
                "工具调用强校验失败：重试一次后模型仍未按要求调用工具"
                "（tool_choice=required）"
            )
        return text, reasoning, calls

    def chat_completion(self, payload: dict) -> dict:
        model = payload.get("model") or self.config.default_model
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        tools = payload.get("tools") or []
        text = reasoning = ""
        calls: list = []
        finish_seen = True
        chat_id = ""
        # P0 空响应重试：上游正常结束但客户端可见输出全空（骨架标记被
        # 扣光/剥光也算空）→ 作废上游会话重跑一次，再空按原样返回。
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
            calls = _parse_tool_calls(text, tools) if tools else []
            # 客户端可见输出 = 剥掉工具标记后的正文（思考算有输出）
            visible = _strip_tool_block(text) if tools and not calls else text
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
            text, reasoning, calls = self._forced_retry(payload, tools, first_reply=text)
        if not calls and tools and not text.strip() and reasoning.strip():
            # P0 思考双通道：正文可见输出为空时才扫 reasoning 里的完整
            # DSML 块——思考里的草稿调用绝不提取（shell 级误执行风险）
            rcalls = _parse_tool_calls(reasoning, tools, require_marker=True)
            if rcalls:
                self.logger.warning("思考里发现完整工具调用（正文为空），已提取下发")
                calls = rcalls
                reasoning = _strip_tool_block(reasoning)
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

        # 空正文兜底：Kimi 全系实测会把最终答案写进思考块（正文空白），
        # content 为空时用 reasoning 顶上，保证客户端拿得到答案
        content, reasoning = reasoning_bridge.finalize_reasoning(
            _strip_tool_block(text),
            reasoning,
            legacy=self.config.legacy_reasoning_in_content,
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

        tools = payload.get("tools") or []
        if payload.get("tool_choice") == "none":
            tools = []
        forced = bool(tools) and tool_bridge.is_forced_tool_choice(payload)

        # tool_choice=required 强校验模式：不直播（客户端要求必须拿到工具
        # 调用，直播一段注定作废的正文没有意义），解析失败先重试一次，
        # 再失败按明确错误返回
        if forced:
            text = reasoning = ""
            try:
                for ev in self._stream_events(payload):
                    text, reasoning = ev["text"], ev["reasoning"]
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Kimi 流式请求失败: %s", exc)
                err = {
                    "error": {"message": f"上游错误: {exc}", "type": "upstream_error"}
                }
                yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            calls = _parse_tool_calls(text, tools)
            if not calls:
                try:
                    text, reasoning, calls = self._forced_retry(payload, tools, first_reply=text)
                except KimiError as exc:
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

        # 有工具时扣住标记文本，不让 DSML/旧标记漏进 content（增量口径）
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
                tool_bridge.DeltaToolBuffer(
                    [tool_bridge.DSML_TOOL_CALLS_OPEN, TOOL_MARKER],
                    fragment_re=tool_bridge.MARKER_FRAGMENT_RE,
                    open_re=tool_bridge.MARKER_OPEN_LOOSE_RE,
                )
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
                self.logger.warning("Kimi 流式请求失败: %s", exc)
                err = {"error": {"message": f"上游错误: {exc}", "type": "upstream_error"}}
                yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return

            # 工具调用：网页版没有原生 function calling，改成解析约定标记后按
            # OpenAI 流式 tool_calls 分片下发，客户端才能真的执行本地工具。
            calls = (
                _parse_tool_calls(buf.text if buf is not None else sent_text, tools)
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

        # P0 思考双通道：正文可见输出为空时才扫 reasoning 里的完整 DSML 块
        # ——思考里的草稿调用绝不提取（shell 级误执行风险）
        if tools and not calls and not emitted_tail.strip() and sent_reasoning:
            rcalls = _parse_tool_calls(sent_reasoning, tools)
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
                "Kimi 输出了工具调用标记但未解析成调用（工具名不在声明清单"
                "或 JSON 残缺），已按普通文本剥离。片段: %r",
                (_strip_tool_block(buf.text) or buf.text)[-200:],
            )

        # 普通回答：放出缓冲器扣住的尾巴（标记被拆包也没关系），
        # 再做空正文兜底——思考里就是答案时补发 content
        if buf is not None:
            tail = _strip_tool_block(buf.flush())
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
        yield chunk({}, usage=_usage(payload, sent_text, sent_reasoning))
        if not finish_seen:
            self.logger.warning(
                "上游流提前中断（续写未救回），按截断返回 finish=length"
            )
        yield chunk({}, finish="stop" if finish_seen else "length")
        yield "data: [DONE]\n\n"
