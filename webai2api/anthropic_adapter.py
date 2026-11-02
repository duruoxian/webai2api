"""Anthropic Messages 协议适配层（/v1/messages 端点）。

把 Anthropic 协议（Claude Code / ZCode 等客户端原生格式）翻译成内部
OpenAI 形状，复用既有网关链路（鉴权/额度/选号/重试/用量统计）；响应
与流式事件再翻译回 Anthropic 格式。流式事件结构参考 Rfym21/Qwen2API
的 anthropic 控制器（协议逆向自 Claude Code 真实流量）。

四平台的上游适配（工具调用/流式/会话）完全复用，本模块只做**入口
协议翻译**，不碰任何平台 client。
"""

from __future__ import annotations

import json
import uuid

__all__ = [
    "anthropic_to_openai",
    "openai_to_anthropic",
    "anthropic_error",
    "openai_error_to_anthropic",
    "AnthropicStreamTranslator",
    "STOP_REASON_MAP",
]

# OpenAI finish_reason → Anthropic stop_reason
STOP_REASON_MAP = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "length": "max_tokens",
    "content_filter": "end_turn",
}


def _text_of(content) -> str:
    """Anthropic content 块数组/字符串 → 纯文本（拼接 text 块）。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = ""
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                out += item.get("text") or ""
        return out
    return str(content)


def _blocks(content) -> list:
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    if content:
        return [{"type": "text", "text": str(content)}]
    return []


def anthropic_to_openai(body: dict) -> dict:
    """Anthropic Messages 请求 → 内部 OpenAI chat payload。

    支持：system（字符串/块数组）、消息块数组（text/tool_use/
    tool_result）、tools（input_schema 形状）、tool_choice 四态、
    max_tokens/temperature/top_p/stream 透传。
    """
    messages: list = []
    pending_tool_ids: list = []  # 最近一条 assistant 的 tool_call id 队列
    system = body.get("system")
    sys_text = _text_of(system) if system else ""
    if sys_text:
        messages.append({"role": "system", "content": sys_text})

    for m in body.get("messages") or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        blocks = _blocks(m.get("content"))
        tool_results = [b for b in blocks if b.get("type") == "tool_result"]
        tool_uses = [b for b in blocks if b.get("type") == "tool_use"]
        text = "".join(b.get("text") or "" for b in blocks if b.get("type") == "text")

        if role == "assistant":
            msg: dict = {"role": "assistant"}
            if tool_uses:
                msg["content"] = text or None
                msg["tool_calls"] = [
                    {
                        "id": b.get("id") or f"call_{uuid.uuid4().hex[:16]}",
                        "type": "function",
                        "function": {
                            "name": b.get("name") or "",
                            "arguments": json.dumps(
                                b.get("input") or {}, ensure_ascii=False
                            ),
                        },
                    }
                    for b in tool_uses
                ]
                pending_tool_ids = [tc["id"] for tc in msg["tool_calls"]]
            else:
                msg["content"] = text
            messages.append(msg)
            continue

        # user（或意外角色）：tool_result 块逐条转成 role=tool，
        # 其余文本合并为 user——保持相对顺序由「tool 在前、文本在后」近似
        for tr in tool_results:
            rc = tr.get("content")
            tool_text = _text_of(rc) if not isinstance(rc, str) else rc
            tid = str(tr.get("tool_use_id") or "")
            # 上游按 assistant.tool_calls[].id 匹配回填：id 对不上时部分适配层
            # （GLM）认不出工具结果，会原样重发上一次的 tool_call。客户端本就
            # 应回传同一 id：在 id 在队列里时精确配对（乱序块也不会互换结果，
            # 2026-11-01 审计 B2）；错位时才退回依序对齐。
            if tid and pending_tool_ids:
                if tid in pending_tool_ids:
                    pending_tool_ids.remove(tid)
                else:
                    tid = pending_tool_ids.pop(0)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tid,
                    "content": tool_text,
                }
            )
        # 非文本/工具块（image/document 等）：本网关上游不支持多模态上行，
        # 但绝不能让「纯图片消息」整条蒸发——blocks 非空时三条转换分支都
        # 不命中，该轮对话直接消失，客户端与模型对不上账（2026-11-01 审计
        # A2）。降级为占位文本保住轮次结构。
        placeholders = []
        for b in blocks:
            bt = b.get("type")
            if bt == "image":
                placeholders.append("[图片]")
            elif bt == "document":
                placeholders.append("[文档]")
        if placeholders:
            text = (text + "\n" if text else "") + " ".join(placeholders)
        if text:
            messages.append({"role": "user", "content": text})
        if not blocks:
            messages.append({"role": role or "user", "content": ""})

    out: dict = {
        "model": body.get("model") or "",
        "messages": messages,
        "stream": bool(body.get("stream")),
    }
    if body.get("max_tokens"):
        out["max_tokens"] = body["max_tokens"]
    if body.get("temperature") is not None:
        out["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        out["top_p"] = body["top_p"]
    if body.get("stop_sequences"):
        # Anthropic stop_sequences → OpenAI stop（2026-11-01 审计 B7；
        # 上游网页端对 stop 的支持程度不一，透传由适配层自行取舍）
        ss = body["stop_sequences"]
        if isinstance(ss, list) and ss:
            out["stop"] = [str(x) for x in ss if x][:4]

    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        out["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.get("name") or "",
                    "description": t.get("description") or "",
                    "parameters": t.get("input_schema") or {"type": "object"},
                },
            }
            for t in tools
            if isinstance(t, dict) and t.get("name")
        ]

    tc = body.get("tool_choice")
    # Anthropic 官方形态是对象 {"type":"auto"|"any"|"tool"}（2026-11-01
    # 审计 B6：此前只认字符串，{"type":"any"} 三分支全不命中，「必须调用
    # 工具」静默降级为 auto）。字符串分支保留作宽容兜底。
    if isinstance(tc, dict):
        ttype = tc.get("type")
        if ttype == "any":
            out["tool_choice"] = "required"
        elif ttype == "tool":
            out["tool_choice"] = {
                "type": "function",
                "function": {"name": tc.get("name") or ""},
            }
        elif ttype in ("auto", "none"):
            out["tool_choice"] = ttype
    elif tc == "any":
        out["tool_choice"] = "required"
    elif tc in ("auto", "none"):
        out["tool_choice"] = tc
    return out


def _fake_signature() -> str:
    """thinking 块的合成签名（不透明占位符）。

    Anthropic 官方 thinking 块必带 signature 字段，Claude 系客户端依赖
    它存在。我们的思考来自网页模型，没有真实签名——生成无意义占位符
    （做法与 Qwen2API 一致），保证客户端兼容性。
    """
    return f"webai2api_{uuid.uuid4().hex}"


def openai_to_anthropic(resp: dict, model: str = "") -> dict:
    """OpenAI 非流式响应 → Anthropic message 对象。"""
    choices = resp.get("choices") or []
    ch = choices[0] if choices else {}
    msg = ch.get("message") or {}
    content_blocks: list = []
    reasoning = msg.get("reasoning_content")
    if reasoning:
        content_blocks.append(
            {"type": "thinking", "thinking": reasoning, "signature": _fake_signature()}
        )
    text = msg.get("content")
    if text:
        content_blocks.append({"type": "text", "text": text})
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        try:
            inp = json.loads(fn.get("arguments") or "{}")
        except Exception:
            inp = {}
        content_blocks.append(
            {
                "type": "tool_use",
                "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:16]}",
                "name": fn.get("name") or "",
                "input": inp,
            }
        )
    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})
    fin = ch.get("finish_reason") or "stop"
    u = resp.get("usage") or {}
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model or resp.get("model") or "",
        "content": content_blocks,
        "stop_reason": STOP_REASON_MAP.get(fin, "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(u.get("prompt_tokens") or 0),
            "output_tokens": int(u.get("completion_tokens") or 0),
        },
    }


def anthropic_error(status: int, err_type: str, message: str):
    """Anthropic 错误格式（与 OpenAI 的 {"error":...} 结构不同）。"""
    body = {"type": "error", "error": {"type": normalize_error_type(err_type), "message": message}}
    return status, body


_ANTHROPIC_ERROR_TYPES = {
    "invalid_request_error",
    "authentication_error",
    "permission_error",
    "not_found_error",
    "request_too_large",
    "rate_limit_error",
    "api_error",
    "overloaded_error",
}


def normalize_error_type(err_type: str) -> str:
    """收敛到 Anthropic 合法 error.type 集合。

    OpenAI 侧的 internal_error / upstream_error / unavailable 等自造
    type 原样透传会被严格客户端判为畸形错误（2026-11-02 审查）。
    """
    t = str(err_type or "").strip()
    return t if t in _ANTHROPIC_ERROR_TYPES else "api_error"


def openai_error_to_anthropic(payload: dict) -> dict:
    """网关 OpenAI 错误体 → Anthropic 错误体。"""
    err = payload.get("error") or {}
    return {
        "type": "error",
        "error": {
            "type": normalize_error_type(err.get("type") or "api_error"),
            "message": err.get("message") or "上游错误",
        },
    }


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class AnthropicStreamTranslator:
    """把内部 OpenAI 流 chunk 翻译成 Anthropic SSE 事件流。

    输入：OpenAI chat chunk 的 JSON 对象（逐个喂 `feed`），结束时调
    `finish`。输出：Anthropic SSE 字符串（event: + data:，已含空行）。

    块管理：thinking / text / tool_use 三种 content block 按需开启，
    类型切换时自动闭合旧块。OpenAI 的 tool_calls.arguments 分片映射
    为 input_json_delta。上游错误映射为 event: error。
    """

    def __init__(self, model: str = "") -> None:
        self._model = model
        self._started = False
        self._index = -1
        self._type: str | None = None
        self._stop_reason = "end_turn"
        self._in_tokens = 0
        self._out_tokens = 0
        self._text_parts: list[str] = []
        # 当前 tool_use 块的调用 id（OpenAI 流式把 id 放在 tool_call 对象上；
        # id 变化 = 新调用，必须切新 content_block，否则第 2 个起全部并进
        # 第 1 块（2026-10-30 审计 GW-1）。GW-1 首修只改了 feed 的触发条件，
        # _switch 的同类短路未处理，并行第二条仍整体丢失（2026-11-01 审计
        # A1 运行时复现）——_switch 现感知 tool→tool 换 id）
        self._tool_id = ""
        # 最近一次见到的 (id, name)，args 被其它块打断后重开时用
        # （否则重开的 tool_use 块无名，客户端按 name 查工具必败）
        self._last_tool_meta: dict = {}
        self._errored = False

    # ── 内部：确保 message_start 已发 ──
    def _start_events(self) -> list[str]:
        if self._started:
            return []
        self._started = True
        return [
            _sse(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": f"msg_{uuid.uuid4().hex[:24]}",
                        "type": "message",
                        "role": "assistant",
                        "model": self._model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                },
            )
        ]

    def _switch(self, want: str, tool_meta: dict | None = None) -> list[str]:
        out: list[str] = []
        if self._type == want:
            # 例外：tool_use → tool_use 且 id 变化（并行调用）——必须闭合
            # 旧块开新块；其余同类切换是 no-op（2026-11-01 审计 A1）
            if not (
                want == "tool_use"
                and (tool_meta or {}).get("id")
                and self._tool_id
                and tool_meta["id"] != self._tool_id
            ):
                return out
        if self._type is not None:
            if self._type == "thinking":
                # thinking 块闭合前必须发 signature_delta（Claude 系客户端依赖）
                out.append(
                    self._delta(
                        {"type": "signature_delta", "signature": _fake_signature()}
                    )
                )
            out.append(
                _sse(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": self._index},
                )
            )
        self._index += 1
        if want == "text":
            block: dict = {"type": "text", "text": ""}
        elif want == "thinking":
            block = {"type": "thinking", "thinking": ""}
        else:
            block = {
                "type": "tool_use",
                "id": (tool_meta or {}).get("id") or f"toolu_{uuid.uuid4().hex[:16]}",
                "name": (tool_meta or {}).get("name") or "",
                "input": {},
            }
        out.append(
            _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self._index,
                    "content_block": block,
                },
            )
        )
        self._type = want
        if want == "tool_use":
            self._tool_id = block["id"]
        return out

    def _delta(self, delta: dict) -> str:
        return _sse(
            "content_block_delta",
            {"type": "content_block_delta", "index": self._index, "delta": delta},
        )

    # ── 对外：喂一个 OpenAI chunk ──
    def feed(self, obj: dict) -> list[str]:
        if self._errored:
            return []
        events = self._start_events()
        if obj.get("error"):
            self._errored = True
            events.append(_sse("error", openai_error_to_anthropic(obj)))
            return events

        u = obj.get("usage")
        if u:
            self._in_tokens = int(u.get("prompt_tokens") or self._in_tokens)
            self._out_tokens = int(u.get("completion_tokens") or self._out_tokens)

        choices = obj.get("choices") or []
        if not choices:
            return events
        ch = choices[0]
        d = ch.get("delta") or {}

        r = d.get("reasoning_content")
        if r:
            events += self._switch("thinking")
            events.append(self._delta({"type": "thinking_delta", "thinking": r}))
        c = d.get("content")
        if c:
            events += self._switch("text")
            self._text_parts.append(str(c))
            events.append(self._delta({"type": "text_delta", "text": c}))
        for tc in d.get("tool_calls") or []:
            fn = tc.get("function") or {}
            tid = str(tc.get("id") or fn.get("id") or "")
            name = str(fn.get("name") or "")
            if tid:
                prev = self._last_tool_meta
                if prev.get("id") == tid:
                    # 同一调用的后续分片常不带 name：合并保留，别抹掉
                    self._last_tool_meta = {
                        "id": tid, "name": name or prev.get("name") or ""
                    }
                else:
                    self._last_tool_meta = {"id": tid, "name": name}
            elif name:
                self._last_tool_meta = {"id": "", "name": name}
            if tid and tid != self._tool_id:
                events += self._switch("tool_use", {"id": tid, "name": name})
                self._tool_id = tid
            elif not tid and name and self._type != "tool_use":
                # 无 id 带 name：按新调用开块（每轮单调用形态）
                events += self._switch("tool_use", {"id": "", "name": name})
            args = fn.get("arguments")
            if args:
                if self._type != "tool_use":
                    # args 被其它块（思考/正文穿插）打断后恢复：用记住的
                    # meta 重开，否则重开的块无名无 id，客户端无法配对
                    # （2026-11-01 审计 D8）
                    events += self._switch("tool_use", dict(self._last_tool_meta))
                events.append(
                    self._delta({"type": "input_json_delta", "partial_json": args})
                )
        fin = ch.get("finish_reason")
        if fin:
            self._stop_reason = STOP_REASON_MAP.get(fin, "end_turn")
        return events

    # ── 对外：收尾 ──
    def finish(self) -> list[str]:
        if self._errored:
            return []
        events = self._start_events()
        if self._index < 0:
            # 全程零内容块：补一个空 text 块再收尾——与非流式空回复补空
            # text 一致；message_stop 前没有任何 content_block_start 会被
            # 严格客户端判为畸形流（2026-11-02 审查）
            self._index = 0
            self._type = "text"
            events.append(
                _sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
            )
        if self._type is not None:
            if self._type == "thinking":
                events.append(
                    self._delta(
                        {"type": "signature_delta", "signature": _fake_signature()}
                    )
                )
            events.append(
                _sse(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": self._index},
                )
            )
            self._type = None
        events.append(
            _sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": self._stop_reason, "stop_sequence": None},
                    "usage": {
                        "input_tokens": self._in_tokens,
                        "output_tokens": self._out_tokens,
                    },
                },
            )
        )
        events.append(_sse("message_stop", {"type": "message_stop"}))
        return events
