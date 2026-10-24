"""网页版模型的 OpenAI tools 桥接（Kimi / 智谱清言 / 通义千问共用）。

各网页平台没有原生 function calling，桥接约定分三步：

1. 注入   适配器把工具清单写进 system prompt，要求模型用约定标记输出
   「一次」工具调用。标记格式按平台对齐其模型的训练习惯，命中率更高：
     Kimi / 智谱   行内标记 ``<<<TOOL_CALL>>>{json}``
     千问          Hermes 风格标签 ``<tool_call>{json}</tool_call>``
   （Qwen 系列在训练时就见过这个格式，输出更稳。）

2. 增量   网页会话是增量协议：每次请求只把「上一条 assistant 之后、
   上游还没见过的内容」发给会话。工具链能不能多轮，全看这一步：
   客户端执行完工具会送回 role=tool 的结果消息，`increment_text()`
   把它渲染成 <tool_response> 块作为本轮增量，模型基于真实执行结果
   继续——要么再调一个工具，要么给出最终回答。没有这一步，工具
   结果永远到不了模型眼里，多轮链路是断的。

3. 解析   流式输出经 `ToolStreamBuffer` 扣住标记文本不漏进 content，
   结束后 `parse_tool_calls()` 解析成标准 OpenAI tool_calls 下发；
   客户端执行工具后进入下一轮，回到第 2 步。

设计约束：
- 一轮只解析一个工具调用（网页模型并行调用不可靠）；
- 只放行 tools 清单里声明过的工具名，模型幻觉出来的名字一律丢弃；
- `tool_choice="none"` 时适配器应完全不注入、完全不解析。
"""
from __future__ import annotations

import json
import re
import uuid

__all__ = [
    "text_of",
    "tool_names",
    "tool_name_by_id",
    "render_tool_result",
    "increment_text",
    "parse_tool_calls",
    "strip_tool_markup",
    "ToolStreamBuffer",
]


def text_of(msg: dict) -> str:
    """提取消息文本；content 为字符串或 OpenAI 多段数组都兼容。"""
    if not isinstance(msg, dict):
        return ""
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


def tool_names(tools: list) -> set:
    """tools 清单里声明过的函数名，用于放行校验。"""
    names = set()
    for t in tools or []:
        if isinstance(t, dict):
            fn = t.get("function") or t
            name = fn.get("name")
            if name:
                names.add(name)
    return names


def tool_name_by_id(messages: list) -> dict:
    """从 payload 里的 assistant.tool_calls 建立 tool_call_id → 函数名映射，
    用于把 role=tool 结果对回它的工具名。"""
    out = {}
    for m in messages or []:
        if isinstance(m, dict) and m.get("role") == "assistant":
            for c in m.get("tool_calls") or []:
                if isinstance(c, dict):
                    fn = c.get("function") or {}
                    if c.get("id") and fn.get("name"):
                        out[c["id"]] = fn["name"]
    return out


def render_tool_result(name: str, content: str, *, tag: str = "tool_result") -> str:
    """把一次工具执行结果渲染成模型可读的标签块。"""
    return f'<{tag} tool="{name}">\n{content}\n</{tag}>'


def increment_text(messages: list, *, result_tag: str = "tool_result") -> str:
    """计算本轮应发给网页会话的增量文本。

    规则（与网页会话的增量协议对齐）：
    - 会话里还没有 assistant（首轮）：发最后一条 user；
    - 最后一条 assistant 之后有 role=tool 结果：把结果渲染成
      `<result_tag>` 块，尾随的新 user 消息一并附上——这是工具链
      多轮的关键增量；
    - 只有普通 user 跟在后面（普通多轮）：发最后一条 user，与
      历史行为完全一致。
    """
    msgs = [m for m in (messages or []) if isinstance(m, dict)]
    last_assist = -1
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant":
            last_assist = i

    if last_assist < 0:
        for m in reversed(msgs):
            if m.get("role") == "user":
                return text_of(m)
        return ""

    id2name = tool_name_by_id(msgs[: last_assist + 1])
    parts: list = []
    for m in msgs[last_assist + 1:]:
        role = m.get("role")
        if role == "tool":
            content = text_of(m)
            if content.strip():
                name = id2name.get(m.get("tool_call_id") or "",
                                   m.get("name") or "tool")
                parts.append(render_tool_result(name, content, tag=result_tag))
        elif role == "user":
            t = text_of(m)
            if t.strip():
                parts.append(t)
    if parts:
        return "\n\n".join(parts)

    # assistant 之后没有新内容（不该发生）：退回最后一条 user，不改变普通多轮行为
    for m in reversed(msgs):
        if m.get("role") == "user":
            return text_of(m)
    return ""


def _loads_loose(raw: str):
    """宽松 JSON 解析：剥代码围栏、截取花括号、去尾逗号、修全角标点。
    网页模型输出的 JSON 常带这些小毛病，硬解析会白白丢调用。"""
    raw = (raw or "").strip().strip("`").strip()
    if raw.startswith("json"):
        raw = raw[4:].strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        pass
    s, e = raw.find("{"), raw.rfind("}")
    if s < 0 or e <= s:
        return None
    frag = raw[s:e + 1]
    for candidate in (
        frag,
        re.sub(r",\s*([}\]])", r"\1", frag),          # 尾逗号
        frag.replace("，", ",").replace("：", ":"),    # 全角标点
        re.sub(r",\s*([}\]])", r"\1",
               frag.replace("，", ",").replace("：", ":")),
    ):
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def parse_tool_calls(text: str, tools: list, *, tag: str | None = None,
                     marker: str | None = None, limit: int = 1) -> list:
    """从模型输出解析工具调用，返回标准 OpenAI tool_calls 结构。

    - tag:    标签形式 ``<tag>{json}</tag>``（DOTALL，模型最常输出）
    - marker: 行内标记 ``<marker>{json}``（取该行剩余部分）
    两者都提供时标签优先、标记兜底。只放行 tools 里声明过的名字；
    一轮最多返回 limit 个调用（默认 1，网页模型并行调用不可靠）。
    """
    allowed = tool_names(tools)
    if not text or not allowed:
        return []

    blobs: list = []
    if tag:
        blobs += re.findall(rf"<{tag}>(.*?)</{tag}>", text, re.S)
    if marker:
        for line in text.splitlines():
            idx = line.find(marker)
            if idx >= 0:
                blobs.append(line[idx + len(marker):])

    calls = []
    for raw in blobs:
        obj = _loads_loose(raw)
        if not isinstance(obj, dict):
            continue
        name = obj.get("name") or obj.get("tool") or ""
        if not name or name not in allowed:
            continue
        args = obj.get("arguments", obj.get("args", {}))
        if not isinstance(args, str):
            args = json.dumps(args or {}, ensure_ascii=False)
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:16]}",
            "type": "function",
            "function": {"name": name, "arguments": args},
        })
        if len(calls) >= limit:
            break
    return calls


def strip_tool_markup(text: str, *, tag: str | None = None,
                      marker: str | None = None) -> str:
    """从文本里剥掉工具标记（标签块 / 标记行），用于纯文本回退。"""
    if not text:
        return text or ""
    out = text
    if tag:
        # 连同行尾换行一起移除，避免剥完留下一串空行
        out = re.sub(rf"<{tag}>.*?</{tag}>\n?", "", out, flags=re.S)
    if marker:
        out = "\n".join(ln for ln in out.splitlines() if marker not in ln)
    return out.strip()


class ToolStreamBuffer:
    """流式输出里扣住工具标记，保证标记文本不漏进 content 增量。

    用法：每个流事件把**全文快照**喂给 `feed()`，拿到的返回值是
    「可以安全下发给客户端的完整前缀」；结束时分两种情况：
    - `opener_found` 为真：不再吐 content，直接解析 tool_calls；
    - 为假：`flush()` 把扣住的尾巴吐干净，正常 finish。
    未发现 opener 时也会回扣一小段尾巴，防止标记恰好被拆在两次
    delta 之间漏出去。
    """

    def __init__(self, openers: list) -> None:
        self._openers = [o for o in (openers or []) if o]
        self._text = ""
        self._emitted = 0
        self._open_at = -1

    def feed(self, text: str) -> str:
        self._text = text or ""
        if self._open_at < 0 and self._openers:
            for op in self._openers:
                i = self._text.find(op)
                if 0 <= i and (self._open_at < 0 or i < self._open_at):
                    self._open_at = i
        if self._open_at >= 0:
            out = self._text[self._emitted:self._open_at]
            self._emitted = self._open_at
            return out
        # 回扣一个完整 opener 的长度：完整出现的 opener 一定还没被发出
        hold = max((len(op) for op in self._openers), default=1)
        safe_end = max(self._emitted, len(self._text) - hold)
        out = self._text[self._emitted:safe_end]
        self._emitted = safe_end
        return out

    def flush(self) -> str:
        out = self._text[self._emitted:]
        self._emitted = len(self._text)
        return out

    @property
    def opener_found(self) -> bool:
        return self._open_at >= 0

    @property
    def text(self) -> str:
        return self._text
