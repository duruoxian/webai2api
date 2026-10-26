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
    "strip_call_json",
    "is_first_turn",
    "is_forced_tool_choice",
    "required_nudge",
    "assistant_fingerprint",
    "conv_fingerprint",
    "ToolStreamBuffer",
    "DeltaToolBuffer",
]


def assistant_fingerprint(msg: dict) -> str:
    """单条 assistant 消息的规范化指纹（会话 key 的防串台锚点）。

    tool_calls 做 JSON 规范化（解析后重排 key），保证「我们发出的调用」
    与「客户端历史回传的调用」指纹一致；纯文本回复取正文。
    """
    if not isinstance(msg, dict):
        return ""
    tc = msg.get("tool_calls")
    if tc:
        try:
            norm = []
            for c in tc:
                if isinstance(c, dict):
                    fn = c.get("function") or {}
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except Exception:
                        args = fn.get("arguments") or ""
                    norm.append({"name": fn.get("name") or "", "arguments": args})
            return json.dumps(norm, ensure_ascii=False, sort_keys=True)
        except Exception:
            pass
    return text_of(msg)


def conv_fingerprint(messages: list, model: str = "", key_id: str = "") -> str:
    """客户端会话指纹（会话缓存 key，三家共用）。

    组成：system 前 200 字 + 首条 user 前 500 字 + **首条 assistant 指纹**
    + model + key_id。首条 assistant 是跨会话串台的关键防线：不同客户端
    会话的 system 与开场白可能完全相同（实测 opencode 新会话撞了几天前
    旧会话的上游上下文，模型生成出旧会话的内容），而模型对相同输入的
    回复几乎必然不同。首轮请求还没有 assistant（指纹不含它 → 必然查不到
    旧会话 → 新建）；多轮请求带首轮回复原文（客户端历史回传），以此锚定。
    """
    import hashlib

    system = first = asst = ""
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "system" and not system:
            system = text_of(m)
        elif role == "user" and not first:
            first = text_of(m)
        elif role == "assistant" and not asst:
            asst = assistant_fingerprint(m)
    raw = (
        f"{system[:200]}\x00{first[:500]}\x00{asst[:200]}\x00"
        f"{model or ''}\x00{key_id or ''}"
    )
    return hashlib.sha1(raw.encode("utf-8", "ignore")).hexdigest()


def strip_call_json(text: str) -> str:
    """剥掉正文中「调用形态」的裸 JSON（{"name":..,"arguments":..}）。

    网页模型有时输出平台内置工具（如千问的 browser）的调用 JSON——
    不带协议标记、名字也不在工具清单里，无法转成 tool_calls，原样
    下发就是用户看到的「正文里一坨 JSON」。调用形态判据：顶层对象
    同时含 name 与 arguments 键。
    """
    if not text or "{" not in text:
        return text
    out = text
    while True:
        i = out.find('{"name"')
        if i < 0:
            break
        blob = _scan_json_blob(out, i)
        if not blob:
            break
        obj = _loads_loose(blob)
        if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
            out = out[:i] + out[i + len(blob) :]
        else:
            break
    return out


def is_first_turn(messages: list) -> bool:
    """会话首轮判定：messages 里还没有任何 assistant 消息。

    首轮**必须新建上游会话、绝不复用缓存**：不同客户端会话的首轮
    指纹经常相同（同样的 system + 同样的开场白，如 opencode 每个新
    会话都发「你好」），复用会让模型接着旧会话的上下文聊——实测
    表现为「新开会话却记得上一个会话干过什么」。
    """
    for m in messages or []:
        if isinstance(m, dict) and m.get("role") == "assistant":
            return False
    return True


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


def render_assistant_tool_calls(
    calls: list, text: str = "", *, marker: str = "<<<TOOL_CALL>>>"
) -> str:
    """把 assistant.tool_calls 还原成模型当初输出的调用文本（压平用）。

    全量拼平重发时，工具调用轮的 assistant.content 往往为空：如果把调用
    丢掉，模型就看不到「自己刚刚请求过什么」，后续 <tool_result> 失去
    归属，实测会原地重发同一个工具调用。
    """
    lines = []
    for c in calls or []:
        if not isinstance(c, dict):
            continue
        fn = c.get("function") or {}
        name = fn.get("name") or ""
        if not name:
            continue
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except Exception:
            args = fn.get("arguments") or {}
        lines.append(
            marker + json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
        )
    body = "\n".join(lines)
    if text and body:
        return f"{text}\n{body}"
    return body or text


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
    for m in msgs[last_assist + 1 :]:
        role = m.get("role")
        if role == "tool":
            content = text_of(m)
            if content.strip():
                name = id2name.get(m.get("tool_call_id") or "", m.get("name") or "tool")
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
    frag = raw[s : e + 1]
    for candidate in (
        frag,
        re.sub(r",\s*([}\]])", r"\1", frag),  # 尾逗号
        frag.replace("，", ",").replace("：", ":"),  # 全角标点
        re.sub(r",\s*([}\]])", r"\1", frag.replace("，", ",").replace("：", ":")),
    ):
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def _scan_json_blob(text: str, start: int) -> str:
    """从 start 起找第一个 '{'，做大括号配平扫描，返回完整 {...} 片段。

    字符串字面量里的花括号不计入配平；流被截断（没等到配对的 '}'）
    时返回已见部分，交给 _loads_loose 兜底修复。
    """
    s = text.find("{", start)
    if s < 0:
        return ""
    depth = 0
    in_str = False
    esc = False
    for i in range(s, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[s : i + 1]
    return text[s:]


def _marker_blobs(text: str, marker: str) -> list:
    """收集行内标记后的 JSON 片段：配平扫描，跨行 JSON 也能取全。

    旧实现按行截取，模型把 JSON 写成多行时只能拿到第一个 '{'，
    整个调用被丢弃后 JSON 又漏进正文——这是 qwen/kimi 实测
    「工具 JSON 当正文输出」的直接原因。
    """
    out: list = []
    idx = text.find(marker)
    while idx >= 0:
        blob = _scan_json_blob(text, idx + len(marker))
        if blob:
            out.append(blob)
        idx = text.find(marker, idx + len(marker))
    return out


def _bare_blobs(text: str, allowed: set) -> list:
    """裸 JSON 兜底：扫描全文的花括号块，只留长得像调用的。

    判据（两条都要，避免误伤）：
    - name / tool 键命中 tools 白名单（幻觉名字不放行）；
    - 必须带 arguments / args 键——模型回显工具定义时内层也有
      {"name": ...}，但没有 arguments，不算调用。
    """
    out: list = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        blob = _scan_json_blob(text, i)
        if not blob:
            break
        obj = _loads_loose(blob)
        if isinstance(obj, dict):
            name = obj.get("name") or obj.get("tool") or ""
            if name and name in allowed and ("arguments" in obj or "args" in obj):
                out.append(blob)
                i += len(blob)
                continue
        i += 1
    return out


def _fix_arg_names(args: dict, tool: dict | None) -> dict:
    """把 arguments 的参数名校正到工具 schema 的形参名。

    小模型（如 omni-flash）经常不守 schema：要 filePath 它给 path。
    校正规则（保守）：错名不在 schema 里，且与某个形参**大小写不敏感
    子串匹配唯一命中**时才改名；有歧义或无命中保持原样。
    """
    if not isinstance(args, dict) or not isinstance(tool, dict):
        return args
    params = tool.get("function", tool).get("parameters") or {}
    props = params.get("properties")
    if not isinstance(props, dict) or not props:
        return args
    keys = {k.lower(): k for k in props}
    fixed = {}
    for k, v in args.items():
        if k in props:
            fixed[k] = v
            continue
        low = k.lower()
        hits = [orig for low_k, orig in keys.items() if low in low_k]
        fixed[hits[0] if len(hits) == 1 else k] = v
    return fixed


def _build_calls(
    blobs: list, allowed: set, limit: int, tools: list | None = None
) -> list:
    """把 JSON 片段列表规整成标准 OpenAI tool_calls（白名单放行）。"""
    by_name = {}
    for t in tools or []:
        if isinstance(t, dict):
            fn = t.get("function") or t
            if fn.get("name"):
                by_name[fn["name"]] = t
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
            args = json.dumps(
                _fix_arg_names(args or {}, by_name.get(name)), ensure_ascii=False
            )
        calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:16]}",
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
        )
        if len(calls) >= limit:
            break
    return calls


def parse_tool_calls(
    text: str,
    tools: list,
    *,
    tag: str | None = None,
    marker: str | None = None,
    limit: int = 1,
) -> list:
    """从模型输出解析工具调用，返回标准 OpenAI tool_calls 结构。

    - tag:    标签形式 ``<tag>{json}</tag>``（DOTALL，模型最常输出）
    - marker: 行内标记 ``<marker>{json}``（配平扫描，支持跨行 JSON）
    两者都提供时标签优先、标记兜底；都解析不出时再做**裸 JSON 兜底**
    （模型不带任何标记直接把调用 JSON 写进正文的情形，qwen 全系
    实测高发）。只放行 tools 里声明过的名字；一轮最多返回 limit 个
    调用（默认 1，网页模型并行调用不可靠）。
    """
    allowed = tool_names(tools)
    if not text or not allowed:
        return []

    blobs: list = []
    if tag:
        blobs += re.findall(rf"<{tag}>(.*?)</{tag}>", text, re.S)
    if marker:
        blobs += _marker_blobs(text, marker)

    calls = _build_calls(blobs, allowed, limit, tools)
    if calls:
        return calls
    return _build_calls(_bare_blobs(text, allowed), allowed, limit, tools)


def strip_tool_markup(
    text: str, *, tag: str | None = None, marker: str | None = None
) -> str:
    """从文本里剥掉工具标记（标签块 / 标记 + 其 JSON），用于纯文本回退。

    marker 模式旧实现按行删，多行 JSON 的续行会残留在正文里；
    现在标记 + 配平出的整个 JSON 片段一起删。
    """
    if not text:
        return text or ""
    out = text
    if tag:
        # 连同行尾换行一起移除，避免剥完留下一串空行
        out = re.sub(rf"<{tag}>.*?</{tag}>\n?", "", out, flags=re.S)
    if marker:
        while True:
            idx = out.find(marker)
            if idx < 0:
                break
            blob = _scan_json_blob(out, idx + len(marker))
            end = idx + len(marker) + (len(blob) if blob else 0)
            # 连同紧随的换行一起删，避免剥完留下空行（与 tag 模式一致）
            if end < len(out) and out[end] == "\n":
                end += 1
            out = out[:idx] + out[end:]
        out = "\n".join(ln for ln in out.splitlines() if marker not in ln)
    return out.strip()


def is_forced_tool_choice(payload: dict) -> bool:
    """tool_choice 是否强制调用工具："required" 或指定函数对象。"""
    tc = payload.get("tool_choice")
    if tc == "required":
        return True
    return isinstance(tc, dict) and (tc.get("type") == "function" or "function" in tc)


REQUIRED_NUDGE = (
    "上一次回复没有调用任何工具，这不符合要求。"
    "这次必须调用一个工具，不要直接用文字回答，也不要解释。"
    "只输出一次工具调用，格式严格为：\n{example}"
)


def required_nudge(*, tag: str | None = None, marker: str | None = None) -> str:
    """tool_choice=required 重试时追加的加强指令（含平台各自的调用示例）。"""
    if tag:
        example = f'<{tag}>{{"name":"工具名","arguments":{{"参数名":值}}}}</{tag}>'
    else:
        example = (
            f"{marker or '<<<TOOL_CALL>>>'}"
            f'{{"name":"工具名","arguments":{{"参数名":值}}}}'
        )
    return REQUIRED_NUDGE.format(example=example)


class ToolStreamBuffer:
    """流式输出里扣住工具标记（**全文快照口径**）。

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
            out = self._text[self._emitted : self._open_at]
            self._emitted = self._open_at
            return out
        # 回扣一个完整 opener 的长度：完整出现的 opener 一定还没被发出
        hold = max((len(op) for op in self._openers), default=1)
        safe_end = max(self._emitted, len(self._text) - hold)
        out = self._text[self._emitted : safe_end]
        self._emitted = safe_end
        return out

    def flush(self) -> str:
        out = self._text[self._emitted :]
        self._emitted = len(self._text)
        return out

    @property
    def opener_found(self) -> bool:
        return self._open_at >= 0

    @property
    def text(self) -> str:
        return self._text


class DeltaToolBuffer:
    """工具标记扣留器（**增量口径**），与 ToolStreamBuffer 行为等价。

    差别只在喂法：`feed_delta(piece)` 接收**新增量**而非全文快照，内部
    自行累计。给「流循环先做无损前缀差、再扣留标记」的链路用——快照
    口径的 buf 会回扣尾部，输出非单调，嵌套在多层前缀差里会把下层
    错位咬字（GLM 实测）。
    """

    def __init__(self, openers: list) -> None:
        self._openers = [o for o in (openers or []) if o]
        self._full = ""
        self._emitted = 0
        self._open_at = -1

    def feed_delta(self, piece: str) -> str:
        if piece:
            self._full += piece
        if self._open_at < 0 and self._openers:
            for op in self._openers:
                i = self._full.find(op)
                if 0 <= i and (self._open_at < 0 or i < self._open_at):
                    self._open_at = i
        if self._open_at >= 0:
            out = self._full[self._emitted : self._open_at]
            self._emitted = self._open_at
            return out
        hold = max((len(op) for op in self._openers), default=1)
        safe_end = max(self._emitted, len(self._full) - hold)
        out = self._full[self._emitted : safe_end]
        self._emitted = safe_end
        return out

    def flush(self) -> str:
        out = self._full[self._emitted :]
        self._emitted = len(self._full)
        return out

    @property
    def opener_found(self) -> bool:
        return self._open_at >= 0

    @property
    def text(self) -> str:
        return self._full
