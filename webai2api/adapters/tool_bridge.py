"""网页版模型的 OpenAI tools 桥接（Kimi / 智谱清言 / 通义千问共用）。

各网页平台没有原生 function calling 接口，桥接约定分三步：

1. 注入   适配器把工具清单写进 system prompt，要求模型用约定格式输出
   「一次」工具调用。格式按平台对齐其模型的训练习惯，命中率更高：
     Kimi          DSML XML（``<|DSML|tool_calls>`` + ``<![CDATA[...]]>``）
     智谱          行内标记 ``<<<TOOL_CALL>>>{json}``
     千问          Hermes 风格标签 ``<tool_call>{json}</tool_call>``
   （Qwen 系列在训练时就见过这个格式，输出更稳。）

2. 增量   网页会话是增量协议：每次请求只把「上一条 assistant 之后、
   上游还没见过的内容」发给会话。工具链能不能多轮，全看这一步：
   客户端执行完工具会送回 role=tool 的结果消息，`increment_text()`
   把它渲染成 <tool_result> 块（Kimi 为 DSML tool_result）作为本轮
   增量，模型基于真实执行结果继续——要么再调一个工具，要么给出
   最终回答。没有这一步，工具结果永远到不了模型眼里，多轮链路是断的。

3. 解析   流式输出经 `ToolStreamBuffer` 扣住标记文本不漏进 content，
   结束后 `parse_tool_calls()` 解析成标准 OpenAI tool_calls 下发；
   客户端执行工具后进入下一轮，回到第 2 步。智谱另有上游服务端原生
   tool_calls part，由适配器直接优先采用，不经过本模块解析。

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
    "render_dsml_tool_result",
    "render_dsml_assistant_tool_calls",
    "increment_text",
    "parse_tool_calls",
    "parse_dsml_tool_calls",
    "strip_tool_markup",
    "strip_call_json",
    "is_first_turn",
    "is_forced_tool_choice",
    "SeamTrimmer",
    "SEAM_WINDOW",
    "required_nudge",
    "assistant_fingerprint",
    "conv_fingerprint",
    "ToolStreamBuffer",
    "DeltaToolBuffer",
    "DSML_TOOL_CALLS_OPEN",
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
    # 文本分支归一：store 侧存的是原始累积文本（含工具标记），客户端历史
    # 回传的是清洗/修剪后的交付文本——不归一则首个文本回复被清洗改动时
    # 第 2 轮指纹必 miss（2026-10-30 审计 GLM P2-7/KIMI-F2）。剥标记 +
    # 空白折叠后两侧收敛到同一规范形。
    txt = text_of(msg)
    try:
        txt = strip_tool_markup(txt, dsml=True)
    except Exception:
        pass
    return re.sub(r"\s+", " ", txt).strip()


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


# ── DSML（Kimi / 智谱原生工具协议）────────────────────────
# 上游网页模型在调用平台内置工具时输出的是 DSML XML，而不是 JSON：
#   <|DSML|tool_calls>
#     <|DSML|invoke name="write">
#       <|DSML|parameter name="filePath"><![CDATA[D:\a.md]]></|DSML|parameter>
#     </|DSML|invoke>
#   </|DSML|tool_calls>
# 字符串走 CDATA（换行、引号、花括号都不用转义），对象/数组用嵌套
# parameter 与 <item> 表达。让模型沿用训练里见过的格式，比自造标记稳。

DSML_TOOL_CALLS_OPEN = "<|DSML|tool_calls>"

_DSML_BLOCK_RE = re.compile(
    r"<\|DSML\|tool_calls\b[^>]*>(.*?)</\|DSML\|tool_calls\s*>", re.I | re.S
)
_DSML_INVOKE_RE = re.compile(
    r"<\|DSML\|invoke\b([^>]*)>(.*?)</\|DSML\|invoke\s*>", re.I | re.S
)
_DSML_PARAM_OPEN_RE = re.compile(r"<\|DSML\|parameter\b([^<>]*?)>", re.I)
_DSML_PARAM_CLOSE_RE = re.compile(r"</\|DSML\|parameter\s*>", re.I)
_DSML_CDATA_RE = re.compile(r"<!\[CDATA\[(.*?)\]\]>", re.S)
_DSML_ITEM_RE = re.compile(r"<item\b[^>]*>(.*?)</item>", re.I | re.S)
# 健壮性护栏：模型输出可被提示注入诱导出病态 DSML（几百层嵌套 → 递归爆栈；
# 大量未闭合开标签 → 正则二次放大 CPU）。超过任一阈值即放弃 DSML 解析，
# 回退 JSON/文本路径——正常的工具调用（哪怕携带几十 KB 文件内容）远够不着。
_DSML_MAX_DEPTH = 24
_DSML_MAX_OPENERS = 512
_DSML_MAX_TEXT = 1_000_000


def _dsml_openers(text: str) -> int:
    """统计 DSML 开标签出现次数（纯字符串计数，线性——不能用带 lazy
    尾段的 INVOKE 正则：未闭合输入会让计数器本身变成二次复杂度）。"""
    if not text:
        return 0
    low = text.lower()
    return (low.count("<|dsml|parameter")
            + low.count("<|dsml|invoke")
            + low.count("<|dsml|tool_calls"))


def _dsml_attr(attrs: str, key: str = "name") -> str:
    m = re.search(rf'{key}\s*=\s*"([^"]*)"', attrs or "", re.I)
    if m is None:
        m = re.search(rf"{key}\s*=\s*'([^']*)'", attrs or "", re.I)
    return (m.group(1) if m else "").strip()


def _dsml_cdata(s: str) -> str | None:
    """整段由 CDATA 构成的参数值：取出文本（含 ]]> 的转义拼接）。"""
    if not s.startswith("<![CDATA["):
        return None
    return "".join(_DSML_CDATA_RE.findall(s))


def _dsml_value(raw: str, depth: int = 0):
    """DSML 参数值 → Python 值：CDATA 字符串 / 嵌套对象 / <item> 数组 /
    JSON 字面量 / 原样文本。depth 防深度递归爆栈（上限 _DSML_MAX_DEPTH，
    超限按原样文本返回）。"""
    s = (raw or "").strip()
    if not s:
        return ""
    cdata = _dsml_cdata(s)
    if cdata is not None:
        return cdata
    if depth < _DSML_MAX_DEPTH and _DSML_PARAM_OPEN_RE.search(s):
        return _dsml_params(s, depth + 1)
    if depth < _DSML_MAX_DEPTH:
        items = _DSML_ITEM_RE.findall(s)
        if items and not _DSML_ITEM_RE.sub("", s).strip():
            return [_dsml_value(x, depth + 1) for x in items]
    try:
        return json.loads(s)
    except Exception:
        return s


def _dsml_param_end(text: str, start: int) -> tuple[int, int]:
    """找与 start 处 parameter 配对的结束位置（支持同名嵌套）。"""
    depth = 1
    pos = start
    while True:
        mo = _DSML_PARAM_OPEN_RE.search(text, pos)
        mc = _DSML_PARAM_CLOSE_RE.search(text, pos)
        if mc is None:
            return len(text), len(text)
        if mo is not None and mo.start() < mc.start():
            depth += 1
            pos = mo.end()
        else:
            depth -= 1
            pos = mc.end()
            if depth == 0:
                return mc.start(), mc.end()


def _dsml_params(text: str, depth: int = 0) -> dict:
    out: dict = {}
    pos = 0
    while True:
        m = _DSML_PARAM_OPEN_RE.search(text, pos)
        if m is None:
            return out
        name = _dsml_attr(m.group(1))
        end, pos = _dsml_param_end(text, m.end())
        if name:
            out[name] = _dsml_value(text[m.end() : end], depth + 1)


def _dsml_blocks(text: str) -> list:
    """取出 DSML 调用块；模型漏写根闭合标签时退化为「从 opener 到文末」。"""
    blocks = _DSML_BLOCK_RE.findall(text or "")
    if blocks:
        return blocks
    m = re.search(r"<\|DSML\|tool_calls\b", text or "", re.I)
    return [text[m.start() :]] if m else []


def parse_dsml_tool_calls(
    text: str, tools: list, *, limit: int = 1
) -> list:
    """把模型的 DSML 工具调用解析成标准 OpenAI tool_calls。

    只放行 tools 清单里声明过的名字；参数名走与 JSON 路径相同的
    schema 校正（`_fix_arg_names`）；一轮最多 limit 个调用。
    病态输入护栏（可被提示注入诱导）：嵌套超限按文本折叠（见
    _dsml_value），开标签数/文本长度超限或递归仍爆栈时整体放弃
    DSML 解析返回空——上游回复绝不因解析失败变成 502。
    """
    allowed = tool_names(tools)
    if not text or not allowed:
        return []
    if (len(text) > _DSML_MAX_TEXT
            or _dsml_openers(text) > _DSML_MAX_OPENERS):
        return []
    by_name = {}
    for t in tools or []:
        if isinstance(t, dict):
            fn = t.get("function") or t
            if fn.get("name"):
                by_name[fn["name"]] = t
    calls: list = []
    try:
        for block in _dsml_blocks(text):
            for m in _DSML_INVOKE_RE.finditer(block):
                name = _dsml_attr(m.group(1))
                if not name or name not in allowed:
                    continue
                args = _fix_arg_names(_dsml_params(m.group(2)), by_name.get(name))
                calls.append(
                    {
                        "id": f"call_{uuid.uuid4().hex[:16]}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args, ensure_ascii=False),
                        },
                    }
                )
                if len(calls) >= limit:
                    return calls
    except RecursionError:
        return []
    return calls


def render_dsml_assistant_tool_calls(calls: list, text: str = "") -> str:
    """把 assistant.tool_calls 还原成 DSML 调用块（压平历史用）。"""
    blocks = []
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
            args = {}
        if not isinstance(args, dict):
            args = {"value": args}
        lines = [
            DSML_TOOL_CALLS_OPEN,
            f'  <|DSML|invoke name="{name}">',
        ]
        for k, v in args.items():
            lines.append(f"    {_dsml_param(k, v)}")
        lines += ["  </|DSML|invoke>", "</|DSML|tool_calls>"]
        blocks.append("\n".join(lines))
    body = "\n".join(blocks)
    if text and body:
        return f"{text}\n{body}"
    return body or text


def _dsml_param(name: str, value) -> str:
    """渲染单个 DSML 参数（字符串走 CDATA，容器递归）。"""
    tag = f'<|DSML|parameter name="{name}">'
    if isinstance(value, str):
        safe = value.replace("]]>", "]]]]><![CDATA[>")
        return f"{tag}<![CDATA[{safe}]]></|DSML|parameter>"
    if isinstance(value, dict):
        inner = "".join(_dsml_param(k, v) for k, v in value.items())
        return f"{tag}{inner}</|DSML|parameter>"
    if isinstance(value, list):
        inner = "".join(
            f"<item>{_dsml_param_value(v)}</item>" for v in value
        )
        return f"{tag}{inner}</|DSML|parameter>"
    return f"{tag}{json.dumps(value, ensure_ascii=False)}</|DSML|parameter>"


def _dsml_param_value(value) -> str:
    if isinstance(value, str):
        safe = value.replace("]]>", "]]]]><![CDATA[>")
        return f"<![CDATA[{safe}]]>"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return json.dumps(value, ensure_ascii=False)


def render_dsml_tool_result(name: str, content: str, call_id: str = "") -> str:
    """工具执行结果渲染成 DSML 块（与调用格式同族，模型训练时见过）。"""
    safe = (content or "").replace("]]>", "]]]]><![CDATA[>")
    cid = call_id or "unknown"
    return (
        f'<|DSML|tool_result call_id="{cid}" name="{name}">'
        f"<content><![CDATA[{safe}]]></content></|DSML|tool_result>"
    )


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


def increment_text(
    messages: list, *, result_tag: str = "tool_result", dsml: bool = False
) -> str:
    """计算本轮应发给网页会话的增量文本。

    规则（与网页会话的增量协议对齐）：
    - 会话里还没有 assistant（首轮）：发最后一条 user；
    - 最后一条 assistant 之后有 role=tool 结果：把结果渲染成
      `<result_tag>` 块（`dsml=True` 时渲染成 DSML tool_result），
      尾随的新 user 消息一并附上——这是工具链多轮的关键增量；
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
                if dsml:
                    parts.append(
                        render_dsml_tool_result(
                            name, content, m.get("tool_call_id") or ""
                        )
                    )
                else:
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
    dsml: bool = False,
    limit: int = 1,
    require_marker: bool = False,
) -> list:
    """从模型输出解析工具调用，返回标准 OpenAI tool_calls 结构。

    - dsml:   DSML XML 块（Kimi/智谱原生格式，``<|DSML|tool_calls>``）
    - tag:    标签形式 ``<tag>{json}</tag>``（DOTALL，模型最常输出）
    - marker: 行内标记 ``<marker>{json}``（配平扫描，支持跨行 JSON）
    按 dsml → tag → marker 顺序尝试；都解析不出时再做**裸 JSON 兜底**
    （模型不带任何标记直接把调用 JSON 写进正文的情形，qwen 全系
    实测高发）。只放行 tools 里声明过的名字；一轮最多返回 limit 个
    调用（默认 1，网页模型并行调用不可靠）。
    """
    allowed = tool_names(tools)
    if not text or not allowed:
        return []

    if dsml:
        calls = parse_dsml_tool_calls(text, tools, limit=limit)
        if calls:
            return calls

    blobs: list = []
    if tag:
        blobs += re.findall(rf"<{tag}>(.*?)</{tag}>", text, re.S)
    if marker:
        blobs += _marker_blobs(text, marker)

    calls = _build_calls(blobs, allowed, limit, tools)
    if calls:
        return calls
    if require_marker:
        # 思考双通道扫描专用：只认完整标记块（DSML/标签/标记行），
        # 裸 JSON 一律不提取——思考里的「草稿调用」（模型自言自语的
        # 合法 JSON）绝不能变成真实执行（2026-10-30 审计 KIMI-3/QWEN-3）
        return []
    return _build_calls(_bare_blobs(text, allowed), allowed, limit, tools)


def strip_tool_markup(
    text: str,
    *,
    tag: str | None = None,
    marker: str | None = None,
    dsml: bool = False,
) -> str:
    """从文本里剥掉工具标记（标签块 / 标记 + 其 JSON / DSML 块）。

    marker 模式旧实现按行删，多行 JSON 的续行会残留在正文里；
    现在标记 + 配平出的整个 JSON 片段一起删。DSML 块整体删除；
    漏写根闭合标签时从 opener 截断，避免原始标记漏给用户。
    """
    if not text:
        return text or ""
    out = text
    if dsml:
        if len(out) > _DSML_MAX_TEXT or _dsml_openers(out) > _DSML_MAX_OPENERS:
            # 病态输入护栏：跳过 O(n²) 的块正则，直接从首个 opener 截断
            m = re.search(r"<\|DSML\|tool_calls\b", out, re.I)
            out = out[: m.start()] if m else out
        else:
            out = _DSML_BLOCK_RE.sub("", out)
            m = re.search(r"<\|DSML\|tool_calls\b", out, re.I)
            if m:
                out = out[: m.start()]
        out = out.strip()
    if tag:
        # 连同行尾换行一起移除，避免剥完留下一串空行
        out = re.sub(rf"<{tag}>.*?</{tag}>\n?", "", out, flags=re.S)
        # 漏写闭合标签的残块：从 opener 截断（与 dsml 分支同策略），
        # 否则 opener + 半截 JSON 会漏给客户端
        m = re.search(rf"<{tag}>", out)
        if m:
            out = out[: m.start()]
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
        # 兜底：标记还在（后面跟的不是可识别 JSON）时只删标记本身，
        # 不删整行——标记前若有正文，删行会把正文一起丢掉
        out = out.replace(marker, "")
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


def required_nudge(
    *, tag: str | None = None, marker: str | None = None, dsml: bool = False
) -> str:
    """tool_choice=required 重试时追加的加强指令（含平台各自的调用示例）。"""
    if dsml:
        example = (
            f"{DSML_TOOL_CALLS_OPEN}\n"
            '  <|DSML|invoke name="工具名">\n'
            '    <|DSML|parameter name="参数名"><![CDATA[参数值]]></|DSML|parameter>\n'
            "  </|DSML|invoke>\n"
            "</|DSML|tool_calls>"
        )
    elif tag:
        example = f'<{tag}>{{"name":"工具名","arguments":{{"参数名":值}}}}</{tag}>'
    else:
        example = (
            f"{marker or '<<<TOOL_CALL>>>'}"
            f'{{"name":"工具名","arguments":{{"参数名":值}}}}'
        )
    return REQUIRED_NUDGE.format(example=example)


class _ToolHoldBuffer:
    """扣留器公共实现（快照/增量两种口径的基类）。

    状态与推进逻辑只有一份：扫描 opener 首现位置，opener 前照常放行、
    opener 起全部扣住；未发现 opener 时回扣一个完整 opener 长度的尾巴，
    防止标记被拆在两次 delta 之间漏出去。
    """

    def __init__(
        self,
        openers: list,
        fragment_re: "re.Pattern | None" = None,
        open_re: "re.Pattern | None" = None,
    ) -> None:
        self._openers = [o for o in (openers or []) if o]
        self._full = ""
        self._emitted = 0
        self._open_at = -1
        # 降级标记形态（上游吃字符/换写法）：提供给 GLM 这类「模型手写标记」
        # 的链路——fragment_re 负责清洗放行段/收尾残迹；open_re（更严，必须
        # 还剩 <）在精确 opener 找不到时兜底认 opener。kimi/qwen 不传，行为不变。
        self._fragment_re = fragment_re
        self._open_re = open_re

    def _advance(self) -> str:
        """基于当前 _full 放行「可以安全下发的前缀」，推进 _emitted。"""
        if self._open_at < 0 and self._openers:
            for op in self._openers:
                i = self._full.find(op)
                if 0 <= i and (self._open_at < 0 or i < self._open_at):
                    self._open_at = i
            if self._open_at < 0:
                # 降级形态兜底：模型把 opener 写缺了角括号（如 <<TOOL_CALL>>），
                # 精确 find 认不出会连标记带 JSON 漏进正文；按宽松锚认 opener。
                pat = self._open_re or self._fragment_re
                if pat:
                    m = pat.search(self._full, self._emitted)
                    if m:
                        self._open_at = m.start()
        if self._open_at >= 0:
            out = self._full[self._emitted : self._open_at]
            if self._fragment_re:
                # 放行段里的标记残迹：完整降级形态正则剥，半截 opener
                # （断流续写重启标记时被扣住的 <<<TOOL_CA）按前缀判定剥
                out = self._fragment_re.sub("", out)
                cut = self._partial_prefix_len(out)
                if cut:
                    out = out[: len(out) - cut]
            self._emitted = self._open_at
            return out
        hold = max((len(op) for op in self._openers), default=1)
        if self._fragment_re:
            # 半截 opener 也扣住：续写可能把它接完，末尾才 flush 丢弃
            extra = self._partial_prefix_len(self._full[self._emitted :])
            if extra > hold:
                hold = extra
        safe_end = max(self._emitted, len(self._full) - hold)
        out = self._full[self._emitted : safe_end]
        self._emitted = safe_end
        return out

    def _partial_prefix_len(self, tail: str) -> int:
        """tail 的最长后缀中构成某 opener「括号弹性前缀」的长度。

        只认有标记气味的开头（< 或 |）：正文恰以 TOOL_CA 结尾这类
        无害情形不误吞。fragment_re 未启用时恒 0（kimi/qwen 行为不变）。
        """
        if not self._fragment_re:
            return 0
        limit = max((len(op) for op in self._openers), default=0)
        best = 0
        for op in self._openers:
            core = op.strip("<>")
            if not core:
                continue
            for cut in range(min(len(tail), limit), best, -1):
                s = tail[len(tail) - cut :]
                if not s.startswith(("<", "|")):
                    continue
                s2 = s.lstrip("<")
                if core.startswith(s2):
                    best = cut
                    break
        return best

    def flush(self) -> str:
        out = self._full[self._emitted :]
        self._emitted = len(self._full)
        if self._fragment_re:
            # 收尾仍扣着的半截 opener 必然是被掐断的残迹，丢弃
            cut = self._partial_prefix_len(out)
            if cut:
                out = out[: len(out) - cut]
        return out

    def flush_prefix(self) -> str:
        """calls 成立时取走「opener 之前被保守扣住的正文尾巴」。

        无 opener（native 调用场景）：返回全部未放行的正文并清空水位；
        有 opener：返回 _emitted 到 _open_at 之间未放行的部分（正常流式
        下为空串），水位推进到 opener——opener 之后的标记/JSON 区域
        不放行（已解析成 tool_calls）。
        """
        if self._open_at < 0:
            out = self._full[self._emitted :]
            self._emitted = len(self._full)
            return out
        out = self._full[self._emitted : self._open_at]
        self._emitted = self._open_at
        return out

    @property
    def opener_found(self) -> bool:
        return self._open_at >= 0

    @property
    def text(self) -> str:
        return self._full


# ── 降级标记形态识别（2026-10-30，对齐 deepseek DSML 零残留）──────────
# GLM/Kimi 的工具标记全靠 < 尖括号，上游会吃字符、流会被切碎、模型会写缺
# 角括号——精确匹配认不出时标记/碎片整段漏进正文（deepseek 2026-10-30 同
# 族报障的 GLM 版）。两个正则分工：
# - MARKER_FRAGMENT_RE（清洗用）：允许括号全缺，任何残迹都剥；
# - MARKER_OPEN_LOOSE_RE（锚点兜底）：必须还剩 <，防止正文里的
#   「TOOL_CALL>」字样被误当 opener 吞掉后续内容。
MARKER_FRAGMENT_RE = re.compile(
    r"(?:<{0,3}\s*TOOL_CALL\s*>{1,3}"            # <<<TOOL_CALL>>> 缺角括号形态
    r"|(?:</?)?\|?DSML\|?tool_calls\s*>{0,3}"    # <|DSML|tool_calls> 残缺/孤立闭合
    r")"
)
MARKER_OPEN_LOOSE_RE = re.compile(
    r"(?:<{1,3}\s*TOOL_CALL\s*>{1,3}|<\|?DSML\|?tool_calls\s*>)"
)


def scrub_marker_fragments(text: str) -> str:
    """剥掉降级/残缺/孤立的标记碎片，防止漏给客户端。"""
    if not text:
        return text or ""
    return MARKER_FRAGMENT_RE.sub("", text)


def strip_truncated_marker_tail(text: str, openers: list) -> str:
    """剥掉文本末尾「有标记气味的半截 opener」。

    上游掐断/模型写一半时，非流式终态文本可能停在 <<<TOOL_CA 这类
    半截上。只认以 < 或 | 开头、且剩余部分是某 opener 核心词前缀的
    尾巴（正文单独的 < / << 不动——deepseek「单个 | 不丢」同规则）。
    """
    if not text or not openers:
        return text or ""
    limit = max((len(op) for op in openers if op), default=0)
    for cut in range(min(len(text), limit), 1, -1):
        s = text[len(text) - cut :]
        if not s.startswith(("<", "|")):
            continue
        s2 = s.lstrip("<")
        if not s2:
            continue  # 纯尖括号尾巴可能是正文，不吞
        if any(op.strip("<>").startswith(s2) for op in openers if op):
            return text[: len(text) - cut]
    return text


def normalize_tool_marker(text: str, marker: str) -> str:
    """把「标记词对但角括号缺数」的写法归一到精确形态（供精确剥除/解析）。

    只动角括号数量（<<<TOOL_CALL>> → <<<TOOL_CALL>>>），不碰正文。
    """
    if not text or not marker:
        return text or ""
    core = marker.strip("<>")
    left = len(marker) - len(marker.lstrip("<"))
    right = len(marker) - len(marker.rstrip(">"))
    if not core or left == 0 or right == 0:
        return text or ""
    return re.sub(
        rf"<{{1,{left}}}\s*{re.escape(core)}\s*>{{1,{right}}}", marker, text
    )


class ToolStreamBuffer(_ToolHoldBuffer):
    """流式输出里扣住工具标记（**全文快照口径**）。

    用法：每个流事件把**全文快照**喂给 `feed()`，拿到的返回值是
    「可以安全下发给客户端的完整前缀」；结束时分两种情况：
    - `opener_found` 为真：不再吐 content，直接解析 tool_calls；
    - 为假：`flush()` 把扣住的尾巴吐干净，正常 finish。
    """

    def feed(self, text: str) -> str:
        self._full = text or ""
        return self._advance()


class DeltaToolBuffer(_ToolHoldBuffer):
    """工具标记扣留器（**增量口径**），与 ToolStreamBuffer 行为等价。

    差别只在喂法：`feed_delta(piece)` 接收**新增量**而非全文快照，内部
    自行累计。给「流循环先做无损前缀差、再扣留标记」的链路用——快照
    口径的 buf 会回扣尾部，输出非单调，嵌套在多层前缀差里会把下层
    错位咬字（GLM 实测）。
    """

    def feed_delta(self, piece: str) -> str:
        if piece:
            self._full += piece
        return self._advance()


# ── 续写接缝重叠剪除（2026-10-29 自 deepseek 上移共用）──────────
# 模型「继续」常把断点前最后一段逐字重说一遍。复读形态：从已发尾巴的
# 某处开始、逐字复读到尾巴末尾，再接新内容。
SEAM_WINDOW = 200


class SeamTrimmer:
    """续写接缝重叠剪除：增量开头只要「出现在尾巴里」就可能是复读的开头，
    继续扣住；分叉时提交最长后缀-前缀重叠。

    无损性：剪掉的任何字节都与客户端已收到内容逐字相同——剪除永不丢
    新内容。窗口 SEAM_WINDOW 之外不比较（保守上限）。`committed` 是已经
    放行的累计文本，供「全文快照」口径的调用方拼装单调快照（扣住期间
    快照不增长即不向客户端发新字）。
    """

    def __init__(self, tail: str) -> None:
        self._tail = (tail or "")[-SEAM_WINDOW:]
        self._lead = ""
        self._done = not self._tail
        self._committed = ""

    @property
    def committed(self) -> str:
        return self._committed

    def feed(self, piece: str) -> str:
        """喂入续写增量，返回可安全下发的部分（重叠段被剪掉）。"""
        if self._done:
            self._committed += piece or ""
            return piece
        self._lead += piece or ""
        tail, lead = self._tail, self._lead
        if len(lead) <= len(tail) and lead in tail:
            return ""  # 仍是「尾巴某处开始复读」的候选，继续扣
        j = 0
        for k in range(min(len(lead), len(tail)), 0, -1):
            if tail.endswith(lead[:k]):
                j = k
                break
        self._done = True
        out = lead[j:]
        self._committed += out
        return out

    def finish(self) -> str:
        """续写轮收尾：仍扣着的一段必然是已发尾巴内的逐字复读，丢弃。"""
        self._done = True
        return ""
