"""
OpenAI 兼容工具调用支持。

DeepSeek Web API 是纯文本流 —— 没有原生 function calling。
本模块通过「哨兵协议」+ DSML 在文本层模拟。

## 协议（两套并行，都兼容）

### 1. 哨兵格式（我们提示词教的）

    <|tool_calls|>
    {"name": "get_weather", "arguments": {"city": "Beijing"}}
    <|/tool_calls|>

### 2. DSML 格式（DeepSeek 原生，模型先验最强）

    <｜｜DSML｜｜ calls>
    <｜｜DSML｜｜ invoke name="get_weather">
    <｜｜DSML｜｜ parameter name="city" string="true">Beijing</｜｜DSML｜｜ parameter>
    </｜｜DSML｜｜ invoke>
    </｜｜DSML｜｜ calls>

## 解析容忍矩阵（本次修复）

| 维度        | 接受                                       |
|-------------|--------------------------------------------|
| 竖线        | `|` (U+007C)、`｜` (U+FF5C)，单/双/混合均可 |
| DSML 大小写 | `DSML` / `dsml` / `Dsml` …                 |
| 标签名大小写 | `invoke` / `INVOKE` / `Invoke` …           |
| 标签内空格  | `<｜｜DSML｜｜ invoke>` / `<｜｜DSML｜｜invoke>` 均可 |
| 属性值引号  | `name="x"` / `name='x'` / `name=x`         |
| 属性间隔    | `name = "x"` / `name="x"`                  |
| 属性值内 `>` | `command="a > b"` 安全剥离                  |
| 缺 `</invoke>` | 在下一个 `<invoke` 处隐式切分           |
| 缺 `</calls>`  | `finalize` 兜底解析                    |
| 孤立关闭标签  | SCAN 状态静默跳过，不吐到 content         |

## 关键设计：Artifacts 零容忍

客户端会把流里的 `content` 和 `tool_calls` 都渲染给用户。
**任何 DSML 相关的文本碎片出现在 content 里都是视觉污染**。

原则：

  1. 进入 DSML 状态后，只有两种出路：解析出 tool_calls 或静默丢弃
  2. SCAN 状态遇到孤立的 DSML 闭合标签，跳过不吐
  3. 空 DSML 块、残缺 DSML 块，静默丢弃
  4. `finalize` 的 DSML 残余也优先尝试解析，失败则丢弃

## 工具名匹配策略

litellm / Claude Code 侧传下来的工具名可能是 `Bash`（首字母大写）
或 `bash_command`（带下划线），而模型输出 `name="bash"`。
字符串不完全相等 ≠ 幻觉 → 宽容匹配 + debug 日志，绝不硬拒收。

## 解析优先级

    _try_dsml > _try_sentinel > _try_code_blocks > _try_embedded_json > _try_bare_json
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Optional

logger = logging.getLogger("deepseek2api.tool_calling")

# ═══════════════════════════════════════════════════════════════
# 解析行为开关（由 DeepSeekWebClient 初始化时按 AppConfig 设置一次）
# ═══════════════════════════════════════════════════════════════

_TOOL_PARSE_CONFIG = {
    # 工具名严格匹配：模型输出的工具名不在 tools 声明中时直接丢弃并告警。
    # 关闭则恢复"宽容放行"（未知工具名也会透给客户端）。
    "strict_names": True,
    # 是否启用"代码块 / 内嵌 JSON / 裸 JSON 碰运气"式解析。
    # 关闭后只认 DSML / 哨兵等明确标记，解析结果更可预期，
    # 也避免把回答里的普通 JSON/代码示例误判成工具调用。
    "json_fallback": False,
}


def configure_tool_parsing(
    *, strict_names: bool = True, json_fallback: bool = False
) -> None:
    """设置工具调用解析行为。应在服务启动、创建 client 时调用一次。"""
    _TOOL_PARSE_CONFIG["strict_names"] = bool(strict_names)
    _TOOL_PARSE_CONFIG["json_fallback"] = bool(json_fallback)


# ═══════════════════════════════════════════════════════════════
# 哨兵常量
# ═══════════════════════════════════════════════════════════════

TOOL_OPEN_SINGULAR = "<|tool_call|>"
TOOL_OPEN_PLURAL = "<|tool_calls|>"
TOOL_CLOSE_SINGULAR = "<|/tool_call|>"
TOOL_CLOSE_PLURAL = "<|/tool_calls|>"

_OPEN_VARIANTS = (TOOL_OPEN_PLURAL, TOOL_OPEN_SINGULAR)
_CLOSE_VARIANTS = (TOOL_CLOSE_PLURAL, TOOL_CLOSE_SINGULAR)

# ═══════════════════════════════════════════════════════════════
# DSML 正则（★ 容忍所有常见变体 ★）
# ═══════════════════════════════════════════════════════════════

# 竖线：全角 U+FF5C 或半角 U+007C，允许混合、允许 1~2 个
_BAR = "[|\uff5c]"

# DSML 标记：竖线 + DSML + 竖线，中间允许空格，大小写不敏感
_DSML_MARKER = rf"{_BAR}+\s*DSML\s*{_BAR}+"

# 标签属性：双引号 / 单引号 / 裸值 均可，值里可含 >（在引号内）
_ATTRS = r'(?:"[^"]*"|\'[^\']*\'|[^>"\'])*'

# 标签：< / </ + 标记 + 标签名 + 属性 + >
_CALLS_OPEN_RE = re.compile(
    rf"<\s*{_DSML_MARKER}\s*calls\b({_ATTRS})>",
    re.IGNORECASE,
)
_CALLS_CLOSE_RE = re.compile(
    rf"</\s*{_DSML_MARKER}\s*calls\s*>",
    re.IGNORECASE,
)
_INVOKE_OPEN_RE = re.compile(
    rf"<\s*{_DSML_MARKER}\s*invoke\b({_ATTRS})>",
    re.IGNORECASE,
)
_INVOKE_CLOSE_RE = re.compile(
    rf"</\s*{_DSML_MARKER}\s*invoke\s*>",
    re.IGNORECASE,
)
_PARAM_OPEN_RE = re.compile(
    rf"<\s*{_DSML_MARKER}\s*parameter\b({_ATTRS})>",
    re.IGNORECASE,
)
_PARAM_CLOSE_RE = re.compile(
    rf"</\s*{_DSML_MARKER}\s*parameter\s*>",
    re.IGNORECASE,
)

# 属性名值对：name="x" / name='x' / name=x
_ATTR_PAIR_RE = re.compile(
    r'([a-zA-Z_][a-zA-Z0-9_-]*)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\s>]+))'
)

# 任意 DSML 起始标签（SCAN 状态检测用）
# ★ `(?:</?)?` 是给「降级形态」留的后门 ★
# DeepSeek 网页端会把标记前的 `<` 吃掉（实测上游直接下发 `｜｜DSML｜｜ calls>`，
# 客户端渲染成 `| | DSML | | calls>`）；断流续写轮也只能看到后半截。
# 不认这种形态 → 残片当普通文本吐进正文。
_ANY_DSML_OPEN_RE = re.compile(
    rf"(?:</?)?\s*{_DSML_MARKER}\s*(?:calls|invoke)\b",
    re.IGNORECASE,
)

# 孤立关闭标签（SCAN 状态剔除用）
_NOISE_RE = re.compile(
    rf"</\s*{_DSML_MARKER}\s*(?:calls|invoke|parameter)\s*>",
    re.IGNORECASE,
)

# 降级碎片兜底：`｜｜DSML｜｜ calls>`、`</｜｜DSML｜｜ calls>` 等
# 「标记 + calls 标签名 + 属性 + >」的残片（无论有没有 `<`）。
# 只扫 calls 层：invoke/parameter 开标签保留（残缺块「保守不剥」是既有契约，
# 完整块在剥块循环里已经处理）。
_ORPHAN_TAG_RE = re.compile(
    rf"(?:</?)?\s*{_DSML_MARKER}\s*(?:calls|tool_calls)\b[^<>\n]{{0,300}}>",
    re.IGNORECASE,
)

# 降级开标签：`</｜｜DSML｜｜ invoke name="read">` —— 开标签位置写成了 `</`。
# 只在「带 name= 属性」时纠正，真正的闭合标签 `</...invoke>` 不受影响。
_MALFORMED_OPEN_RE = re.compile(
    rf"</(\s*{_DSML_MARKER}\s*(?:calls|invoke|parameter)\b[^<>]*\bname\s*=)",
    re.IGNORECASE,
)

# 尾部可能是标记前缀的形态：`<` / `</` / `｜` / `｜｜DS` / `<｜｜DSM` / `｜｜DSML｜｜ cal` …
# 上游按 token 切片，标记会碎成多块；不扣住就会先吐半截、整体再也认不出。
# 注意：竖线之间只允许空格/制表符、且用 \Z 收尾（`$` 会放过行尾换行，
# 把普通表格行的 `|\n` 误扣成标记前缀，正文尾巴就被吞了）。
_TAG_WORDS = ("calls", "invoke", "tool_calls", "parameter")
_TAG_PREFIXES = sorted(
    {w[:k] for w in _TAG_WORDS for k in range(1, len(w) + 1)},
    key=len,
    reverse=True,
)
_MARKER_PARTIAL_RE = re.compile(
    r"^(?:"
    r"</?"
    r"|(?:</?)?[ \t]*(?:[|\uff5c][ \t]*)+"
    r"(?:D|DS|DSM|DSML)?[ \t]*(?:[|\uff5c][ \t]*)*"
    rf"(?:{'|'.join(_TAG_PREFIXES)})?"
    r")\Z",
    re.IGNORECASE,
)
_MARKER_PARTIAL_MAX = 24


def _fix_malformed_openers(text: str) -> str:
    """把 `</…invoke name=` 这类降级开标签纠回 `<…invoke name=`。"""
    if not text or "</" not in text:
        return text
    return _MALFORMED_OPEN_RE.sub(r"<\1", text)


# 任意 DSML 标签起点（不含标签名限制）：用来在「参数缺闭合」时截断值。
_ANY_DSML_TAG_RE = re.compile(rf"</?\\?\s*{_DSML_MARKER}", re.IGNORECASE)

# 参数值里的残片：`D:\study<｜｜DSML｜｜>` / `D:\study<\｜｜DSML｜｜>` → 整段删。
_TAG_RESIDUE_RE = re.compile(
    rf"(?:</?)?\\?\s*{_DSML_MARKER}[^<>\n]{{0,300}}>",
    re.IGNORECASE,
)


def _strip_marker_residue(value: str) -> str:
    """从参数值里清掉 DSML 残片。

    上游漏写 `</parameter>` 时，后一个标签会被吃进值里，实测把
    `workdir` 污染成 `D:\\study<｜｜DSML｜｜>`（agent 报
    "Working directory does not exist"）。这里做最后一道闸。
    """
    if not value or "DSML" not in value.upper():
        return value
    out = _TAG_RESIDUE_RE.sub("", value)
    if out == value:
        return value
    # 去掉残片留下的裸尖括号尾巴（`D:\study<\…` → `D:\study<` → `D:\study`），
    # 以及块内换行带来的尾随空白
    return re.sub(r"<\\?\s*$", "", out).rstrip()


def _marker_partial_start(buf: str) -> int:
    """返回 buf 末尾「可能正在拼 DSML 标记」的起点；没有则 -1。"""
    n = len(buf)
    start = max(0, n - _MARKER_PARTIAL_MAX)
    for i in range(start, n):
        seg = buf[i:]
        if "<" not in seg and "|" not in seg and "\uff5c" not in seg:
            continue
        if _MARKER_PARTIAL_RE.match(seg):
            return i
    return -1


def _looks_like_marker_prefix(seg: str) -> bool:
    """收尾判断：这段碎片「像不像」DSML 标记的开头。

    只扣住真正有 DSML 气味（尖括号 / DSML 字样 / ≥2 根竖线）的尾巴；
    单个 `|` 可能是 Markdown 表格行的最后一个字符，误丢就是吞正文。
    """
    return (
        "<" in seg or "DSML" in seg.upper() or seg.count("|") + seg.count("\uff5c") >= 2
    )


# 提示词里仍用全角，因为这是 DeepSeek 原生先验
_DSML = "\uff5c\uff5cDSML\uff5c\uff5c"

# ═══════════════════════════════════════════════════════════════
# 长度常量
# ═══════════════════════════════════════════════════════════════

_MAX_OPEN_LEN = max(len(v) for v in _OPEN_VARIANTS)
_MAX_CLOSE_LEN = max(len(v) for v in _CLOSE_VARIANTS)

_ANCHORS = (
    '{"name"',
    '{"tool_calls"',
    '{"function"',
    '{ "name"',
    '{ "tool_calls"',
    '{ "function"',
)
_MAX_ANCHOR_LEN = max(len(a) for a in _ANCHORS)

# DSML 标签最长前缀的保守上界
_MAX_DSML_TAG_LEN = 64

_SAFE_KEEP = max(_MAX_OPEN_LEN, _MAX_ANCHOR_LEN, _MAX_DSML_TAG_LEN)

# ═══════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════


@dataclass(slots=True)
class ParsedToolCall:
    id: str
    name: str
    arguments: str

    def to_openai(self) -> dict:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass(slots=True)
class ToolParseResult:
    calls: list[ParsedToolCall] = field(default_factory=list)
    clean_text: str = ""


@dataclass(slots=True)
class StreamAction:
    kind: str
    text: str = ""
    tool_calls: Optional[list[ParsedToolCall]] = None


# ═══════════════════════════════════════════════════════════════
# 残留剥离（非流式 Artifacts 零容忍）
# ═══════════════════════════════════════════════════════════════


def strip_dsml_residue(text: str) -> str:
    """把「解析不出调用」的完整 DSML/哨兵块从正文剥掉。

    流式检测器天然扣住标记不漏；非流式 parse_tool_calls 解析失败时
    clean_text 会原样带回标记原文，客户端就会看到一坨 DSML。按本模块
    「Artifacts 零容忍」原则，**完整**块静默删除（残缺尾部不剥——保守，
    交给续写前的 _drop_trailing_partial_markup 处理）。剥完为空即代表
    「客户端可见输出为空」，供空响应重试判定使用。
    """
    if not text:
        return text
    out = _fix_malformed_openers(text)

    # 完整 DSML calls 块（含缺 invoke 的空壳块）
    while True:
        m_open = _CALLS_OPEN_RE.search(out)
        if not m_open:
            break
        m_close = _CALLS_CLOSE_RE.search(out, m_open.end())
        if not m_close:
            break
        out = out[: m_open.start()] + out[m_close.end() :]

    # 孤立 DSML invoke 块（缺 calls 包裹但 invoke 完整）
    while True:
        m_open = _INVOKE_OPEN_RE.search(out)
        if not m_open:
            break
        m_close = _INVOKE_CLOSE_RE.search(out, m_open.end())
        if not m_close:
            break
        out = out[: m_open.start()] + out[m_close.end() :]

    # 哨兵块
    for open_v in _OPEN_VARIANTS:
        while True:
            i = out.find(open_v)
            if i < 0:
                break
            closed = None
            for cv in _CLOSE_VARIANTS:
                j = out.find(cv, i + len(open_v))
                if j >= 0:
                    closed = (cv, j)
                    break
            if closed is None:
                break
            out = out[:i] + out[closed[1] + len(closed[0]) :]

    # 降级碎片兜底：`｜｜DSML｜｜ calls>` / `</｜｜DSML｜｜ calls>` 等
    # 「标记 + calls + >」的残片（含没有 `<` 的形态）一律剥掉——
    # 用户实测看到的就是这行 `| | DSML | | calls>`。
    out = _ORPHAN_TAG_RE.sub("", out)
    # 孤立闭合标签（invoke / parameter / calls）同样没有正文价值
    out = _NOISE_RE.sub("", out)
    return out


# ═══════════════════════════════════════════════════════════════
# tool_choice=required 强校验重试指令
# ═══════════════════════════════════════════════════════════════


def build_required_nudge(tool_name: str = "") -> str:
    """required 强校验重试时追加给模型的加强指令（全角 DSML 示例）。

    与 GLM 的 tool_bridge.required_nudge 同语义；示例必须用本平台的
    全角竖线 DSML 形态，模型才认得。tool_name 非空时点名必须调谁。
    """
    D = _DSML
    example = (
        f"<{D} calls>\n"
        f'<{D} invoke name="工具名">\n'
        f'<{D} parameter name="参数名" string="true">参数值</{D} parameter>\n'
        f"</{D} invoke>\n"
        f"</{D} calls>"
    )
    demand = f"这次必须调用 `{tool_name}`。" if tool_name else "这次必须调用一个工具。"
    return (
        "上一次回复没有调用任何工具，这不符合要求。"
        + demand
        + "不要用文字回答，不要解释，只输出一次工具调用，"
        "它必须是回复的全部内容，格式严格为：\n" + example
    )


# ═══════════════════════════════════════════════════════════════
# 属性解析
# ═══════════════════════════════════════════════════════════════


def _parse_attrs(s: str) -> dict[str, str]:
    """容忍 name="x" / name='x' / name=x 三种写法。"""
    result: dict[str, str] = {}
    if not s:
        return result
    for m in _ATTR_PAIR_RE.finditer(s):
        key = m.group(1)
        val = m.group(2)
        if val is None:
            val = m.group(3)
        if val is None:
            val = m.group(4)
        result[key] = val
    return result


def _decode_dsml_value(raw: str, is_string: bool | None):
    """is_string 三态：True=强制字符串；False=强制 JSON 解析；None=省略
    （协议教模型数字/对象/数组省略本属性）→ 先试 JSON、失败回退字符串。"""
    if is_string is True:
        return _strip_marker_residue(raw)
    stripped = raw.strip()
    if not stripped:
        return ""
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        if is_string is False:
            return raw
        return _strip_marker_residue(raw)


# ═══════════════════════════════════════════════════════════════
# 提示词构建（保持全角竖线，与 DeepSeek 原生先验一致）
# ═══════════════════════════════════════════════════════════════


def _tool_name(tool: dict) -> str:
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    return str(fn.get("name", "")).strip()


def _build_tool_prompt_impl(tools, tool_choice="auto"):
    if not tools:
        return ""

    D = _DSML

    lines = [
        "",
        "==== TOOL CALLING PROTOCOL ====",
        "",
        "!!! CRITICAL FORMAT RULE !!!",
        "When you need a tool, the FIRST characters of your response",
        f"MUST be the literal token `<{D} calls>`.",
        "If you write any sentence before it, the call is INVALID and",
        "the system WILL NOT execute the tool.",
        "",
        "WRONG (the tool will NOT run):",
        "  Sure, I'll check the weather for you.",
        '  {"name": "get_weather", "arguments": {"city": "Beijing"}}',
        "",
        "RIGHT:",
        f"  <{D} calls>",
        f'  <{D} invoke name="get_weather">',
        f'  <{D} parameter name="city" string="true">Beijing</{D} parameter>',
        f"  </{D} invoke>",
        f"  </{D} calls>",
        "",
        "Multi-tool example (multiple invokes INSIDE ONE calls block):",
        f"  <{D} calls>",
        f'  <{D} invoke name="read">',
        f'  <{D} parameter name="file_path" string="true">/etc/hosts</{D} parameter>',
        f"  </{D} invoke>",
        f'  <{D} invoke name="bash">',
        f'  <{D} parameter name="command" string="true">ls -la</{D} parameter>',
        f"  </{D} invoke>",
        f"  </{D} calls>",
        "",
        "Rules:",
        f"  - No text, no markdown, no code fences outside the <{D} calls> block.",
        f'  - ONE <{D} calls> block per response; wrap EACH tool call in its own <{D} invoke name="...">.',
        f'  - Every argument uses <{D} parameter name="..." string="true|false">value</{D} parameter>.',
        '  - Use string="true" for text values; string="false" (or omit) for numbers/objects/arrays.',
        "  - Do NOT emit <|tool_calls|> or <|tool_call|> sentinel tokens.",
        f"  - Do NOT emit a trailing <{D} calls> block with no invokes.",
        "  - A tool call MISSING any required argument is INVALID and will be DISCARDED.",
        "    Write every required parameter BEFORE closing the invoke; never emit an empty",
        "    invoke and try to fill it in later — you get one shot per response.",
        "  - Do NOT write tool calls inside your thinking. Drafting invokes in the",
        f"    reasoning is wasted: only the <{D} calls> block in your visible reply",
        "    is executed, and a half-written draft gets discarded entirely.",
        "  - When the user asks you to create/write a file, you MUST call the",
        "    file-writing tool. Do NOT paste the file content into your reply",
        "    and consider the task done.",
        "  - For edit tools with an old-text argument (old_string etc.): copy the",
        "    old text VERBATIM from the most recent read of that file. Never",
        "    reconstruct it from memory; if unsure or the previous edit failed,",
        "    read the file section again first.",
        "  - Use EXACT tool names: "
        + ", ".join(_tool_name(t) for t in tools if isinstance(t, dict)),
        "",
        "## Tools",
        "",
    ]

    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = str(fn.get("name", "")).strip()
        if not name:
            continue
        desc = str(fn.get("description", "")).strip()
        params = fn.get("parameters") or {}
        props = params.get("properties") if isinstance(params, dict) else None
        required = (
            set(params.get("required", [])) if isinstance(params, dict) else set()
        )

        lines.append(f"### {name}")
        if desc:
            lines.append(desc)
        if props:
            lines.append("Parameters:")
            for pname, pinfo in props.items():
                pinfo = pinfo if isinstance(pinfo, dict) else {}
                ptype = pinfo.get("type", "string")
                pdesc = pinfo.get("description", "")
                req = " (required)" if pname in required else ""
                lines.append(f"  - {pname} ({ptype}{req}): {pdesc}")
        lines.append("")

    if tool_choice == "required":
        lines.append(">> You MUST call at least one tool in every response.")
    elif isinstance(tool_choice, dict):
        fn_name = (tool_choice.get("function") or {}).get("name", "")
        if fn_name:
            lines.append(f">> You MUST call the tool `{fn_name}`.")

    return "\n".join(lines)


@lru_cache(maxsize=128)
def _build_tool_prompt_cached(tools_json: str, choice_json: str) -> str:
    """按 tools 内容缓存拼好的协议提示词：同一套 tools 只拼一次。

    agent 场景下每轮请求都带同一套 tools，不缓存就是每轮重复几千 token
    的字符串拼接。
    """
    try:
        tools = json.loads(tools_json)
    except json.JSONDecodeError:
        tools = []
    try:
        choice = json.loads(choice_json)
    except json.JSONDecodeError:
        choice = "auto"
    return _build_tool_prompt_impl(tools, choice)


def build_tool_prompt(tools, tool_choice="auto"):
    """构建工具调用协议提示词（带缓存）。"""
    if not tools:
        return ""
    try:
        tools_json = json.dumps(tools, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return _build_tool_prompt_impl(tools, tool_choice)
    try:
        choice_json = json.dumps(
            tool_choice, sort_keys=True, ensure_ascii=False, default=str
        )
    except (TypeError, ValueError):
        choice_json = '"auto"'
    return _build_tool_prompt_cached(tools_json, choice_json)


# ═══════════════════════════════════════════════════════════════
# 工具名匹配
# ═══════════════════════════════════════════════════════════════


def _norm_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def _match_tool_name(name: str, known: set[str]) -> bool:
    if not known:
        return True
    if name in known:
        return True
    n = _norm_name(name)
    return any(_norm_name(k) == n for k in known)


def _normalize_call(data: Any, tools: Optional[list[dict]]) -> Optional[ParsedToolCall]:
    """宽容策略：白名单只做软匹配，不匹配也放行（打 debug 日志）。"""
    if not isinstance(data, dict):
        return None

    name = data.get("name")
    if not isinstance(name, str) or not name:
        fn = data.get("function")
        if isinstance(fn, dict):
            name = fn.get("name")
    if not isinstance(name, str) or not name:
        return None

    if tools is not None:
        known = {_tool_name(t) for t in tools if isinstance(t, dict)}
        known.discard("")
        if known and not _match_tool_name(name, known):
            if _TOOL_PARSE_CONFIG["strict_names"]:
                # 严格模式：未知工具名直接丢弃。宽容放行会把模型的幻觉
                # 透给 agent，agent 再报"未知工具"，链路更难排查。
                logger.warning(
                    "tool name %r 不在声明的工具列表 %r 中，已丢弃（严格模式）",
                    name,
                    sorted(known),
                )
                return None
            logger.debug(
                "tool name %r not in known set %r — accepted anyway (lenient match)",
                name,
                sorted(known),
            )

    args = data.get("arguments")
    if args is None:
        fn = data.get("function")
        if isinstance(fn, dict):
            args = fn.get("arguments")
    # 参数值里的 DSML 残片最后清一遍（防止污染路径/命令）
    args = _scrub_args(args)
    args_str = _coerce_arguments(args)

    # ★ 必填参数校验：模型常先吐 invoke 骨架、参数还没写完就被截断，
    # 结果是一个 arguments={} 的空调用。这种调用交给 agent 只会换来
    # 「Invalid arguments for tool: Missing key」再重来一轮，不如当场丢弃，
    # 让上层把这轮当普通文本处理（模型下一轮会重新组织完整调用）。
    if tools is not None:
        missing = _missing_required(name, args_str, tools)
        if missing:
            logger.warning(
                "工具 %r 缺必填参数 %s，丢弃这次工具调用（args=%s）",
                name,
                missing,
                args_str[:200],
            )
            return None

    return ParsedToolCall(
        id=str(data.get("id") or f"call_{uuid.uuid4().hex[:24]}"),
        name=name,
        arguments=args_str,
    )


def _missing_required(name: str, args_str: str, tools: list[dict]) -> list[str]:
    """返回缺失的必填参数名（空列表表示参数齐全）。

    宽容原则：找不到该工具定义、或定义里没写 required，一律放行，
    避免把本来能跑通的调用误杀。
    """
    # 用与 _normalize_call 放行相同的模糊匹配（lower + -/_ 归一）找定义：
    # 此前精确比对，模型输出 `Bash`、声明 `bash` 时放行了调用却找不到定义
    # → 必填校验被整体绕过，空 arguments={} 透传（Missing key {} 回归口子，
    # 2026-10-30 审计 DS-2）
    tool = None
    for t in tools:
        if isinstance(t, dict) and _match_tool_name(name, {_tool_name(t)}):
            tool = t
            break
    if tool is None:
        return []
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    params = fn.get("parameters") or {}
    if not isinstance(params, dict):
        return []
    required = [r for r in (params.get("required") or []) if isinstance(r, str)]
    if not required:
        return []
    try:
        parsed = json.loads(args_str or "{}")
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, dict):
        return list(required)
    missing = []
    for key in required:
        if key not in parsed:
            missing.append(key)
            continue
        val = parsed[key]
        # 空字符串/空列表/空字典对必填项同样无效（模型写了标签没写值）
        if isinstance(val, str) and not val.strip():
            missing.append(key)
        elif isinstance(val, (list, dict)) and not val:
            missing.append(key)
    return missing


def _scrub_args(value: Any) -> Any:
    """递归清理参数值里的 DSML 残片（字符串才动，数字/布尔原样）。"""
    if isinstance(value, str):
        return _strip_marker_residue(value)
    if isinstance(value, dict):
        return {k: _scrub_args(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub_args(v) for v in value]
    return value


def _coerce_arguments(args: Any) -> str:
    if args is None:
        return "{}"
    if isinstance(args, dict):
        return json.dumps(args, ensure_ascii=False)
    if isinstance(args, list):
        return json.dumps(args, ensure_ascii=False)
    if isinstance(args, str):
        s = args.strip()
        if not s:
            return "{}"
        try:
            json.loads(s)
            return s
        except json.JSONDecodeError:
            return json.dumps({"value": args}, ensure_ascii=False)
    try:
        return json.dumps(args, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"


# ═══════════════════════════════════════════════════════════════
# DSML body 解析（★ 核心：缺闭合标签也能救 ★）
# ═══════════════════════════════════════════════════════════════


def _parse_dsml_body(
    body: str, tools: Optional[list[dict]] = None
) -> list[ParsedToolCall]:
    """
    解析 DSML 块内部的若干 invoke。容忍：

      - 缺 </invoke>：在下一个 <invoke 处隐式切分；都没有则吃到末尾
      - 缺 </parameter>：吃到 invoke 末尾
      - 属性无引号：name=foo
    """
    calls: list[ParsedToolCall] = []
    if not body:
        return calls

    pos = 0
    while True:
        m = _INVOKE_OPEN_RE.search(body, pos)
        if not m:
            break

        attrs = _parse_attrs(m.group(1))
        name = attrs.get("name", "")

        # 定位 invoke 结束
        close_m = _INVOKE_CLOSE_RE.search(body, m.end())
        if close_m:
            invoke_inner = body[m.end() : close_m.start()]
            pos = close_m.end()
        else:
            # 缺 </invoke>：下一个 <invoke 隐式切分
            next_inv = _INVOKE_OPEN_RE.search(body, m.end())
            if next_inv:
                invoke_inner = body[m.end() : next_inv.start()]
                pos = next_inv.start()
            else:
                invoke_inner = body[m.end() :]
                pos = len(body)

        arguments: dict = {}
        ppos = 0
        while True:
            pm = _PARAM_OPEN_RE.search(invoke_inner, ppos)
            if not pm:
                break

            p_attrs = _parse_attrs(pm.group(1))
            p_name = p_attrs.get("name", "")
            # string 属性省略 ≠ 字符串：协议提示词教的是「数字/对象/数组
            # 省略本属性」（tool_calling.py:496），此前把省略当 true 导致
            # <parameter name="limit">42</parameter> 解析成 "42"（审计 DS-P2）。
            # None=省略：由 _decode_dsml_value 先试 JSON 再回退字符串
            raw_attr = str(p_attrs.get("string", "")).strip().lower()
            if raw_attr == "":
                p_is_string = None
            else:
                p_is_string = raw_attr in ("true", "1", "yes")

            p_close = _PARAM_CLOSE_RE.search(invoke_inner, pm.end())
            if p_close:
                raw_value = invoke_inner[pm.end() : p_close.start()]
                ppos = p_close.end()
            else:
                # 缺 </parameter>：吃到下一条 DSML 标签为止（原来是吃到
                # invoke 末尾——上游漏闭合时会把后续标签吃进参数值，
                # 实测 workdir 变成 `D:\study<｜｜DSML｜｜>`）
                nxt = _ANY_DSML_TAG_RE.search(invoke_inner, pm.end())
                end = nxt.start() if nxt else len(invoke_inner)
                raw_value = invoke_inner[pm.end() : end]
                # 标签前遗留的裸 `<` / `<\` 不属于参数值
                raw_value = re.sub(r"<\\?\s*$", "", raw_value)
                ppos = end

            if p_name:
                arguments[p_name] = _decode_dsml_value(raw_value, p_is_string)

        call = _normalize_call({"name": name, "arguments": arguments}, tools)
        if call:
            calls.append(call)

    return calls


# ═══════════════════════════════════════════════════════════════
# 非流式解析
# ═══════════════════════════════════════════════════════════════


def _try_dsml(text: str, tools: Optional[list[dict]]) -> ToolParseResult:
    """定位并解析 DSML 块（三种形态兼容）。"""
    if not text:
        return ToolParseResult()

    # 降级开标签（`</…invoke name=`）先纠正，否则整个块解析不出调用
    text = _fix_malformed_openers(text)

    # ── 形态 ①：完整 <calls> ... </calls> ─────────────────
    m_open = _CALLS_OPEN_RE.search(text)
    if m_open:
        m_close = _CALLS_CLOSE_RE.search(text, m_open.end())
        if m_close:
            body = text[m_open.end() : m_close.start()]
            calls = _parse_dsml_body(body, tools)
            if calls:
                clean = (text[: m_open.start()] + text[m_close.end() :]).strip()
                return ToolParseResult(calls=calls, clean_text=clean)

        # 有 calls 开头但无 calls 闭合 → 用最后一个 invoke 收尾
        body = text[m_open.end() :]
        calls = _parse_dsml_body(body, tools)
        if calls:
            last_close = None
            for m in _INVOKE_CLOSE_RE.finditer(text, m_open.end()):
                last_close = m
            if last_close:
                clean = (text[: m_open.start()] + text[last_close.end() :]).strip()
            else:
                clean = text[: m_open.start()].strip()
            return ToolParseResult(calls=calls, clean_text=clean)

    # ── 形态 ②：缺头但有 calls 闭合 ──────────────────────
    m_close = _CALLS_CLOSE_RE.search(text)
    if m_close:
        m_inv = _INVOKE_OPEN_RE.search(text, 0, m_close.start())
        if m_inv:
            body = text[m_inv.start() : m_close.start()]
            calls = _parse_dsml_body(body, tools)
            if calls:
                clean = (text[: m_inv.start()] + text[m_close.end() :]).strip()
                return ToolParseResult(calls=calls, clean_text=clean)

    # ── 形态 ③：只有 invoke ──────────────────────────────
    m_inv = _INVOKE_OPEN_RE.search(text)
    if m_inv:
        body = text[m_inv.start() :]
        calls = _parse_dsml_body(body, tools)
        if calls:
            last_close = None
            for m in _INVOKE_CLOSE_RE.finditer(text, m_inv.start()):
                last_close = m
            if last_close:
                clean = (text[: m_inv.start()] + text[last_close.end() :]).strip()
            else:
                clean = text[: m_inv.start()].strip()
            return ToolParseResult(calls=calls, clean_text=clean)

    return ToolParseResult()


def _try_sentinel(text: str, tools: Optional[list[dict]]) -> ToolParseResult:
    for open_v in _OPEN_VARIANTS:
        open_idx = text.find(open_v)
        if open_idx == -1:
            continue
        close_idx = -1
        close_v = ""
        for cv in _CLOSE_VARIANTS:
            i = text.find(cv, open_idx + len(open_v))
            if i != -1:
                close_idx = i
                close_v = cv
                break
        if close_idx == -1:
            continue
        body = text[open_idx + len(open_v) : close_idx]
        calls = parse_sentinel_body(body, tools)
        if calls:
            clean = (text[:open_idx] + text[close_idx + len(close_v) :]).strip()
            return ToolParseResult(calls=calls, clean_text=clean)
    return ToolParseResult()


def parse_sentinel_body(
    body: str, tools: Optional[list[dict]] = None
) -> list[ParsedToolCall]:
    calls: list[ParsedToolCall] = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        call = _normalize_call(data, tools)
        if call:
            calls.append(call)
    return calls


def _try_code_blocks(text: str, tools: Optional[list[dict]]) -> ToolParseResult:
    pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
    for m in pattern.finditer(text):
        block = m.group(1).strip()
        calls = _parse_json_calls(block, tools)
        if calls:
            clean = (text[: m.start()] + text[m.end() :]).strip()
            return ToolParseResult(calls=calls, clean_text=clean)
    return ToolParseResult()


def _try_embedded_json(text: str, tools: Optional[list[dict]]) -> ToolParseResult:
    if not text:
        return ToolParseResult()

    anchors = (
        '{"name"',
        '{"tool_calls"',
        '{"function"',
        '{ "name"',
        '{ "tool_calls"',
        '{ "function"',
    )

    for anchor in anchors:
        idx = text.find(anchor)
        if idx == -1:
            continue

        depth, in_str, esc, end = 0, False, False, -1
        for i in range(idx, len(text)):
            ch = text[i]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if end == -1:
            continue

        candidate = text[idx:end]
        calls = _parse_json_calls(candidate, tools)
        if calls:
            clean = (text[:idx] + text[end:]).strip()
            return ToolParseResult(calls=calls, clean_text=clean)

    return ToolParseResult()


def _try_bare_json(text: str, tools: Optional[list[dict]]) -> ToolParseResult:
    stripped = text.strip()
    calls = _parse_json_calls(stripped, tools)
    if calls:
        return ToolParseResult(calls=calls, clean_text="")
    return ToolParseResult()


def _parse_json_calls(raw: str, tools: Optional[list[dict]]) -> list[ParsedToolCall]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if isinstance(data, dict):
        if isinstance(data.get("tool_calls"), list):
            return [
                c for c in (_normalize_call(x, tools) for x in data["tool_calls"]) if c
            ]
        single = _normalize_call(data, tools)
        if single:
            return [single]

    if isinstance(data, list):
        return [c for c in (_normalize_call(x, tools) for x in data) if c]

    return []


def parse_tool_calls(
    text: str,
    tools: Optional[list[dict]] = None,
    require_marker: bool = False,
) -> ToolParseResult:
    """
    require_marker=True 时，只有明确的 DSML / 哨兵标记才解析；
    不解析裸 JSON / 代码块 —— 用于「调用方没声明 tools，但模型仍输出了 DSML」的场景。
    """
    if not text:
        return ToolParseResult(clean_text=text)

    result = _try_dsml(text, tools)
    if result.calls:
        return result

    result = _try_sentinel(text, tools)
    if result.calls:
        return result

    if require_marker:
        return ToolParseResult(clean_text=text)

    if not _TOOL_PARSE_CONFIG["json_fallback"]:
        # 默认关闭"代码块 / 内嵌 JSON / 裸 JSON 碰运气"式解析：
        # 只认 DSML / 哨兵等明确标记。回答里的普通 JSON、代码示例
        # 不会被误判成工具调用。如需旧行为，设 TOOL_JSON_FALLBACK=true。
        return ToolParseResult(clean_text=text)

    result = _try_code_blocks(text, tools)
    if result.calls:
        return result

    result = _try_embedded_json(text, tools)
    if result.calls:
        return result

    result = _try_bare_json(text, tools)
    if result.calls:
        return result

    return ToolParseResult(clean_text=text)


# ═══════════════════════════════════════════════════════════════
# 流式检测器（★ 本次修复主体 ★）
# ═══════════════════════════════════════════════════════════════


class StreamingToolDetector:
    """
    有状态流式工具调用检测器。

    状态机：
      SCAN     —— 扫描 DSML / 哨兵 / JSON 锚点；尾部缓冲可能成为前缀的字符
      DSML     —— 累积直到 </…DSML… calls> 或 </…DSML… invoke>，然后整体解析
      SENTINEL —— 累积直到 </|tool_calls|>
      JSON     —— 累积直到括号平衡

    关键行为：
      - 遇到最早的结束标签（calls 或 invoke）就切块解析
      - 多 invoke 的 calls 块会被逐 invoke 切分处理
      - SCAN 状态剥掉孤立关闭标签，不吐到 content
      - 任何块解析失败都静默丢弃，绝不吐文本污染 content
    """

    def __init__(
        self,
        tools: Optional[list[dict]] = None,
        require_marker: bool = False,
    ) -> None:
        self._tools = tools
        self._require_marker = require_marker
        self._state = "SCAN"
        self._buf = ""
        self._closed = False
        self._saw_dsml = False

    @property
    def pending_markup(self) -> bool:
        """是否有未落定的标记块（DSML/哨兵/JSON 收集中）。

        断流自动续写前检查：有残块时续写轮必须换新检测器丢弃残块，
        否则续写内容拼进半开的块里必然损坏。
        """
        return (not self._closed) and (self._state != "SCAN" or bool(self._buf))

    def recover_text_on_cut(self) -> str:
        """断流换新检测器前回收扣住的缓冲。

        SCAN/JSON 态的 _buf 是被保守扣住的普通正文（_safe_prefix_len 对
        未闭合 `<` 之后的文本也扣），直接换检测器会把最多 64 字正文一起
        丢掉（2026-10-30 审计）——去掉末尾标记碎片后放行。DSML/SENTINEL
        态是半截工具调用：丢弃，续写轮会重发完整调用。
        """
        held = self._buf or ""
        state = self._state
        self._buf = ""
        self._state = "SCAN"
        self._closed = True
        if state in ("DSML", "SENTINEL") or not held:
            return ""
        idx = _marker_partial_start(held)
        if idx >= 0 and _looks_like_marker_prefix(held[idx:]):
            held = held[:idx]
        return held

    # ── 主入口 ────────────────────────────────────────────

    def feed(self, delta: str) -> list[StreamAction]:
        if self._closed:
            return [StreamAction(kind="text", text=delta)] if delta else []

        self._buf += delta
        actions: list[StreamAction] = []

        guard = 0
        while self._buf and guard < 256:
            guard += 1
            if self._state == "DSML":
                if not self._step_dsml(actions):
                    break
            elif self._state == "SENTINEL":
                if not self._step_sentinel(actions):
                    break
            elif self._state == "JSON":
                if not self._step_json(actions):
                    break
            else:
                if not self._step_scan(actions):
                    break

        return actions

    def finalize(self) -> list[StreamAction]:
        actions: list[StreamAction] = []
        if not self._buf:
            self._closed = True
            return actions

        if self._state == "DSML":
            body = _fix_malformed_openers(self._buf)
            m_open = _CALLS_OPEN_RE.match(body)
            if m_open:
                body = body[m_open.end() :]

            calls = _parse_dsml_body(body, self._tools)
            if calls:
                actions.append(StreamAction(kind="tool_calls", tool_calls=calls))
            else:
                logger.debug(
                    "finalize: discarding unparseable DSML residual len=%d",
                    len(self._buf),
                )
            self._buf = ""
            self._state = "SCAN"
            self._closed = True
            return actions

        if self._state == "SENTINEL":
            text = self._buf
            for v in _OPEN_VARIANTS:
                if text.startswith(v):
                    text = text[len(v) :]
                    break
            # 与 DSML 一致：残余优先尝试解析，解析不出就静默丢弃，
            # 绝不把未闭合的哨兵标记原文吐进 content。
            calls = parse_sentinel_body(text, self._tools)
            if calls:
                actions.append(StreamAction(kind="tool_calls", tool_calls=calls))
            else:
                logger.debug(
                    "finalize: discarding unparseable SENTINEL residual len=%d",
                    len(self._buf),
                )
            self._buf = ""
            self._state = "SCAN"
            self._closed = True
            return actions

        # JSON 或 SCAN：按文本吐出（可能是不完整的 JSON，客户端自行处理）
        # 但末尾的标记碎片（上游把 `<` / `｜｜DSM` 单独留在流尾）必须丢掉，
        # 不能把裸 `<` 交给客户端。
        if self._buf:
            idx = _marker_partial_start(self._buf)
            if idx >= 0 and _looks_like_marker_prefix(self._buf[idx:]):
                # 只剩 DSML 标记碎片（上游把 `<` / `｜｜DSM` 单独留在流尾）→ 丢掉
                logger.debug(
                    "finalize: discarding partial DSML marker prefix len=%d",
                    len(self._buf) - idx,
                )
                text = self._buf[:idx]
            else:
                text = self._buf
            if text:
                actions.append(StreamAction(kind="text", text=text))
        self._buf = ""
        self._state = "SCAN"
        self._closed = True
        return actions

    # ── SCAN ──────────────────────────────────────────────

    def _step_scan(self, actions: list[StreamAction]) -> bool:
        if not self._buf:
            return False

        idx, kind = self._find_earliest_scan_token()
        if idx == -1:
            # 没有锚点：吐出安全前缀（保留可能成为标签前缀的尾巴）
            safe_len = self._safe_prefix_len()
            if safe_len > 0:
                actions.append(StreamAction(kind="text", text=self._buf[:safe_len]))
                self._buf = self._buf[safe_len:]
            return False

        if idx > 0:
            prefix = self._buf[:idx]
            # 前缀里如果有孤立关闭标签，剥掉再吐
            if _NOISE_RE.search(prefix):
                prefix = _NOISE_RE.sub("", prefix)
            if prefix:
                actions.append(StreamAction(kind="text", text=prefix))
            self._buf = self._buf[idx:]

        if kind == "dsml_open":
            self._state = "DSML"
            self._saw_dsml = True
            return True
        if kind == "dsml_close":
            # 孤立关闭标签：剥掉不吐
            m = _NOISE_RE.match(self._buf)
            if m:
                self._buf = self._buf[m.end() :]
                return True
            return False
        if kind == "sentinel_open":
            self._state = "SENTINEL"
            return True
        if kind == "json_open":
            self._state = "JSON"
            return True
        return False

    def _find_earliest_scan_token(self) -> tuple[int, str]:
        best_idx, best_kind = -1, ""

        # 哨兵始终识别
        for v in _OPEN_VARIANTS:
            i = self._buf.find(v)
            if i != -1 and (best_idx == -1 or i < best_idx):
                best_idx, best_kind = i, "sentinel_open"

        # DSML 始终识别
        m = _ANY_DSML_OPEN_RE.search(self._buf)
        if m and (best_idx == -1 or m.start() < best_idx):
            best_idx, best_kind = m.start(), "dsml_open"

        # 孤立关闭标签：tools 模式剥；无 tools 时当普通文本——
        # 但本流里已经出现过 DSML 标记（说明模型确实在写调用）时，
        # 残留的闭合标签没有任何正文价值，一并剥掉。
        if (not self._require_marker) or self._saw_dsml:
            m = _NOISE_RE.search(self._buf)
            if m and (best_idx == -1 or m.start() < best_idx):
                best_idx, best_kind = m.start(), "dsml_close"

        # JSON 锚点：require_marker=True 或关闭 json_fallback 时忽略
        if not self._require_marker and _TOOL_PARSE_CONFIG["json_fallback"]:
            for a in _ANCHORS:
                i = self._buf.find(a)
                if i != -1 and (best_idx == -1 or i < best_idx):
                    best_idx, best_kind = i, "json_open"

        return best_idx, best_kind

    def _safe_prefix_len(self) -> int:
        """
        返回可安全吐为文本的前缀长度。

        策略：
          - 如果 buf 末尾有未闭合的 "<..."（可能构成 DSML 标签或哨兵），保留它。
          - 如果末尾是 JSON 锚点的部分前缀（如 "{"、"{\"n"），保留它。
          - 否则整个 buf 都可以吐。

        关键：绝不能只看末字符。单个 "｜" 不是锚点开头，
        但它可能是 "<｜｜DSML｜｜" 的中间片段；一旦把它当独立字符吐出去，
        前面的 "<" 也保不住，整个锚点就永远错过。
        """
        buf = self._buf
        n = len(buf)
        if n == 0:
            return 0

        lower = max(0, n - _SAFE_KEEP)

        # ── 1. 找最长的、未闭合的 "<..." 后缀 ──
        # 从末尾往前扫，遇到第一个 "<" 就判断它到末尾是否闭合。
        #   - 未闭合 → 这就是我们要保留的起点
        #   - 已闭合 → 说明末尾的 "<...>" 是完整标签（不属于 DSML 的话，
        #             早就该被 _find_earliest_scan_token 处理掉了）；
        #             再往前找已经没意义，直接退出
        for i in range(n - 1, lower - 1, -1):
            if buf[i] == "<":
                if ">" not in buf[i:]:
                    return i
                break

        # ── 1.5 尾巴是 DSML 标记碎片（可能完全没有 `<`）──
        # 降级形态 `｜｜DSML｜｜ calls>` 不含尖括号，上面的规则扣不住，
        # 不扣就会先把 `｜｜DSM` 半截吐给客户端，整体锚点再也认不出来。
        mstart = _marker_partial_start(buf)
        if mstart != -1:
            return mstart

        # ── 2. JSON 锚点的部分前缀 ──
        # 只在"调用方声明了 tools 且开启 json_fallback"时才会触发 JSON 检测；
        # 关闭时这里恒为 n，不影响纯文本。
        if not self._require_marker and _TOOL_PARSE_CONFIG["json_fallback"]:
            best_json = n
            for a in _ANCHORS:
                max_k = min(len(a), n)
                for k in range(1, max_k + 1):
                    suffix = buf[n - k :]
                    if a.startswith(suffix):
                        best_json = min(best_json, n - k)
            if best_json < n:
                return best_json

        return n

    # ── DSML ──────────────────────────────────────────────

    def _step_dsml(self, actions: list[StreamAction]) -> bool:
        """
        DSML 状态：只找最早的结束标签（</…calls> 或 </…invoke>）。

        - 先出现 </…invoke> 就切一个 invoke，回到 SCAN（下一个 <invoke 会重新进入）
        - 先出现 </…calls> 就切整个块
        - 都不出现 → 继续累积，finalize 兜底
        """
        end_pos = -1
        end_len = 0

        m_calls = _CALLS_CLOSE_RE.search(self._buf)
        if m_calls:
            end_pos = m_calls.start()
            end_len = m_calls.end() - m_calls.start()

        m_invoke = _INVOKE_CLOSE_RE.search(self._buf)
        if m_invoke and (end_pos == -1 or m_invoke.start() < end_pos):
            end_pos = m_invoke.start()
            end_len = m_invoke.end() - m_invoke.start()

        if end_pos == -1:
            return False

        body = self._buf[:end_pos]
        rest = self._buf[end_pos + end_len :]

        # 降级开标签先纠正，再按正常块解析
        body = _fix_malformed_openers(body)

        # 若 body 以 <… calls> 开头，去掉外层 calls 开标签
        m_open = _CALLS_OPEN_RE.match(body)
        if m_open:
            body = body[m_open.end() :]

        calls = _parse_dsml_body(body, self._tools)
        if calls:
            actions.append(StreamAction(kind="tool_calls", tool_calls=calls))
        else:
            logger.debug(
                "_step_dsml: DSML block yielded no calls, discarding body len=%d",
                len(body),
            )

        self._buf = rest
        self._state = "SCAN"
        return True

    # ── SENTINEL ──────────────────────────────────────────

    def _step_sentinel(self, actions: list[StreamAction]) -> bool:
        close_idx, close_v = self._find_close()
        if close_idx == -1:
            return False

        open_v = self._open_at(0)
        if open_v is None:
            actions.append(StreamAction(kind="text", text=self._buf))
            self._buf = ""
            self._state = "SCAN"
            return True

        body = self._buf[len(open_v) : close_idx]
        rest = self._buf[close_idx + len(close_v) :]

        calls = parse_sentinel_body(body, self._tools)
        if calls:
            actions.append(StreamAction(kind="tool_calls", tool_calls=calls))
        else:
            logger.debug("_step_sentinel: no calls, discarding block")

        self._buf = rest
        self._state = "SCAN"
        return True

    # ── JSON ──────────────────────────────────────────────

    def _step_json(self, actions: list[StreamAction]) -> bool:
        end = self._find_json_end(self._buf)
        if end == -1:
            return False

        block = self._buf[:end]
        rest = self._buf[end:]

        calls = _parse_json_calls(block, self._tools)
        if calls:
            actions.append(StreamAction(kind="tool_calls", tool_calls=calls))
        else:
            # 解析失败 → 当普通文本吐（不是工具调用）
            actions.append(StreamAction(kind="text", text=block))

        self._buf = rest
        self._state = "SCAN"
        return True

    # ── 内部工具 ──────────────────────────────────────────

    def _find_close(self) -> tuple[int, str]:
        start = 0
        for v in _OPEN_VARIANTS:
            if self._buf.startswith(v):
                start = len(v)
                break
        best_idx, best_v = -1, ""
        for v in _CLOSE_VARIANTS:
            i = self._buf.find(v, start)
            if i != -1 and (best_idx == -1 or i < best_idx):
                best_idx, best_v = i, v
        return best_idx, best_v

    def _open_at(self, idx: int) -> Optional[str]:
        for v in _OPEN_VARIANTS:
            if self._buf.startswith(v, idx):
                return v
        return None

    @staticmethod
    def _find_json_end(s: str) -> int:
        if not s.startswith("{"):
            return -1
        depth, in_str, esc = 0, False, False
        for i, ch in enumerate(s):
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i + 1
        return -1
