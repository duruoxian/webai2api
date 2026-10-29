"""智谱清言（chatglm.cn）客户端。

把网页版接口封装成 OpenAI 兼容的 chat_completion / stream_chat_completion。
协议（参考开源项目 glm-free-api 的思路，MIT）：
  1. 用 refresh_token 换 access_token（有效期约 1 小时，本客户端内缓存）
  2. POST /chatglm/backend-api/assistant/stream 拿 SSE，逐块解析 parts
  3. 结束后（可选）删除会话，避免污染用户对话列表
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
import uuid
from logging import Logger

from .. import conv_cache, netutil, reasoning_bridge, tool_bridge
from .config import GlmConfig
from .uploader import GlmImageUploader, extract_image_urls

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
        "X-Request-Id": uuid.uuid4().hex,  # 每请求唯一是正常的
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
    """把 OpenAI 多轮消息合并成智谱要的单条 user 内容。

    智谱走「全量压平」策略：整段对话带角色标记重发一遍。role=tool
    的执行结果在这里渲染成 <tool_result> 块并归入 user 轮，模型才能
    看懂「这是工具跑出来的结果」，而不是一段来历不明的裸文本。
    """
    if len(messages) < 2:
        return "".join(_text_of(m) + "\n" for m in messages)
    role_map = {
        "system": "<|system|>",
        "assistant": "<|assistant|>",
        "user": "<|user|>",
    }
    id2name = tool_bridge.tool_name_by_id(messages)
    parts = []
    for m in messages:
        role = m.get("role", "user")
        if role == "tool":
            name = id2name.get(m.get("tool_call_id") or "", m.get("name") or "tool")
            body = tool_bridge.render_tool_result(name, _text_of(m), tag="tool_result")
            parts.append(f"<|user|>\n{body}\n")
            continue
        text = _text_of(m)
        if role == "assistant" and m.get("tool_calls"):
            # 工具调用轮的 assistant.content 通常是 None，只写角色标签会把
            # 整轮压成空行：模型看不到自己发过什么调用，也就认不出后面的
            # <tool_result> 是给谁的，于是把同一个工具再调一遍。
            text = tool_bridge.render_assistant_tool_calls(m["tool_calls"], text)
        parts.append(f"{role_map.get(role, role)}\n{text}\n")
    return "".join(parts) + "<|assistant|>\n"


TOOL_MARKER = "<<<TOOL_CALL>>>"

# ── 断流自动续写 ────────────────────────────────────
# 上游偶尔在长思考/长回答中途掐流且不发 status=finish 帧。检测到断流后，
# 向同一 conversation 发一条「继续」——对话历史里已有模型自己的半截回答，
# 它会自然接着写（无需特殊协议）。续写增量经全新累积器拼进输出，
# 对上层 prefix-diff 链保持单调。最多续写 2 次，失败降级为截断。
MAX_AUTO_CONTINUATIONS = 2
CONTINUE_PROMPT = "继续"

TOOL_INSTRUCTION = """你可以调用以下工具来完成任务：

{tools}

规则：
1. 如果用户的问题需要调用工具，只输出一行，格式严格为：
{marker}{{"name":"工具名","arguments":{{"参数名":值}}}}
2. 一次只调用一个工具。不要输出任何解释文字。
3. 如果不需要工具，直接用自然语言回答，不要输出 {marker}。
4. 你运行在用户的本地电脑上，工具就是你唯一能操作本地文件的方式；
   不要说「我不能访问你的系统盘」，需要写文件时直接调用对应工具。
   用户要求创建/写入/生成文件时，必须调用对应的写文件工具完成，
   不要把文件内容整段贴在回答里就当作已经完成。
5. 收到 <tool_result> 包裹的工具执行结果后，基于真实结果继续：
   需要更多信息就再调用工具，信息够了就给出最终回答（不要再输出标记）。
6. 编辑文件时，old_string 必须逐字复制自最近一次读到的文件内容，
   禁止凭记忆改写；不确定或上次编辑失败时，先重新读取文件再编辑。
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
                lines.append(
                    f"    参数 {k} ({typ}){mark}: {(v or {}).get('description', '')}"
                )
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


def _collapse_cached_text(content: str, messages: list) -> str:
    """上一轮就是工具调用轮时，给增量补一句「已执行」提示。

    模型在上游会话里看到自己刚发过工具调用，而会话里还没有结果（增量
    只发新的一轮）时，实测会原样重调同一个工具。这里在增量前面加一行
    明确的完成声明，把「调用 → 结果」闭合掉。

    只看**最后一条** assistant 是否带 tool_calls：更早轮次的调用早已
    闭合。此前对「历史里出现过任何工具调用」的每一轮增量都加
    「…请基于结果直接给出最终回答」，agent 后续本该继续调工具的轮次
    被这句带着直接文字作答——实测 GLM 因此「忘记」写文件，把文件内容
    当最终回答贴出来（2026-10-28 用户报障）。
    """
    if not content:
        return content
    last = None
    for m in messages or []:
        if isinstance(m, dict) and m.get("role") == "assistant":
            last = m
    names = []
    for c in (last or {}).get("tool_calls") or []:
        fn = (c or {}).get("function") or {}
        if fn.get("name"):
            names.append(fn["name"])
    if not names:
        return content
    head = (
        "[已执行] 上一轮发起的工具调用（"
        + ", ".join(dict.fromkeys(names))
        + "）均已执行完毕，下方 <tool_result> 就是执行结果。请基于结果继续："
        "还需要其他信息或操作就再调用工具，全部完成后再给出最终回答。\n"
    )
    return head + content


def _strip_tool_block(text: str) -> str:
    """去掉文本里的工具调用标记（标记 + 其 JSON / DSML 块，跨行也一并删）。

    上游会吃 `<`、流会被切碎，标记常以缺角括号/孤立闭合的降级形态出现，
    精确剥除前先归一形态，剥完再扫一遍残余碎片（对齐 deepseek 零残留）。
    """
    text = tool_bridge.normalize_tool_marker(text or "", TOOL_MARKER)
    out = tool_bridge.strip_tool_markup(text, marker=TOOL_MARKER, dsml=True)
    out = tool_bridge.scrub_marker_fragments(out)
    return tool_bridge.strip_truncated_marker_tail(
        out, [tool_bridge.DSML_TOOL_CALLS_OPEN, TOOL_MARKER]
    ).strip()


def _parse_tool_calls(text: str, tools: list) -> list:
    """从模型回复里解析工具调用（DSML 优先，再走标记 + 裸 JSON 兜底）。

    先归一降级角括号形态——上游吃 `<` 后标记词还在但括号缺数，
    归一后 DSML/标记路径才能命中。
    """
    return tool_bridge.parse_tool_calls(
        tool_bridge.normalize_tool_marker(text or "", TOOL_MARKER),
        tools,
        marker=TOOL_MARKER,
        dsml=True,
    )


# ── 重复输出去重（问题 3：GLM 回复重复两遍）────────────────
# 智谱上游偶发把整段回答原样输出两遍（第二遍紧跟第一遍）。无论源头是
# 上游 parts 累积还是模型本身，输出层都能兜住：
# - 流式 RepeatGuard：新增量若逐字复述已发全文的开头 → 先扣住；
#   复述完整确认重复就丢弃，中途发散则原样放行（不丢内容）。
# - 非流式 _dedup_repeated：全文恰好 P+P（P 足够长）时折叠回 P。
_DUP_MIN_BLOCK = 80


def _dedup_repeated(text: str) -> str:
    """整段「输出两遍」折叠回一遍：去尾空白后恰好 P+P（|P|≥80 字）取 P。"""
    if not text:
        return text
    core = text.rstrip()
    n = len(core) // 2
    if n >= _DUP_MIN_BLOCK and core[:n] == core[n:]:
        return core[:n]
    return text


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


def _render_part(part: dict, is_silent: bool = False) -> tuple[str, str]:
    """渲染单个 part 的 (text, reasoning)。item 语义与旧 _extract_parts 一致。"""
    text = ""
    reasoning = ""
    content = part.get("content")
    if not isinstance(content, list):
        return text, reasoning
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
        elif (
            t == "execution_output"
            and isinstance(item.get("content"), str)
            and status == "finish"
        ):
            text += item["content"] + "\n"
        elif (
            t == "image" and isinstance(item.get("image"), list) and status == "finish"
        ):
            for img in item["image"]:
                u = img.get("image_url") if isinstance(img, dict) else None
                if isinstance(u, str) and (
                    u.startswith("http://") or u.startswith("https://")
                ):
                    text += f"![图像]({u})"
            text += "\n"
    return text, reasoning


class _GlmParts:
    """智谱 parts 累积器。

    上游 SSE 语义（2026-10-25 抓帧实测确认）：
    - 每帧 parts 携带该 part 的**增量 delta**（每帧只有几个新字）；
    - part 结束时发 status=finish 帧，content 里是该 part 的**全量快照**
      （权威终态，用于校准）；
    - 思考/正文/代码是不同 logic_id 的 part，交错到达；
    - 偶发空的 tool_calls part（st=finish 无内容），忽略；
    - 服务端执行完原生工具后，会下发 content 项 ``type=tool_calls``
      （``{"id","name","arguments"}``），这是上游已经确定的调用意图，
      比从正文里解析标记可靠，优先采用。

    因此按 logic_id 累积 delta，finish 帧用全量校准，全量输出按 part
    首见顺序拼接。此前把 delta 当快照做前缀差，是丢字/咬字/JSON 漏
    正文的总根源。
    """

    def __init__(
        self,
        is_silent: bool = False,
        allowed_tools: set | None = None,
        seen_call_keys: set | None = None,
    ) -> None:
        self._is_silent = is_silent
        self._allowed_tools = allowed_tools
        self._order: list[str] = []  # lid 首见序
        self._text: dict[str, str] = {}  # lid -> 正文累计
        self._think: dict[str, str] = {}  # lid -> 思考累计
        self._code: dict[str, str] = {}  # lid -> 代码累计（finish 全量校准）
        self._extra: dict[str, str] = {}  # lid -> image 等一次性渲染
        # 去重键跨轮携带：续写轮新累积器也要认得上一轮已收录的调用，
        # 否则断流跨轮重发的同一调用会被按新 id 再收一次（双重执行）
        self._call_ids: set[str] = set(seen_call_keys or ())
        self.tool_calls: list[dict] = []  # 服务端原生 tool_calls
        self.text = ""  # 全量正文
        self.reasoning = ""  # 全量思考

    @staticmethod
    def _call_sig(name: str, args: str) -> str:
        """调用去重签名：name + 参数内容。无 id 的调用也能跨帧/跨轮去重。"""
        return f"{name}:{hashlib.md5((args or '').encode('utf-8')).hexdigest()}"

    @staticmethod
    def _calibrate(buf: str, piece: str, status) -> str:
        """finish 帧携带全量快照（权威终态），其余帧是增量 delta。"""
        if status == "finish":
            return piece if piece else buf
        return buf + piece

    def _native_call(self, data) -> None:
        """收集服务端原生 tool_calls（同一调用只记一次，finish 帧会重发）。"""
        if isinstance(data, list):
            for one in data:
                self._native_call(one)
            return
        if not isinstance(data, dict):
            return
        name = str(data.get("name") or "").strip()
        cid = str(data.get("id") or "").strip()
        if not name:
            return
        # 只在客户端明确声明了工具时下发；没声明就当成上游内置能力，
        # 混进 tool_calls 会让客户端收到无法执行的调用
        if not self._allowed_tools or name not in self._allowed_tools:
            return
        args = data.get("arguments", {})
        if not isinstance(args, str):
            args = json.dumps(args or {}, ensure_ascii=False)
        # 去重键用「名字+参数签名」而非自增序号：无 id 调用的旧键
        # f"{name}:{len(...)}" 在首次 append 后就变了，finish 帧重发
        # 同一调用去不了重（客户端双重执行，2026-10-30 审计 GLM-P2-4）
        key = cid or self._call_sig(name, args)
        if key in self._call_ids:
            return
        self._call_ids.add(key)
        self.tool_calls.append(
            {
                "id": cid or f"call_{uuid.uuid4().hex[:16]}",
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
        )

    def consume(self, parts) -> None:
        if not isinstance(parts, list):
            return
        dirty = False
        for part in parts:
            if not isinstance(part, dict):
                continue
            content = part.get("content")
            if not isinstance(content, list):
                continue
            lid = str(part.get("logic_id") or f"\x00anon:{len(self._order)}:{id(part)}")
            if lid not in self._text:
                self._order.append(lid)
                self._text[lid] = ""
                self._think[lid] = ""
                self._extra[lid] = ""
                self._code[lid] = ""
            status = part.get("status")
            for item in content:
                if not isinstance(item, dict):
                    continue
                t = item.get("type")
                if t == "tool_calls":
                    self._native_call(item.get("tool_calls"))
                    continue
                if t == "text":
                    self._text[lid] = self._calibrate(
                        self._text[lid], item.get("text") or "", status
                    )
                    dirty = True
                elif t == "think" and not self._is_silent:
                    self._think[lid] = self._calibrate(
                        self._think[lid], item.get("think") or "", status
                    )
                    dirty = True
                elif t == "code":
                    # code part 与 text/think 同为 delta 帧语义：按 logic_id
                    # 累积 + finish 全量校准，装配时只渲染一次围栏。此前走
                    # _render_part 逐帧包 ```python 围栏、finish 快照再追加
                    # 一遍（内容重复+围栏错乱，2026-10-30 审计 GLM-1）。
                    self._code[lid] = self._calibrate(
                        self._code[lid], item.get("code") or "", status
                    )
                    dirty = True
                else:
                    rendered, _ = _render_part(
                        {"content": [item], "status": status}, self._is_silent
                    )
                    if rendered:
                        self._extra[lid] += rendered
                        dirty = True
        if dirty:
            def _render_code(code: str) -> str:
                return "```python\n" + code + "\n```\n" if code else ""

            self.text = "".join(
                self._text[lid]
                + (_render_code(self._code[lid]) if self._code.get(lid) else "")
                + self._extra[lid]
                for lid in self._order
            )
            self.reasoning = "".join(self._think[lid] for lid in self._order)


class ChatGlmWebClient:
    def __init__(self, config: GlmConfig, logger: Logger) -> None:
        self.config = config
        self.logger = logger
        self._lock = threading.Lock()
        self._access_token = ""
        self._token_expire = 0.0
        # 会话缓存：同一段对话复用 conversation_id，多轮只发增量（与 DeepSeek 行为一致）
        self._conv_cache = conv_cache.ConversationCache(
            config=config,
            logger=logger,
            tag="GLM",
            id_field="conversation_id",
            persist_path=str(getattr(config, "session_store_path", "") or ""),
            bump=self._bump,
        )
        # 同账号上游请求最小间隔
        self._last_call_at: float = 0.0
        self._last_call_lock = threading.Lock()
        # 图片上传（chat_upload），消息含 image_url 时调用
        self.image_uploader = GlmImageUploader(self, logger)

    def upload_headers(self) -> dict[str, str]:
        """图片上传请求头（含签名与 Bearer，供 uploader 复用同一出口）。"""
        return _headers_for(self.config, self._access())

    def _upload_image_items(self, messages: list[dict]) -> list[dict]:
        """把消息里的图片上传并按网页端字段组装 image part 条目。

        消息不含图片时返回空列表；单张失败跳过（纯文本继续），全部走完
        后按顺序编 order。
        """
        urls = extract_image_urls(messages)
        if not urls:
            return []
        items: list[dict] = []
        for i, u in enumerate(urls[:8]):
            item = self.image_uploader.upload(u, name_hint=f"image_{i}.png")
            if item:
                item["order"] = len(items)
                items.append(item)
            else:
                self.logger.warning("Vision | 第 %d 张图片上传失败，已跳过", i)
        if items:
            self.logger.info("Vision | 已上传 %d/%d 张图片", len(items), len(urls))
        return items

    # ── 会话缓存持久化（实现见 adapters/conv_cache.py）──
    def flush_cache(self) -> None:
        """强制落盘（关机前调用）。"""
        self._conv_cache.flush()

    def _cached_conv(self, key: str) -> str:
        return self._conv_cache.cached(key)[0]

    def _store_conv(self, key: str, conv_id: str, used_tokens: int = 0) -> None:
        self._conv_cache.store(key, conv_id, used_tokens=used_tokens)

    def _conv_key(self, payload: dict) -> str:
        """会话指纹（tool_bridge.conv_fingerprint），首条 assistant 防串台。"""
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

    @staticmethod
    def _split(url: str) -> tuple[str, str]:
        return netutil.split_url(url, "chatglm.cn")

    def _conn(self, url: str, timeout: int):
        return netutil.open_conn(
            url,
            timeout,
            egress=getattr(self.config, "egress", "system"),
            proxy=str(getattr(self.config, "proxy", "") or ""),
            default_host="chatglm.cn",
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
                raise GlmError(
                    f"HTTP {resp.status}: {raw[:200].decode('utf-8', 'replace')}"
                )
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
        try:
            conn.request(
                "POST", path, body=json.dumps(body).encode("utf-8"), headers=headers
            )
            resp = conn.getresponse()
        except Exception:
            # 请求中途抛错（重置/超时）必须关连接，否则 socket 泄漏
            conn.close()
            raise
        if resp.status >= 400:
            if resp.status == 429:
                self._bump("rate_limit")
            raw = resp.read()
            conn.close()
            raise GlmError(
                f"HTTP {resp.status}: {raw[:200].decode('utf-8', 'replace')}"
            )
        return conn, resp

    # ── token ──────────────────────────────────────────
    def _access(self) -> str:
        with self._lock:
            if self._access_token and time.time() < self._token_expire:
                return self._access_token
            if not self.config.refresh_token:
                raise GlmError("缺少 refresh_token，请先在面板登录智谱清言账号", 401)
            data = self._post_json(
                self.config.refresh_url,
                {},
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

    def _body(
        self,
        content: str,
        model: str,
        conv_id: str = "",
        effort: str = "",
        image_items: list[dict] | None = None,
    ) -> dict:
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
        # 图片 part：字段名逐字对齐网页端（file_id/image_url/file_size/order），
        # 抓帧实测缺字段或用错字段名上游会静默丢图
        parts: list[dict] = [{"type": "text", "text": content}]
        if image_items:
            parts.append({"type": "image", "image": image_items})
            meta["input_question_type"] = "image_upload"
        return {
            "assistant_id": self._assistant_id(model),
            "conversation_id": conv_id,
            "project_id": "",
            "chat_type": "user_chat",
            "messages": [{"role": "user", "content": parts}],
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
        stale = self._conv_cache.collect_stale()
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
        tools = payload.get("tools") or []
        if tools and payload.get("tool_choice") != "none":
            messages = _inject_tools(messages, tools)
        else:
            tools = []
        is_silent = "silent" in model

        self._throttle()
        self._bump("upstream")
        token = self._access()
        # 命中会话缓存则复用 conversation_id（与 DeepSeek 的多轮增量行为一致）；
        # 首轮绝不复用——新客户端会话的首轮指纹常与旧会话相同，复用会串上下文
        conv_key = self._conv_key(payload)
        cached_conv = (
            ""
            if tool_bridge.is_first_turn(payload.get("messages"))
            else self._cached_conv(conv_key)
        )
        if not cached_conv:
            self._bump("miss")
        # 复用上游会话时只发增量（最后一条 assistant 之后的 tool 结果与新
        # user 文本）：上游会话里已有此前全部轮次，重发全量拼平历史会把
        # 上下文一遍遍重复（长会话越拖越慢、模型越来越乱）。未复用（首轮
        # /缓存失效/轮换）才发全量拼平——与 Kimi/千问的增量协议一致。
        if cached_conv:
            content = tool_bridge.increment_text(messages, result_tag="tool_result")
            if not content.strip():
                content = _prepare_content(messages)
            content = _collapse_cached_text(content, messages)
        else:
            content = _prepare_content(messages)
        # 图片：首轮随全量压平附上全部图片；复用会话轮只看最后一条 user
        # 消息的新图（上游会话里已有历史图片）。任一张失败跳过，纯文本继续。
        try:
            if cached_conv:
                # 只看最后一条 user 消息（不要求 content 是 list）：若最新
                # 一轮是纯文本而取「最后一条带 list 的 user」，会把历史旧图
                # 重新挂到本轮，上游会话里同一张图逐轮膨胀（审计 GLM-P2-8）
                last_user = next(
                    (m for m in reversed(messages) if m.get("role") == "user"),
                    None,
                )
                image_items = (
                    self._upload_image_items([last_user]) if last_user else []
                )
            else:
                image_items = self._upload_image_items(messages)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Vision | 图片处理失败，按纯文本继续: %s", exc)
            image_items = []
        conn, resp = self._open_stream(
            self.config.stream_url,
            self._body(
                content,
                model,
                conv_id=cached_conv,
                effort=str(
                    payload.get("reasoning_effort") or self.config.default_effort
                ),
                image_items=image_items,
            ),
            _headers_for(self.config, token),
            self.config.request_timeout,
        )
        conv_id = ""
        allowed = tool_bridge.tool_names(tools) if tools else None
        acc = _GlmParts(is_silent, allowed_tools=allowed)
        # 跨轮拼接基线：续写轮用全新累积器（避免上游复用 logic_id 时
        # finish 全量校准污染前一轮内容），输出 = 基线 + 本轮累积，对
        # 上层 prefix-diff 链保持单调。
        base_text = ""
        base_reasoning = ""
        all_calls: list[dict] = []
        saw_finish = False
        # 续写接缝剪除（对齐 deepseek/kimi/qwen 的 SeamTrimmer）：模型
        # 「继续」常把断点前最后一段逐字重说一遍，增量开头「出现在已发
        # 尾巴里」就扣住，分叉时提交最长后缀-前缀重叠——剪掉的字节都与
        # 客户端已有内容逐字相同，数学上不丢新内容。仅续写轮激活
        # （None = 首轮直通）。思考侧在接缝处补一个换行，条件收得很窄：
        # 基线结尾与续写开头都是文字字符、且本轮尚未剪掉重叠（剪过说明
        # 分叉点是真实续点，拼接即原貌，补换行反而拆字）；正文永不注入。
        seam_text = None
        seam_reason = None
        seam_fed = [0, 0]  # 本轮 acc.text / acc.reasoning 已喂入 seam 的长度
        round_text_shown = ""  # 本轮已放行正文（seam.committed 镜像）
        round_reason_shown = ""  # 本轮已放行思考（含接缝换行）
        round_reason_fed = 0  # 本轮已喂入思考的原始字节数（判「剪过没有」）
        seam_joiner = ""

        def _seam_feed() -> None:
            nonlocal round_text_shown, round_reason_shown, round_reason_fed
            nonlocal seam_joiner
            if seam_text is None:
                return
            d_text = acc.text[seam_fed[0] :]
            d_reason = acc.reasoning[seam_fed[1] :]
            if not d_text and not d_reason:
                return
            seam_fed[0], seam_fed[1] = len(acc.text), len(acc.reasoning)
            if d_text:
                round_text_shown += seam_text.feed(d_text)
            if d_reason:
                round_reason_fed += len(d_reason)
                out = seam_reason.feed(d_reason)
                if out and seam_joiner:
                    if (
                        base_reasoning
                        and base_reasoning[-1].isalnum()
                        and out[0].isalnum()
                        and round_reason_fed == len(round_reason_shown) + len(out)
                    ):
                        out = seam_joiner + out
                    seam_joiner = ""
                round_reason_shown += out

        def _combined() -> tuple[str, str]:
            if seam_text is None:
                return base_text + acc.text, base_reasoning + acc.reasoning
            return (
                base_text + round_text_shown,
                base_reasoning + round_reason_shown,
            )

        def _drain(resp) -> bool:
            """消费一个上游 SSE 流直到 finish/断流，返回是否见到 finish 帧。"""
            nonlocal conv_id
            for event in self._iter_events(resp):
                if not isinstance(event, dict):
                    continue
                if event.get("conversation_id"):
                    conv_id = event["conversation_id"]
                if isinstance(event.get("parts"), list):
                    acc.consume(event["parts"])
                    _seam_feed()
                if event.get("status") == "finish":
                    return True
                text, reasoning = _combined()
                yield {
                    "text": text,
                    "reasoning": reasoning,
                    "conv_id": conv_id,
                    "tool_calls": all_calls + acc.tool_calls,
                    "finish_seen": saw_finish,
                }
            return False

        try:
            try:
                saw = yield from _drain(resp)
                if saw:
                    saw_finish = True
            finally:
                try:
                    resp.read()
                except Exception:
                    pass
                try:
                    conn.close()
                except Exception:
                    pass

            # ── 断流自动续写 ──
            # 上游没发 status=finish 帧就断流（长思考/长回答被掐）：向同一
            # conversation 发一条「继续」，模型看到自己的半截回答会接着写。
            # 续写轮换全新累积器，输出拼接后对上层保持单调；最多 2 轮。
            _cont_rounds = 0
            while (
                not saw_finish
                and _cont_rounds < MAX_AUTO_CONTINUATIONS
                and conv_id
                and (acc.text or acc.reasoning or acc.tool_calls)
            ):
                _cont_rounds += 1
                self._bump("auto_continued")
                self.logger.warning(
                    "上游流提前中断（未见 finish 帧），自动续写 %d/%d",
                    _cont_rounds,
                    MAX_AUTO_CONTINUATIONS,
                )
                if seam_text is None:
                    # 首轮无接缝，全文已示，直接并入基线
                    base_text += acc.text
                    base_reasoning += acc.reasoning
                else:
                    # 续写轮只并入已放行部分；仍扣着的必然是已发尾巴的
                    # 逐字复读，finish() 丢弃（与 deepseek/kimi 同语义）
                    seam_text.finish()
                    seam_reason.finish()
                    base_text += round_text_shown
                    base_reasoning += round_reason_shown
                all_calls.extend(acc.tool_calls)
                acc = _GlmParts(
                    is_silent,
                    allowed_tools=allowed,
                    seen_call_keys={
                        _GlmParts._call_sig(
                            c["function"]["name"], c["function"]["arguments"]
                        )
                        for c in all_calls
                    },
                )
                seam_text = tool_bridge.SeamTrimmer(
                    base_text[-tool_bridge.SEAM_WINDOW :]
                )
                seam_reason = tool_bridge.SeamTrimmer(
                    base_reasoning[-tool_bridge.SEAM_WINDOW :]
                )
                seam_fed[0] = seam_fed[1] = 0
                round_text_shown = round_reason_shown = ""
                round_reason_fed = 0
                seam_joiner = "\n\n"
                try:
                    c_conn, c_resp = self._open_stream(
                        self.config.stream_url,
                        self._body(
                            CONTINUE_PROMPT,
                            model,
                            conv_id=conv_id,
                            effort=str(
                                payload.get("reasoning_effort")
                                or self.config.default_effort
                            ),
                        ),
                        _headers_for(self.config, token),
                        self.config.request_timeout,
                    )
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning("自动续写失败（按截断返回）: %s", exc)
                    break
                try:
                    saw = yield from _drain(c_resp)
                    if saw:
                        saw_finish = True
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning("自动续写失败（按截断返回）: %s", exc)
                    break
                finally:
                    try:
                        c_resp.read()
                    except Exception:
                        pass
                    try:
                        c_conn.close()
                    except Exception:
                        pass
        finally:
            # 兜底关闭主连接（续写轮各自关闭）
            try:
                resp.read()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass

        if seam_text is None:
            text, reasoning = base_text + acc.text, base_reasoning + acc.reasoning
        else:
            # 收尾：仍扣住的尾巴复读段丢弃，只保留已放行内容
            seam_text.finish()
            seam_reason.finish()
            text, reasoning = (
                base_text + round_text_shown,
                base_reasoning + round_reason_shown,
            )
        all_calls.extend(acc.tool_calls)
        if conv_id:
            if self.config.session_cache_enabled:
                # store key 与客户端多轮回传的查询 key 一致（含本轮 assistant 指纹）
                calls = all_calls or _parse_tool_calls(text, tools)
                asst = {
                    "role": "assistant",
                    "content": None if calls else text,
                    "tool_calls": calls or None,
                }
                store_key = tool_bridge.conv_fingerprint(
                    (payload.get("messages") or []) + [asst],
                    payload.get("model") or "",
                    payload.get("_wapi_key_id") or "",
                )
                self._store_conv(
                    store_key,
                    conv_id,
                    used_tokens=_estimate_tokens(text) + _estimate_tokens(reasoning),
                )
            elif self.config.delete_conversation:
                self._delete_conversation(conv_id, token)
        yield {
            "__done__": True,
            "text": text,
            "reasoning": reasoning,
            "conv_id": conv_id,
            "tool_calls": all_calls,
            "finish_seen": saw_finish,
        }

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

    # ── 上游会话管理（面板「会话管理」用）──────────────
    def list_upstream_sessions(self, max_pages: int = 20) -> list[dict]:
        """拉取该账号在智谱的上游会话列表（分页直到 has_more=False）。

        接口（实测）：
          GET /chatglm/backend-api/assistant/conversation/list
              ?assistant_id=<id>&page=1&page_size=20
          → {"result": {"conversation_list": [...], "has_more": bool}, "status": 0}
        page 与 page_size 均为必填。
        """
        token = self._access()
        headers = _headers_for(self.config, token)
        out: list[dict] = []
        for page in range(1, max(1, int(max_pages)) + 1):
            url = (
                f"{self.config.base_url}/chatglm/backend-api/assistant/conversation/list"
                f"?assistant_id={self.config.assistant_id}&page={page}&page_size=50"
            )
            conn = None
            try:
                conn = self._conn(url, min(self.config.request_timeout, 20))
                conn.request("GET", self._split(url)[1], headers=headers)
                resp = conn.getresponse()
                raw = resp.read()
                payload = json.loads(raw or b"{}")
            except Exception as exc:  # noqa: BLE001
                raise GlmError(f"拉取智谱会话列表失败: {exc}", 502) from exc
            finally:
                # 异常路径也要关连接（此前只在成功路径 close，审计 GLM-P2-9）
                try:
                    if conn is not None:
                        conn.close()
                except Exception:  # noqa: BLE001
                    pass

            result = payload.get("result") or {}
            items = result.get("conversation_list") or []
            out.extend(items)
            if not result.get("has_more") or not items:
                break
        return out

    def purge_upstream_sessions(
        self, *, keep_titles: tuple = (), dry_run: bool = False
    ) -> dict:
        """删除该账号上游全部会话（可保留标题命中的）。

        删除走 /assistant/conversation/delete（一次一个）。
        """
        sessions = self.list_upstream_sessions()
        doomed: list[str] = []
        kept = 0
        for s in sessions:
            title = str(s.get("title") or s.get("name") or "")
            if any(k and k in title for k in keep_titles):
                kept += 1
                continue
            cid = s.get("conversation_id") or s.get("id")
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

        token = self._access()
        for cid in doomed:
            try:
                self._post_json(
                    self.config.delete_url,
                    {
                        "assistant_id": self.config.assistant_id,
                        "conversation_id": cid,
                    },
                    _headers_for(self.config, token),
                    min(self.config.request_timeout, 20),
                )
                res["deleted"] += 1
            except Exception as exc:  # noqa: BLE001
                res["failed"] += 1
                self.logger.warning("删除智谱会话失败 %s: %s", cid[:12], exc)

        for cid in doomed:
            try:
                self._conv_cache.delete_by_session_id(cid)
            except Exception:  # noqa: BLE001
                pass

        self.logger.info(
            "智谱会话清理完成 total=%d deleted=%d failed=%d kept=%d",
            res["total"],
            res["deleted"],
            res["failed"],
            res["kept"],
        )
        return res

    # ── 对外接口 ───────────────────────────────────────
    def _forced_retry(
        self, payload: dict, tools: list, first_reply: str = ""
    ) -> tuple[str, str, list]:
        """tool_choice=required 强校验重试：加强指令再问一次（上限 1 次）。"""
        self.logger.info("GLM tool_choice=required 首轮未调用工具，加强指令重试一次")
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
                "content": tool_bridge.required_nudge(marker=TOOL_MARKER),
            }
        )
        text = reasoning = ""
        native: list = []
        for ev in self._stream_events(retry):
            text, reasoning = ev["text"], ev["reasoning"]
            native = ev.get("tool_calls") or native
        calls = native or _parse_tool_calls(text, tools)
        if not calls:
            raise GlmError(
                "工具调用强校验失败：重试一次后模型仍未按要求调用工具"
                "（tool_choice=required）"
            )
        return text, reasoning, calls

    def chat_completion(self, payload: dict) -> dict:
        model = payload.get("model") or self.config.default_model
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        text = reasoning = ""
        native: list = []
        finish_seen = True
        for ev in self._stream_events(payload):
            text, reasoning = ev["text"], ev["reasoning"]
            native = ev.get("tool_calls") or native
            if ev.get("__done__"):
                finish_seen = bool(ev.get("finish_seen", True))

        # 服务端原生 tool_calls 优先（上游已确定调用意图）；没有再解析正文标记
        tools = payload.get("tools") or []
        calls = native or (_parse_tool_calls(text, tools) if tools else [])
        if not calls and tools and tool_bridge.is_forced_tool_choice(payload):
            text, reasoning, calls = self._forced_retry(payload, tools, first_reply=text)
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
                            "content": (
                                None if native else (_strip_tool_block(text) or None)
                            ),
                            "tool_calls": calls,
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": _usage(payload, text),
            }

        # 整段重复折叠 + 空正文兜底（思考里就是答案时顶上来）
        content, reasoning = reasoning_bridge.finalize_reasoning(
            _strip_tool_block(_dedup_repeated(text)),
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
            "usage": _usage(payload, text),
        }

    def stream_chat_completion(self, payload: dict):
        model = payload.get("model") or self.config.default_model
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        has_tools = bool(payload.get("tools")) and payload.get("tool_choice") != "none"

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

        has_tools = bool(payload.get("tools")) and payload.get("tool_choice") != "none"
        forced = has_tools and tool_bridge.is_forced_tool_choice(payload)

        # tool_choice=required 强校验模式：不直播，解析失败先重试一次，
        # 再失败按明确错误返回（直播一段注定作废的正文没有意义）
        if forced:
            text = reasoning = ""
            native: list = []
            try:
                for ev in self._stream_events(payload):
                    text, reasoning = ev["text"], ev["reasoning"]
                    native = ev.get("tool_calls") or native
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("智谱流式请求失败: %s", exc)
                err = {
                    "error": {"message": f"上游错误: {exc}", "type": "upstream_error"}
                }
                yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            calls = native or _parse_tool_calls(text, payload.get("tools"))
            if not calls:
                try:
                    text, reasoning, calls = self._forced_retry(
                        payload, payload.get("tools") or [], first_reply=text
                    )
                except GlmError as exc:
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
            for i, c in enumerate(calls):
                yield chunk(
                    {
                        "tool_calls": [
                            {
                                "index": i,
                                "id": c["id"],
                                "type": "function",
                                "function": c["function"],
                            }
                        ]
                    }
                )
            yield chunk({}, usage=_usage(payload, text))
            yield chunk({}, finish="tool_calls")
            yield "data: [DONE]\n\n"
            return

        # 有工具时扣住标记文本。链路纪律：**前缀差只对单调全量做**，
        # 扣留器用增量口径（DeltaToolBuffer）——快照口径的 buf 会回扣
        # 尾部使输出非单调，嵌套在前缀差链里会错位咬字（GLM 实测）。
        # fragment_re/open_re：降级标记形态兜底（吃字符/续写重启的碎片不漏）
        buf = (
            tool_bridge.DeltaToolBuffer(
                [TOOL_MARKER, tool_bridge.DSML_TOOL_CALLS_OPEN],
                fragment_re=tool_bridge.MARKER_FRAGMENT_RE,
                open_re=tool_bridge.MARKER_OPEN_LOOSE_RE,
            )
            if has_tools
            else None
        )
        sent_text = ""
        sent_reasoning = ""
        native: list = []
        finish_seen = True
        rs = reasoning_bridge.ReasoningStream(
            legacy=self.config.legacy_reasoning_in_content
        )
        content_streamed = False
        dropped_post_thinking = 0
        try:
            for ev in self._stream_events(payload):
                native = ev.get("tool_calls") or native
                if ev.get("__done__"):
                    finish_seen = bool(ev.get("finish_seen", True))
                d_reason = ev["reasoning"][len(sent_reasoning) :]
                if d_reason:
                    sent_reasoning = ev["reasoning"]
                d_text = ev["text"][len(sent_text) :]
                sent_text = ev["text"]
                d_out = buf.feed_delta(d_text) if buf is not None else d_text
                # GLM 多阶段思考：正文开流后还会冒出后续思考段（答后思考），
                # 透传会让客户端在「答案」后面再渲染思考块（2026-10-28 用户
                # 报障）。答案既已开始，后续思考对 agent 是纯噪音，丢弃。
                if content_streamed and d_reason:
                    dropped_post_thinking += len(d_reason)
                    d_reason = ""
                out_r, out_c = rs.feed(d_reason, d_out)
                if out_r:
                    yield chunk({"reasoning_content": out_r})
                # content_streamed 只由真实正文增量（d_out）置位：legacy 模式
                # 下思考经 rs.feed 包成 <think> 也走 out_c，若据此判定「答案
                # 已开流」，多事件思考的第 2 段起会被当答后思考丢弃
                # （2026-10-30 审计 GLM-P2-6）
                if d_out:
                    content_streamed = True
                if out_c:
                    yield chunk({"content": out_c})
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("智谱流式请求失败: %s", exc)
            err = {"error": {"message": f"上游错误: {exc}", "type": "upstream_error"}}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return
        if dropped_post_thinking:
            self.logger.debug(
                "已丢弃答后思考 %d 字（答案开始后的多阶段思考）",
                dropped_post_thinking,
            )

        if has_tools:
            calls = native or _parse_tool_calls(buf.text, payload.get("tools"))
            if calls:
                # 扣留器在无 opener 时保守扣住末尾 max(opener) 字节；
                # native tool_calls 场景正文里没有标记，这 18 字节是纯正文
                # 尾巴，先放行再发调用（2026-10-30 审计 GLM-2）
                held = buf.flush_prefix()
                if held:
                    yield chunk({"content": held})
                tail = rs.close_think()
                if tail:
                    yield chunk({"content": tail})
                for i, c in enumerate(calls):
                    yield chunk(
                        {
                            "tool_calls": [
                                {
                                    "index": i,
                                    "id": c["id"],
                                    "type": "function",
                                    "function": c["function"],
                                }
                            ]
                        }
                    )
                yield chunk({}, usage=_usage(payload, sent_text))
                yield chunk({}, finish="tool_calls")
                yield "data: [DONE]\n\n"
                return
            # 没解析出工具调用：只放缓冲器扣住的未发尾巴（剥掉标记残迹）
            tail = buf.flush()
            if buf.opener_found:
                # 模型输出了调用标记但没解析成调用（名字不在声明清单 /
                # JSON 残缺）。静默剥掉会造成「模型以为调用了、客户端什么
                # 都没收到」的无头案，留一条带片段的 warning 便于排查。
                self.logger.warning(
                    "GLM 输出了工具调用标记但未解析成调用（工具名不在声明清单"
                    "或 JSON 残缺），已按普通文本剥离。片段: %r",
                    _strip_tool_block(tail)[:200] or tail[:200],
                )
            rel_tail = tool_bridge.scrub_marker_fragments(_strip_tool_block(tail))
        else:
            rel_tail = ""

        if rel_tail:
            out_r, out_c = rs.feed("", rel_tail)
            if out_r:
                yield chunk({"reasoning_content": out_r})
            if out_c:
                yield chunk({"content": out_c})
        out_r, out_c = rs.finish()
        if out_r:
            yield chunk({"reasoning_content": out_r})
        if out_c:
            yield chunk({"content": out_c})

        # 无条件带上 usage（客户端没传 include_usage 也要给，
        # 否则 opencode 等客户端统计不到用量）
        yield chunk({}, usage=_usage(payload, sent_text))
        if not finish_seen:
            self.logger.warning(
                "上游流提前中断（续写未救回），按截断返回 finish=length"
            )
        yield chunk({}, finish="stop" if finish_seen else "length")
        yield "data: [DONE]\n\n"
