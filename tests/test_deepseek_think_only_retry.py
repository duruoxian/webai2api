"""deepseek 光说不做自愈（think-only self-heal）回归。

真实事故（opencode）：deepseek-reasoner 某轮只产出思考——思考里写着
「Let me write the script and run」并起草了工具调用——正文零输出、
无 tool_calls，回合直接结束，用户被迫追问一句才继续。

自愈语义（在镜像兜底之前插一层，宁窄勿宽）：
- 触发（四条同时满足）：has_tools + 本轮无 tool_calls + 客户端可见正文
  为空 + 思考有「工具调用气味」（DSML/哨兵标记的草稿，半角/全角竖线都
  算），且步骤 2.25 的 require_marker 解析提不出完整调用——纯思考型
  回答（无调用气味）不触发，照旧镜像。
- 动作：向**同一上游会话**追加一条 nudge user 消息重试一轮（不是弃
  会话——保留上下文，模型只是忘了发调用）；每个请求最多一次。
- 重试轮仍无调用/正文 → 落回现有镜像兜底（最坏情况=现状）。
- 计费：草稿思考不与重试轮内容双计；镜像兜底保持「镜像后清空
  full_reasoning」语义（DS-P2）。
- 安全线：思考草稿绝不直接提取执行（2.25 require_marker 语义原样）。

桩写法参照 tests/test_deepseek_p0p1.py / test_deepseek_resilience.py。
注意：think-only 的第 1 轮是「正常结束」（有 FINISHED 帧）但只发了
思考——这正是事故形态；ready 事件仍带 response_message_id（真实上游
都有，nudge 的 parent_message_id 指向它）。
"""

from __future__ import annotations

import json
import logging
import threading
from types import SimpleNamespace

from webai2api.adapters.deepseek.services.deepseek_client import (
    THINK_ONLY_NUDGE_PROMPT,
    ConcurrentRequestQueue,
    DeepSeekWebClient,
    _estimate_tokens,
)

D = "｜｜DSML｜｜"  # 全角竖线，DeepSeek 原生 DSML 先验形态

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "跑命令",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]


def _dsml_call(command: str = "ls -la") -> str:
    """完整 DSML 调用（第 2 轮正文里的真调用，检测器可解析）。"""
    return (
        f"<{D} calls>\n"
        f'<{D} invoke name="shell">\n'
        f'<{D} parameter name="command" string="true">{command}</{D} parameter>\n'
        f"</{D} invoke>\n"
        f"</{D} calls>\n"
    )


# 事故草稿形态：思考里起草了哨兵调用但块未闭合（require_marker 提不出
# 完整调用——所以步骤 2.25 不提取，才轮到 think-only 自愈触发）。
DRAFT_SENTINEL = '<|tool_calls|>{"name": "shell", "arguments": {"command": "ls -la"}}'

# 半角 DSML 骨架草稿（无参数无闭合，同样提不出完整调用）。
DRAFT_DSML_SKELETON = '<|DSML| invoke name="shell">'


def _events_think_only(text: str, msg_id: int = 1001) -> list[dict]:
    """第 1 轮事故形态：只发思考、正常 FINISHED（回合直接结束）。"""
    return [
        {"_event_type": "ready", "response_message_id": msg_id},
        {
            "p": "response/fragments",
            "o": "APPEND",
            "v": [{"id": "f1", "type": "THINK", "content": ""}],
        },
        {"v": text},
        {"data": {"p": "response/status", "v": "FINISHED"}},
    ]


def _events_content(text: str, msg_id: int = 1002) -> list[dict]:
    return [
        {"_event_type": "ready", "response_message_id": msg_id},
        {
            "p": "response/fragments",
            "o": "APPEND",
            "v": [{"id": "f2", "type": "RESPONSE", "content": ""}],
        },
        {"v": text},
        {"data": {"p": "response/status", "v": "FINISHED"}},
    ]


def _stub_client(rounds) -> DeepSeekWebClient:
    """rounds: 每次上游调用返回一组事件；同时记录每次开流的 resolved。"""
    if rounds and isinstance(rounds[0], dict):
        rounds = [rounds]  # 单轮便捷写法

    c = DeepSeekWebClient.__new__(DeepSeekWebClient)
    c.config = SimpleNamespace(
        strip_citations=False,
        delete_conversation=False,
        debug_dump_all=False,
        default_model="deepseek-chat",
        request_timeout=10,
        search_enabled=False,
    )
    c.logger = logging.getLogger("ds.thinkonly.test")
    c.request_queue = ConcurrentRequestQueue(
        c.logger, wait_timeout=1, max_concurrency=1
    )
    c._stats_lock = threading.Lock()
    c._stats = {}
    c.session_cache = SimpleNamespace(get=lambda k: None, delete=lambda k: None, size=0)
    c.stored = []  # (text, tool_calls, used_tokens, cache_key, mode)
    c.opened = []  # 每次 _open_chat_stream 收到的 resolved
    c.invalidated = []
    c.resolve_calls = {"n": 0}
    c.upstream_calls = {"n": 0}

    def _resolve(payload):
        c.resolve_calls["n"] += 1
        return {
            "model": "deepseek-chat",
            "ds_payload": {"prompt": "hi"},
            "mode": "fresh",
            "session_id": "sess-1",
        }

    def _open(resolved):
        c.upstream_calls["n"] += 1
        c.opened.append(resolved)
        if c.upstream_calls["n"] > len(rounds):
            raise RuntimeError("no more upstream rounds (stub)")
        return object(), "sess-1"

    def _iter(response):
        return iter(rounds[min(c.upstream_calls["n"], len(rounds)) - 1])

    c._resolve_conversation = _resolve
    c._open_chat_stream = _open
    c._iter_sse_events = _iter
    c._invalidate_and_delete_session = (
        lambda resolved: c.invalidated.append(resolved.get("session_id"))
    )
    c._delete_session = lambda sid: None
    c._store_after_response = (
        lambda resolved, sid, text, msg_id, tool_calls=None, used_tokens=0,
               upstream_total=0: c.stored.append(
            (text, tool_calls, used_tokens, resolved.get("cache_key"),
             resolved.get("mode"))
        )
    )
    return c


def _fail_open_on_round(round_no: int):
    """包装 _open：第 round_no 次开流抛硬错误（模拟 nudge 轮打不开）。"""
    def _wrap(c):
        orig = c._open_chat_stream

        def _open(resolved):
            c.upstream_calls["n"] += 1
            c.opened.append(resolved)
            if c.upstream_calls["n"] >= round_no:
                raise RuntimeError("nudge upstream hard down (stub)")
            return object(), "sess-1"

        c._open_chat_stream = _open

    return _wrap


def _payload(**extra) -> dict:
    base = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "帮我看看目录"}],
    }
    base.update(extra)
    return base


def _parse_stream(chunks_raw: list[str]) -> list[dict]:
    out = []
    for sse in chunks_raw:
        sse = sse.strip()
        if sse.startswith("data: ") and sse != "data: [DONE]":
            out.append(json.loads(sse[6:]))
    return out


def _stream_parts(chunks: list[dict]):
    content = "".join(
        x["choices"][0]["delta"].get("content", "")
        for x in chunks
        if x.get("choices")
    )
    reasoning = "".join(
        x["choices"][0]["delta"].get("reasoning_content", "")
        for x in chunks
        if x.get("choices")
    )
    calls = []
    finish = None
    usage = None
    slots: dict[int, dict] = {}
    for x in chunks:
        if x.get("usage"):
            usage = x["usage"]
        if not x.get("choices"):
            continue
        d = x["choices"][0].get("delta") or {}
        if x["choices"][0].get("finish_reason"):
            finish = x["choices"][0]["finish_reason"]
        for t in d.get("tool_calls") or []:
            slot = slots.setdefault(t["index"], {"name": None, "arguments": ""})
            fn = t.get("function") or {}
            if fn.get("name"):
                slot["name"] = fn["name"]
            slot["arguments"] += fn.get("arguments") or ""
    for idx in sorted(slots):
        calls.append({"name": slots[idx]["name"], "arguments": slots[idx]["arguments"]})
    return content, reasoning, calls, finish, usage


def _assert_nudge_resolved(c, idx: int = 1) -> None:
    """第 idx 次开流必须是 nudge 轮：同会话、parent 指向上一条、一次性
    哨兵 cache_key、nudge 文案进了 prompt。"""
    resolved = c.opened[idx]
    assert resolved["mode"] == "cached", "nudge 轮应复用同一会话（cached），不是弃会话重开"
    assert resolved["session_id"] == "sess-1"
    assert (
        resolved["ds_payload"]["chat_session_id"] == "sess-1"
    ), "nudge 必须追加到同一上游会话"
    assert resolved["ds_payload"]["parent_message_id"] == 1001
    assert resolved["cache_key"].startswith("__think_only_nudge__")
    assert THINK_ONLY_NUDGE_PROMPT in resolved["ds_payload"]["prompt"]


# ═══════════════════════════════════════════════════════════
# 流式
# ═══════════════════════════════════════════════════════════


def test_stream_think_draft_nudge_recovers_tool_calls():
    """场景 1：思考含草稿 + 正文空 → nudge 重试，第 2 轮发来调用 →
    最终下发 tool_calls、无镜像、同会话、不弃会话。"""
    c = _stub_client([_events_think_only(DRAFT_SENTINEL), _events_content(_dsml_call())])
    content, reasoning, calls, finish, usage = _stream_parts(
        _parse_stream(list(c.stream_chat_completion(_payload(tools=TOOLS))))
    )
    assert c.upstream_calls["n"] == 2, "应触发 nudge 重试（第 2 次上游调用）"
    _assert_nudge_resolved(c)
    assert c.resolve_calls["n"] == 1, "nudge 是同会话追加，不是弃会话重解析"
    assert c.invalidated == [], "绝不弃会话"
    assert calls and calls[0]["name"] == "shell"
    assert json.loads(calls[0]["arguments"]) == {"command": "ls -la"}
    assert finish == "tool_calls"
    # 检测器对 DSML 块周边的纯空白残文放行是既有行为；这里断言的语义是
    # 「没有把草稿思考镜像进正文」——正文不得含任何实义字符
    assert content.strip() == "", "重试成功后不得把草稿思考镜像进正文"
    # 用量：草稿思考只计一次（重试轮草稿+思考合计）
    assert c.stored, "最终结果应入库"
    stored_text, stored_calls, used_tokens, cache_key, mode = c.stored[0]
    assert stored_calls, "入库应带 tool_calls"
    assert cache_key is None and mode == "fresh", "入库取回原始 resolved（非哨兵 key）"
    assert used_tokens == _estimate_tokens(DRAFT_SENTINEL + "\n\n") + _estimate_tokens(
        stored_text
    )


def test_stream_retry_still_think_only_falls_back_to_mirror():
    """场景 2：重试轮仍光说不做 → 落回镜像兜底；草稿+重试思考合计
    只镜像/计费一次（full_reasoning 镜像后清空，DS-P2 语义不变）。"""
    r2 = "第二轮还是没有正文"
    c = _stub_client([_events_think_only(DRAFT_SENTINEL), _events_think_only(r2, msg_id=1002)])
    content, reasoning, calls, finish, usage = _stream_parts(
        _parse_stream(list(c.stream_chat_completion(_payload(tools=TOOLS))))
    )
    assert c.upstream_calls["n"] == 2
    _assert_nudge_resolved(c)
    assert calls == []
    assert finish == "stop"
    combined = DRAFT_SENTINEL + "\n\n" + r2
    assert content == combined, "镜像兜底：草稿+重试轮思考合并顶成正文"
    # usage chunk：镜像后 reasoning 清空 → 草稿只计一次
    assert usage is not None
    assert usage["completion_tokens"] == _estimate_tokens(combined)
    stored_text, stored_calls, used_tokens, _ck, _mode = c.stored[0]
    assert stored_text == combined
    assert used_tokens == _estimate_tokens(combined), "镜像后不得把同一段文本算两遍"


def test_stream_no_smell_mirrors_without_retry():
    """场景 3：reasoning 无调用气味（纯思考型回答）→ 不重试，直接镜像。"""
    c = _stub_client([_events_think_only("答案就在思考里，不需要工具")])
    content, reasoning, calls, finish, usage = _stream_parts(
        _parse_stream(list(c.stream_chat_completion(_payload(tools=TOOLS))))
    )
    assert c.upstream_calls["n"] == 1, "无调用气味绝不触发 nudge"
    assert content == "答案就在思考里，不需要工具"
    assert calls == []
    assert finish == "stop"


def test_stream_no_tools_no_nudge():
    """场景 4：请求未声明 tools → 不触发（哪怕思考里有草稿气味）。"""
    c = _stub_client([_events_think_only(DRAFT_SENTINEL)])
    content, reasoning, calls, finish, usage = _stream_parts(
        _parse_stream(list(c.stream_chat_completion(_payload())))
    )
    assert c.upstream_calls["n"] == 1, "未声明 tools 不触发自愈"
    assert calls == []
    assert content == DRAFT_SENTINEL, "照旧镜像兜底"
    assert finish == "stop"


def test_stream_self_heal_at_most_once():
    """场景 5：自愈最多一次——重试轮再犯也不再 nudge（不循环）。"""
    c = _stub_client([_events_think_only(DRAFT_SENTINEL), _events_think_only(DRAFT_SENTINEL, msg_id=1002)])
    content, reasoning, calls, finish, usage = _stream_parts(
        _parse_stream(list(c.stream_chat_completion(_payload(tools=TOOLS))))
    )
    assert c.upstream_calls["n"] == 2, "第 2 轮再犯不得发起第 2 次 nudge"
    assert c._stats.get("think_only_retry") == 1
    combined = DRAFT_SENTINEL + "\n\n" + DRAFT_SENTINEL
    assert content == combined
    assert finish == "stop"


def test_stream_nudge_open_failure_falls_back_to_mirror():
    """nudge 轮打不开（上游硬错误）→ 就地落回镜像兜底，绝不把本可
    镜像的回答变成报错。"""
    c = _stub_client([_events_think_only(DRAFT_SENTINEL), _events_content("x")])
    _fail_open_on_round(2)(c)
    content, reasoning, calls, finish, usage = _stream_parts(
        _parse_stream(list(c.stream_chat_completion(_payload(tools=TOOLS))))
    )
    assert c.upstream_calls["n"] == 2
    assert calls == []
    assert content == DRAFT_SENTINEL, "开流失败按原样镜像（无接缝空行）"
    assert finish == "stop"
    errors = [x for x in _parse_stream([])]  # 占位保持风格一致
    assert not errors
    stored_text, _tc, used_tokens, _ck, _mode = c.stored[0]
    assert stored_text == DRAFT_SENTINEL
    assert used_tokens == _estimate_tokens(DRAFT_SENTINEL)


def test_stream_dsml_skeleton_draft_also_triggers():
    """半角 DSML 骨架草稿（无参数无闭合）同样是「调用气味」，触发自愈。"""
    c = _stub_client([_events_think_only(DRAFT_DSML_SKELETON), _events_content(_dsml_call())])
    content, reasoning, calls, finish, usage = _stream_parts(
        _parse_stream(list(c.stream_chat_completion(_payload(tools=TOOLS))))
    )
    assert c.upstream_calls["n"] == 2
    assert calls and calls[0]["name"] == "shell"
    assert finish == "tool_calls"
    assert content.strip() == "", "无镜像正文（检测器空白残文除外）"


# ═══════════════════════════════════════════════════════════
# 非流式
# ═══════════════════════════════════════════════════════════


def test_non_stream_think_draft_nudge_recovers_tool_calls():
    """非流式场景 1：nudge 重试拿到真调用；入库/结果取回原始 resolved。"""
    c = _stub_client([_events_think_only(DRAFT_SENTINEL), _events_content(_dsml_call())])
    out = c.chat_completion(_payload(tools=TOOLS))
    msg = out["choices"][0]["message"]
    assert c.upstream_calls["n"] == 2
    _assert_nudge_resolved(c)
    assert c.resolve_calls["n"] == 1
    assert msg["tool_calls"][0]["function"]["name"] == "shell"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {
        "command": "ls -la"
    }
    assert msg["content"] is None, "重试成功后无镜像正文"
    assert out["choices"][0]["finish_reason"] == "tool_calls"
    stored_text, stored_calls, used_tokens, cache_key, mode = c.stored[0]
    assert stored_calls, "入库应带 tool_calls"
    assert cache_key is None and mode == "fresh", "最终入库取回原始 resolved"
    # clean_text 为空 → used_tokens 全部来自草稿+重试思考，且只计一次
    assert used_tokens == _estimate_tokens(DRAFT_SENTINEL + "\n\n")


def test_non_stream_retry_still_think_only_falls_back_to_mirror():
    """非流式场景 2：重试轮仍空 → 镜像兜底；镜像后 reasoning_content
    清空（客户端不会在思考与正文里看到重复的同一份答案）。"""
    r2 = "非流式第二轮也没正文"
    c = _stub_client([_events_think_only(DRAFT_SENTINEL), _events_think_only(r2, msg_id=1002)])
    out = c.chat_completion(_payload(tools=TOOLS))
    msg = out["choices"][0]["message"]
    assert c.upstream_calls["n"] == 2
    combined = DRAFT_SENTINEL + "\n\n" + r2
    assert msg["content"] == combined
    assert not msg.get("reasoning_content"), "镜像后 full_reasoning 必须清空（DS-P2）"
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"]["completion_tokens"] == _estimate_tokens(combined)
    stored_text, _tc, used_tokens, _ck, _mode = c.stored[0]
    assert stored_text == combined
    assert used_tokens == _estimate_tokens(combined), "草稿只计一次"


def test_non_stream_no_smell_mirrors_without_retry():
    """非流式场景 3：无调用气味 → 不重试直接镜像。"""
    c = _stub_client([_events_think_only("纯思考型回答，无需调用工具")])
    out = c.chat_completion(_payload(tools=TOOLS))
    msg = out["choices"][0]["message"]
    assert c.upstream_calls["n"] == 1
    assert msg["content"] == "纯思考型回答，无需调用工具"
    assert not msg.get("tool_calls")


def test_non_stream_no_tools_no_nudge():
    """非流式场景 4：未声明 tools → 不触发。"""
    c = _stub_client([_events_think_only(DRAFT_SENTINEL)])
    out = c.chat_completion(_payload())
    msg = out["choices"][0]["message"]
    assert c.upstream_calls["n"] == 1
    assert msg["content"] == DRAFT_SENTINEL
    assert not msg.get("tool_calls")


def test_non_stream_self_heal_at_most_once():
    """非流式场景 5：自愈最多一次，重试轮再犯直接镜像。"""
    c = _stub_client([_events_think_only(DRAFT_SENTINEL), _events_think_only(DRAFT_SENTINEL, msg_id=1002)])
    out = c.chat_completion(_payload(tools=TOOLS))
    assert c.upstream_calls["n"] == 2, "不得发起第 2 次 nudge"
    assert c._stats.get("think_only_retry") == 1
    combined = DRAFT_SENTINEL + "\n\n" + DRAFT_SENTINEL
    assert out["choices"][0]["message"]["content"] == combined
    assert out["choices"][0]["finish_reason"] == "stop"


def test_non_stream_nudge_open_failure_falls_back_to_mirror():
    """非流式：nudge 轮打不开 → 就地镜像兜底（最坏情况=现状）。"""
    c = _stub_client([_events_think_only(DRAFT_SENTINEL), _events_content("x")])
    _fail_open_on_round(2)(c)
    out = c.chat_completion(_payload(tools=TOOLS))
    msg = out["choices"][0]["message"]
    assert c.upstream_calls["n"] == 2
    assert msg["content"] == DRAFT_SENTINEL
    assert not msg.get("reasoning_content")
    assert out["choices"][0]["finish_reason"] == "stop"


# ═══════════════════════════════════════════════════════════
# 气味检测 / nudge 文案单元
# ═══════════════════════════════════════════════════════════


def test_smell_regex_half_and_full_width_variants():
    from webai2api.adapters.deepseek.services.deepseek_client import (
        _reasoning_has_tool_call_draft,
    )

    positives = [
        "<|tool_calls|>",          # 半角哨兵
        "<|tool_call|>",           # 半角单数哨兵
        "<|/tool_calls|>",         # 半角闭合
        "<｜tool_calls｜>",        # 全角竖线哨兵
        "<｜DSML｜ calls>",        # 全角 DSML 开块
        '<|DSML| invoke name="x">',  # 半角 DSML invoke
        "</｜DSML｜ invoke>",      # 全角 DSML 闭合
        "｜DSML｜parameter",       # 无尖括号的标记碎片
        f"<{D} calls>",            # 双全角竖线原生形态
    ]
    for s in positives:
        assert _reasoning_has_tool_call_draft(s), f"应识别为草稿气味: {s!r}"
    negatives = [
        "",
        "我会调用 shell 工具来查看目录",
        "先写脚本再运行它",
        "tool_call 而已，没有任何标记",
        "Let me write the script and run it",
    ]
    for s in negatives:
        assert not _reasoning_has_tool_call_draft(s), f"普通思考不应触发: {s!r}"


def test_nudge_prompt_covers_required_points():
    assert "没有输出正文" in THINK_ONLY_NUDGE_PROMPT
    assert "没有发出任何工具调用" in THINK_ONLY_NUDGE_PROMPT
    assert "工具调用" in THINK_ONLY_NUDGE_PROMPT
    assert "正文回答" in THINK_ONLY_NUDGE_PROMPT
    assert "不要只输出思考" in THINK_ONLY_NUDGE_PROMPT
