"""deepseek 流式韧性测试。

背景（2026-10-27 诊断）：DeepSeek 网页版上游会在长思考/大上下文时中途
掐流，且 reasoner 常把答案整个写进思考块。此前流式收尾既不检测
「没收到 finish 帧就断流」，也没有「正文为空把思考顶上来」的兜底
（Kimi/GLM 都有），用户侧表现为：回复中断、全部/部分藏在思考里、
工具参数 JSON 流一半被掐成残缺。

这里用桩事件直接驱动 DeepSeekWebClient 的流式/非流式生成器，
验证三条新防线：
1. 上游没发 finish 帧 → finish_reason="length"（不再装作正常 stop）；
2. 正文一个字没发、思考有货 → 思考顶成正文下发；
3. 正常带 finish 的流行为完全不变。
"""
from __future__ import annotations

import json
import logging
import threading
from types import SimpleNamespace

from webai2api.adapters.deepseek.services import tool_calling as tc
from webai2api.adapters.deepseek.services.deepseek_client import (
    ConcurrentRequestQueue,
    DeepSeekWebClient,
)


def _events_reasoning_only_no_finish():
    """思考片段 + 掐流：只发 reasoning，永远等不到 finish 帧。"""
    return [
        {"_event_type": "ready"},
        {"p": "response/fragments", "o": "APPEND",
         "v": [{"id": "f1", "type": "THINK", "content": ""}]},
        {"v": "用户问1+1，"},
        {"v": "答案是2。我要只输出数字。"},
        # —— 上游在此断流，没有 response/status=FINISHED ——
    ]


def _events_normal():
    """正常流：思考 + 正文 + finish 帧。"""
    return [
        {"_event_type": "ready"},
        {"p": "response/fragments", "o": "APPEND",
         "v": [{"id": "f1", "type": "THINK", "content": ""}]},
        {"v": "想一下"},
        {"p": "response/fragments", "o": "APPEND",
         "v": [{"id": "f2", "type": "RESPONSE", "content": ""}]},
        {"v": "正文答案"},
        {"data": {"p": "response/status", "v": "FINISHED"}},
    ]


def _events_truncated_mid_content():
    """正文发一半被掐：有 reasoning 有半截 content，无 finish。"""
    return [
        {"_event_type": "ready"},
        {"p": "response/fragments", "o": "APPEND",
         "v": [{"id": "f1", "type": "THINK", "content": ""}]},
        {"v": "思考若干"},
        {"p": "response/fragments", "o": "APPEND",
         "v": [{"id": "f2", "type": "RESPONSE", "content": ""}]},
        {"v": "正文开头"},
        # —— 掐流 ——
    ]


def _stub_client(rounds) -> DeepSeekWebClient:
    """rounds: 每次上游调用返回一组事件（断流续写测试用多组）。"""
    if rounds and isinstance(rounds[0], dict):
        rounds = [rounds]          # 单轮便捷写法
    state = {"n": 0}

    c = DeepSeekWebClient.__new__(DeepSeekWebClient)
    c.config = SimpleNamespace(
        strip_citations=False, delete_conversation=False,
        debug_dump_all=False, default_model="deepseek-chat",
        request_timeout=10, search_enabled=False,
    )
    c.logger = logging.getLogger("ds.resilience.test")
    c.request_queue = ConcurrentRequestQueue(c.logger, wait_timeout=1, max_concurrency=1)
    c._stats_lock = threading.Lock()
    c._stats = {}
    c.session_cache = SimpleNamespace(
        get=lambda k: None, delete=lambda k: None, size=0)
    c.stored = []
    c.upstream_calls = {"n": 0}

    def _open(resolved):
        c.upstream_calls["n"] += 1
        if c.upstream_calls["n"] > len(rounds):
            raise RuntimeError("no more upstream rounds (stub)")
        return object(), "sess-1"

    def _iter(response):
        return iter(rounds[min(c.upstream_calls["n"], len(rounds)) - 1])

    c._resolve_conversation = lambda payload: {
        "model": "deepseek-chat", "ds_payload": {"prompt": "hi"},
        "mode": "fresh", "session_id": "sess-1",
    }
    c._open_chat_stream = _open
    c._iter_sse_events = _iter
    c._store_after_response = (
        lambda resolved, sid, text, msg_id, tool_calls=None:
        c.stored.append((text, tool_calls))
    )
    return c


def _payload() -> dict:
    return {"model": "deepseek-chat",
            "messages": [{"role": "user", "content": "1+1等于几"}]}


def _parse_stream(chunks_raw: list[str]) -> list[dict]:
    out = []
    for sse in chunks_raw:
        sse = sse.strip()
        if sse.startswith("data: ") and sse != "data: [DONE]":
            out.append(json.loads(sse[6:]))
    return out


def test_stream_truncated_promotes_reasoning_and_marks_length():
    c = _stub_client([_events_reasoning_only_no_finish()])
    chunks = _parse_stream(list(c.stream_chat_completion(_payload())))
    contents = [x["choices"][0]["delta"].get("content", "")
                for x in chunks if x.get("choices")]
    assert "".join(contents) == "用户问1+1，答案是2。我要只输出数字。"
    finishes = [x["choices"][0].get("finish_reason") for x in chunks
                if x.get("choices") and x["choices"][0].get("finish_reason")]
    assert finishes == ["length"], f"截断流应返回 length，实际 {finishes}"
    # 兜底顶上来的正文也要进会话存储（多轮上下文一致）
    assert c.stored and c.stored[0][0] == "用户问1+1，答案是2。我要只输出数字。"


def test_stream_normal_finish_unchanged():
    c = _stub_client([_events_normal()])
    chunks = _parse_stream(list(c.stream_chat_completion(_payload())))
    contents = "".join(x["choices"][0]["delta"].get("content", "")
                       for x in chunks if x.get("choices"))
    assert contents == "正文答案"
    finishes = [x["choices"][0].get("finish_reason") for x in chunks
                if x.get("choices") and x["choices"][0].get("finish_reason")]
    assert finishes == ["stop"]


def test_stream_truncated_mid_content_keeps_partial_text_and_marks_length():
    c = _stub_client([_events_truncated_mid_content()])
    chunks = _parse_stream(list(c.stream_chat_completion(_payload())))
    contents = "".join(x["choices"][0]["delta"].get("content", "")
                       for x in chunks if x.get("choices"))
    assert "正文开头" in contents
    finishes = [x["choices"][0].get("finish_reason") for x in chunks
                if x.get("choices") and x["choices"][0].get("finish_reason")]
    assert finishes == ["length"]
    # 有正文时不应把思考再重复塞进正文
    assert "思考若干" not in contents


def test_non_stream_truncated_promotes_and_marks_length():
    c = _stub_client([_events_reasoning_only_no_finish()])
    out = c.chat_completion(_payload())
    msg = out["choices"][0]["message"]
    assert msg["content"] == "用户问1+1，答案是2。我要只输出数字。"
    assert out["choices"][0]["finish_reason"] == "length"
    # 对齐 Kimi/GLM 的 finalize_reasoning 语义：顶成正文后 reasoning 清空，
    # 客户端不会在思考与正文里看到重复的同一份答案
    assert not msg.get("reasoning_content")


def test_non_stream_normal_stop():
    c = _stub_client([_events_normal()])
    out = c.chat_completion(_payload())
    msg = out["choices"][0]["message"]
    assert msg["content"] == "正文答案"
    assert out["choices"][0]["finish_reason"] == "stop"


def test_truncated_partial_dsml_does_not_emit_broken_tool_call():
    """工具参数流到一半被掐：残缺 DSML 被丢弃，绝不给客户端半个调用。"""
    events = [
        {"_event_type": "ready"},
        {"p": "response/fragments", "o": "APPEND",
         "v": [{"id": "f1", "type": "RESPONSE", "content": ""}]},
        {"v": '<|tool_call|>{"name": "write", "arguments": {"file'},
        # —— 掐流：DSML 未闭合 ——
    ]
    c = _stub_client(events)
    chunks = _parse_stream(list(c.stream_chat_completion(
        {**_payload(), "tools": [{"type": "function", "function": {
            "name": "write", "parameters": {"type": "object",
            "properties": {"file": {"type": "string"}}}}}]},
    )))
    tool_chunks = [x for x in chunks if x.get("choices")
                   and x["choices"][0]["delta"].get("tool_calls")]
    assert tool_chunks == []
    finishes = [x["choices"][0].get("finish_reason") for x in chunks
                if x.get("choices") and x["choices"][0].get("finish_reason")]
    assert finishes == ["length"]


# ── 断流自动续写 ────────────────────────────────────
def _truncated_with_msgid():
    """第 1 轮：正文发一半 + ready 带消息 id，无 finish → 触发续写。"""
    return [
        {"_event_type": "ready", "response_message_id": 1001},
        {"p": "response/fragments", "o": "APPEND",
         "v": [{"id": "f1", "type": "RESPONSE", "content": ""}]},
        {"v": "回答开头，"},
        # —— 掐流 ——
    ]


def _continuation_completes():
    """第 2 轮（续写）：把剩余正文说完并正常 finish。"""
    return [
        {"_event_type": "ready", "response_message_id": 1002},
        {"p": "response/fragments", "o": "APPEND",
         "v": [{"id": "f2", "type": "RESPONSE", "content": ""}]},
        {"v": "答案的剩余部分"},
        {"data": {"p": "response/status", "v": "FINISHED"}},
    ]


def test_stream_truncated_auto_continues_to_complete_answer():
    """断流自动续写：续写增量拼进同一 SSE，客户端看到一条完整回答，
    finish 回到 stop（不再是 length）。"""
    c = _stub_client([_truncated_with_msgid(), _continuation_completes()])
    chunks = _parse_stream(list(c.stream_chat_completion(_payload())))
    contents = "".join(x["choices"][0]["delta"].get("content", "")
                       for x in chunks if x.get("choices"))
    assert contents == "回答开头，答案的剩余部分"
    finishes = [x["choices"][0].get("finish_reason") for x in chunks
                if x.get("choices") and x["choices"][0].get("finish_reason")]
    assert finishes == ["stop"], f"续写成功应返回 stop，实际 {finishes}"
    # 存储的是合并后的完整正文（下一轮前缀哈希才能匹配客户端历史）
    assert c.stored and c.stored[0][0] == "回答开头，答案的剩余部分"
    assert c.upstream_calls["n"] == 2


def test_stream_truncated_no_msgid_falls_back_to_length():
    """上游连 ready（消息 id）都没发就断流：无法续写，退回 length。"""
    c = _stub_client([_truncated_with_msgid(), _continuation_completes()])
    # 挖掉 ready 的 response_message_id → 无法续写
    c_rounds = [_truncated_with_msgid()]
    for ev in c_rounds[0]:
        ev.pop("response_message_id", None)
    c = _stub_client(c_rounds)
    chunks = _parse_stream(list(c.stream_chat_completion(_payload())))
    finishes = [x["choices"][0].get("finish_reason") for x in chunks
                if x.get("choices") and x["choices"][0].get("finish_reason")]
    assert finishes == ["length"]
    assert c.upstream_calls["n"] == 1


def test_stream_truncated_continuation_fails_gracefully():
    """续写请求失败：按截断收场（length），绝不把已输出的流变成报错。"""
    c = _stub_client([_truncated_with_msgid(), _continuation_completes()])
    # 第二轮 _open 直接抛错
    orig_open = c._open_chat_stream

    def _open_fail(resolved):
        c.upstream_calls["n"] += 1
        if c.upstream_calls["n"] >= 2:
            raise RuntimeError("continuation upstream down")
        return object(), "sess-1"
    c._open_chat_stream = _open_fail
    chunks = _parse_stream(list(c.stream_chat_completion(_payload())))
    contents = "".join(x["choices"][0]["delta"].get("content", "")
                       for x in chunks if x.get("choices"))
    assert "回答开头，" in contents
    finishes = [x["choices"][0].get("finish_reason") for x in chunks
                if x.get("choices") and x["choices"][0].get("finish_reason")]
    assert finishes == ["length"]
    errors = [x for x in chunks if x.get("error")]
    assert errors == [], "续写失败不能变成 SSE error 事件"


def test_stream_truncated_mid_tool_call_recovered_by_continuation():
    """工具标记流到一半被掐：残块丢弃，续写轮重发完整调用 → 照常拿到
    完整 tool_calls（此前这种断流=客户端收到残缺 JSON/直接丢调用）。"""
    D = tc._DSML  # DeepSeek 真实 DSML 标记（全角竖线）
    r1 = [
        {"_event_type": "ready", "response_message_id": 2001},
        {"p": "response/fragments", "o": "APPEND",
         "v": [{"id": "f1", "type": "RESPONSE", "content": ""}]},
        {"v": f'<{D} calls><{D} invoke name="write">'
              f'<{D} parameter name="filePath" string="true">a'},
        # —— 掐流：DSML 半截 ——
    ]
    r2 = [
        {"_event_type": "ready", "response_message_id": 2002},
        {"p": "response/fragments", "o": "APPEND",
         "v": [{"id": "f2", "type": "RESPONSE", "content": ""}]},
        {"v": f'<{D} calls><{D} invoke name="write">'
              f'<{D} parameter name="filePath" string="true">a.txt</{D} parameter>'
              f'</{D} invoke></{D} calls>'},
        {"data": {"p": "response/status", "v": "FINISHED"}},
    ]
    tools = [{"type": "function", "function": {
        "name": "write", "parameters": {"type": "object",
        "properties": {"filePath": {"type": "string"}}}}}]
    c = _stub_client([r1, r2])
    chunks = _parse_stream(list(c.stream_chat_completion(
        {**_payload(), "tools": tools})))
    calls = []
    for x in chunks:
        if x.get("choices"):
            tcd = x["choices"][0]["delta"].get("tool_calls")
            if tcd:
                calls.extend(tcd)
    assert calls, "续写后应拿到完整 tool_calls"
    head = calls[0]
    assert head["function"]["name"] == "write"
    finishes = [x["choices"][0].get("finish_reason") for x in chunks
                if x.get("choices") and x["choices"][0].get("finish_reason")]
    assert finishes == ["tool_calls"]


def test_non_stream_truncated_auto_continues():
    c = _stub_client([_truncated_with_msgid(), _continuation_completes()])
    out = c.chat_completion(_payload())
    msg = out["choices"][0]["message"]
    assert msg["content"] == "回答开头，答案的剩余部分"
    assert out["choices"][0]["finish_reason"] == "stop"


def test_non_stream_truncated_partial_tool_markup_dropped_before_continue():
    """非流式：残缺 DSML 在续写前被清场，续写轮的完整块正常解析。"""
    D = tc._DSML
    r1 = [
        {"_event_type": "ready", "response_message_id": 3001},
        {"p": "response/fragments", "o": "APPEND",
         "v": [{"id": "f1", "type": "RESPONSE", "content": ""}]},
        {"v": f'<{D} calls><{D} invoke name="write">'
              f'<{D} parameter name="filePath" string="true">a'},
    ]
    r2 = [
        {"_event_type": "ready", "response_message_id": 3002},
        {"p": "response/fragments", "o": "APPEND",
         "v": [{"id": "f2", "type": "RESPONSE", "content": ""}]},
        {"v": f'<{D} calls><{D} invoke name="write">'
              f'<{D} parameter name="filePath" string="true">a.txt</{D} parameter>'
              f'</{D} invoke></{D} calls>'},
        {"data": {"p": "response/status", "v": "FINISHED"}},
    ]
    tools = [{"type": "function", "function": {
        "name": "write", "parameters": {"type": "object",
        "properties": {"filePath": {"type": "string"}}}}}]
    c = _stub_client([r1, r2])
    out = c.chat_completion({**_payload(), "tools": tools})
    calls = out["choices"][0]["message"].get("tool_calls") or []
    assert calls and calls[0]["function"]["name"] == "write"
    assert out["choices"][0]["finish_reason"] == "tool_calls"
