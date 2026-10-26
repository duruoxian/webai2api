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
from types import SimpleNamespace

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


def _stub_client(events) -> DeepSeekWebClient:
    c = DeepSeekWebClient.__new__(DeepSeekWebClient)
    c.config = SimpleNamespace(
        strip_citations=False, delete_conversation=False,
        debug_dump_all=False, default_model="deepseek-chat",
        request_timeout=10,
    )
    c.logger = logging.getLogger("ds.resilience.test")
    c.request_queue = ConcurrentRequestQueue(c.logger, wait_timeout=1, max_concurrency=1)
    c.stored = []
    c._resolve_conversation = lambda payload: {
        "model": "deepseek-chat", "ds_payload": {"prompt": "hi"},
        "mode": "fresh", "session_id": "sess-1",
    }
    c._open_chat_stream = lambda resolved: (object(), "sess-1")
    c._iter_sse_events = lambda response: iter(events)
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
    c = _stub_client(_events_reasoning_only_no_finish())
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
    c = _stub_client(_events_normal())
    chunks = _parse_stream(list(c.stream_chat_completion(_payload())))
    contents = "".join(x["choices"][0]["delta"].get("content", "")
                       for x in chunks if x.get("choices"))
    assert contents == "正文答案"
    finishes = [x["choices"][0].get("finish_reason") for x in chunks
                if x.get("choices") and x["choices"][0].get("finish_reason")]
    assert finishes == ["stop"]


def test_stream_truncated_mid_content_keeps_partial_text_and_marks_length():
    c = _stub_client(_events_truncated_mid_content())
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
    c = _stub_client(_events_reasoning_only_no_finish())
    out = c.chat_completion(_payload())
    msg = out["choices"][0]["message"]
    assert msg["content"] == "用户问1+1，答案是2。我要只输出数字。"
    assert out["choices"][0]["finish_reason"] == "length"
    # 对齐 Kimi/GLM 的 finalize_reasoning 语义：顶成正文后 reasoning 清空，
    # 客户端不会在思考与正文里看到重复的同一份答案
    assert not msg.get("reasoning_content")


def test_non_stream_normal_stop():
    c = _stub_client(_events_normal())
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
