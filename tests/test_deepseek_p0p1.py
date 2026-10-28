"""deepseek P0/P1 修复回归（2026-10-29，针对 museAI.txt 长文核对的真缺口）。

- P0-1 空响应重试：上游正常结束但正文/思考/调用全空 → 弃会话重试一次；
  骨架 DSML 被丢弃导致客户端一无所获的轮次同样命中。
- P0-2 思考双通道：正文可见输出为空时才扫 reasoning 里的完整 DSML 块
  （收紧版——思考里的草稿调用绝不提取，防 shell 级误执行）。
- P1-3 required 强校验：tool_choice=required 拿到纯文本 → 加强指令重试
  一次，仍不行明确报错；流式走「非流式结果翻制 SSE」形态，不直播。
- P1-4 续写接缝重叠剪除：只裁「已发尾巴的逐字后缀」，数学上不丢新内容。
- 非流式 DSML 残留剥离：解析不出调用的完整块不漏进 content。
"""

from __future__ import annotations

import json
import logging
import threading
from types import SimpleNamespace

import pytest

from webai2api.adapters.deepseek.services import tool_calling as tc
from webai2api.adapters.deepseek.services.deepseek_client import (
    ConcurrentRequestQueue,
    DeepSeekWebClient,
    UpstreamAPIError,
    _SeamTrimmer,
)
from webai2api.adapters.deepseek.services.tool_calling import (
    build_required_nudge,
    strip_dsml_residue,
)

D = "｜｜DSML｜｜"  # 全角竖线，DeepSeek 原生先验形态

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
    return (
        f"<{D} calls>\n"
        f'<{D} invoke name="shell">\n'
        f'<{D} parameter name="command" string="true">{command}</{D} parameter>\n'
        f"</{D} invoke>\n"
        f"</{D} calls>\n"
    )


def _dsml_skeleton() -> str:
    """骨架调用：缺必填参数 command（v0.1.43 起被丢弃的形态）。"""
    return (
        f"<{D} calls>\n"
        f'<{D} invoke name="shell">\n'
        f"</{D} invoke>\n"
        f"</{D} calls>\n"
    )


def _events_empty_finish():
    return [{"_event_type": "ready"}, {"data": {"p": "response/status", "v": "FINISHED"}}]


def _events_content(text: str):
    return [
        {"_event_type": "ready"},
        {
            "p": "response/fragments",
            "o": "APPEND",
            "v": [{"id": "f1", "type": "RESPONSE", "content": ""}],
        },
        {"v": text},
        {"data": {"p": "response/status", "v": "FINISHED"}},
    ]


def _events_reasoning(text: str):
    return [
        {"_event_type": "ready"},
        {
            "p": "response/fragments",
            "o": "APPEND",
            "v": [{"id": "f1", "type": "THINK", "content": ""}],
        },
        {"v": text},
        {"data": {"p": "response/status", "v": "FINISHED"}},
    ]


def _stub_client(rounds):
    if rounds and isinstance(rounds[0], dict):
        rounds = [rounds]
    state = {"n": 0}
    c = DeepSeekWebClient.__new__(DeepSeekWebClient)
    c.config = SimpleNamespace(
        strip_citations=False,
        delete_conversation=False,
        debug_dump_all=False,
        default_model="deepseek-chat",
        request_timeout=10,
        search_enabled=False,
    )
    c.logger = logging.getLogger("ds.p0p1.test")
    c.request_queue = ConcurrentRequestQueue(c.logger, wait_timeout=1, max_concurrency=1)
    c._stats_lock = threading.Lock()
    c._stats = {}
    c.session_cache = SimpleNamespace(get=lambda k: None, delete=lambda k: None, size=0)
    c.stored = []
    c.invalidated = []
    c.upstream_calls = {"n": 0}

    def _open(resolved):
        c.upstream_calls["n"] += 1
        if c.upstream_calls["n"] > len(rounds):
            raise RuntimeError("no more upstream rounds (stub)")
        return object(), f"sess-{c.upstream_calls['n']}"

    def _iter(response):
        return iter(rounds[min(c.upstream_calls["n"], len(rounds)) - 1])

    def _resolve(payload):
        return {
            "model": "deepseek-chat",
            "ds_payload": {"prompt": "hi"},
            "mode": "fresh",
            "session_id": f"sess-{c.upstream_calls['n'] + 1}",
        }

    c._resolve_conversation = _resolve
    c._open_chat_stream = _open
    c._iter_sse_events = _iter
    c._invalidate_and_delete_session = (
        lambda resolved: c.invalidated.append(resolved.get("session_id"))
    )
    c._delete_session = lambda sid: None
    c._store_after_response = (
        lambda resolved, sid, text, msg_id, tool_calls=None, used_tokens=0,
               upstream_total=0: c.stored.append((text, tool_calls))
    )
    return c


def _payload(**extra) -> dict:
    base = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "1+1等于几"}],
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
    calls = []
    finish = None
    slots: dict[int, dict] = {}
    for x in chunks:
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
    return content, calls, finish


# ── P0-1 空响应重试 ──────────────────────────────────────────────


def test_stream_empty_response_retried_with_fresh_session():
    c = _stub_client([_events_empty_finish(), _events_content("重试后的答案")])
    content, calls, finish = _stream_parts(
        _parse_stream(list(c.stream_chat_completion(_payload())))
    )
    assert content == "重试后的答案"
    assert finish == "stop"
    assert c.upstream_calls["n"] == 2
    assert len(c.invalidated) == 1  # 弃掉了第一次的空会话


def test_stream_empty_both_rounds_returns_empty_without_error():
    c = _stub_client([_events_empty_finish(), _events_empty_finish()])
    content, calls, finish = _stream_parts(
        _parse_stream(list(c.stream_chat_completion(_payload())))
    )
    assert content == ""
    assert finish == "stop"
    assert c.upstream_calls["n"] == 2  # 只重试一次


def test_stream_skeleton_dsml_triggers_retry_and_recovers():
    """骨架调用被丢弃 → 客户端一无所获 → 弃会话重试拿到完整调用。"""
    c = _stub_client([_events_content(_dsml_skeleton()), _events_content(_dsml_call())])
    content, calls, finish = _stream_parts(
        _parse_stream(list(c.stream_chat_completion(_payload(tools=TOOLS))))
    )
    assert calls and calls[0]["name"] == "shell"
    assert json.loads(calls[0]["arguments"]) == {"command": "ls -la"}
    assert finish == "tool_calls"
    assert c.upstream_calls["n"] == 2
    assert "DSML" not in content  # 残块不漏进正文


def test_non_stream_empty_response_retried():
    c = _stub_client([_events_empty_finish(), _events_content("非流式重试答案")])
    out = c.chat_completion(_payload())
    assert out["choices"][0]["message"]["content"] == "非流式重试答案"
    assert c.upstream_calls["n"] == 2


def test_non_stream_skeleton_dsml_retried_not_leaked():
    """骨架 DSML 剥离后为空 → 重试；残留绝不以原文漏进 content。"""
    c = _stub_client([_events_content(_dsml_skeleton()), _events_content("好的")])
    out = c.chat_completion(_payload(tools=TOOLS))
    msg = out["choices"][0]["message"]
    assert msg["content"] == "好的"
    assert "DSML" not in (msg["content"] or "")
    assert c.upstream_calls["n"] == 2


# ── P0-2 思考双通道 ──────────────────────────────────────────────


def test_stream_tool_call_hidden_in_reasoning_extracted():
    """正文空、思考里有完整 DSML → 提取成 tool_calls（不再卡死）。"""
    c = _stub_client([_events_reasoning(f"先想一下。\n{_dsml_call()}")])
    content, calls, finish = _stream_parts(
        _parse_stream(list(c.stream_chat_completion(_payload(tools=TOOLS))))
    )
    assert calls and calls[0]["name"] == "shell"
    assert json.loads(calls[0]["arguments"]) == {"command": "ls -la"}
    assert finish == "tool_calls"
    assert c.upstream_calls["n"] == 1  # 提取成功，不触发空响应重试


def test_non_stream_tool_call_hidden_in_reasoning_extracted():
    c = _stub_client([_events_reasoning(f"思考。\n{_dsml_call()}")])
    out = c.chat_completion(_payload(tools=TOOLS))
    msg = out["choices"][0]["message"]
    assert msg["content"] is None
    tcalls = msg["tool_calls"]
    assert tcalls[0]["function"]["name"] == "shell"
    assert json.loads(tcalls[0]["function"]["arguments"]) == {"command": "ls -la"}


def test_reasoning_without_complete_block_still_promoted():
    """思考里没有完整调用块 → 维持原空正文兜底（思考顶成正文）。"""
    c = _stub_client([_events_reasoning("答案就在思考里")])
    content, calls, finish = _stream_parts(
        _parse_stream(list(c.stream_chat_completion(_payload(tools=TOOLS))))
    )
    assert content == "答案就在思考里"
    assert calls == []
    assert finish == "stop"


def test_nonstream_reasoning_draft_with_prose_content_not_extracted():
    """正文有实际内容时，思考里的调用是「草稿」——绝不提取（收紧版红线）。"""
    c = _stub_client([_events_normal_with_prose_and_think()])
    out = c.chat_completion(_payload(tools=TOOLS))
    msg = out["choices"][0]["message"]
    assert msg["content"] == "我直接文字回答"
    assert not msg.get("tool_calls")


def _events_normal_with_prose_and_think():
    return [
        {"_event_type": "ready"},
        {
            "p": "response/fragments",
            "o": "APPEND",
            "v": [{"id": "f1", "type": "THINK", "content": ""}],
        },
        {"v": f"我可以调用工具：{_dsml_call()}"},
        {
            "p": "response/fragments",
            "o": "APPEND",
            "v": [{"id": "f2", "type": "RESPONSE", "content": ""}],
        },
        {"v": "我直接文字回答"},
        {"data": {"p": "response/status", "v": "FINISHED"}},
    ]


# ── P1-3 required 强校验 ─────────────────────────────────────────


def test_non_stream_required_retry_recovers_call():
    c = _stub_client([_events_content("我直接文字回答"), _events_content(_dsml_call())])
    out = c.chat_completion(_payload(tools=TOOLS, tool_choice="required"))
    msg = out["choices"][0]["message"]
    assert msg["tool_calls"][0]["function"]["name"] == "shell"
    assert c.upstream_calls["n"] == 2


def test_non_stream_required_exhausted_raises():
    c = _stub_client([_events_content("纯文本一"), _events_content("纯文本二")])
    with pytest.raises(UpstreamAPIError):
        c.chat_completion(_payload(tools=TOOLS, tool_choice="required"))


def test_stream_required_recovers_call():
    c = _stub_client([_events_content("纯文本"), _events_content(_dsml_call())])
    content, calls, finish = _stream_parts(
        _parse_stream(list(c.stream_chat_completion(_payload(tools=TOOLS, tool_choice="required"))))
    )
    assert calls and calls[0]["name"] == "shell"
    assert finish == "tool_calls"
    assert content == ""  # 强校验形态不直播注定作废的正文


def test_stream_required_exhausted_raises():
    c = _stub_client([_events_content("纯文本一"), _events_content("纯文本二")])
    with pytest.raises(UpstreamAPIError):
        list(c.stream_chat_completion(_payload(tools=TOOLS, tool_choice="required")))


# ── P1-4 续写接缝剪除 ────────────────────────────────────────────


def test_seam_pure_repeat_discarded():
    t = _SeamTrimmer("断点前最后一句。")
    assert t.feed("断点前最后一句。") == ""
    assert t.finish() == ""


def test_seam_partial_repeat_trimmed_new_content_kept():
    t = _SeamTrimmer("……答案是2。")
    assert t.feed("答案是2。") == ""  # 是「尾巴后缀」的开头，扣住
    assert t.feed("，因为1+1=2") == "，因为1+1=2"  # 分叉：剪掉已发的「答案是2。」


def test_seam_no_overlap_passthrough():
    t = _SeamTrimmer("abcdef")
    assert t.feed("xyz") == "xyz"


def test_seam_holds_across_small_pieces():
    t = _SeamTrimmer("断点前最后一句。")
    assert t.feed("断") == ""
    assert t.feed("点前最后") == ""
    assert t.feed("一句。") == ""
    assert t.feed("然后是新内容") == "然后是新内容"


def test_seam_repeat_longer_than_window():
    t = _SeamTrimmer("x" * 200)
    assert t.feed("x" * 250) == "x" * 50  # 窗口外的新增量照常放行


def test_seam_empty_tail_passthrough():
    t = _SeamTrimmer("")
    assert t.feed("任意") == "任意"


# ── 工具件单元 ───────────────────────────────────────────────────


def test_strip_dsml_residue_removes_blocks_keeps_prose():
    text = f"前面的回答。\n{_dsml_call()}后面的字。"
    out = strip_dsml_residue(text)
    assert "DSML" not in out and "invoke" not in out
    assert "前面的回答" in out and "后面的字" in out
    # 纯块 → 剥到只剩空白
    assert not strip_dsml_residue(_dsml_skeleton()).strip()
    # 残缺块（无闭合）保守不剥
    partial = f"<{D} calls>\n<{D} invoke name=\"shell\">"
    assert "invoke" in strip_dsml_residue(partial)
    # 无标记文本原样
    assert strip_dsml_residue("普通回答") == "普通回答"


def test_build_required_nudge_format():
    nudge = build_required_nudge()
    assert "必须调用一个工具" in nudge
    assert f"<{D} calls>" in nudge
    named = build_required_nudge("shell")
    assert "`shell`" in named
