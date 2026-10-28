"""kimi / qwen 与 deepseek/GLM 修复对齐的回归（2026-10-29）。

- 用量口径：completion 必须计入思考（此前只算正文导致低估）。
- P0 空响应重试：正文/思考/调用全空且正常结束 → 弃上游会话重跑一次。
- P0 思考双通道：正文可见输出为空时才扫 reasoning 里的完整调用块
  （收紧版——草稿调用绝不提取）。
- P1 续写接缝剪除：SeamTrimmer 上移 tool_bridge 三家共用。
- P1 解析失败可观测性：扣住标记却解析不出调用时留带片段的 warning。
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

from webai2api.adapters.kimi.client import KimiWebClient, _estimate_tokens
from webai2api.adapters.qwen.client import QwenWebClient
from webai2api.adapters.tool_bridge import SeamTrimmer

KIMI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "写文件",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["filePath", "content"],
            },
        },
    }
]

KIMI_DSML_CALL = (
    "<|DSML|tool_calls>\n"
    '  <|DSML|invoke name="write">\n'
    '    <|DSML|parameter name="filePath"><![CDATA[t.txt]]></|DSML|parameter>\n'
    '    <|DSML|parameter name="content"><![CDATA[hi]]></|DSML|parameter>\n'
    "  </|DSML|invoke>\n"
    "</|DSML|tool_calls>\n"
)

QWEN_TOOLS = KIMI_TOOLS
QWEN_TOOL_CALL = (
    '<tool_call>{"name": "write", "arguments": '
    '{"filePath": "t.txt", "content": "hi"}}</tool_call>'
)


def _done(text: str, reasoning: str = "", chat_id: str = "c1", finish=True):
    return {
        "__done__": True,
        "text": text,
        "reasoning": reasoning,
        "chat_id": chat_id,
        "finish_seen": finish,
    }


class _RecorderCache:
    def __init__(self):
        self.deleted = []

    def delete_by_session_id(self, sid):
        self.deleted.append(sid)
        return 1


def _kimi(script):
    """script: 每次 _stream_events 调用返回一组事件（可迭代的可迭代）。"""
    c = KimiWebClient.__new__(KimiWebClient)
    c.config = SimpleNamespace(
        default_model="kimi-k2.6",
        legacy_reasoning_in_content=False,
        session_cache_enabled=True,
        delete_conversation=False,
    )
    c.logger = logging.getLogger("test.kimi.parity")
    c._conv_cache = _RecorderCache()
    it = iter(script)

    def _events(payload):
        return iter(next(it))

    c._stream_events = _events
    return c


def _qwen(script):
    c = QwenWebClient.__new__(QwenWebClient)
    c.config = SimpleNamespace(
        default_model="qwen3.8-max",
        legacy_reasoning_in_content=False,
        session_cache_enabled=True,
    )
    c.logger = logging.getLogger("test.qwen.parity")
    c._conv_cache = _RecorderCache()
    it = iter(script)

    def _events(payload):
        return iter(next(it))

    c._stream_events = _events
    return c


def _payload(**extra):
    base = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
    }
    base.update(extra)
    return base


def _parse_stream(chunks):
    out = []
    for sse in chunks:
        sse = sse.strip()
        if sse.startswith("data: ") and sse != "data: [DONE]":
            out.append(json.loads(sse[6:]))
    return out


# ── 用量口径 ─────────────────────────────────────────────────────


def test_kimi_usage_counts_reasoning():
    c = _kimi([[_done("a" * 400, "思" * 300)]])  # 100 + 200 token
    out = c.chat_completion(_payload())
    u = out["usage"]
    assert u["completion_tokens"] == 100 + 200


def test_qwen_usage_counts_reasoning():
    c = _qwen([[_done("a" * 400, "思" * 300)]])
    out = c.chat_completion(_payload())
    u = out["usage"]
    assert u["completion_tokens"] == 100 + 200


def test_kimi_stream_usage_chunk_counts_reasoning():
    c = _kimi([[
        {"text": "a" * 400, "reasoning": "思" * 300, "chat_id": "c1"},
        _done("a" * 400, "思" * 300),
    ]])
    chunks = _parse_stream(list(c.stream_chat_completion(_payload())))
    usage = [x.get("usage") for x in chunks if x.get("usage")]
    assert usage and usage[-1]["completion_tokens"] == 100 + 200


# ── P0 思考双通道 ────────────────────────────────────────────────


def test_kimi_tool_call_hidden_in_reasoning_extracted():
    c = _kimi([[_done("", f"先想一下。\n{KIMI_DSML_CALL}")]])
    out = c.chat_completion(_payload(tools=KIMI_TOOLS))
    msg = out["choices"][0]["message"]
    assert msg["tool_calls"][0]["function"]["name"] == "write"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {
        "filePath": "t.txt",
        "content": "hi",
    }
    assert "DSML" not in (msg.get("reasoning_content") or ""), "块应从思考中剥除"


def test_kimi_stream_tool_call_hidden_in_reasoning_extracted():
    c = _kimi([[
        {"text": "", "reasoning": f"想一想\n{KIMI_DSML_CALL}", "chat_id": "c1"},
        _done("", f"想一想\n{KIMI_DSML_CALL}"),
    ]])
    chunks = _parse_stream(list(c.stream_chat_completion(_payload(tools=KIMI_TOOLS))))
    calls = []
    for x in chunks:
        for t in (x["choices"][0].get("delta") or {}).get("tool_calls") or []:
            calls.append(t["function"]["name"])
    assert calls == ["write"]


def test_qwen_tool_call_hidden_in_reasoning_extracted():
    c = _qwen([[_done("", f"想一想。\n{QWEN_TOOL_CALL}")]])
    out = c.chat_completion(_payload(tools=QWEN_TOOLS))
    msg = out["choices"][0]["message"]
    assert msg["tool_calls"][0]["function"]["name"] == "write"
    assert "<tool_call>" not in (msg.get("reasoning_content") or "")


def test_kimi_reasoning_without_complete_block_still_promoted():
    """思考里没有完整调用块 → 维持空正文兜底（思考顶成正文）。"""
    c = _kimi([[_done("", "答案就在思考里")]])
    out = c.chat_completion(_payload(tools=KIMI_TOOLS))
    msg = out["choices"][0]["message"]
    assert msg["content"] == "答案就在思考里"
    assert not msg.get("tool_calls")


# ── P0 空响应重试 ────────────────────────────────────────────────


def test_kimi_empty_response_retried_with_fresh_session():
    c = _kimi([
        [_done("", "", chat_id="c1")],
        [_done("重试后的答案", chat_id="c2")],
    ])
    out = c.chat_completion(_payload())
    assert out["choices"][0]["message"]["content"] == "重试后的答案"
    assert c._conv_cache.deleted == ["c1"], "应作废第一次的空会话"


def test_kimi_stream_empty_response_retried():
    c = _kimi([
        [_done("", "", chat_id="c1")],
        [_done("流式重试答案", chat_id="c2")],
    ])
    chunks = _parse_stream(list(c.stream_chat_completion(_payload())))
    content = "".join(
        x["choices"][0]["delta"].get("content", "") for x in chunks if x.get("choices")
    )
    assert content == "流式重试答案"
    assert c._conv_cache.deleted == ["c1"]


def test_kimi_empty_both_rounds_returns_empty():
    c = _kimi([
        [_done("", "", chat_id="c1")],
        [_done("", "", chat_id="c2")],
    ])
    out = c.chat_completion(_payload())
    assert out["choices"][0]["message"]["content"] == ""
    assert c._conv_cache.deleted == ["c1"]  # 只重试一次


def test_qwen_empty_response_retried_with_fresh_session():
    c = _qwen([
        [_done("", "", chat_id="q1")],
        [_done("qwen 重试答案", chat_id="q2")],
    ])
    out = c.chat_completion(_payload())
    assert out["choices"][0]["message"]["content"] == "qwen 重试答案"
    assert c._conv_cache.deleted == ["q1"]


# ── P1 解析失败可观测性 ──────────────────────────────────────────


def test_kimi_parse_failure_logs_warning(caplog):
    bad = "<|DSML|tool_calls>\n  <|DSML|invoke name=\"not_a_tool\">\n  </|DSML|invoke>\n</|DSML|tool_calls>\n"
    # 两轮同样的事件：第一轮解析失败会触发空响应重试（可见输出为空）
    c = _kimi([[_done(bad)], [_done(bad)]])
    with caplog.at_level(logging.WARNING, logger="test.kimi.parity"):
        list(c.stream_chat_completion(_payload(tools=KIMI_TOOLS)))
    assert any("未解析成调用" in r.message for r in caplog.records)


# ── P1 SeamTrimmer（共用版）──────────────────────────────────────


def test_seam_trimmer_committed_tracks_passed_text():
    t = SeamTrimmer("……答案是2。")
    assert t.feed("答案是2。") == ""
    assert t.committed == ""
    assert t.feed("，因为1+1=2") == "，因为1+1=2"
    assert t.committed == "，因为1+1=2"


def test_seam_trimmer_pure_repeat_commit_stays_empty():
    t = SeamTrimmer("断点前最后一句。")
    assert t.feed("断点前最后一句。") == ""
    assert t.finish() == ""
    assert t.committed == ""


# ── 防呆：估算函数 sanity ────────────────────────────────────────


def test_estimate_tokens_sanity():
    assert _estimate_tokens("") == 0
    assert _estimate_tokens("a" * 400) == 100
