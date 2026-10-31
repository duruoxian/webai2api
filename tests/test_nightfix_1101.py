"""2026-11-01 夜间批次1回归（UI 外的适配器/网关修复）。

- K-1/Q-1：kimi/qwen 流式 forced 分支 usage 块引用未赋值变量 → UnboundLocalError
  （required 流式必炸，且每次把账号冻 30 分钟）。
- K-2：kimi _stream_events 空正文思考型回答在入库锚点处引用未定义 reasoning。
- K-3/Q-4：非流式「思考顶正文」置换后 usage 用旧变量，漏计整段答案。
- Q-2/D-1：空判定 200 字滑窗误判（qwen 流式；deepseek 流式四处）。
- T-1：DeltaToolBuffer.flush 吞掉表格行尾 | / 单个 <（带工具流式的所有回答）。
- G-1：GLM 非流式不过滤 tool_choice=none → 正文举例的调用被当真执行。
- G-2：GLM _usage 不计思考。
- G-3：GLM 无空响应弃会话重试（四家唯一缺口）+ 提示词补思考禁令。
- G-4(D-4)：deepseek 空响应重试轮不重置 full_text（骨架幽灵文本进 usage/入库）。
- D-2：deepseek 工具协议缺【身份与环境】锚定块（四家唯一缺口）。
- 网关：流式全失败路径引用未绑定 last_exc（NameError）；stream_err 恒空；
  internal（代码级）错误冻结账号并换号空转；粘性键工具轮只记名不记参数。
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from webai2api.adapters import tool_bridge
from webai2api.adapters.chatglm.client import (
    TOOL_INSTRUCTION,
    ChatGlmWebClient,
    _usage as glm_usage,
)
from webai2api.adapters.deepseek.services.tool_calling import build_tool_prompt
from webai2api.adapters.kimi.client import KimiWebClient
from webai2api.adapters.kimi.config import KimiConfig
from webai2api.adapters.qwen.client import QwenWebClient
from webai2api.app import build_app
from webai2api.gateway import _conv_key, _classify_error
from webai2api.providers import ProviderRegistry
from webai2api.store import Store

TOOLS = [
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

GLM_MARKER_CALL = (
    '<<<TOOL_CALL>>>{"name": "write", "arguments": '
    '{"filePath": "t.txt", "content": "hi"}}'
)


def _done(text="", reasoning="", chat_id="c1", finish=True, conv_id=""):
    return {
        "__done__": True,
        "text": text,
        "reasoning": reasoning,
        "chat_id": chat_id,
        "conv_id": conv_id,
        "finish_seen": finish,
    }


class _RecorderCache:
    def __init__(self):
        self.deleted = []

    def delete_by_session_id(self, sid):
        self.deleted.append(sid)
        return 1


def _kimi(script):
    c = KimiWebClient.__new__(KimiWebClient)
    c.config = SimpleNamespace(
        default_model="kimi-k2.6",
        legacy_reasoning_in_content=False,
        session_cache_enabled=True,
        delete_conversation=False,
    )
    c.logger = logging.getLogger("test.kimi.nightfix")
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
    c.logger = logging.getLogger("test.qwen.nightfix")
    c._conv_cache = _RecorderCache()
    it = iter(script)

    def _events(payload):
        return iter(next(it))

    c._stream_events = _events
    return c


def _glm(script):
    c = ChatGlmWebClient.__new__(ChatGlmWebClient)
    c.config = SimpleNamespace(
        default_model="GLM-5.3-Flash",
        legacy_reasoning_in_content=False,
    )
    c.logger = logging.getLogger("test.glm.nightfix")
    c._conv_cache = _RecorderCache()
    it = iter(script)

    def _events(payload):
        return iter(next(it))

    c._stream_events = _events
    return c


def _payload(**extra):
    base = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    base.update(extra)
    return base


def _sse_objs(chunks):
    out = []
    for sse in chunks:
        sse = sse.strip()
        if sse.startswith("data: ") and sse != "data: [DONE]":
            out.append(json.loads(sse[6:]))
    return out


def _finish_list(objs):
    return [
        c2["finish_reason"]
        for o in objs
        for c2 in o.get("choices") or []
        if c2.get("finish_reason")
    ]


# ── K-1 / Q-1：forced 流式 usage 块 NameError ────────────────────


def test_kimi_forced_stream_usage_chunk_ok():
    c = _kimi([[{"text": KIMI_DSML_CALL, "reasoning": "想一想", "chat_id": "c1"},
                _done(KIMI_DSML_CALL, "想一想")]])
    objs = _sse_objs(
        c.stream_chat_completion(
            _payload(tools=TOOLS, tool_choice="required", stream=True)
        )
    )
    usage = [o for o in objs if o.get("usage")]
    assert usage, "usage 块必须存在（旧代码在此处 UnboundLocalError）"
    assert usage[0]["usage"]["completion_tokens"] >= 2, "思考必须计入"
    assert _finish_list(objs)[-1] == "tool_calls"


def test_qwen_forced_stream_usage_chunk_ok():
    call = '{"name": "write", "arguments": {"filePath": "t.txt", "content": "hi"}}'
    c = _qwen([[{"text": call, "reasoning": "想一想", "chat_id": "c1"},
                _done(call, "想一想")]])
    objs = _sse_objs(
        c.stream_chat_completion(
            _payload(tools=TOOLS, tool_choice="required", stream=True)
        )
    )
    usage = [o for o in objs if o.get("usage")]
    assert usage, "usage 块必须存在（旧代码在此处 UnboundLocalError）"
    assert usage[0]["usage"]["completion_tokens"] >= 2
    assert _finish_list(objs)[-1] == "tool_calls"


# ── K-3 / Q-4：非流式思考顶正文后 usage 口径 ─────────────────────


def test_kimi_nonstream_usage_counts_promoted_reasoning():
    c = _kimi([[_done("", "这是藏在思考里的完整答案内容")]])
    out = c.chat_completion(_payload(tools=TOOLS))
    msg = out["choices"][0]["message"]
    assert "完整答案" in (msg.get("content") or "")
    assert out["usage"]["completion_tokens"] >= 5, "顶上来的思考必须计入 usage"


def test_qwen_nonstream_usage_counts_promoted_reasoning():
    c = _qwen([[_done("", "这是藏在思考里的完整答案内容")]])
    out = c.chat_completion(_payload(tools=TOOLS))
    msg = out["choices"][0]["message"]
    assert "完整答案" in (msg.get("content") or "")
    assert out["usage"]["completion_tokens"] >= 5


# ── Q-2：qwen 流式空判定全量口径 ─────────────────────────────────


def test_qwen_stream_empty_judgment_uses_full_visible():
    """正文以 200+ 空白结尾：滑窗口径会误判全空 → 弃会话重试 → 脚本耗尽
    报错；全量口径下应正常收场且只有一份回答。"""
    long_blank_tail = "回答正文" + " " * 300
    c = _qwen([[{"text": long_blank_tail, "reasoning": "", "chat_id": "c1"},
                _done(long_blank_tail)]])
    objs = _sse_objs(c.stream_chat_completion(_payload(stream=True)))
    errs = [o for o in objs if o.get("error")]
    assert not errs, f"不应误判空响应触发重试: {errs}"
    content = "".join(
        c2["delta"].get("content") or ""
        for o in objs
        for c2 in o.get("choices") or []
    )
    assert content == long_blank_tail, "回答必须恰好一份"


# ── T-1：flush 不吞表格竖线 / 单个 < ─────────────────────────────


def _buf():
    return tool_bridge.DeltaToolBuffer(
        ["<<<TOOL_CALL>>>", tool_bridge.DSML_TOOL_CALLS_OPEN],
        fragment_re=tool_bridge.MARKER_FRAGMENT_RE,
        open_re=tool_bridge.MARKER_OPEN_LOOSE_RE,
    )


def test_flush_keeps_table_pipe_tail():
    buf = _buf()
    out = ""
    for piece in ("对比结果：\n\n", "| 方法 | 速度 |\n", "|---|---|\n", "| A | 快 |"):
        out += buf.feed_delta(piece)
    out += buf.flush()
    assert out.endswith("快 |"), f"表格行尾 | 被吞: {out!r}"


def test_flush_keeps_single_lt_tail():
    buf = _buf()
    buf.feed_delta("比较: 1 ")
    buf.feed_delta("<")
    out = buf.flush()
    assert out == "比较: 1 <", f"单个 < 被吞: {out!r}"


def test_flush_still_drops_partial_opener():
    buf = _buf()
    buf.feed_delta("abc ")
    buf.feed_delta("<<<TOOL_CA")
    out = buf.flush()
    assert out == "abc ", f"半截 opener 残迹应丢弃: {out!r}"


def test_strip_truncated_marker_tail_keeps_lt_pipe():
    openers = ["<<<TOOL_CALL>>>", tool_bridge.DSML_TOOL_CALLS_OPEN]
    assert tool_bridge.strip_truncated_marker_tail("x <|", openers) == "x <|"
    assert tool_bridge.strip_truncated_marker_tail("行尾 |", openers) == "行尾 |"
    assert (
        tool_bridge.strip_truncated_marker_tail("abc <<<TOOL_CA", openers)
        == "abc "
    )


# ── G-1 / G-2 / G-3：GLM ────────────────────────────────────────


def test_glm_nonstream_tool_choice_none_not_parsed():
    text = f"举例说明：\n{GLM_MARKER_CALL}\n以上只是格式演示。"
    c = _glm([[{"text": text, "reasoning": "", "chat_id": "c1"},
               _done(text, "", conv_id="conv-x")]])
    out = c.chat_completion(_payload(tools=TOOLS, tool_choice="none"))
    msg = out["choices"][0]["message"]
    assert not msg.get("tool_calls"), "tool_choice=none 时绝不能解析出调用"
    assert "TOOL_CALL" not in (msg.get("content") or "")


def test_glm_usage_counts_reasoning():
    p = {"messages": [{"role": "user", "content": "hi"}]}
    base = glm_usage(p, "答案正文")["completion_tokens"]
    with_think = glm_usage(p, "答案正文", "一段思考内容")["completion_tokens"]
    assert with_think > base, "思考必须计入 completion"


def test_glm_nonstream_empty_response_retry():
    c = _glm([
        [_done("", "", conv_id="conv-empty")],
        [{"text": "第二次的回答", "reasoning": "", "chat_id": "c2"},
         _done("第二次的回答", "", conv_id="conv-real")],
    ])
    out = c.chat_completion(_payload())
    assert out["choices"][0]["message"]["content"] == "第二次的回答"
    assert c._stream_events_calls == 2 if hasattr(c, "_stream_events_calls") else True
    assert "conv-empty" in c._conv_cache.deleted, "空响应必须弃会话"


def test_glm_stream_empty_response_retry():
    c = _glm([
        [_done("", "", conv_id="conv-empty")],
        [{"text": "重试后的回答", "reasoning": "", "chat_id": "c2"},
         _done("重试后的回答", "", conv_id="conv-real")],
    ])
    objs = _sse_objs(c.stream_chat_completion(_payload(stream=True)))
    errs = [o for o in objs if o.get("error")]
    assert not errs, f"空响应重试不应报错: {errs}"
    content = "".join(
        c2["delta"].get("content") or ""
        for o in objs
        for c2 in o.get("choices") or []
    )
    assert content == "重试后的回答"
    assert "conv-empty" in c._conv_cache.deleted


def test_glm_stream_normal_answer_single_pass():
    """有输出的回答绝不触发重试（_stream_events 只跑一次）。"""
    calls = {"n": 0}
    script = [[{"text": "一次就好", "reasoning": "思考", "chat_id": "c1"},
               _done("一次就好", "思考", conv_id="conv-1")]]

    c = _glm(script)
    orig = c._stream_events

    def _counted(payload):
        calls["n"] += 1
        return orig(payload)

    c._stream_events = _counted
    objs = _sse_objs(c.stream_chat_completion(_payload(stream=True)))
    assert calls["n"] == 1
    content = "".join(
        c2["delta"].get("content") or ""
        for o in objs
        for c2 in o.get("choices") or []
    )
    assert "一次就好" in content


def test_glm_prompt_forbids_thinking_tool_calls():
    assert "思考" in TOOL_INSTRUCTION
    assert "禁止" in TOOL_INSTRUCTION


# ── D-2 / G-4：deepseek ─────────────────────────────────────────


def test_deepseek_prompt_has_identity_anchor():
    prompt = build_tool_prompt(TOOLS, "auto")
    low = prompt.lower()
    assert "identity & environment" in low
    assert "sandbox" in low
    assert "coding agent" in low


# ── Q-6：qwen content list 形态 ──────────────────────────────────


def test_qwen_content_list_rendered_not_repr():
    ev = {
        "choices": [
            {
                "delta": {
                    "content": [
                        {"type": "text", "text": "看这张图 "},
                        {"type": "image", "image_url": {"url": "http://x/y.png"}},
                    ]
                }
            }
        ]
    }
    text, reasoning = QwenWebClient._parse_event(ev)
    assert text == "看这张图 ![图像](http://x/y.png)"
    assert "[{'type'" not in text, "绝不能把 Python repr 当正文"


def test_qwen_content_plain_string_unchanged():
    ev = {"choices": [{"delta": {"content": "普通文本"}}]}
    text, _ = QwenWebClient._parse_event(ev)
    assert text == "普通文本"


# ── 网关：流式全失败 / internal 分类 / 粘性键 ─────────────────────

TOKEN_HDR = "X-Admin-Token"
AUTH_ERR = RuntimeError(
    "获取 PoW challenge 失败: 上游返回 code=40003, "
    "msg=Authorization Failed (invalid token)"
)


def _resp(text: str) -> dict:
    return {
        "model": "deepseek-chat",
        "choices": [
            {"finish_reason": "stop", "message": {"role": "assistant", "content": text}}
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


class _AC:
    def __init__(self, name: str, client):
        self.account = {
            "id": f"acc-{name}",
            "name": name,
            "provider": "deepseek",
            "enabled": 1,
        }
        self.client = client
        self.calls = 0
        self.bad_marks = 0

    def mark_ok(self):
        pass

    def mark_bad(self, seconds=0, exc=None):
        self.bad_marks += 1

    def cooldown_left(self):
        return 0


class _BoomAuthClient:
    def __init__(self, ac: _AC):
        self.ac = ac

    def chat_completion(self, payload):
        self.ac.calls += 1
        raise AUTH_ERR

    def stream_chat_completion(self, payload):
        self.ac.calls += 1
        raise AUTH_ERR
        yield  # pragma: no cover


class _BoomInternalClient:
    """代码级 bug：TypeError（非账号/上游问题）。"""

    def __init__(self, ac: _AC):
        self.ac = ac

    def chat_completion(self, payload):
        self.ac.calls += 1
        raise TypeError("cannot access local variable 'sent_reasoning'")

    def stream_chat_completion(self, payload):
        self.ac.calls += 1
        raise TypeError("cannot access local variable 'sent_reasoning'")
        yield  # pragma: no cover


def _make_app(tmp_path: Path, picks: list):
    store = Store(tmp_path / "data" / "webai2api.db")
    logger = logging.getLogger("webai2api.test.nightfix")
    registry = ProviderRegistry(store, logger)
    app = build_app(store, registry, logger)
    store.add_account("deepseek", "一号", {"auth_token": "tok-a"})
    store.add_account("deepseek", "二号", {"auth_token": "tok-b"})
    registry.refresh()
    it = iter(picks)

    def _pick(*a, **k):
        try:
            return next(it)
        except StopIteration:
            return None

    registry.pick = _pick  # type: ignore[assignment]
    registry.pick_by_id = _pick  # type: ignore[assignment]
    client = TestClient(app, client=("127.0.0.1", 50000))
    r = client.post("/admin/api/login", json={"password": "admin"})
    tok = r.json()["token"]
    r = client.post(
        "/admin/api/keys", json={"name": "测试Key"}, headers={TOKEN_HDR: tok}
    )
    assert r.json().get("ok") is True, r.text
    return client, r.json()["key"]["key"]


def test_classify_internal_for_code_errors():
    assert _classify_error(TypeError("x")) == "internal"
    assert _classify_error(KeyError("x")) == "internal"
    assert _classify_error(RuntimeError("HTTP 500 bad")) == "server"
    assert _classify_error(RuntimeError("429 too many")) == "rate"


def test_stream_all_accounts_fail_returns_502_not_500(tmp_path):
    """流式全失败：旧代码在 log_usage 处引用未绑定的 last_exc 直接
    NameError → 500。修复后 502 + 真实原因。"""
    boom = _AC("一号", None)
    boom.client = _BoomAuthClient(boom)
    client, key = _make_app(tmp_path, [boom])
    r = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "deepseek-chat",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 502, f"应 502 而非内部错误: {r.status_code} {r.text[:200]}"
    assert boom.calls == 1


def test_internal_error_breaks_fast_without_freezing_accounts(tmp_path):
    """代码级 bug：不换号空转、不冷却任何账号，500 + 真实异常透出。"""
    boom = _AC("一号", None)
    boom.client = _BoomInternalClient(boom)
    spare = _AC("二号", None)
    spare.client = _BoomInternalClient(spare)
    client, key = _make_app(tmp_path, [boom, spare])
    r = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert boom.calls == 1, "internal 立即失败，不换号空转"
    assert spare.calls == 0
    assert boom.bad_marks == 0, "代码 bug 不是账号的错，绝不冷却"
    assert spare.bad_marks == 0
    assert r.status_code == 500
    assert "网关内部错误" in r.json()["error"]["message"]


def test_conv_key_tool_signature_includes_arguments():
    asst_a = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "write",
                    "arguments": '{"filePath": "a.txt", "content": "A"}',
                },
            }
        ],
    }
    asst_b = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "write",
                    "arguments": '{"filePath": "b.txt", "content": "B"}',
                },
            }
        ],
    }
    p1 = {"model": "m", "messages": [{"role": "user", "content": "hi"}, asst_a]}
    p2 = {"model": "m", "messages": [{"role": "user", "content": "hi"}, asst_b]}
    assert _conv_key(p1, "k") != _conv_key(p2, "k"), "参数不同必须不同键"
    # 空白/键序免疫：同一参数的不同序列化形态必须同键
    asst_a2 = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "write",
                    "arguments": '{ "content" : "A" , "filePath" : "a.txt" }',
                },
            }
        ],
    }
    p3 = {"model": "m", "messages": [{"role": "user", "content": "hi"}, asst_a2]}
    assert _conv_key(p1, "k") == _conv_key(p3, "k"), "规范化后应同键"


# ── K-2：kimi 空正文思考型回答（真实 _stream_events，mock 上游）──────


def _frame(flag: int, payload: bytes) -> bytes:
    return (0).to_bytes(1, "big") + len(payload).to_bytes(4, "big") + payload \
        if flag == 0 else (flag).to_bytes(1, "big") + len(payload).to_bytes(4, "big") + payload


def _json_frame(obj: dict) -> bytes:
    return _frame(0, json.dumps(obj, ensure_ascii=False).encode())


class _ThinkOnlyHandler(BaseHTTPRequestHandler):
    """正文块为空、思考块有完整答案、正常结束（done + 0x02）。"""

    def log_message(self, *a):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "application/connect+json")
        self.end_headers()
        for ev in [
            {"op": "set", "chat": {"id": "chat-think"}},
            {
                "op": "set",
                "message": {
                    "id": "msg-a",
                    "role": "assistant",
                    "status": "MESSAGE_STATUS_GENERATING",
                },
            },
            {"op": "set", "mask": "block.think",
             "block": {"id": "t1", "think": {"content": ""}}},
            {"op": "append", "mask": "block.think.content",
             "block": {"id": "t1", "think": {"content": "思考里的完整答案"}}},
            {"done": {}},
        ]:
            self.wfile.write(_json_frame(ev))
        self.wfile.write(_frame(0x02, b"{}"))

    do_GET = do_POST


@pytest.fixture()
def think_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _ThinkOnlyHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_kimi_think_only_reply_no_nameerror(think_server):
    """正文空白+思考有货：入库锚点处旧代码引用未定义 reasoning 直接
    UnboundLocalError（K-2），整条请求变 502 且空响应重试被绕过。"""
    cfg = KimiConfig(
        base_url=think_server,
        auth_token="t",
        egress="direct",
        anti_rate_limit_enabled=False,
        session_store_path="",
    )
    c = KimiWebClient(cfg, logging.getLogger("test.kimi.thinkonly"))
    out = c.chat_completion({
        "model": "kimi-k2.6",
        "messages": [{"role": "user", "content": "hi"}],
        "_wapi_key_id": "key-1",
    })
    msg = out["choices"][0]["message"]
    assert "完整答案" in (msg.get("content") or ""), msg
    assert out["choices"][0]["finish_reason"] == "stop"
