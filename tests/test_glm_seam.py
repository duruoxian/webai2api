"""GLM 续写接缝剪除 + 思考接缝换行（2026-10-29 用户报障「思考中断」）回归。

上游 GLM 网页端会掐断长思考/长回答流（无 finish 帧），自动续写向同一
会话发「继续」救回。本文件锁定续写轮的两条接缝行为：

1. SeamTrimmer 剪除：续写轮开头逐字复读已发尾巴的部分被剪掉——与
   deepseek/kimi/qwen 同款，剪掉的字节都与客户端已有内容逐字相同，
   数学上不丢新内容；
2. 思考接缝换行：仅当基线结尾与续写开头都是文字字符、且本轮没有剪掉
   重叠（剪过 = 分叉点是真实续点，拼接即原貌）时补一个换行；正文永不注入。

桩服务两轮行为：第 1 次 POST 返回不带 finish 帧的半截流（模拟被掐），
第 2 次 POST（续写轮）返回复读 + 新内容 + finish 帧。
"""
from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from webai2api.adapters.chatglm.client import ChatGlmWebClient
from webai2api.adapters.chatglm.config import GlmConfig


def _frame(conv: str, lid: str, piece: str, *, finish: bool = False) -> dict:
    # think/text 字段名不同，按 lid 前缀构造
    if lid.startswith("t"):
        content = [{"type": "think", "think": piece}]
    else:
        content = [{"type": "text", "text": piece}]
    part = {
        "logic_id": lid,
        "status": "finish" if finish else "init",
        "content": content,
    }
    frame: dict = {"parts": [part]}
    if conv:
        frame["conversation_id"] = conv
    if finish:
        frame["status"] = "finish"
    return frame


def _to_bytes(frames: list[dict]) -> bytes:
    out = b""
    for f in frames:
        out += f"data: {json.dumps(f, ensure_ascii=False)}\n\n".encode()
    return out + b"data: [DONE]\n\n"


class _SeamHandler(BaseHTTPRequestHandler):
    round1_bytes = b""
    round2_bytes = b""
    hits = 0

    def log_message(self, *a):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        if self.path.endswith("/user/refresh"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps({"code": 0, "result": {"access_token": "tok"}}).encode()
            )
            return
        type(self).hits += 1
        payload = self.round1_bytes if self.hits <= 1 else self.round2_bytes
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(payload)

    do_GET = do_POST


def _make_server(round1: list[dict], round2: list[dict]):
    class H(_SeamHandler):
        pass

    H.round1_bytes = _to_bytes(round1)
    H.round2_bytes = _to_bytes(round2)
    H.hits = 0
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return f"http://127.0.0.1:{srv.server_address[1]}", srv


def _client(base: str) -> ChatGlmWebClient:
    cfg = GlmConfig(
        refresh_token="rt",
        base_url=base,
        anti_rate_limit_enabled=False,
        session_store_path="",
        egress="direct",
    )
    return ChatGlmWebClient(cfg, logging.getLogger("glm.seam"))


def _stream_collect(c: ChatGlmWebClient) -> tuple[str, str, str | None]:
    reasoning = ""
    content = ""
    finish_reason = None
    for chunk in c.stream_chat_completion({
        "model": "GLM-5.3-Flash",
        "stream": True,
        "messages": [{"role": "user", "content": "写一份长分析"}],
    }):
        if not chunk.startswith("data: ") or chunk == "data: [DONE]\n\n":
            continue
        obj = json.loads(chunk[len("data: "):])
        for ch in obj.get("choices") or []:
            d = ch.get("delta") or {}
            reasoning += d.get("reasoning_content") or ""
            content += d.get("content") or ""
            if ch.get("finish_reason"):
                finish_reason = ch["finish_reason"]
    return reasoning, content, finish_reason


def test_continuation_trims_repeated_reasoning_tail():
    """续写轮复读已发思考尾巴：复读段被剪掉，新内容完整，且不补换行。"""
    base, srv = _make_server(
        [_frame("conv-seam", "t1", "正在分析"), _frame("", "t1", "这个问")],
        [
            _frame("", "t2", "这个问"),
            _frame("", "t2", "题的约束，继续推演"),
            _frame("", "t2", "这个问题的约束，继续推演", finish=True),
        ],
    )
    try:
        reasoning, content, finish = _stream_collect(_client(base))
    finally:
        srv.shutdown()
    assert reasoning == "正在分析这个问题的约束，继续推演"
    assert "这个问这个" not in reasoning
    assert "\n\n" not in reasoning
    # 正文全程为空时收尾兜底把思考镜像进 content（既有设计），镜像值
    # 同样应是剪除后的终态
    assert content == "正在分析这个问题的约束，继续推演"
    assert finish == "stop"


def test_continuation_reasoning_joiner_on_fresh_start():
    """续写另起话头（无复读）且两侧都是文字字符：思考接缝补换行。"""
    base, srv = _make_server(
        [_frame("conv-seam", "t1", "正在分析这个问题")],
        [
            _frame("", "t2", "换个角度推演"),
            _frame("", "t2", "换个角度推演", finish=True),
        ],
    )
    try:
        reasoning, content, _ = _stream_collect(_client(base))
    finally:
        srv.shutdown()
    assert reasoning == "正在分析这个问题\n\n换个角度推演"
    # 收尾兜底镜像（与 reasoning 一致）
    assert content == "正在分析这个问题\n\n换个角度推演"


def test_no_joiner_when_reasoning_ends_with_punctuation():
    """基线思考以标点收尾：拼接本身通顺，不补换行。"""
    base, srv = _make_server(
        [_frame("conv-seam", "t1", "第一段思考，结论成立。")],
        [
            _frame("", "t2", "接下来验证边界"),
            _frame("", "t2", "接下来验证边界", finish=True),
        ],
    )
    try:
        reasoning, _, _ = _stream_collect(_client(base))
    finally:
        srv.shutdown()
    assert reasoning == "第一段思考，结论成立。接下来验证边界"


def test_text_seam_trims_repeat_and_never_injects_newline():
    """正文续写：复读段剪除，且任何情况下不向正文注入换行。"""
    base, srv = _make_server(
        [_frame("conv-seam", "b1", "答案是 ABC"), _frame("", "b1", "DEF")],
        [
            _frame("", "b2", "DEF"),
            _frame("", "b2", "GHI"),
            _frame("", "b2", "DEFGHI", finish=True),
        ],
    )
    try:
        _, content, finish = _stream_collect(_client(base))
    finally:
        srv.shutdown()
    assert content == "答案是 ABCDEFGHI"
    assert "\n" not in content
    assert finish == "stop"


def test_pure_repeat_continuation_drops_held_tail():
    """续写轮整段都是已发尾巴的逐字复读：扣住后整段丢弃，不产生重复。"""
    base, srv = _make_server(
        [_frame("conv-seam", "t1", "正在分析这个问")],
        [
            _frame("", "t2", "这个问"),
            _frame("", "t2", "这个问", finish=True),
        ],
    )
    try:
        reasoning, _, finish = _stream_collect(_client(base))
    finally:
        srv.shutdown()
    assert reasoning == "正在分析这个问"
    assert finish == "stop"


def test_nonstream_path_gets_trimmed_reasoning():
    """非流式路径同样吃到剪除后的终态（chat_completion 消费同一生成器）。"""
    base, srv = _make_server(
        [_frame("conv-seam", "t1", "正在分析这个问")],
        [
            _frame("", "t2", "这个问"),
            _frame("", "t2", "题的约束，继续推演"),
            _frame("", "t2", "这个问题的约束，继续推演", finish=True),
        ],
    )
    try:
        resp = _client(base).chat_completion({
            "model": "GLM-5.3-Flash",
            "messages": [{"role": "user", "content": "写一份长分析"}],
        })
    finally:
        srv.shutdown()
    msg = resp["choices"][0]["message"]
    # 思考里就是答案时非流式收尾把思考顶进 content（空正文兜底，既有设计）
    assert msg.get("content") == "正在分析这个问题的约束，继续推演"


def test_single_round_passthrough_unchanged():
    """一轮正常结束（见到 finish 帧）：不进续写分支，输出与上游逐字一致。"""
    base, srv = _make_server(
        [
            _frame("conv-ok", "t1", "完整思考"),
            _frame("", "b1", "完整答案"),
            _frame("", "b1", "完整答案", finish=True),
        ],
        [],
    )
    try:
        reasoning, content, finish = _stream_collect(_client(base))
    finally:
        srv.shutdown()
    assert reasoning == "完整思考"
    assert content == "完整答案"
    assert finish == "stop"
