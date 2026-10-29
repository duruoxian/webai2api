"""GLM 工具标记降级形态零残留回归（对齐 deepseek DSML 降级修复的同族问题）。

GLM 的工具标记（<<<TOOL_CALL>>> / <|DSML|tool_calls>）全靠尖括号，上游会吃
`<`、流会被掐断切碎、模型会写缺角括号——修复前精确匹配认不出，标记/碎片整段
漏进客户端正文。本文件锁定：

1. 断流续写重启标记：被扣住的半截 opener（<<<TOOL_CA）不随续写放行段漏出；
2. 流末半截标记：收尾丢弃，有标记气味的才丢（正文单个 `<`/`|` 不误伤）；
3. 降级完整 opener（<<TOOL_CALL>>）：按碎片正则认 opener，扣住；
4. 非流式：归一角括号后剥除/解析照常命中，孤立闭合标签不漏；
5. kimi/qwen 共享的 _ToolHoldBuffer 默认不启用（行为不变）。
"""
from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from webai2api.adapters.chatglm.client import ChatGlmWebClient, _strip_tool_block
from webai2api.adapters.chatglm.config import GlmConfig
from webai2api.adapters.tool_bridge import (
    DeltaToolBuffer,
    MARKER_FRAGMENT_RE,
    normalize_tool_marker,
    scrub_marker_fragments,
)

MARKER = "<<<TOOL_CALL>>>"


def _write_invoke(path: str, content: str) -> str:
    # 真实协议格式（TOOL_INSTRUCTION/required_nudge 教授的形态）：
    # <|DSML|invoke> + <|DSML|parameter> + CDATA
    return (
        f"{MARKER}\n<|DSML|tool_calls>\n"
        f'<|DSML|invoke name="write">\n'
        f'<|DSML|parameter name="filePath"><![CDATA[{path}]]></|DSML|parameter>\n'
        f'<|DSML|parameter name="content"><![CDATA[{content}]]></|DSML|parameter>\n'
        f"</|DSML|invoke>\n</|DSML|tool_calls>"
    )


class _DegradedHandler(BaseHTTPRequestHandler):
    """可编程上游：rounds 每项 (deltas, finish)——finish=False 即断流续写。"""

    rounds: list = []
    hits: int = 0

    def log_message(self, *a):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        if self.path.endswith("/user-api/user/refresh"):
            self._json({"code": 0, "result": {"access_token": "tok"}})
            return
        if not self.path.endswith("/assistant/stream"):
            self.send_response(404)
            self.end_headers()
            return
        type(self).hits += 1
        idx = min(self.hits, len(self.rounds)) - 1
        deltas, finish = self.rounds[idx]
        parts_frames = []
        for j, d in enumerate(deltas):
            frame = {"parts": [{"logic_id": "b1", "status": "init",
                                "content": [{"type": "text", "text": d}]}]}
            if self.hits == 1 and j == 0:
                frame["conversation_id"] = "conv-degraded"
            parts_frames.append(frame)
        if finish:
            parts_frames.append({
                "status": "finish",
                "parts": [{"logic_id": "b1", "status": "finish",
                           "content": [{"type": "text",
                                        "text": "".join(deltas)}]}],
            })
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for f in parts_frames:
            self.wfile.write(f"data: {json.dumps(f, ensure_ascii=False)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        if not finish:
            self.close_connection = True

    def _json(self, obj):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(obj).encode())


@pytest.fixture()
def degraded_server():
    _DegradedHandler.rounds = []
    _DegradedHandler.hits = 0
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _DegradedHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _client(base: str) -> ChatGlmWebClient:
    cfg = GlmConfig(
        refresh_token="rt",
        base_url=base,
        anti_rate_limit_enabled=False,
        session_store_path="",
        egress="direct",
    )
    return ChatGlmWebClient(cfg, logging.getLogger("glm.degraded"))


def _stream(c: ChatGlmWebClient, messages) -> tuple[str, list]:
    content = ""
    calls = []
    for chunk in c.stream_chat_completion({
        "model": "GLM-5.3-Flash",
        "stream": True,
        "messages": messages,
        "tools": [{
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
        }],
    }):
        if not chunk.startswith("data: ") or chunk == "data: [DONE]\n\n":
            continue
        obj = json.loads(chunk[len("data: "):])
        for ch in obj.get("choices") or []:
            d = ch.get("delta") or {}
            content += d.get("content") or ""
            for tc in d.get("tool_calls") or []:
                calls.append(tc["function"])
    return content, calls


def test_stream_cut_then_marker_restart_no_fragment_leak(degraded_server):
    """轮 1 被掐在标记半截、轮 2 重启完整标记：半截不漏进正文，调用照常解析。"""
    _DegradedHandler.rounds = [
        (["我来写入文件\n<<<TOOL_CA"], False),  # 断流（无 finish 帧）
        ([_write_invoke("D:\\x.txt", "hi")], True),  # 续写重启完整标记
    ]
    c = _client(degraded_server)
    content, calls = _stream(c, [{"role": "user", "content": "写文件"}])
    assert "TOOL" not in content and "<<<" not in content
    assert content.startswith("我来写入文件")
    assert calls and calls[0]["name"] == "write"


def test_stream_truncated_marker_tail_dropped(degraded_server):
    """流末停在半截标记：收尾丢弃残迹，前段正文完整。"""
    _DegradedHandler.rounds = [(["回答正文，请看：\n<<<TOOL_CA"], True)]
    c = _client(degraded_server)
    content, calls = _stream(c, [{"role": "user", "content": "写文件"}])
    assert content.startswith("回答正文，请看：")
    assert "TOOL" not in content and "<<<" not in content
    assert not calls


def test_stream_degraded_full_opener_held(degraded_server):
    """缺角括号的完整标记（<<TOOL_CALL>>）：按碎片正则认 opener，不漏正文。"""
    degraded = _write_invoke("D:\\x.txt", "hi").replace(MARKER, "<<TOOL_CALL>>", 1)
    _DegradedHandler.rounds = [([f"好的\n{degraded}"], True)]
    c = _client(degraded_server)
    content, calls = _stream(c, [{"role": "user", "content": "写文件"}])
    assert "TOOL_CALL" not in content
    assert calls and calls[0]["name"] == "write"


def test_prose_with_bracket_like_text_not_eaten(degraded_server):
    """正文里的孤立 `<`、`|`、TOOL_CA 字样不误吞（空白修剪是既有行为）。"""
    _DegradedHandler.rounds = [(["表格 | 结尾 < 和 TOOL_CA 字样保留", "，完毕"], True)]
    c = _client(degraded_server)
    content, _ = _stream(c, [{"role": "user", "content": "写文件"}])
    assert "表格" in content and "TOOL_CA 字样保留" in content and "完毕" in content
    assert "<<<" not in content and "DSML" not in content


def test_nonstream_degraded_marker_stripped_and_parsed(degraded_server):
    """非流式：缺角括号标记 + 完整 DSML → 调用解析成功且正文无标记残迹。"""
    degraded = _write_invoke("D:\\x.txt", "hi").replace(MARKER, "<<TOOL_CALL>>", 1)
    _DegradedHandler.rounds = [([f"好的\n{degraded}"], True)]
    c = _client(degraded_server)
    resp = c.chat_completion({
        "model": "GLM-5.3-Flash",
        "messages": [{"role": "user", "content": "写文件"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "write",
                "description": "写文件",
                "parameters": {
                    "type": "object",
                    "properties": {"filePath": {"type": "string"},
                                   "content": {"type": "string"}},
                    "required": ["filePath", "content"],
                },
            },
        }],
    })
    msg = resp["choices"][0]["message"]
    assert "TOOL_CALL" not in (msg.get("content") or "")
    assert msg.get("tool_calls") and msg["tool_calls"][0]["function"]["name"] == "write"


def test_nonstream_orphan_closer_stripped():
    """孤立闭合标签（opener 被上游吃掉）：不漏进正文。"""
    assert _strip_tool_block("正文\n</|DSML|tool_calls>") == "正文"
    assert _strip_tool_block("正文\n|DSML|tool_calls>") == "正文"


def test_normalize_and_scrub_units():
    assert normalize_tool_marker("a\n<<TOOL_CALL>>\nb", MARKER) == \
        "a\n<<<TOOL_CALL>>>\nb"
    assert normalize_tool_marker("没有标记的正文", MARKER) == "没有标记的正文"
    assert scrub_marker_fragments("正文\n</|DSML|tool_calls>") == "正文\n"
    assert scrub_marker_fragments("普通 < 和 | 保留") == "普通 < 和 | 保留"


def test_default_buffer_behavior_unchanged():
    """kimi/qwen 共享的扣留器不传 fragment_re：无精确 opener 时按旧规则扣尾。"""
    b = DeltaToolBuffer([MARKER])
    assert b.feed_delta("正文\n<<<TOOL_CA") == ""
    assert b.flush() == "正文\n<<<TOOL_CA"  # 旧行为：flush 原样放行全部扣留
