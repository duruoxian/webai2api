"""GLM agent 体验三修（2026-10-28 用户报障）回归。

1. 「答完又冒思考」：GLM 多阶段思考在正文开流后仍会下发后续 think
   part，流式路径原样透传 reasoning_content，客户端在答案后面再渲染
   思考块。修复后答案开流即抑制后续思考（答前思考不受影响）。
2. 「忘记生成文件」：_collapse_cached_text 原来对历史里出现过任何
   工具调用的每一轮增量都加「…请基于结果直接给出最终回答」，把该
   继续调工具的轮次带偏成直接文字作答。修复后只看最后一条 assistant。
3. 协议提示词补「写文件必须调工具 / old_string 逐字复制」规则
   （GLM TOOL_INSTRUCTION + deepseek DSML 协议提示词）。
"""
from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from webai2api.adapters.chatglm.client import (
    TOOL_INSTRUCTION,
    ChatGlmWebClient,
    _collapse_cached_text,
)
from webai2api.adapters.chatglm.config import GlmConfig
from webai2api.adapters.deepseek.services.tool_calling import build_tool_prompt


# ── 1. _collapse_cached_text 只看最后一条 assistant ─────────────


def _asst_tool_calls(name: str = "write") -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
        ],
    }


def test_collapse_head_added_when_last_assistant_has_tool_calls():
    msgs = [
        {"role": "user", "content": "q"},
        _asst_tool_calls(),
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]
    out = _collapse_cached_text("增量文本", msgs)
    assert out.startswith("[已执行]")
    assert "write" in out
    assert "增量文本" in out


def test_collapse_head_not_added_when_tool_calls_are_old_history():
    """历史里出现过工具调用、但最后一轮 assistant 是普通文本 → 不加头。

    这是「忘记生成文件」的病灶：agent 会话里几乎每段历史都有工具调用，
    旧行为给每一轮增量都加「请直接给出最终回答」，模型于是不再调工具。
    """
    msgs = [
        {"role": "user", "content": "q"},
        _asst_tool_calls(),
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        {"role": "assistant", "content": "上一轮的文字回答"},
        {"role": "user", "content": "生成文件"},
    ]
    out = _collapse_cached_text("增量文本", msgs)
    assert out == "增量文本"


def test_collapse_head_not_added_without_tool_history():
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "q2"},
    ]
    assert _collapse_cached_text("增量文本", msgs) == "增量文本"
    assert _collapse_cached_text("", msgs) == ""


# ── 2. 答后思考抑制（流式 e2e，mock 上游 SSE）────────────────────


class _ThinkTextThinkHandler(BaseHTTPRequestHandler):
    """think → text → think → finish 的多阶段思考流。"""

    captured: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        if self.path.endswith("/user/refresh"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps({"code": 0, "result": {"access_token": "tok"}}).encode()
            )
            return

        def frame(part: dict) -> bytes:
            return f"data: {json.dumps(part, ensure_ascii=False)}\n\n".encode()

        events = [
            frame({
                "conversation_id": "conv-post",
                "parts": [{"logic_id": "t1", "status": "init",
                           "content": [{"type": "think", "think": "答前思考"}]}],
            }),
            frame({
                "parts": [{"logic_id": "b1", "status": "init",
                           "content": [{"type": "text", "text": "最终答案"}]}],
            }),
            frame({
                "parts": [{"logic_id": "t2", "status": "init",
                           "content": [{"type": "think", "think": "答后思考"}]}],
            }),
            frame({
                "status": "finish",
                "parts": [
                    {"logic_id": "t1", "status": "finish",
                     "content": [{"type": "think", "think": "答前思考"}]},
                    {"logic_id": "b1", "status": "finish",
                     "content": [{"type": "text", "text": "最终答案"}]},
                    {"logic_id": "t2", "status": "finish",
                     "content": [{"type": "think", "think": "答后思考"}]},
                ],
            }),
        ]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for f in events:
            self.wfile.write(f)
        self.wfile.write(b"data: [DONE]\n\n")

    do_GET = do_POST


@pytest.fixture()
def glm_server():
    _ThinkTextThinkHandler.captured = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _ThinkTextThinkHandler)
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
    return ChatGlmWebClient(cfg, logging.getLogger("glm.post-answer"))


def test_stream_suppresses_post_answer_thinking(glm_server):
    """答案开流后到达的思考不再下发：reasoning 只含答前思考，content 完整。"""
    c = _client(glm_server)
    reasoning = ""
    content = ""
    for chunk in c.stream_chat_completion({
        "model": "GLM-5.3-Flash",
        "stream": True,
        "messages": [{"role": "user", "content": "你好"}],
    }):
        if not chunk.startswith("data: ") or chunk == "data: [DONE]\n\n":
            continue
        obj = json.loads(chunk[len("data: "):])
        for ch in obj.get("choices") or []:
            d = ch.get("delta") or {}
            reasoning += d.get("reasoning_content") or ""
            content += d.get("content") or ""

    assert content == "最终答案"
    assert "答前思考" in reasoning
    assert "答后思考" not in reasoning


def test_nonstream_still_keeps_full_reasoning(glm_server):
    """非流式路径不受影响：reasoning 字段保留全部思考（顺序由客户端渲染）。"""
    c = _client(glm_server)
    resp = c.chat_completion({
        "model": "GLM-5.3-Flash",
        "messages": [{"role": "user", "content": "你好"}],
    })
    msg = resp["choices"][0]["message"]
    assert "最终答案" in (msg.get("content") or "")
    assert "答后思考" in (msg.get("reasoning_content") or "")


# ── 3. 协议提示词补强 ────────────────────────────────────────────


def test_glm_tool_instruction_has_write_and_verbatim_rules():
    assert "写文件工具" in TOOL_INSTRUCTION
    assert "逐字复制" in TOOL_INSTRUCTION


def test_deepseek_tool_prompt_has_write_and_verbatim_rules():
    tools = [{
        "type": "function",
        "function": {
            "name": "edit",
            "description": "编辑文件",
            "parameters": {
                "type": "object",
                "properties": {"old_string": {"type": "string"}},
                "required": ["old_string"],
            },
        },
    }]
    prompt = build_tool_prompt(tools, "auto")
    assert "file-writing tool" in prompt
    assert "VERBATIM" in prompt
