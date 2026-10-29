"""GLM 图片上传（chat_upload）+ 消息 image part 接入回归。

2026-10-30 真机抓帧确认的上游协议：
  1. POST /chatglm/productivity-api/file/chat_upload (multipart: file/from/assistant_id)
     → result.{file_id, file_url, file_name, file_size}
  2. 消息 content 追加 {"type":"image","image":[{file_name,file_id,image_url,file_size,order}]}
     （字段名逐字对齐网页端，错字段名上游会静默丢图）
  3. meta_data.input_question_type 带图时为 image_upload

本文件锁定：流式请求体带 image part、上传 LRU 缓存、上传失败降级纯文本、
opencode 配置生成器的图片能力按真实能力声明。
"""
from __future__ import annotations

import base64
import json
import logging
import struct
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from webai2api.adapters.chatglm.client import ChatGlmWebClient
from webai2api.adapters.chatglm.config import GlmConfig
from webai2api.adapters.chatglm.uploader import extract_image_urls
from webai2api.admin_api import _build_opencode_snippet


def red_png(w=48, h=48) -> str:
    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * w for _ in range(h))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
           + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


class _VisionHandler(BaseHTTPRequestHandler):
    captured_bodies: list = []
    upload_count: int = 0
    upload_fail: bool = False

    def log_message(self, *a):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        if self.path.endswith("/user-api/user/refresh"):
            self._json({"code": 0, "result": {"access_token": "tok"}})
            return
        if self.path.endswith("/file/chat_upload"):
            type(self).upload_count += 1
            if self.upload_fail:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"boom")
                return
            self._json({
                "status": 0,
                "message": "success",
                "result": {
                    "file_id": "file_abc123",
                    "file_url": "https://t1.chatglm.cn/file/file_abc123.png?sign=x",
                    "file_name": "image_0.png",
                    "file_size": len(raw),
                    "file_type": "png",
                },
            })
            return
        if self.path.endswith("/assistant/stream"):
            type(self).captured_bodies.append(json.loads(raw.decode("utf-8")))
            frames = [
                {"conversation_id": "conv-vision", "parts": []},
                {"parts": [{"logic_id": "b1", "status": "init",
                            "content": [{"type": "text", "text": "收到"}]}]},
                {"status": "finish",
                 "parts": [{"logic_id": "b1", "status": "finish",
                            "content": [{"type": "text", "text": "收到"}]}]},
            ]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for f in frames:
                self.wfile.write(
                    f"data: {json.dumps(f, ensure_ascii=False)}\n\n".encode()
                )
            self.wfile.write(b"data: [DONE]\n\n")
            return
        self.send_response(404)
        self.end_headers()

    def _json(self, obj):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(obj, ensure_ascii=False).encode())


@pytest.fixture()
def vision_server():
    _VisionHandler.captured_bodies = []
    _VisionHandler.upload_count = 0
    _VisionHandler.upload_fail = False
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _VisionHandler)
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
    return ChatGlmWebClient(cfg, logging.getLogger("glm.vision"))


def _last_stream_body() -> dict:
    return _VisionHandler.captured_bodies[-1]


def test_stream_request_carries_image_part(vision_server):
    c = _client(vision_server)
    out = ""
    for chunk in c.stream_chat_completion({
        "model": "GLM-5.3-Flash",
        "stream": True,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "图里是什么颜色？"},
                {"type": "image_url", "image_url": {"url": red_png()}},
            ],
        }],
    }):
        if chunk.startswith("data: ") and chunk != "data: [DONE]\n\n":
            obj = json.loads(chunk[len("data: "):])
            for ch in obj.get("choices") or []:
                out += (ch.get("delta") or {}).get("content") or ""
    assert out == "收到"

    body = _last_stream_body()
    parts = body["messages"][0]["content"]
    assert [p["type"] for p in parts] == ["text", "image"]
    img = parts[1]["image"][0]
    assert img["file_id"] == "file_abc123"
    assert img["image_url"].startswith("https://t1.chatglm.cn/file/")
    assert img["file_name"]
    assert img["order"] == 0
    assert body["meta_data"]["input_question_type"] == "image_upload"


def test_upload_cache_same_image_only_uploads_once(vision_server):
    c = _client(vision_server)
    img = red_png()
    for q in ("第一问", "第二问"):
        c.chat_completion({
            "model": "GLM-5.3-Flash",
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": q},
                            {"type": "image_url", "image_url": {"url": img}}],
            }],
        })
    assert _VisionHandler.upload_count == 1
    assert len(_VisionHandler.captured_bodies) == 2
    for body in _VisionHandler.captured_bodies:
        parts = body["messages"][0]["content"]
        assert parts[1]["type"] == "image"


def test_upload_failure_degrades_to_text(vision_server):
    _VisionHandler.upload_fail = True
    c = _client(vision_server)
    resp = c.chat_completion({
        "model": "GLM-5.3-Flash",
        "messages": [{
            "role": "user",
            "content": [{"type": "text", "text": "纯文字问题"},
                        {"type": "image_url", "image_url": {"url": red_png()}}],
        }],
    })
    assert "收到" in resp["choices"][0]["message"]["content"]
    body = _last_stream_body()
    parts = body["messages"][0]["content"]
    assert [p["type"] for p in parts] == ["text"]
    assert body["meta_data"]["input_question_type"] == "xxxx"


def test_text_only_request_untouched(vision_server):
    c = _client(vision_server)
    c.chat_completion({
        "model": "GLM-5.3-Flash",
        "messages": [{"role": "user", "content": "你好"}],
    })
    body = _last_stream_body()
    assert _VisionHandler.upload_count == 0
    assert body["messages"][0]["content"][0]["type"] == "text"
    assert body["meta_data"]["input_question_type"] == "xxxx"


def test_extract_image_urls_forms_and_order():
    msgs = [
        {"role": "user", "content": "字符串内容忽略"},
        {"role": "user", "content": [
            {"type": "text", "text": "t"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
            {"type": "image_url", "image_url": "https://x/y.png"},
        ]},
    ]
    urls = extract_image_urls(msgs)
    assert urls == ["data:image/png;base64,AAA", "https://x/y.png"]


def test_opencode_snippet_vision_by_real_capability():
    s = _build_opencode_snippet(
        "http://127.0.0.1:8800", "key",
        ["deepseek-chat", "GLM-5.3-Flash", "kimi-k2.6", "qwen3.8-plus"],
    )
    def input_of(model):
        import re
        m = re.search(
            re.escape(json.dumps(model, ensure_ascii=False))
            + r': \{[^}]*"input": (\[[^\]]*\])', s)
        return m.group(1) if m else ""
    assert input_of("deepseek-chat") == '["image","text"]'
    assert input_of("GLM-5.3-Flash") == '["image","text"]'
    assert input_of("kimi-k2.6") == '["text"]'
    assert input_of("qwen3.8-plus") == '["text"]'
