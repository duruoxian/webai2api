"""2026-11-01 多账号自动切换 e2e（用户点名场景）。

- A 号死（auth 类错误）→ 网关自动换 B 号完成本次请求（已有回归覆盖）；
  本文件补：**换号后粘住幸存者**——第 2 轮直接命中 B 号，死号不再被碰。
- 带内错误流（适配器把上游错误包成 data: {"error":...} 而非抛异常）：
  流式换号链同样要接住（_take_until_content 抛出 → 换号）。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi.testclient import TestClient

from webai2api.app import build_app
from webai2api.providers import ProviderRegistry
from webai2api.store import Store

TOKEN_HDR = "X-Admin-Token"
AUTH_ERR = RuntimeError("上游返回 code=40003, msg=Authorization Failed (invalid token)")


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

    def mark_ok(self):
        pass

    def mark_bad(self, seconds=0, exc=None):
        pass

    def cooldown_left(self):
        return 0


class _BoomAuth:
    def __init__(self, ac: _AC):
        self.ac = ac

    def chat_completion(self, payload):
        self.ac.calls += 1
        raise AUTH_ERR

    def stream_chat_completion(self, payload):
        self.ac.calls += 1
        # 带内错误形态：适配器把上游错误包成 SSE 错误块而非抛异常
        yield (
            "data: "
            + json.dumps({"error": {"message": "Authorization Failed", "type": "upstream_error"}},
                         ensure_ascii=False)
            + "\n\n"
        )
        yield "data: [DONE]\n\n"


class _Good:
    def __init__(self, ac: _AC, text: str = "B号的回答"):
        self.ac = ac
        self._text = text

    def chat_completion(self, payload):
        self.ac.calls += 1
        return _resp(self._text)

    def stream_chat_completion(self, payload):
        self.ac.calls += 1
        for piece in self._text:
            yield (
                "data: "
                + json.dumps({"choices": [{"index": 0, "delta": {"content": piece}}]},
                             ensure_ascii=False)
                + "\n\n"
            )
        yield (
            "data: "
            + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            + "\n\n"
        )
        yield "data: [DONE]\n\n"


def _make_app(tmp_path: Path, picks: list):
    store = Store(tmp_path / "data" / "webai2api.db")
    logger = logging.getLogger("webai2api.test.stickyfailover")
    registry = ProviderRegistry(store, logger)
    app = build_app(store, registry, logger)
    store.add_account("deepseek", "死号", {"auth_token": "tok-a"})
    store.add_account("deepseek", "活号", {"auth_token": "tok-b"})
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
    r = client.post("/admin/api/keys", json={"name": "测试Key"}, headers={TOKEN_HDR: tok})
    assert r.json().get("ok") is True, r.text
    return client, r.json()["key"]["key"]


def _msg(content):
    return {"role": "user", "content": content}


def test_failover_round2_sticks_to_survivor(tmp_path):
    boom = _AC("死号", None)
    boom.client = _BoomAuth(boom)
    good = _AC("活号", None)
    good.client = _Good(good)
    client, key = _make_app(tmp_path, [boom, good, good])

    # 第 1 轮：撞死号 → 自动换活号完成
    r1 = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "deepseek-chat", "messages": [_msg("hi")]},
    )
    assert r1.status_code == 200, r1.text
    assert r1.json()["choices"][0]["message"]["content"] == "B号的回答"
    assert boom.calls == 1 and good.calls == 1

    # 第 2 轮（同会话回传首轮回复）：必须直接命中活号，死号不再被碰
    r2 = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "deepseek-chat",
            "messages": [_msg("hi"), {"role": "assistant", "content": "B号的回答"}, _msg("继续")],
        },
    )
    assert r2.status_code == 200, r2.text
    assert boom.calls == 1, "粘性应指向幸存者，死号不再被尝试"
    assert good.calls == 2


def test_inband_error_stream_fails_over(tmp_path):
    """带内错误块（非异常）也要触发流式换号。"""
    boom = _AC("死号", None)
    boom.client = _BoomAuth(boom)
    good = _AC("活号", None)
    good.client = _Good(good, "流式回答")
    client, key = _make_app(tmp_path, [boom, good])

    r = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "deepseek-chat",
            "stream": True,
            "messages": [_msg("hi")],
        },
    )
    assert r.status_code == 200, r.text
    body = r.text
    assert "流" in body and "答" in body, "换号后必须由活号出完整流"
    assert "Authorization Failed" not in body, "死号的错误块不能透给客户端"
    assert boom.calls == 1 and good.calls == 1


def test_inband_error_stream_failover_round2_sticks(tmp_path):
    boom = _AC("死号", None)
    boom.client = _BoomAuth(boom)
    good = _AC("活号", None)
    good.client = _Good(good, "流式回答")
    client, key = _make_app(tmp_path, [boom, good, good])

    r1 = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "deepseek-chat", "stream": True, "messages": [_msg("hi")]},
    )
    assert r1.status_code == 200 and "流" in r1.text and "答" in r1.text
    r2 = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "deepseek-chat",
            "stream": True,
            "messages": [_msg("hi"), {"role": "assistant", "content": "流式回答"}, _msg("继续")],
        },
    )
    assert r2.status_code == 200 and "流" in r2.text and "答" in r2.text
    assert boom.calls == 1, "第 2 轮粘住幸存者"
    assert good.calls == 2
