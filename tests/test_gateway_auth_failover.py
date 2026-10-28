"""多账号池 auth 失败必须换号重试的回归（2026-10-29 真机冒烟抓到）。

场景：deepseek 池两个账号，A 号 token 过期（Authorization Failed / invalid token）。
旧行为：gateway 对 auth 类失败直接 break（注释写着「换号也没用」）——那个假设
在单账号时代成立，多账号池里各号凭证独立，撞上死号本次请求直接 502，用户侧
表现为 opencode/ZCode 里工具链随机失败。修复后：排除死号、换下一个账号把
本次请求救回来；note_auth_failure 的「连续两次自动停用」仍然生效。
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

AUTH_ERR = RuntimeError(
    "获取 PoW challenge 失败: 上游返回 data 为空"
    "（code=40003, msg=Authorization Failed (invalid token)）"
)


def _resp(text: str) -> dict:
    return {
        "model": "deepseek-chat",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": text},
            }
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


class _BoomClient:
    """token 过期的账号：每次调用抛 auth 类错误。"""

    def __init__(self, ac: _AC):
        self.ac = ac

    def chat_completion(self, payload):
        self.ac.calls += 1
        raise AUTH_ERR

    def stream_chat_completion(self, payload):
        self.ac.calls += 1
        raise AUTH_ERR
        yield  # pragma: no cover  生成器语法要求


class _GoodClient:
    def __init__(self, ac: _AC, text: str = "B号的回答"):
        self.ac = ac
        self._text = text

    def chat_completion(self, payload):
        self.ac.calls += 1
        return _resp(self._text)

    def stream_chat_completion(self, payload):
        self.ac.calls += 1
        for piece in ("B", "号", "的", "回", "答"):
            yield (
                "data: "
                + json.dumps(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": piece},
                                "finish_reason": None,
                                "logprobs": None,
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
                + "\n\n"
            )
        yield (
            "data: "
            + json.dumps(
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            )
            + "\n\n"
        )
        yield "data: [DONE]\n\n"


def _make_app(tmp_path: Path, picks: list):
    store = Store(tmp_path / "data" / "webai2api.db")
    logger = logging.getLogger("webai2api.test.authfailover")
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
            return None  # 与真 registry 一致：全被 exclude 时无号可选

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


def test_nonstream_auth_failover_to_healthy_account(tmp_path):
    boom = _AC("死号", _BoomClient.__new__(_BoomClient))
    boom.client = _BoomClient(boom)
    good = _AC("活号", None)
    good.client = _GoodClient(good)
    client, key = _make_app(tmp_path, [boom, good])

    r = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["message"]["content"] == "B号的回答"
    assert boom.calls == 1, "死号应被尝试并排除"
    assert good.calls == 1, "活号应接管本次请求"


def test_stream_auth_failover_to_healthy_account(tmp_path):
    boom = _AC("死号", None)
    boom.client = _BoomClient(boom)
    good = _AC("活号", None)
    good.client = _GoodClient(good)
    client, key = _make_app(tmp_path, [boom, good])

    r = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "deepseek-chat",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200, r.text
    body = r.text
    assert "B" in body and "答" in body
    assert boom.calls == 1
    assert good.calls == 1


def test_single_dead_account_still_fails_fast(tmp_path):
    boom = _AC("死号", None)
    boom.client = _BoomClient(boom)
    client, key = _make_app(tmp_path, [boom])

    r = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 502, "唯一的号死了：应快速失败而不是原地空转"
    assert boom.calls == 1
