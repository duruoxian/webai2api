"""GLM / Kimi 上游会话清理功能测试（桩驱动，不打真实上游）。"""

from __future__ import annotations

import json
import logging

from webai2api.adapters.chatglm.client import ChatGlmWebClient
from webai2api.adapters.kimi.client import KimiWebClient


class _FakeCache:
    def __init__(self):
        self.deleted = []

    def delete_by_session_id(self, sid):
        self.deleted.append(sid)
        return 1


# ── GLM ──────────────────────────────────────────────


def _glm_client(pages, delete_ok=True):
    from types import SimpleNamespace

    c = object.__new__(ChatGlmWebClient)
    c.logger = logging.getLogger("glm-test")
    c.config = SimpleNamespace(
        base_url="https://chatglm.cn",
        assistant_id="aid-1",
        delete_url="https://chatglm.cn/chatglm/backend-api/assistant/conversation/delete",
        request_timeout=20,
        sign_secret="secret",
    )
    c._conv_cache = _FakeCache()
    c._access = lambda: "tok"
    c._post_json = lambda url, body, headers, timeout: (
        c.deleted.append(body)
        if delete_ok
        else (_ for _ in ()).throw(RuntimeError("x"))
    )
    c.deleted = []
    state = {"page": 0}

    def _conn(url, timeout):
        class _C:
            def request(self, method, path, headers=None):
                pass

            def getresponse(self):
                class _R:
                    def read(self_inner):
                        payload = pages[min(state["page"], len(pages) - 1)]
                        state["page"] += 1
                        return json.dumps(payload).encode()

                return _R()

            def close(self):
                pass

        return _C()

    c._conn = _conn
    c._split = lambda url: ("chatglm.cn", url.split("chatglm.cn", 1)[1])
    return c


def _glm_page(items, has_more=False):
    return {"status": 0, "result": {"conversation_list": items, "has_more": has_more}}


def test_glm_list_sessions_paginates():
    c = _glm_client(
        [
            _glm_page([{"id": "a", "title": "A"}], has_more=True),
            _glm_page([{"id": "b", "title": "B"}]),
        ]
    )
    assert [s["id"] for s in c.list_upstream_sessions()] == ["a", "b"]


def test_glm_purge_deletes_and_invalidates_cache():
    c = _glm_client([_glm_page([{"id": "a", "title": "A"}])])
    res = c.purge_upstream_sessions()
    assert res["deleted"] == 1
    assert c.deleted[0]["conversation_id"] == "a"
    assert c._conv_cache.deleted == ["a"]


def test_glm_keep_titles():
    c = _glm_client(
        [_glm_page([{"id": "a", "title": "重要资料"}, {"id": "b", "title": "测试"}])]
    )
    res = c.purge_upstream_sessions(keep_titles=("重要",))
    assert res["kept"] == 1
    assert res["deleted"] == 1


def test_glm_dry_run_no_delete():
    c = _glm_client([_glm_page([{"id": "a", "title": "A"}])])
    res = c.purge_upstream_sessions(dry_run=True)
    assert res["matched"] == 1 and res["deleted"] == 0
    assert c.deleted == []


# ── Kimi ─────────────────────────────────────────────


def _kimi_client(chats):
    from types import SimpleNamespace

    c = object.__new__(KimiWebClient)
    c.logger = logging.getLogger("kimi-test")
    c.config = SimpleNamespace(request_timeout=20)
    c._conv_cache = _FakeCache()
    c.rpc_calls = []

    def _rpc(method, body, timeout=20):
        c.rpc_calls.append((method, body))
        if method == "ListChats":
            return {"chats": chats}
        return {}

    c._rpc = _rpc
    return c


def test_kimi_list_sessions():
    c = _kimi_client([{"id": "k1", "name": "写文件"}])
    assert c.list_upstream_sessions()[0]["id"] == "k1"


def test_kimi_purge_deletes_and_invalidates_cache():
    c = _kimi_client([{"id": "k1", "name": "A"}])
    res = c.purge_upstream_sessions()
    assert res["deleted"] == 1
    assert ("DeleteChat", {"chatId": "k1"}) in c.rpc_calls
    assert c._conv_cache.deleted == ["k1"]


def test_kimi_keep_titles():
    c = _kimi_client([{"id": "k1", "name": "重要"}, {"id": "k2", "name": "测试"}])
    res = c.purge_upstream_sessions(keep_titles=("重要",))
    assert res["kept"] == 1 and res["deleted"] == 1


def test_kimi_dry_run():
    c = _kimi_client([{"id": "k1", "name": "A"}])
    res = c.purge_upstream_sessions(dry_run=True)
    assert res["matched"] == 1 and res["deleted"] == 0
    assert all(m != "DeleteChat" for m, _ in c.rpc_calls)
