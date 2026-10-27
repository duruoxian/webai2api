"""上游会话清理功能测试（桩驱动，不打真实上游）。"""

from __future__ import annotations

import json

from webai2api.adapters.deepseek.services.deepseek_client import DeepSeekWebClient


class _FakeResp:
    def __init__(self, payload):
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self, *a):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *e):
        return False


class _FakeCache:
    def __init__(self):
        self.deleted = []

    def delete_by_session_id(self, sid):
        self.deleted.append(sid)
        return 1


def _make_client(pages, delete_ok=True):
    """构造只用于会话管理的轻量客户端。"""
    from types import SimpleNamespace

    import logging

    client = object.__new__(DeepSeekWebClient)
    client.config = SimpleNamespace(
        base_url="https://chat.deepseek.com",
        session_list_url="https://chat.deepseek.com/api/v0/chat_session/fetch_page",
        session_delete_url="https://chat.deepseek.com/api/v0/chat_session/delete",
    )
    client.logger = logging.getLogger("session-test")
    client.auth = SimpleNamespace(
        get_auth_headers=lambda: {}, absorb_response_cookies=lambda r: None
    )
    client._proxy_sessions = set()
    import threading

    client._proxy_sessions_lock = threading.Lock()
    client.session_cache = _FakeCache()
    client._bump = lambda k: None

    state = {"page": 0, "deleted_batches": []}

    def _urlopen(request, timeout=15):
        if "fetch_page" in request.full_url:
            payload = pages[min(state["page"], len(pages) - 1)]
            state["page"] += 1
            return _FakeResp(payload)
        # delete
        body = json.loads(request.data.decode())
        state["deleted_batches"].append(body["chat_session_ids"])
        if not delete_ok:
            raise RuntimeError("boom")
        return _FakeResp({"code": 0})

    client._urlopen = _urlopen
    return client, state


def _page(sessions, has_more=False):
    return {
        "code": 0,
        "data": {"biz_data": {"chat_sessions": sessions, "has_more": has_more}},
    }


def test_list_sessions_follows_pagination():
    client, state = _make_client(
        [
            _page(
                [{"id": "a", "title": "A"}, {"id": "b", "title": "B"}], has_more=True
            ),
            _page([{"id": "c", "title": "C"}], has_more=False),
        ]
    )
    sessions = client.list_upstream_sessions()
    assert [s["id"] for s in sessions] == ["a", "b", "c"]


def test_dry_run_does_not_delete():
    client, state = _make_client([_page([{"id": "a", "title": "A"}])])
    res = client.purge_upstream_sessions(dry_run=True)
    assert res["total"] == 1
    assert res["matched"] == 1
    assert res["deleted"] == 0
    assert state["deleted_batches"] == []


def test_purge_deletes_and_invalidates_cache():
    client, state = _make_client([_page([{"id": "a", "title": "A"}])])
    res = client.purge_upstream_sessions()
    assert res["deleted"] == 1
    assert state["deleted_batches"] == [["a"]]
    assert client.session_cache.deleted == ["a"]


def test_keep_titles_preserved():
    client, state = _make_client(
        [_page([{"id": "a", "title": "重要资料"}, {"id": "b", "title": "测试"}])]
    )
    res = client.purge_upstream_sessions(keep_titles=("重要",))
    assert res["kept"] == 1
    assert res["deleted"] == 1
    assert state["deleted_batches"] == [["b"]]


def test_batches_are_chunked_by_20():
    sessions = [{"id": "id%02d" % i, "title": "t"} for i in range(45)]
    client, state = _make_client([_page(sessions)])
    res = client.purge_upstream_sessions()
    assert res["deleted"] == 45
    assert [len(b) for b in state["deleted_batches"]] == [20, 20, 5]
