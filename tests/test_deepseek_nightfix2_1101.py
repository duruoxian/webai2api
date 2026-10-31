"""2026-11-01 夜间批次5回归（deepseek 未覆盖面审计）。

- 连接租约「检查+置位」原子化：并发 _get_http_conn 不得租出同一条共享连接。
- HIF：部分失败不缓存 600s、失败不覆盖并发取到的有效值、请求走 egress。
- curl_cffi 的 Headers 只有 get_list：Set-Cookie 吸收双兼容。
- conv 指纹碰撞守卫：同 key 不同前缀哈希 → fresh，不接别人的会话。
- close() flush 会话缓存（dispose 链自动识别）。
"""

from __future__ import annotations

import io
import json
import logging
import threading
import time
from types import SimpleNamespace

import pytest

from webai2api.adapters.deepseek.services.deepseek_auth import DeepSeekAuthManager
from webai2api.adapters.deepseek.services.session_cache import (
    SessionCache,
    SessionState,
    prefix_hash,
)
import webai2api.adapters.deepseek.services.deepseek_client as ds_mod
from webai2api.adapters.deepseek.services.deepseek_client import DeepSeekWebClient


# ── Set-Cookie 吸收：get_list-only（curl_cffi 形态）──────────────


class _CffiHeaders:
    """模拟 curl_cffi Headers：只有 get_list，没有 get_all。"""

    def __init__(self, cookies):
        self._cookies = cookies

    def get_list(self, name):
        if name == "Set-Cookie":
            return list(self._cookies)
        return []

    def get(self, name):
        return self._cookies[0] if self._cookies else None


class _FakeResponse:
    def __init__(self, cookies, body=b"{}"):
        self.headers = _CffiHeaders(cookies)
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _auth_manager() -> DeepSeekAuthManager:
    m = DeepSeekAuthManager.__new__(DeepSeekAuthManager)
    m.config = SimpleNamespace(
        auth_token="t", device_id="dev", user_agent="UA", settings_token="",
        cookie="seed=1", hif_leim_url="https://x/leim", hif_dliq_url="https://x/dliq",
        egress="direct", proxy="", debug_dump_all=False,
    )
    m.logger = logging.getLogger("ds.auth.test")
    from webai2api.adapters.deepseek.services.cookie_jar import CookieJar

    m.cookie_jar = CookieJar(seed_cookie="seed=1", logger=m.logger)
    m._lock = threading.RLock()
    m._hif_tokens = None
    return m


def test_absorb_cookies_with_cffi_headers():
    m = _auth_manager()
    m.absorb_response_cookies(
        _FakeResponse(["HWWAFSESID=abc; Path=/", "HWWAFSESTIME=123; Path=/"])
    )
    header = m.cookie_jar.build_cookie_header()
    assert "HWWAFSESID=abc" in header, f"curl_cffi 形态的 Set-Cookie 必须被吸收: {header}"
    assert "HWWAFSESTIME=123" in header


# ── HIF：部分失败不缓存 / 空结果不缓存 / 成功缓存 ────────────────


def _hif_body(value: str) -> bytes:
    return json.dumps({"data": {"biz_data": {"value": value}}}).encode()


def test_hif_partial_failure_not_cached(monkeypatch):
    m = _auth_manager()
    calls = {"n": 0}

    def fake_open(url, timeout=10):
        calls["n"] += 1
        if url.endswith("/leim"):
            return _FakeResponse([], _hif_body("LEIM1"))
        raise RuntimeError("dliq network down")

    monkeypatch.setattr(m, "_open_hif_url", fake_open)
    tokens = m.get_hif_tokens()
    assert tokens.leim == "LEIM1" and tokens.dliq == ""
    assert m._hif_tokens is None, "部分失败绝不能进缓存（此前缓存 600s）"

    # 网络恢复：下次调用立即重试并缓存完整结果
    def fake_open2(url, timeout=10):
        return _FakeResponse([], _hif_body("DLIQ1") if url.endswith("/dliq")
                             else _hif_body("LEIM2"))

    monkeypatch.setattr(m, "_open_hif_url", fake_open2)
    tokens2 = m.get_hif_tokens()
    assert tokens2.leim == "LEIM2" and tokens2.dliq == "DLIQ1"
    assert m._hif_tokens is not None and m._hif_tokens.dliq == "DLIQ1"


def test_hif_total_failure_not_cached_and_no_crash(monkeypatch):
    m = _auth_manager()

    def fake_open(url, timeout=10):
        raise RuntimeError("down")

    monkeypatch.setattr(m, "_open_hif_url", fake_open)
    tokens = m.get_hif_tokens()
    assert tokens.leim == "" and tokens.dliq == ""
    assert m._hif_tokens is None


# ── 连接租约原子性 ───────────────────────────────────────────────


def _conn_client() -> DeepSeekWebClient:
    c = DeepSeekWebClient.__new__(DeepSeekWebClient)
    c.config = SimpleNamespace(egress="direct", proxy="")
    c.logger = logging.getLogger("ds.conn.test")
    c._http_conn = None
    c._http_host = None
    c._http_proxy = ""
    c._conn_busy = False
    c._conn_lock = threading.Lock()
    return c


def test_conn_lease_never_double_shared():
    c = _conn_client()
    shared_ids = []
    lock = threading.Lock()
    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()
        conn, shared = c._get_http_conn("chat.deepseek.com", 15)
        if shared:
            with lock:
                shared_ids.append(id(conn))
        time.sleep(0.02)
        c._release_conn(conn, shared)

    for _ in range(25):
        shared_ids.clear()
        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start(); t2.start(); t1.join(); t2.join()
        assert len(shared_ids) <= 1, "同一条共享连接绝不能租给两个线程"


# ── 指纹碰撞守卫 ────────────────────────────────────────────────


def _resolve_client(tmp_path):
    c = DeepSeekWebClient.__new__(DeepSeekWebClient)
    c.config = SimpleNamespace(
        default_model="deepseek-chat", max_turns_per_session=40,
        vision_enabled=False,
    )
    c.logger = logging.getLogger("ds.collide.test")
    c.cache_enabled = True
    c.session_cache = SessionCache(max_size=10, persist_path=None)
    c._stats_lock = threading.Lock()
    c._stats = {}
    monkeypatch_totem = None
    return c


def test_fingerprint_collision_goes_fresh(tmp_path, monkeypatch):
    c = _resolve_client(tmp_path)
    monkeypatch.setattr(c, "_upload_images_from_messages", lambda msgs: [])
    monkeypatch.setattr(
        ds_mod, "convert_openai_to_deepseek", lambda payload, config: {"prompt": "x"}
    )
    monkeypatch.setattr(
        ds_mod, "convert_openai_to_deepseek_incremental",
        lambda payload, config: {"prompt": "x"},
    )
    sys_msg = {"role": "system", "content": "固定系统提示"}
    first_user = {"role": "user", "content": "模板化开场白"}
    msgs_a = [sys_msg, first_user]
    # 对话 A 第一轮完成：存档 hash = A 的前缀 + assistant 回答
    key_a = __import__(
        "webai2api.adapters.deepseek.services.session_cache", fromlist=["conv_fingerprint"]
    ).conv_fingerprint(msgs_a, "deepseek-chat", "key-1")
    stored_hash = prefix_hash(
        [sys_msg, first_user, {"role": "assistant", "content": "A的回答"}],
        model="deepseek-chat",
    )
    c.session_cache.put(
        key_a,
        SessionState(
            session_id="sess-A", turn_count=1, msg_count=2, history_hash=stored_hash
        ),
    )
    # 对话 B：同 system + 同首条 user（模板撞车），直接发第二问
    msgs_b = [sys_msg, first_user, {"role": "assistant", "content": "B的回答"},
              {"role": "user", "content": "B 的问题"}]
    # 2 msg == stored 2：msg_count 守卫挡不住；哈希不同必须 fresh
    payload_b = {"model": "deepseek-chat", "messages": msgs_b, "_wapi_key_id": "key-1"}
    resolved = c._resolve_conversation(payload_b)
    assert resolved["mode"] == "fresh", "碰撞请求绝不接进存档会话"
    assert resolved["session_id"] == ""
    # 存档不被删（原主人 A 还要用）：同前缀的 A 延续轮仍能命中
    msgs_a2 = [sys_msg, first_user, {"role": "assistant", "content": "A的回答"},
               {"role": "user", "content": "A 的问题"}]
    payload_a2 = {"model": "deepseek-chat", "messages": msgs_a2, "_wapi_key_id": "key-1"}
    resolved_a2 = c._resolve_conversation(payload_a2)
    assert resolved_a2["mode"] == "cached", "A 的正常延续仍命中"
    assert resolved_a2["session_id"] == "sess-A"


def test_legit_continuation_still_hits(tmp_path, monkeypatch):
    c = _resolve_client(tmp_path)
    monkeypatch.setattr(c, "_upload_images_from_messages", lambda msgs: [])
    monkeypatch.setattr(
        ds_mod, "convert_openai_to_deepseek_incremental",
        lambda payload, config: {"prompt": "x"},
    )
    sys_msg = {"role": "system", "content": "sys"}
    u1 = {"role": "user", "content": "hi"}
    msgs1 = [sys_msg, u1]
    key = __import__(
        "webai2api.adapters.deepseek.services.session_cache", fromlist=["conv_fingerprint"]
    ).conv_fingerprint(msgs1, "deepseek-chat", "")
    stored_hash = prefix_hash(
        [sys_msg, u1, {"role": "assistant", "content": "答"}], model="deepseek-chat"
    )
    c.session_cache.put(
        key, SessionState(session_id="sess-1", turn_count=1, msg_count=2,
                          history_hash=stored_hash)
    )
    msgs2 = [sys_msg, u1, {"role": "assistant", "content": "答"}, {"role": "user", "content": "继续"}]
    resolved = c._resolve_conversation(
        {"model": "deepseek-chat", "messages": msgs2}
    )
    assert resolved["mode"] == "cached", "正规延续（p_hash==history_hash）必须命中"


# ── close() flush 会话缓存 ───────────────────────────────────────


def test_close_flushes_session_cache(tmp_path):
    c = DeepSeekWebClient.__new__(DeepSeekWebClient)
    c.logger = logging.getLogger("ds.close.test")
    persist = tmp_path / "ds_sessions" / "acc-x.json"
    c.session_cache = SessionCache(max_size=10, persist_path=persist)
    c.session_cache.put(
        "k", SessionState(session_id="sess-1", turn_count=1)
    )  # 落盘节流 5s 内未必写盘
    c._http_conn = None
    c._http_host = None
    c._http_proxy = ""
    c._conn_busy = False
    c._conn_lock = threading.Lock()
    c._browser = None
    c.close()
    assert persist.exists(), "close 必须 flush 会话缓存"
    data = json.loads(persist.read_text(encoding="utf-8"))
    assert "k" in data and data["k"]["session_id"] == "sess-1"
