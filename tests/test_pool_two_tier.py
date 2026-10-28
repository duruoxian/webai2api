"""两级选号回归（2026-10-29 夜间审计 B1）。

场景：库里 health='error'/'expired' 的账号（凭证过期/被停），在 600s
内存冷却过期后会重新进入轮询——多账号池里请求反复撞死号（当日冒烟
实测 deepseek 工具链 502 的根因之一）。修复后：health 坏的账号不参与
轮询、只作好号全冷却时的兜底；粘性指向坏号时改选健康号（会话自动
重建，成功后粘性更新）。重新登录或面板健康检查通过会把 health 置回
ok，账号恢复可选。

附带：qwen `_open_stream` 请求中途抛错必须关连接（socket 泄漏回归）。
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

from webai2api.adapters.base import GenericClientPool
from webai2api.store import Store


def _make_pool(tmp_path, healths: dict):
    store = Store(tmp_path / "data" / "webai2api.db")
    ids = {}
    for name, health in healths.items():
        acc = store.add_account("kimi", name, {"auth_token": "tok"})
        if health != "ok":
            store.update_account(acc["id"], health=health)
        ids[name] = acc["id"]

    logger = logging.getLogger("test.pool.tiertier")
    pool = GenericClientPool(
        store,
        "kimi",
        logger,
        build_config=lambda s, a: SimpleNamespace(daily_request_limit=0),
        client_factory=lambda cfg, logger_: SimpleNamespace(),
    )
    pool.refresh()
    return pool, ids


def test_pick_skips_error_health_accounts(tmp_path):
    pool, ids = _make_pool(tmp_path, {"好号": "ok", "死号": "error"})
    picked = {pool.pick().account["name"] for _ in range(10)}
    assert picked == {"好号"}, f"坏号不应进入轮询，实际抽到 {picked}"


def test_pick_prefers_healthy_on_sticky_to_dead(tmp_path):
    pool, ids = _make_pool(tmp_path, {"好号": "ok", "死号": "error"})
    for _ in range(5):
        ac = pool.pick(prefer=ids["死号"])
        assert ac.account["name"] == "好号", "粘性指向死号时应改选健康号"


def test_pick_falls_back_to_bad_when_no_healthy(tmp_path):
    pool, _ = _make_pool(tmp_path, {"死号": "error"})
    ac = pool.pick()
    assert ac is not None and ac.account["name"] == "死号", (
        "只剩坏号时仍应返回（快速失败给出可行动的错误），而不是无号可用"
    )


def test_pick_recovers_after_health_reset(tmp_path):
    pool, ids = _make_pool(tmp_path, {"好号": "ok", "死号": "error"})
    picked = {pool.pick().account["name"] for _ in range(10)}
    assert picked == {"好号"}
    store = pool.store
    store.update_account(ids["死号"], health="ok")
    pool._last_sync = 0.0  # 强制重新同步（绕过 5 秒节流）
    picked = {pool.pick().account["name"] for _ in range(10)}
    assert picked == {"好号", "死号"}, "health 置回 ok 后应恢复可选"


def test_pick_expired_health_also_deprioritized(tmp_path):
    pool, _ = _make_pool(tmp_path, {"好号": "ok", "过期号": "expired"})
    picked = {pool.pick().account["name"] for _ in range(10)}
    assert picked == {"好号"}


def test_qwen_open_stream_closes_conn_on_request_error(tmp_path, monkeypatch):
    """_open_stream 的 conn.request/getresponse 抛错时必须 close 连接。"""
    from webai2api.adapters.qwen.client import QwenWebClient

    c = QwenWebClient.__new__(QwenWebClient)
    closed = []

    class _Conn:
        def request(self, *a, **k):
            raise ConnectionResetError("reset")

        def close(self):
            closed.append(True)

    c._conn = lambda url, timeout: _Conn()
    c._split = lambda url: ("host", "/path")

    import urllib.error

    try:
        c._open_stream(
            "https://host/path", {}, {"Content-Type": "application/json"}, 5
        )
    except ConnectionResetError:
        pass
    else:  # pragma: no cover
        raise AssertionError("应当把原始异常继续抛出")
    assert closed, "异常路径必须关闭连接"
