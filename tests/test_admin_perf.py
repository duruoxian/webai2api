"""面板增删卡顿根因的回归测试。

事故（2026-10-26 实测定位）：LoginManager.state() 对 status=success 的
账号**每次轮询**都全量 registry.refresh()（重建四家池）。state 接口被
面板轮询 + 每次增删后必刷，且对每个账号各调一次 → 7 账号实测
3~5 秒。修复语义：每个账号的一次登录只触发一次刷新（新登录开始时
重置），刷新放后台线程。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from webai2api import login as login_mod
from webai2api.login import LoginManager
from webai2api.store import Store


class _CountingRegistry:
    def __init__(self) -> None:
        self.refreshes = 0

    def refresh(self, provider=None) -> None:
        self.refreshes += 1


def _wait_refreshes(reg: _CountingRegistry, want: int, timeout: float = 5.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if reg.refreshes >= want:
            time.sleep(0.05)   # 等线程彻底退出
            return True
        time.sleep(0.02)
    return False


def _mk_manager(monkeypatch, tmp_path: Path, reg: _CountingRegistry) -> LoginManager:
    import logging

    st = tmp_path / "login_state"
    st.mkdir(exist_ok=True)

    def fake_state_path(acc_id: str) -> Path:
        return st / f"{acc_id}.json"

    monkeypatch.setattr(login_mod, "state_path", fake_state_path)
    store = Store(tmp_path / "data" / "webai2api.db")
    mgr = LoginManager(store, reg, logging.getLogger("t"))
    return mgr


def test_success_triggers_refresh_only_once(tmp_path, monkeypatch):
    reg = _CountingRegistry()
    mgr = _mk_manager(monkeypatch, tmp_path, reg)
    acc = "acc-a"
    sp = login_mod.state_path(acc)

    sp.write_text(json.dumps({"status": "success"}), encoding="utf-8")
    mgr.state(acc)
    assert _wait_refreshes(reg, 1), "首次 success 应触发一次刷新"
    n = reg.refreshes

    # 后续轮询（面板每几秒一次 + 每次增删后）：success 状态还在，但不再刷
    for _ in range(5):
        mgr.state(acc)
    time.sleep(0.2)
    assert reg.refreshes == n, "success 未变不应重复刷新"


def test_new_login_round_reapplies(tmp_path, monkeypatch):
    reg = _CountingRegistry()
    mgr = _mk_manager(monkeypatch, tmp_path, reg)
    acc = "acc-b"
    sp = login_mod.state_path(acc)

    sp.write_text(json.dumps({"status": "success"}), encoding="utf-8")
    mgr.state(acc)
    assert _wait_refreshes(reg, 1)

    # 重新登录：状态离开 success → 下次 success 要再次刷新（新凭证生效）
    sp.write_text(json.dumps({"status": "waiting"}), encoding="utf-8")
    mgr.state(acc)
    sp.write_text(json.dumps({"status": "success"}), encoding="utf-8")
    mgr.state(acc)
    assert _wait_refreshes(reg, 2), "新一轮登录成功应再次刷新"


def test_error_and_idle_never_refresh(tmp_path, monkeypatch):
    reg = _CountingRegistry()
    mgr = _mk_manager(monkeypatch, tmp_path, reg)
    sp = login_mod.state_path("acc-c")
    sp.write_text(json.dumps({"status": "error"}), encoding="utf-8")
    mgr.state("acc-c")
    sp.write_text(json.dumps({"status": "idle"}), encoding="utf-8")
    mgr.state("acc-c")
    time.sleep(0.2)
    assert reg.refreshes == 0


# ── Store.get_many（state 接口 37 键合一查询）──────────────

def test_get_many_overrides_and_defaults(tmp_path):
    store = Store(tmp_path / "data" / "webai2api.db")
    store.set_setting("glm_search", "true")
    store.set_setting("qwen_timeout", "180")
    out = store.get_many({
        "glm_search": "false",        # 有值 → 覆盖
        "qwen_timeout": "120",        # 有值 → 覆盖
        "not_set_key": "default1",    # 无值 → 默认
    })
    assert out == {"glm_search": "true", "qwen_timeout": "180",
                   "not_set_key": "default1"}


def test_get_many_empty_and_missing(tmp_path):
    store = Store(tmp_path / "data" / "webai2api.db")
    assert store.get_many({}) == {}
    out = store.get_many({"a": "1"})
    assert out == {"a": "1"}
