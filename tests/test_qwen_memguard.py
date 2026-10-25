"""千问中继内存看护测试（不发网络、不启动真实中继）。"""
from __future__ import annotations

import os
import threading

import pytest

from webai2api import qwen_relay as R


def test_should_recycle_truth_table():
    ok = {"ok": True, "active": False}
    busy = {"ok": True, "active": True}
    assert R._should_recycle(1200, 900, ok) is True        # 超限且空闲 → 回收
    assert R._should_recycle(1200, 900, busy) is False     # 超限但忙 → 等下轮
    assert R._should_recycle(500, 900, ok) is False        # 未超限
    assert R._should_recycle(1200, 0, ok) is False         # limit<=0 = 关闭看护
    assert R._should_recycle(1200, 900, {}) is False       # 健康未知不冒险
    assert R._should_recycle(1200, 900, {"ok": False}) is False  # 中继未就绪


def test_tree_rss_measures_self():
    rss = R.relay_tree_rss_mb(os.getpid())
    assert rss > 0, "测自身进程树应大于 0"
    assert R.relay_tree_rss_mb(0) == 0.0
    assert R.relay_tree_rss_mb(-5) == 0.0


@pytest.mark.skipif(os.name != "nt", reason="Windows 树枚举")
def test_win_tree_contains_self():
    assert os.getpid() in R._win_tree_pids(os.getpid())


class _FakeRuntime:
    ready = True
    logged_in = True
    url = "https://chat.qwen.ai"
    acc_id = "acc-x"
    _active = False

    def has_active(self):
        return self._active


def test_health_payload_carries_active_flag():
    rt = _FakeRuntime()
    p = R._health_payload(rt)
    assert p["ok"] is True and p["active"] is False
    rt._active = True
    assert R._health_payload(rt)["active"] is True


def test_memory_guard_is_singleton():
    """重复启动（多 app 实例/测试）不会堆线程。"""

    class _Store:
        def get_setting(self, k, d):
            return d

    before = R._GUARD_STARTED.is_set()
    R.start_memory_guard(_Store())          # 可能已在别的测试里启动
    R.start_memory_guard(_Store())          # 再启动一次
    assert R._GUARD_STARTED.is_set()
    guards = [t for t in threading.enumerate()
              if t.name == "qwen-relay-memguard"]
    assert len(guards) <= 1, f"看护线程出现多份: {len(guards)}"
    _ = before
