"""数据目录的版本分流测试（lite「授权消失」事故的回归防护）。

事故链：提权安装器拉起 lite → exe 目录（Program Files）对管理员可写
→ user_data_dir() 按旧规则「exe 同级优先」在安装目录另建空库 →
用户面板授权/账号全部「消失」。修复后安装态 lite 永远用用户本地目录。
"""
from __future__ import annotations

import sys
from pathlib import Path

import webai2api.paths as paths


def test_frozen_lite_always_uses_profile_dir(monkeypatch, tmp_path):
    """安装态 lite：即便 exe 目录可写，也绝不用它（防提权实例另建空库）。"""
    fake_exe_dir = tmp_path / "program-files-like"   # 可写的假安装目录
    fake_exe_dir.mkdir()
    monkeypatch.setattr(paths, "is_frozen", lambda: True)
    monkeypatch.setattr(paths, "_is_lite", lambda: True)
    monkeypatch.setattr(paths, "_exe_dir", lambda: fake_exe_dir)
    monkeypatch.setattr(paths, "_fallback_data_dir",
                        lambda: tmp_path / "profile" / "data")
    d = paths.user_data_dir()
    assert d == tmp_path / "profile" / "data"
    assert d.exists()
    # 安装目录里不允许出现 data
    assert not (fake_exe_dir / "data").exists()


def test_frozen_pro_prefers_writable_exe_dir(monkeypatch, tmp_path):
    """安装态 pro：行为不变，exe 同级可写就用（便携/开发态设计）。"""
    monkeypatch.setattr(paths, "is_frozen", lambda: True)
    monkeypatch.setattr(paths, "_is_lite", lambda: False)
    monkeypatch.setattr(paths, "_exe_dir", lambda: tmp_path)
    monkeypatch.setattr(paths, "_fallback_data_dir",
                        lambda: tmp_path / "should-not-be-used")
    assert paths.user_data_dir() == tmp_path / "data"


def test_fallback_dir_naming(monkeypatch, tmp_path):
    """回退目录命名：lite 与 pro 分家，互不污染。"""
    monkeypatch.setattr(paths, "_is_lite", lambda: True)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert paths._fallback_data_dir() == tmp_path / "webai2api-lite" / "data"
    monkeypatch.setattr(paths, "_is_lite", lambda: False)
    assert paths._fallback_data_dir() == tmp_path / "webai2api" / "data"


def test_source_state_unchanged():
    """源码态（测试环境）：仍是 exe/项目根 data，回归防护不误伤开发流。"""
    assert not paths.is_frozen()
    d = paths.user_data_dir()
    assert d == Path(paths.__file__).resolve().parent.parent / "data"
