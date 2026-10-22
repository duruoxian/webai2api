"""登录窗口清理逻辑：精确杀本账号残留进程 + cookie 兜底清除 + 出口角标 JS。"""
from __future__ import annotations

import json
from pathlib import Path

import webai2api.login_webview as lw


class _Run:
    def __init__(self, stdout=""):
        self.stdout = stdout


def test_kill_only_matching_storage(tmp_path, monkeypatch) -> None:
    storage = tmp_path / "acc-1"
    calls: list[list[str]] = []

    def fake_run(cmd, **kw):
        if cmd[0] == "powershell":
            procs = [
                {"ProcessId": 111,
                 "CommandLine": f'--user-data-dir="{storage}\\EBWebView" --other'},
                {"ProcessId": 222,
                 "CommandLine": r'--user-data-dir="C:\Other\acc-2\EBWebView"'},
                {"ProcessId": 333, "CommandLine": None},
            ]
            return _Run(json.dumps(procs))
        calls.append(cmd)
        return _Run()

    monkeypatch.setattr(lw.subprocess, "run", fake_run)
    monkeypatch.setattr(lw.time, "sleep", lambda s: None)
    lw._kill_webview_processes_for(storage)

    assert len(calls) == 1, calls          # 只杀匹配本账号目录的 111
    assert calls[0][2] == "111"


def test_kill_tolerates_powershell_failure(tmp_path, monkeypatch) -> None:
    def boom(cmd, **kw):
        raise OSError("no powershell")
    monkeypatch.setattr(lw.subprocess, "run", boom)
    lw._kill_webview_processes_for(tmp_path)  # 不应抛异常


def test_purge_login_state_removes_session_files(tmp_path) -> None:
    default = tmp_path / "EBWebView" / "Default"
    default.mkdir(parents=True)
    (default / "Cookies").write_bytes(b"x")
    (default / "Login Data").write_bytes(b"x")
    ls = default / "Local Storage"
    ls.mkdir()
    (ls / "leveldb").mkdir()
    (ls / "leveldb" / "000003.log").write_bytes(b"secret")
    keep = default / "Preferences"
    keep.write_text("{}", encoding="utf-8")

    lw._purge_login_state(tmp_path)

    assert not (default / "Cookies").exists()
    assert not (default / "Login Data").exists()
    assert not ls.exists()
    assert keep.exists()


def test_purge_login_state_missing_dir_is_noop(tmp_path) -> None:
    lw._purge_login_state(tmp_path / "nope")   # 不应抛异常


def test_egress_badge_js_is_single_expression() -> None:
    js = lw._egress_badge_js("direct")
    assert js.startswith("(function(){")
    assert "直连" in js
    # JSON 字面量注入安全：换 mode 只改文案不改结构
    assert "跟随系统代理" in lw._egress_badge_js("system")
    assert "自定义代理" in lw._egress_badge_js("custom")
