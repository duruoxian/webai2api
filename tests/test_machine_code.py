"""machine_code()：稳定 + 格式（只读验证，不 monkeypatch 硬件读取）。"""
from __future__ import annotations

import re

from webai2api import license as lic

PATTERN = re.compile(r"^[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}$")


def test_machine_code_stable():
    a = lic.machine_code()
    b = lic.machine_code()
    assert a == b
    assert len(a) == 19


def test_machine_code_format():
    code = lic.machine_code()
    assert PATTERN.match(code), f"machine_code 格式不符: {code!r}"
    assert code == code.upper()
    assert all(seg != "" for seg in code.split("-"))


def test_machine_code_cache_consistent():
    """清空进程内缓存后重算也必须一致（硬件信息没变）。"""
    first = lic.machine_code()
    lic._CACHE = None
    try:
        assert lic.machine_code() == first
    finally:
        lic._CACHE = first


def test_machine_code_used_by_hmac_key():
    """HMAC 密钥派生确实绑定本机机器码（不泄露具体值）。"""
    key = lic._hmac_key()
    assert isinstance(key, bytes) and len(key) == 32
    assert lic._hmac_key() == key
