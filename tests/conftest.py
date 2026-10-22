"""pytest 公共装置。

安全边界（非常重要）：
- 绝不调用 license.activate() / license.write_record()：前者会写本机影子文件
  （%LOCALAPPDATA%\\webai2api\\.activation.json），后者会写注册表
  HKCU\\Software\\webai2api\\license。
- 把影子文件路径、注册表读写、user_data_dir 全部 monkeypatch 到 tmp_path，
  保证测试不会读写作者/用户的真实授权状态。
- 需要测试"未安装授权"状态时用 `isolated_license` 装置（它会清空授权缓存并
  屏蔽所有真实副本）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webai2api import license as _license  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def license_sandbox(tmp_path, monkeypatch):
    """把授权模块的「持久化出口」全部改到 tmp_path（no-op / 沙箱）。"""
    shadow = tmp_path / "shadow" / ".activation.json"
    monkeypatch.setattr(_license, "_shadow_path", lambda: shadow)
    monkeypatch.setattr(_license, "_shadow_write", lambda rec: None)
    monkeypatch.setattr(_license, "_reg_write", lambda rec: None)
    monkeypatch.setattr(_license, "_reg_read", lambda: None)
    monkeypatch.setattr(_license, "_mirror", lambda rec: None)
    monkeypatch.setattr(_license, "_floor_reg_read", lambda: None)
    monkeypatch.setattr(_license, "_floor_reg_write", lambda data: None)
    monkeypatch.setattr(_license, "_LIC_CACHE", {"key": None, "doc": None})
    yield tmp_path
    _license._LIC_CACHE["key"] = None
    _license._LIC_CACHE["doc"] = None


@pytest.fixture
def isolated_license(tmp_path, monkeypatch):
    """完全隔离：三份副本都不存在（模拟全新机器）。"""
    shadow = tmp_path / "shadow" / ".activation.json"
    monkeypatch.setattr(_license, "_shadow_path", lambda: shadow)
    monkeypatch.setattr(_license, "_shadow_read", lambda: None)
    monkeypatch.setattr(_license, "_shadow_write", lambda rec: None)
    monkeypatch.setattr(_license, "_reg_read", lambda: None)
    monkeypatch.setattr(_license, "_reg_write", lambda rec: None)
    monkeypatch.setattr(_license, "_mirror", lambda rec: None)
    monkeypatch.setattr(_license, "_floor_reg_read", lambda: None)
    monkeypatch.setattr(_license, "_floor_reg_write", lambda data: None)
    monkeypatch.setattr(_license, "_LIC_CACHE", {"key": None, "doc": None})
    yield tmp_path
    _license._LIC_CACHE["key"] = None
    _license._LIC_CACHE["doc"] = None


@pytest.fixture
def signed_codes() -> dict:
    """作者私钥签发的真实授权码（生成一次后缓存，见 fixtures/signed_codes.json）。

    fixtures/ 含真实签发码，已被 .gitignore 排除，绝不可提交到公开仓库；
    在没有该文件的机器（如公开 clone / CI）上跳过依赖它的用例。
    """
    import json

    path = FIXTURES / "signed_codes.json"
    if not path.exists():
        pytest.skip("缺少测试夹具 signed_codes.json（含真实签发码，不入库）")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def author_seed() -> bytes:
    """作者私钥 seed（只在本机存在；不存在则跳过依赖它的用例）。"""
    from webai2api import issuer

    seed = issuer.load_author_seed()
    if seed is None:
        pytest.skip("本机没有作者私钥，跳过签发相关用例")
    return seed
