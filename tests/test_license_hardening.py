"""授权离线加强的安全回归（pepper + v3 + 地板 + 防降级 + edition 后门）。

关键：绝不触碰真实注册表/影子文件——用 license_sandbox / isolated_license 装置。
"""
from __future__ import annotations

import json
import time

from webai2api import license as lic
from webai2api import issuer


def _rec(signed_codes, activated_at, v=3, key="valid"):
    c = signed_codes[key]
    rec = {
        "v": v,
        "code": c["code"],
        "code_id": c["id"],
        "days": int(c["days"]),
        "issued": c["issued"],
        "activated_at": float(activated_at),
        "machine": lic.machine_code(),
    }
    rec["sig"] = lic._sign_record(rec) if v >= 3 else _sign_v2(rec)
    return rec


def _sign_v2(rec):
    import hashlib
    import hmac
    body = {k: v for k, v in rec.items() if k != "sig"}
    raw = json.dumps(body, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return hmac.new(lic._hmac_key_legacy(), raw, hashlib.sha256).hexdigest()


def test_v3_signature_requires_pepper(signed_codes):
    """v3 HMAC 必须混入 pepper；无 pepper 的旧公式签不出合法 v3。"""
    assert len(lic._pepper()) >= 32
    now = time.time()
    v3 = _rec(signed_codes, now - 86400, v=3)
    assert lic._sig_ok(v3) is True
    # 攻击者用公开的旧公式（仅机器码）伪造 v3 → 验签失败
    forged = dict(v3)
    forged["activated_at"] = now  # 想把时间拨回今天续命
    forged["sig"] = _sign_v2(forged)  # 旧公式签，对不上 v3
    assert lic._sig_ok(forged) is False


def test_legacy_v2_still_valid_within_grace(signed_codes):
    """已发货客户的 v2 记录（旧公式）在宽限期内仍可信，并可自动升级。"""
    now = time.time()
    v2 = _rec(signed_codes, now - 2 * 86400, v=2)
    st = lic._verify_record(v2)
    assert st["valid"] is True
    assert st.get("upgraded") is True


def test_legacy_v2_rejected_after_grace(signed_codes, monkeypatch):
    monkeypatch.setattr(lic.time, "time", lambda: lic.V2_GRACE_TS + 3600)
    v2 = _rec(signed_codes, time.time() - 86400, v=2)
    st = lic._verify_record(v2)
    assert st["valid"] is False and st["code"] == "legacy"


def test_downgrade_v2_ignored_when_v3_present(signed_codes):
    """同码已有 v3 权威记录时，伪造的更早/更晚 v2 一律被无视。"""
    now = time.time()
    v3 = _rec(signed_codes, now - 5 * 86400, v=3)
    # 攻击者放一份 v2 记录，activated_at 更晚（想重置倒计时）
    evil_v2 = _rec(signed_codes, now, v=2)
    groups = {v3["code_id"]: [v3, evil_v2]}
    best = min(
        [r for r in groups[v3["code_id"]] if int(r["v"]) >= 3]
        or groups[v3["code_id"]],
        key=lambda r: float(r["activated_at"]),
    )
    assert best["v"] == 3
    assert best["activated_at"] == v3["activated_at"]


def test_activation_floor_blocks_countdown_reset(signed_codes, license_sandbox):
    """删光副本后把 activated_at 拨晚想续命 → 被'最早激活地板'按回原值。"""
    now = time.time()
    first_act = now - 25 * 86400
    rec1 = _rec(signed_codes, first_act, v=3)
    lic._floor_write(rec1["code_id"], first_act)
    (license_sandbox / "license.key").write_text(
        json.dumps(rec1, ensure_ascii=False), encoding="utf-8")

    st = lic.load_license(license_sandbox)
    assert st["valid"] is True

    # 攻击：删 data 文件，伪造一份 activated_at=现在 的合法 v3（假设其搞到了 pepper）
    (license_sandbox / "license.key").unlink(missing_ok=True)
    lic._LIC_CACHE["key"] = None
    forged = _rec(signed_codes, now, v=3)
    (license_sandbox / "license.key").write_text(
        json.dumps(forged, ensure_ascii=False), encoding="utf-8")

    st = lic.load_license(license_sandbox)
    assert st["valid"] is True
    # 倒计时被地板钉回 25 天前，续命失败
    assert st["activated_at"] <= now - 20 * 86400


def test_edition_frozen_ignores_env_and_datadir(tmp_path, monkeypatch):
    """打包态：环境变量与数据目录 edition.txt 都不能把 lite 伪造成 pro。"""
    from webai2api import edition as ed

    monkeypatch.setattr(ed, "is_frozen", lambda: True)
    monkeypatch.setattr(ed, "resource_path",
                        lambda *p: tmp_path / "res" / p[0])
    monkeypatch.setattr(ed, "user_data_dir", lambda: tmp_path / "data")
    # 内嵌资源声明为 lite
    (tmp_path / "res").mkdir()
    (tmp_path / "res" / "edition.txt").write_text("lite", encoding="utf-8")
    # 攻击者放 pro 到数据目录 + 设环境变量
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "edition.txt").write_text("pro", encoding="utf-8")
    monkeypatch.setenv("WEBAI2API_EDITION", "pro")

    ed._CACHE = None
    try:
        assert ed.is_lite() is True
    finally:
        ed._CACHE = None


def test_edition_source_allows_env(tmp_path, monkeypatch):
    """源码态仍允许环境变量覆盖（方便开发）。"""
    from webai2api import edition as ed

    monkeypatch.setattr(ed, "is_frozen", lambda: False)
    monkeypatch.setattr(ed, "resource_path",
                        lambda *p: tmp_path / "nope" / p[0])
    monkeypatch.setattr(ed, "user_data_dir", lambda: tmp_path / "data")
    monkeypatch.setenv("WEBAI2API_EDITION", "pro")
    ed._CACHE = None
    try:
        assert ed.is_lite() is False
    finally:
        ed._CACHE = None
