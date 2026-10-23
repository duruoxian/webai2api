"""授权码绑定机器：绑定码只认签发时那台机器，通用码仍兼容。"""
from __future__ import annotations

import time

from webai2api import issuer, license as lic

MACHINE_A = "AAAA-BBBB-CCCC-DDDD"
MACHINE_B = "1111-2222-3333-4444"


def test_v1_code_still_parses(author_seed):
    """旧通用码（10 字节 payload）必须继续可用。"""
    made = issuer.make_code(author_seed, days=30)
    info = issuer.verify_code(made["code"], lic.public_key())
    assert info is not None
    assert info["bound"] is False
    assert info["days"] == 30
    assert info["machine_digest"] == ""


def test_bound_code_embeds_digest(author_seed):
    made = issuer.make_code(author_seed, days=30, machine_code=MACHINE_A)
    assert made["bound"] is True
    info = issuer.verify_code(made["code"], lic.public_key())
    assert info is not None and info["bound"] is True
    assert info["machine_digest"] == issuer.machine_digest(MACHINE_A).hex().upper()


def test_bound_code_rejected_on_other_machine(author_seed, license_sandbox, monkeypatch):
    made = issuer.make_code(author_seed, days=30, machine_code=MACHINE_B)
    monkeypatch.setattr(lic, "machine_code", lambda: MACHINE_A)
    rec = {
        "v": lic.LICENSE_VERSION,
        "code": made["code"],
        "code_id": made["id"],
        "days": made["days"],
        "issued": made["issued"],
        "activated_at": time.time() - 3600,
        "machine": MACHINE_A,
    }
    rec["sig"] = lic._sign_record(rec)
    st = lic._verify_record(rec)
    assert st["valid"] is False
    assert st["code"] == "other_device"
    assert "不是为这台机器签发" in st["reason"]


def test_bound_code_accepted_on_target_machine(author_seed, license_sandbox, monkeypatch):
    made = issuer.make_code(author_seed, days=30, machine_code=MACHINE_A)
    monkeypatch.setattr(lic, "machine_code", lambda: MACHINE_A)
    rec = {
        "v": lic.LICENSE_VERSION,
        "code": made["code"],
        "code_id": made["id"],
        "days": made["days"],
        "issued": made["issued"],
        "activated_at": time.time() - 3600,
        "machine": MACHINE_A,
    }
    rec["sig"] = lic._sign_record(rec)
    st = lic._verify_record(rec)
    assert st["valid"] is True and st["code"] == "ok"


def test_activate_rejects_wrong_machine(author_seed, isolated_license, monkeypatch):
    made = issuer.make_code(author_seed, days=30, machine_code=MACHINE_B)
    monkeypatch.setattr(lic, "machine_code", lambda: MACHINE_A)
    res = lic.activate(isolated_license, made["code"])
    assert res["ok"] is False
    assert "不是为这台机器签发" in res["error"]


def test_machine_digest_normalizes_format():
    """带不带分隔符、大小写不同都应得到同一摘要。"""
    a = issuer.machine_digest("abcd-efgh-ijkl-mnop")
    b = issuer.machine_digest("ABCDEFGHIJKLMNOP")
    c = issuer.machine_digest("abcd efgh ijkl mnop")
    assert a == b == c


def test_general_code_has_no_binding(author_seed):
    made = issuer.make_code(author_seed, days=7, machine_code="")
    assert made["bound"] is False
    assert made["machine_digest"] == ""
