"""license._verify_record / license.status 的字段契约。

覆盖本次改动新增的字段：
    activated / expires        精确到分钟的本地时间 "%Y-%m-%d %H:%M"
    activated_at / expires_ts  原始时间戳
    seconds_left               剩余秒数（过期后钳到 0）
以及 valid / expired / tampered 三条路径。
"""
from __future__ import annotations

import re
import time

import pytest

from webai2api import license as lic
from webai2api import issuer

TS_FMT = "%Y-%m-%d %H:%M"
FMT_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$")


def _make_rec(signed_codes: dict, activated_at: float, key: str = "valid") -> dict:
    """按 write_record 的格式构造记录（但不落盘、不写注册表）。"""
    c = signed_codes[key]
    rec = {
        "v": lic.LICENSE_VERSION,
        "code": c["code"],
        "code_id": c["id"],
        "days": int(c["days"]),
        "issued": c["issued"],
        "activated_at": float(activated_at),
        "machine": lic.machine_code(),
    }
    rec["sig"] = lic._sign_record(rec)
    return rec


def test_fixture_code_is_signed_by_author_key(signed_codes):
    """fixtures 里的授权码必须经内置公钥验签通过，否则后面所有用例都无意义。"""
    pub = lic.public_key()
    assert pub is not None
    for item in signed_codes.values():
        assert issuer.verify_code(item["code"], pub) is not None


def test_valid_record_fields(signed_codes):
    activated = time.time() - 3 * 86400 - 3600  # 3 天 1 小时前激活，30 天有效期
    rec = _make_rec(signed_codes, activated)
    st = lic._verify_record(rec)

    assert st["valid"] is True
    assert st["code"] == "ok"
    assert st["reason"] == ""
    assert st["days"] == 30
    assert st["days_left"] == 26
    assert st["code_id"] == signed_codes["valid"]["id"]

    # 精确到分钟的时间字符串
    assert st["activated"] == time.strftime(TS_FMT, time.localtime(activated))
    assert st["activated"] == time.strftime("%Y-%m-%d %H:%M", time.localtime(activated))
    assert st["expires"] == time.strftime(TS_FMT, time.localtime(activated + 30 * 86400))
    assert FMT_RE.match(st["activated"]) and FMT_RE.match(st["expires"])
    assert len(st["activated"]) == 16

    # 时间戳与剩余秒数
    assert st["activated_at"] == pytest.approx(activated)
    assert st["expires_ts"] == pytest.approx(activated + 30 * 86400)
    assert st["expires_ts"] > st["activated_at"]
    remain = st["expires_ts"] - time.time()
    assert abs(st["seconds_left"] - remain) <= 2


def test_expired_record_fields(signed_codes):
    activated = time.time() - 40 * 86400  # 30 天有效期，已过期 10 天
    rec = _make_rec(signed_codes, activated)
    st = lic._verify_record(rec)

    assert st["valid"] is False
    assert st["code"] == "expired"
    assert st["reason"] == f"授权已于 {st['expires']} 到期"
    assert st["seconds_left"] == 0
    assert "days_left" not in st
    # 过期路径同样带完整时间字段（面板要靠它显示上次激活/到期）
    assert st["activated"] == time.strftime(TS_FMT, time.localtime(activated))
    assert st["expires"] == time.strftime(TS_FMT, time.localtime(activated + 30 * 86400))
    assert st["activated_at"] == pytest.approx(activated)
    assert st["expires_ts"] == pytest.approx(activated + 30 * 86400)
    assert st["expires_ts"] < time.time()


def test_expires_is_exact_activation_plus_days(signed_codes):
    """expires 取的是「激活时刻 + 天数」，不是「到期日前一秒」。"""
    activated = time.time() - 3600
    rec = _make_rec(signed_codes, activated)
    st = lic._verify_record(rec)
    assert st["expires"] == time.strftime(TS_FMT, time.localtime(activated + 30 * 86400))
    assert st["activated"][:10] == time.strftime("%Y-%m-%d", time.localtime(activated))


def test_just_expired_no_days_left(signed_codes):
    activated = time.time() - 30 * 86400 - 5  # 过期 5 秒
    st = lic._verify_record(_make_rec(signed_codes, activated))
    assert st["valid"] is False
    assert st["code"] == "expired"
    assert st["seconds_left"] == 0
    assert "days_left" not in st


def test_tampered_record(signed_codes):
    rec = _make_rec(signed_codes, time.time() - 86400)
    rec["days"] = 36500  # 手改天数 → HMAC 必然对不上
    st = lic._verify_record(rec)
    assert st["valid"] is False
    assert st["code"] == "tampered"
    assert st["reason"] == "授权文件被修改，请重新输入授权码"
    # 被拒绝的记录不应泄露激活/到期字段
    assert "expires" not in st and "seconds_left" not in st


def test_tampered_signature_field(signed_codes):
    rec = _make_rec(signed_codes, time.time() - 86400)
    rec["sig"] = "0" * 64
    st = lic._verify_record(rec)
    assert st["valid"] is False and st["code"] == "tampered"


def test_other_device_before_hmac(signed_codes):
    """换机器要报 other_device，而不是 tampered（顺序契约）。"""
    rec = _make_rec(signed_codes, time.time() - 86400)
    rec["machine"] = "AAAA-BBBB-CCCC-DDDD"
    st = lic._verify_record(rec)
    assert st["valid"] is False
    assert st["code"] == "other_device"
    assert st["machine"] == "AAAA-BBBB-CCCC-DDDD"


def test_bad_code_when_signature_invalid(license_sandbox):
    """码本身不是作者签的（长度/结构对但签名错）→ bad_code。"""
    act = time.time() - 3600
    rec = {
        "v": lic.LICENSE_VERSION,
        "code": "WB2.AUAm2yMAHgidAA." + "A" * 86,
        "code_id": "4026DB23",
        "days": 30,
        "issued": "2024-01-15",
        "activated_at": act,
        "machine": lic.machine_code(),
    }
    rec["sig"] = lic._sign_record(rec)
    st = lic._verify_record(rec)
    assert st["valid"] is False and st["code"] == "bad_code"


def test_bad_record_without_activated(signed_codes):
    rec = _make_rec(signed_codes, 0)
    st = lic._verify_record(rec)
    assert st["valid"] is False and st["code"] == "bad_record"


def test_legacy_version(signed_codes):
    rec = _make_rec(signed_codes, time.time() - 3600)
    rec["v"] = 1
    rec["sig"] = lic._sign_record(rec)
    st = lic._verify_record(rec)
    assert st["valid"] is False and st["code"] == "legacy"


def test_status_exposes_new_fields(license_sandbox, signed_codes):
    """status() 汇总版必须透传 activated_at / expires_ts / seconds_left。"""
    activated = time.time() - 2 * 86400
    rec = _make_rec(signed_codes, activated)
    lic.write_record = _no_write  # 防御：万一被测代码调用了 write_record
    (license_sandbox / "license.key").write_text(_dump(rec), encoding="utf-8")

    class _Store:
        def __init__(self):
            self.kv = {}

        def get_setting(self, k, d=""):
            return self.kv.get(k, d)

        def set_setting(self, k, v):
            self.kv[k] = v

    st = lic.status(license_sandbox, _Store())
    assert st["valid"] is True
    assert st["activated"] == time.strftime(TS_FMT, time.localtime(activated))
    assert st["expires"] == time.strftime(TS_FMT, time.localtime(activated + 30 * 86400))
    assert st["activated_at"] == pytest.approx(activated)
    assert st["expires_ts"] == pytest.approx(activated + 30 * 86400)
    assert st["seconds_left"] > 0
    assert st["clock_ok"] is True and st["days"] == 30


def test_status_missing_when_fully_isolated(isolated_license):
    st = lic.status(isolated_license)
    assert st["valid"] is False
    assert st["code"] == "missing"
    assert st["expires"] == "" and st["activated"] == ""
    assert st["activated_at"] == 0.0 and st["expires_ts"] == 0.0
    assert st["seconds_left"] == 0


def test_status_clock_strikes_invalidates(license_sandbox, signed_codes):
    rec = _make_rec(signed_codes, time.time() - 3600)
    lic.write_record = _no_write
    (license_sandbox / "license.key").write_text(_dump(rec), encoding="utf-8")

    class _Store:
        def get_setting(self, k, d=""):
            return {"license_last_seen": str(time.time() + 86400),
                    "license_clock_strikes": "3"}.get(k, d)

        def set_setting(self, k, v):
            pass

    st = lic.status(license_sandbox, _Store())
    assert st["valid"] is False and st["code"] == "clock_tamper"
    assert st["clock_ok"] is False


# ── 小工具（故意不用 write_record，避免写注册表/影子文件） ──
def _dump(rec: dict) -> str:
    import json

    return json.dumps(rec, ensure_ascii=False, indent=2)


def _no_write(*a, **kw):
    raise AssertionError("测试不得调用 write_record()")
