"""issuer.make_code / parse_code / verify_code 的往返与结构契约。

授权码格式：WB2.<payload b64url>.<sig b64url>
payload 固定 10 字节：
    [0] 版本=1 | [1:5] 码ID | [5:7] 天数(u16 BE) | [7:9] 签发日(u16 BE) | [9] 版本类型
"""
from __future__ import annotations

import base64
import struct
import time

import pytest

from webai2api import ed25519, issuer


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode((s + "=" * (-len(s) % 4)).encode("ascii"))


@pytest.fixture
def seed() -> bytes:
    # 固定 seed：不为每个用例现场读作者私钥，也不会写任何文件
    return bytes(range(32))


@pytest.fixture
def pub(seed) -> bytes:
    return ed25519.public_key(seed)


def test_payload_is_ten_bytes(seed):
    info = issuer.make_code(seed, 30)
    parts = info["code"].split(".")
    assert parts[0] == "WB2"
    assert len(parts) == 3
    payload = _b64d(parts[1])
    assert len(payload) == 10
    assert len(_b64d(parts[2])) == 64  # Ed25519 签名


def test_payload_layout(seed):
    info = issuer.make_code(seed, 90, edition=issuer.EDITION_PRO,
                            issued_at=1705276800.0)  # 2024-01-15 00:00 UTC
    ver, cid, days, issued_day, edition = struct.unpack(
        ">B4sHHB", _b64d(info["code"].split(".")[1]))
    assert ver == issuer.VERSION == 1
    assert cid.hex().upper() == info["id"]
    assert days == 90
    assert issued_day == 1705276800 // 86400 - 17532
    assert edition == issuer.EDITION_PRO
    assert issuer._date_of(issued_day) == "2024-01-15"
    assert info["days"] == 90 and info["issued"] == "2024-01-15"


def test_roundtrip(seed, pub):
    made = issuer.make_code(seed, 365)
    parsed = issuer.parse_code(made["code"])
    assert parsed is not None
    assert parsed["id"] == made["id"]
    assert parsed["days"] == 365
    assert parsed["edition"] == issuer.EDITION_LITE
    assert parsed["issued"] == made["issued"]

    verified = issuer.verify_code(made["code"], pub)
    assert verified is not None
    assert verified["id"] == made["id"]
    assert verified["payload"] == parsed["payload"]


def test_roundtrip_with_whitespace(seed, pub):
    made = issuer.make_code(seed, 7)
    dirty = f"  {made['code']}\r\n"
    assert issuer.verify_code(dirty, pub) is not None


def test_unique_ids(seed):
    ids = {issuer.make_code(seed, 1)["id"] for _ in range(20)}
    assert len(ids) == 20


@pytest.mark.parametrize("days", [0, -5, 1, 3650, 99999])
def test_days_clamped(seed, pub, days):
    made = issuer.make_code(seed, days)
    assert 1 <= made["days"] <= 3650
    assert issuer.verify_code(made["code"], pub)["days"] == made["days"]


def test_tampered_payload_rejected(seed, pub):
    made = issuer.make_code(seed, 30)
    magic, payload, sig = made["code"].split(".")
    raw = bytearray(_b64d(payload))
    raw[5:7] = struct.pack(">H", 3650)  # 把 30 天改成 3650 天
    tampered = f"{magic}.{issuer._b64u(bytes(raw))}.{sig}"
    assert issuer.parse_code(tampered) is not None  # 结构仍合法
    assert issuer.verify_code(tampered, pub) is None  # 但验签失败


def test_tampered_signature_rejected(seed, pub):
    made = issuer.make_code(seed, 30)
    magic, payload, sig = made["code"].split(".")
    raw = bytearray(_b64d(sig))
    raw[0] ^= 0x01
    assert issuer.verify_code(f"{magic}.{payload}.{issuer._b64u(bytes(raw))}", pub) is None


@pytest.mark.parametrize("bad", [
    "", "WB2", "WB2.abc", "WB2.abc.def", "XXX.AUAm2yMAHgidAA.AAAA",
    "WB2.!!!.!!!", "not-a-code",
])
def test_parse_rejects_malformed(bad):
    assert issuer.parse_code(bad) is None
    assert issuer.verify_code(bad, bytes(32)) is None


def test_wrong_public_key(seed):
    made = issuer.make_code(seed, 30)
    other_pub = ed25519.public_key(bytes([7] * 32))
    assert issuer.verify_code(made["code"], other_pub) is None


def test_verify_without_public_key(seed):
    made = issuer.make_code(seed, 30)
    assert issuer.verify_code(made["code"], None) is None


def test_public_key_is_32_bytes(seed):
    assert len(ed25519.public_key(seed)) == 32


def test_payload_edition_byte(seed):
    for edition in (issuer.EDITION_LITE, issuer.EDITION_PRO):
        info = issuer.make_code(seed, 30, edition=edition)
        assert issuer.parse_code(info["code"])["edition"] == edition


def test_issued_day_matches_today(seed):
    made = issuer.make_code(seed, 30)  # 不传 issued_at → 今天
    expected = time.strftime("%Y-%m-%d", time.gmtime())
    assert made["issued"] == expected
