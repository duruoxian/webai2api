"""授权码：签发与校验（作者侧 + 客户端共用）。

授权码格式（一行，约 105 字符）：

    WB2.<payload>.<signature>

payload 为 10 字节紧凑二进制，base64url 无填充：
    [0]      版本 = 1
    [1:5]    授权码 ID（随机 4 字节）
    [5:7]    有效天数（uint16，大端）
    [7:9]    签发日（距 2020-01-01 的天数，uint16，大端）
    [9]      版本类型（0 = 客户版）

签名：Ed25519(payload)，base64url 无填充。

授权码本身**不含机器码**，客户无需把机器码发给作者；
客户端首次激活时把「激活时间 + 本机机器码」写入本地授权文件，
之后其他机器拿同一个码激活会被拒绝。
"""
from __future__ import annotations

import base64
import os
import struct
import time
from pathlib import Path

from . import ed25519

MAGIC = "WB2"
VERSION = 1
EPOCH_DAYS = 17532  # 2020-01-01 距 Unix 纪元的天数
EDITION_LITE = 0
EDITION_PRO = 1


def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")


def _b64ud(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("ascii"))


def _day_index(ts: float | None = None) -> int:
    ts = time.time() if ts is None else ts
    return int(ts // 86400) - EPOCH_DAYS


def _date_of(day_index: int) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime((day_index + EPOCH_DAYS) * 86400))


# ── 签发 ────────────────────────────────────────────
def make_code(seed: bytes, days: int, edition: int = EDITION_LITE,
              issued_at: float | None = None) -> dict:
    """用私钥签一个授权码。返回 {code, id, days, issued, expires}。"""
    days = max(1, min(int(days), 3650))
    code_id = os.urandom(4)
    issued_day = _day_index(issued_at)
    payload = struct.pack(
        ">B4sHHB", VERSION, code_id, days, issued_day & 0xFFFF, edition & 0xFF
    )
    sig = ed25519.sign(seed, payload)
    code = f"{MAGIC}.{_b64u(payload)}.{_b64u(sig)}"
    return {
        "code": code,
        "id": code_id.hex().upper(),
        "days": days,
        "issued": _date_of(issued_day),
        "expires_days": days,
    }


def parse_code(code: str) -> dict | None:
    """解析授权码结构（不验签）。"""
    code = (code or "").strip().replace("\r", "").replace("\n", "")
    parts = code.split(".")
    if len(parts) != 3 or parts[0].upper() != MAGIC:
        return None
    try:
        payload = _b64ud(parts[1])
    except Exception:
        return None
    if len(payload) != 10:
        return None
    ver, cid, days, issued_day, edition = struct.unpack(">B4sHHB", payload)
    if ver != VERSION:
        return None
    return {
        "code": code,
        "payload": payload,
        "sig": _b64ud(parts[2]),
        "id": cid.hex().upper(),
        "days": days,
        "issued": _date_of(issued_day),
        "edition": edition,
    }


def verify_code(code: str, pubkey: bytes | None) -> dict | None:
    """验签并返回解析结果；失败返回 None。"""
    info = parse_code(code)
    if info is None or pubkey is None:
        return None
    try:
        if not ed25519.verify(pubkey, info["payload"], info["sig"]):
            return None
    except Exception:
        return None
    return info


# ── 作者私钥查找 ────────────────────────────────────
def author_key_path() -> Path | None:
    """查找作者私钥；找不到返回 None（客户端正常用户没有）。"""
    env = os.environ.get("WEBAI2API_AUTHOR_KEY")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    candidates.append(Path(base) / "webai2api-author" / "private_key.txt")
    candidates.append(Path(__file__).resolve().parent.parent / "tools" / "private_key.txt")
    for p in candidates:
        try:
            if p.is_file():
                return p
        except Exception:
            continue
    return None


def load_author_seed() -> bytes | None:
    p = author_key_path()
    if not p:
        return None
    try:
        raw = p.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    try:
        seed = bytes.fromhex(raw)
    except Exception:
        return None
    return seed if len(seed) == 32 else None
