"""纯 Python 的 Ed25519 签名 / 验签（无第三方依赖）。

只用于离线授权文件：私钥留在作者本地，软件内置公钥。
实现参考 RFC 8032 的描述（affine 坐标，够用且短）。
"""
from __future__ import annotations

import hashlib

Q = 2**255 - 19
L = 2**252 + 27742317777372353535851937790883648493
_D = -121665 * pow(121666, Q - 2, Q) % Q
_I = pow(2, (Q - 1) // 4, Q)


def _sha512_int(data: bytes) -> int:
    return int.from_bytes(hashlib.sha512(data).digest(), "little")


def _inv(x: int) -> int:
    return pow(x, Q - 2, Q)


def _recover_x(y: int) -> int:
    xx = (y * y - 1) * _inv(_D * y * y + 1) % Q
    x = pow(xx, (Q + 3) // 8, Q)
    if (x * x - xx) % Q != 0:
        x = x * _I % Q
    if x % 2 != 0:
        x = Q - x
    return x


_BY = 4 * _inv(5) % Q
_B = (_recover_x(_BY), _BY)


def _add(p, q):
    x1, y1 = p
    x2, y2 = q
    t = _D * x1 * x2 * y1 * y2 % Q
    x3 = (x1 * y2 + x2 * y1) * _inv(1 + t) % Q
    y3 = (y1 * y2 + x1 * x2) * _inv(1 - t) % Q
    return (x3, y3)


def _mul(p, e: int):
    if e == 0:
        return (0, 1)
    if e == 1:
        return p
    half = _mul(p, e // 2)
    out = _add(half, half)
    if e & 1:
        out = _add(out, p)
    return out


def _encode(p) -> bytes:
    x, y = p
    return (y | ((x & 1) << 255)).to_bytes(32, "little")


def _decode(data: bytes):
    if len(data) != 32:
        raise ValueError("point must be 32 bytes")
    n = int.from_bytes(data, "little")
    sign = n >> 255
    y = n & ((1 << 255) - 1)
    x = _recover_x(y)
    if x & 1 != sign:
        x = Q - x
    if (-x * x + y * y - 1 - _D * x * x * y * y) % Q != 0:
        raise ValueError("invalid point")
    return (x, y)


def public_key(seed: bytes) -> bytes:
    h = hashlib.sha512(seed).digest()
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    return _encode(_mul(_B, a))


def sign(seed: bytes, msg: bytes) -> bytes:
    h = hashlib.sha512(seed).digest()
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    pk = _encode(_mul(_B, a))
    r = _sha512_int(h[32:] + msg) % L
    rp = _encode(_mul(_B, r))
    s = (r + _sha512_int(rp + pk + msg) % L * a) % L
    return rp + s.to_bytes(32, "little")


def verify(pk: bytes, msg: bytes, sig: bytes) -> bool:
    if len(pk) != 32 or len(sig) != 64:
        return False
    try:
        a = _decode(pk)
        r = _decode(sig[:32])
    except Exception:
        return False
    s = int.from_bytes(sig[32:], "little")
    if s >= L:
        return False
    h = _sha512_int(sig[:32] + pk + msg) % L
    left = _mul(_B, s)
    right = _add(r, _mul(a, h))
    return left == right
