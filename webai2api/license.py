"""离线授权：机器码 + Ed25519 签名授权文件 + 到期校验。

授权文件（license.key）是 JSON：
    {
      "machine": "ABCD-EFGH-IJKL-MNOP",
      "expires": "2027-10-20",
      "edition": "lite",
      "name": "张三",
      "sig": "<base64 64字节签名>"
    }
签名覆盖除 sig 之外所有字段的规范化 JSON。

私钥只留在作者本地（tools/keygen.py），软件内置公钥。
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from . import ed25519

# 内置公钥（base64）。私钥对应的公钥在 tools/keygen.py 生成后填入。
PUBLIC_KEY_B64 = "KALlewBGC52ZocSmOiE8sQ/SyJQWLbyHUS0W94ybcaY="

LICENSE_FILE = "license.key"
CLOCK_SKEW_TOLERANCE = 6 * 3600  # 允许系统时间比上次运行早 6 小时


def _b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


def public_key() -> bytes | None:
    if PUBLIC_KEY_B64.startswith("REPLACE"):
        return None
    try:
        return _b64d(PUBLIC_KEY_B64)
    except Exception:
        return None


# ── 机器码 ──────────────────────────────────────────
def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=15,
            encoding="utf-8", errors="ignore",
            creationflags=0x08000000 if os.name == "nt" else 0,
        )
        return (out.stdout or "").strip()
    except Exception:
        return ""


def _win_hw_ids() -> list[str]:
    ids: list[str] = []
    ps = (
        "$ErrorActionPreference='SilentlyContinue';"
        "(Get-CimInstance Win32_Processor | Select-Object -First 1).ProcessorId;"
        "(Get-CimInstance Win32_BaseBoard | Select-Object -First 1).SerialNumber;"
        "(Get-CimInstance Win32_DiskDrive | Where-Object {$_.Index -eq 0} | "
        "Select-Object -First 1).SerialNumber"
    )
    out = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps])
    for line in out.splitlines():
        line = line.strip()
        if line and line.lower() not in ("serialnumber", "processorid"):
            ids.append(line)
    return ids


def _generic_hw_ids() -> list[str]:
    ids: list[str] = []
    for p in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            ids.append(Path(p).read_text(encoding="utf-8").strip())
        except Exception:
            pass
    for p in ("/sys/class/dmi/id/product_uuid", "/sys/class/dmi/id/board_serial"):
        try:
            ids.append(Path(p).read_text(encoding="utf-8").strip())
        except Exception:
            pass
    return ids


_CACHE: str | None = None


def machine_code() -> str:
    """CPU + 主板 + 系统盘序列号的哈希，形如 ABCD-EFGH-IJKL-MNOP。"""
    global _CACHE
    if _CACHE:
        return _CACHE
    parts = _win_hw_ids() if os.name == "nt" else _generic_hw_ids()
    parts = [p for p in parts if p]
    if not parts:
        # 兜底：主机名 + MAC + 平台
        parts = [platform.node(), str(platform.machine()), f"{id(platform)}"]
    raw = "|".join(parts).encode("utf-8", "ignore")
    h = hashlib.sha256(raw).hexdigest().upper()
    code = "-".join(h[i:i + 4] for i in range(0, 16, 4))
    _CACHE = code
    return code


# ── 授权文件 ────────────────────────────────────────
def license_path(data_dir: Path) -> Path:
    return data_dir / LICENSE_FILE


def _payload_of(doc: dict) -> bytes:
    body = {k: v for k, v in doc.items() if k != "sig"}
    return json.dumps(body, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def load_license(data_dir: Path) -> dict:
    """读取并校验授权文件，返回 {valid, reason, machine, expires, days_left, name}。"""
    path = license_path(data_dir)
    if not path.exists():
        return {"valid": False, "reason": "未找到授权文件", "code": "missing"}

    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"valid": False, "reason": "授权文件格式错误", "code": "bad_format"}

    pk = public_key()
    if pk is None:
        # 未配置公钥（开发模式）：只校验到期时间
        return _check_expiry(doc, skip_signature=True)

    sig = doc.get("sig") or ""
    try:
        ok = ed25519.verify(pk, _payload_of(doc), _b64d(sig))
    except Exception:
        ok = False
    if not ok:
        return {"valid": False, "reason": "授权签名无效（文件被修改过）",
                "code": "bad_signature"}

    return _check_expiry(doc)


def _check_expiry(doc: dict, skip_signature: bool = False) -> dict:
    exp = str(doc.get("expires") or "").strip()
    name = str(doc.get("name") or "").strip()
    base = {"machine": str(doc.get("machine") or ""), "expires": exp,
            "name": name, "edition": str(doc.get("edition") or "lite"),
            "unverified": skip_signature}

    # 机器码
    if base["machine"] and base["machine"].upper() != machine_code():
        return {**base, "valid": False, "code": "wrong_machine",
                "reason": "授权与本机不匹配，请把本机机器码发给作者"}

    if not exp:
        return {**base, "valid": False, "code": "no_expiry", "reason": "授权缺少到期日期"}

    try:
        end = time.mktime(time.strptime(exp, "%Y-%m-%d")) + 86400  # 当天 23:59:59
    except Exception:
        return {**base, "valid": False, "code": "bad_expiry", "reason": "到期日期格式错误"}

    now = time.time()
    if now >= end:
        return {**base, "valid": False, "code": "expired",
                "reason": f"授权已于 {exp} 到期"}
    days = max(0, int((end - now) // 86400))
    return {**base, "valid": True, "code": "ok", "reason": "", "days_left": days,
            "expires_ts": end}


# ── 防改系统时钟 ────────────────────────────────────
def check_clock(store) -> dict:
    """记录上次运行时间；若系统时间被往回拨，累计异常次数。

    返回 {ok, rollback, strikes}。strikes >= 3 视为可疑。
    """
    now = time.time()
    try:
        last = float(store.get_setting("license_last_seen", "0") or 0)
        strikes = int(store.get_setting("license_clock_strikes", "0") or 0)
    except Exception:
        last, strikes = 0.0, 0
    rollback = bool(last and now < last - CLOCK_SKEW_TOLERANCE)
    if rollback:
        strikes += 1
    elif now > last:
        strikes = 0
        try:
            store.set_setting("license_last_seen", str(now))
        except Exception:
            pass
    try:
        store.set_setting("license_clock_strikes", str(strikes))
    except Exception:
        pass
    return {"ok": not rollback, "rollback": rollback, "strikes": strikes}


def status(data_dir: Path, store=None) -> dict:
    """综合状态：授权 + 时钟。"""
    lic = load_license(data_dir)
    clock = check_clock(store) if store is not None else {"ok": True, "strikes": 0}
    if clock.get("strikes", 0) >= 3:
        lic = {**lic, "valid": False, "code": "clock_tamper",
               "reason": "检测到系统时间被多次回拨，请校正系统时间"}
    return {
        "valid": bool(lic.get("valid")),
        "reason": lic.get("reason", ""),
        "code": lic.get("code", ""),
        "expires": lic.get("expires", ""),
        "days_left": lic.get("days_left", 0),
        "name": lic.get("name", ""),
        "machine": machine_code(),
        "clock_ok": clock.get("ok", True),
        "unverified": lic.get("unverified", False),
    }
