"""离线授权：授权码（天数）+ 首次激活锁定本机 + 剩余天数校验。

授权流程：
1. 作者签发授权码（一行文本，见 issuer.py），码里只有「有效天数 + 签发日 + 码 ID」
2. 客户在设置页粘贴授权码 → 本地验签 → 记录「激活时间 + 本机机器码」
3. 之后每次启动：剩余天数 = 有效天数 - (今天 - 激活日)
4. 同一个码换到别的机器 → 机器码不符 → 拒绝（提示已在其他设备激活）
5. 续费：新码覆盖旧记录，重新从激活当天计算

授权文件 license.key 是 JSON，字段用 HMAC 保护，手改会失效。
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from . import ed25519, issuer

# 内置公钥（base64），与作者私钥配对。
PUBLIC_KEY_B64 = "KALlewBGC52ZocSmOiE8sQ/SyJQWLbyHUS0W94ybcaY="

LICENSE_FILE = "license.key"
LICENSE_VERSION = 2
CLOCK_SKEW_TOLERANCE = 6 * 3600  # 允许系统时间比上次运行早 6 小时


def _b64d(s: str) -> bytes:
    import base64
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
    # 1) Windows 安装 ID：读取注册表，不依赖 PowerShell，重装系统才会变
    try:
        import winreg
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Cryptography",
            0, winreg.KEY_READ | winreg.KEY_WOW64_64KEY,
        ) as k:
            guid, _ = winreg.QueryValueEx(k, "MachineGuid")
            if guid:
                ids.append(str(guid).strip())
    except Exception:
        pass
    # 2) CPU / 主板：硬件标识（不含硬盘，换盘不影响授权）
    ps = (
        "$ErrorActionPreference='SilentlyContinue';"
        "(Get-CimInstance Win32_Processor | Select-Object -First 1).ProcessorId;"
        "(Get-CimInstance Win32_BaseBoard | Select-Object -First 1).SerialNumber"
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
    """CPU + 主板 + Windows 安装 ID 的哈希，形如 ABCD-EFGH-IJKL-MNOP。"""
    global _CACHE
    if _CACHE:
        return _CACHE
    parts = _win_hw_ids() if os.name == "nt" else _generic_hw_ids()
    parts = [p for p in parts if p]
    if not parts:
        # 兜底：主机名 + MAC + 平台（必须稳定，不能用 id()）
        parts = [platform.node(), str(platform.machine()), sys.platform]
        try:
            import uuid
            mac = uuid.getnode()
            if mac and (mac >> 40) % 2 == 0:  # 跳过随机生成的 MAC
                parts.append(f"{mac:012x}")
        except Exception:
            pass
    raw = "|".join(parts).encode("utf-8", "ignore")
    h = hashlib.sha256(raw).hexdigest().upper()
    code = "-".join(h[i:i + 4] for i in range(0, 16, 4))
    _CACHE = code
    return code


# ── 授权文件 ────────────────────────────────────────
def license_path(data_dir: Path) -> Path:
    return data_dir / LICENSE_FILE


def _hmac_key() -> bytes:
    return hashlib.sha256(("wb2:" + machine_code()).encode("utf-8")).digest()


def _sign_record(rec: dict) -> str:
    body = {k: v for k, v in rec.items() if k != "sig"}
    raw = json.dumps(body, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return hmac.new(_hmac_key(), raw, hashlib.sha256).hexdigest()


def _read_record(data_dir: Path) -> dict | None:
    path = license_path(data_dir)
    if not path.exists():
        return None
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return rec if isinstance(rec, dict) else None


def write_record(data_dir: Path, code_info: dict, activated_at: float | None = None) -> dict:
    data_dir.mkdir(parents=True, exist_ok=True)
    rec = {
        "v": LICENSE_VERSION,
        "code": code_info["code"],
        "code_id": code_info["id"],
        "days": int(code_info["days"]),
        "issued": code_info["issued"],
        "activated_at": float(activated_at if activated_at is not None else time.time()),
        "machine": machine_code(),
    }
    rec["sig"] = _sign_record(rec)
    path = license_path(data_dir)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    _LIC_CACHE["key"] = None
    return rec


_LIC_CACHE: dict = {"key": None, "doc": None}


def load_license(data_dir: Path) -> dict:
    """读取并校验授权文件，返回状态字典。

    结果按文件 (mtime, size) 缓存：验签约 0.2s，不能每个请求都做。
    """
    path = license_path(data_dir)
    if not path.exists():
        return {"valid": False, "reason": "未授权，请在设置里输入授权码",
                "code": "missing"}

    try:
        stt = path.stat()
        cache_key = (str(path), stt.st_mtime_ns, stt.st_size)
    except OSError:
        cache_key = None

    if cache_key is not None and _LIC_CACHE["key"] == cache_key:
        return _LIC_CACHE["doc"]

    rec = _read_record(data_dir)
    if rec is None:
        result = {"valid": False, "reason": "授权文件格式错误", "code": "bad_format"}
    else:
        result = _verify_record(rec)

    if cache_key is not None:
        _LIC_CACHE["key"] = cache_key
        _LIC_CACHE["doc"] = result
    return result


def _verify_record(rec: dict) -> dict:
    if int(rec.get("v") or 0) != LICENSE_VERSION:
        return {"valid": False, "reason": "授权文件版本过旧，请重新输入授权码",
                "code": "legacy"}

    # 1) 机器码一致（授权码只能在首台机器上用）
    #    先查机器码，再验 HMAC：换机后 HMAC 必然对不上，要给出正确原因
    rec_machine = str(rec.get("machine") or "").upper()
    cur = machine_code()
    if rec_machine and rec_machine != cur:
        return {"valid": False, "code": "other_device",
                "reason": "此授权码已在其他设备激活", "machine": rec_machine}

    # 2) 本地记录没有被改过
    try:
        expect = _sign_record(rec)
    except Exception:
        expect = ""
    if not hmac.compare_digest(expect, str(rec.get("sig") or "")):
        return {"valid": False, "reason": "授权文件被修改，请重新输入授权码",
                "code": "tampered"}

    # 3) 授权码签名有效
    info = issuer.verify_code(str(rec.get("code") or ""), public_key())
    if info is None:
        return {"valid": False, "reason": "授权码无效", "code": "bad_code"}

    days = int(info["days"])
    try:
        activated = float(rec.get("activated_at") or 0)
    except Exception:
        activated = 0.0
    if activated <= 0:
        return {"valid": False, "reason": "授权记录缺少激活时间，请重新输入授权码",
                "code": "bad_record"}

    expires_ts = activated + days * 86400
    now = time.time()
    base = {
        "code_id": info["id"],
        "days": days,
        "activated": time.strftime("%Y-%m-%d", time.localtime(activated)),
        "expires": time.strftime("%Y-%m-%d", time.localtime(expires_ts - 1)),
        "machine": cur,
    }

    if now >= expires_ts:
        return {**base, "valid": False, "code": "expired",
                "reason": f"授权已于 {base['expires']} 到期"}

    left = max(0, int((expires_ts - now) // 86400))
    return {**base, "valid": True, "code": "ok", "reason": "",
            "days_left": left, "expires_ts": expires_ts}


# ── 激活 ────────────────────────────────────────────
def activate(data_dir: Path, code: str) -> dict:
    """首次激活或续费。返回 {ok, error, ...状态}。"""
    info = issuer.verify_code((code or "").strip(), public_key())
    if info is None:
        return {"ok": False, "error": "授权码无效，请确认复制完整"}

    rec = _read_record(data_dir)
    if rec is not None:
        cur = machine_code()
        rec_machine = str(rec.get("machine") or "").upper()
        if rec_machine and rec_machine != cur:
            return {"ok": False, "error": "此授权码已在其他设备激活"}
        same = _verify_record(rec)
        if same.get("valid") and str(rec.get("code_id")) == info["id"]:
            return {"ok": True, "renewed": False, **same}

    write_record(data_dir, info)
    st = load_license(data_dir)
    if not st.get("valid"):
        return {"ok": False, "error": st.get("reason") or "授权无效"}
    return {"ok": True, "renewed": rec is not None, **st}


# ── 防改系统时钟 ────────────────────────────────────
_CLOCK_LAST_WRITE = 0.0
_CLOCK_WRITE_INTERVAL = 60.0  # 最多每分钟落库一次，避免每个请求都写数据库


def check_clock(store) -> dict:
    """记录上次运行时间；若系统时间被往回拨，累计异常次数。

    返回 {ok, rollback, strikes}。strikes >= 3 视为可疑。
    """
    global _CLOCK_LAST_WRITE
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
        if now - _CLOCK_LAST_WRITE >= _CLOCK_WRITE_INTERVAL:
            try:
                store.set_setting("license_last_seen", str(now))
                _CLOCK_LAST_WRITE = now
            except Exception:
                pass
    if rollback:
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
        "activated": lic.get("activated", ""),
        "days": lic.get("days", 0),
        "days_left": lic.get("days_left", 0),
        "code_id": lic.get("code_id", ""),
        "machine": machine_code(),
        "clock_ok": clock.get("ok", True),
    }
