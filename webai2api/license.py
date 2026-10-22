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
from .paths import resource_path

# 内置公钥（base64），与作者私钥配对。
PUBLIC_KEY_B64 = "KALlewBGC52ZocSmOiE8sQ/SyJQWLbyHUS0W94ybcaY="

LICENSE_FILE = "license.key"
LICENSE_VERSION = 3
# v2 是旧公式（无 pepper）：已发货客户的记录仍然有效，验证通过后自动升级为 v3
SUPPORTED_VERSIONS = (2, 3)
CLOCK_SKEW_TOLERANCE = 6 * 3600  # 允许系统时间比上次运行早 6 小时

_PEPPER_CACHE: str | None = None

# 已发货客户旧授权文件（v2 公式）的宽限截止：2028-01-01。
# 在此之前 v2 记录仍可验证并自动升级为 v3；之后伪造 v2 续命被彻底关闭。
V2_GRACE_TS = 1830297600


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


def _pepper() -> str:
    """构建期秘密盐（不进 git）。打包版随 exe 内嵌；源码态读 tools/。

    没有它就无法伪造 v3 授权文件的 HMAC——旧版密钥纯由本机机器码推导，
    熟客可在本地重算签名把 activated_at 无限往后拨。
    """
    global _PEPPER_CACHE
    if _PEPPER_CACHE is not None:
        return _PEPPER_CACHE
    val = ""
    candidates = (
        resource_path("webai2api", "license_pepper.txt"),
        resource_path("license_pepper.txt"),
        Path(__file__).resolve().parent.parent / "tools" / "license_pepper.txt",
    )
    for p in candidates:
        try:
            if p.is_file():
                raw = p.read_text(encoding="utf-8").strip()
                if len(raw) >= 32:
                    val = raw
                    break
        except Exception:
            pass
    _PEPPER_CACHE = val
    return val


def _hmac_key() -> bytes:
    return hashlib.sha256(
        ("wb3:" + _pepper() + ":" + machine_code()).encode("utf-8")).digest()


def _hmac_key_legacy() -> bytes:
    """v2 旧公式（仅机器码）：只用于验证已发货客户的旧授权文件。"""
    return hashlib.sha256(("wb2:" + machine_code()).encode("utf-8")).digest()


def _record_body(rec: dict) -> bytes:
    body = {k: v for k, v in rec.items() if k != "sig"}
    return json.dumps(body, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _sign_record(rec: dict) -> str:
    return hmac.new(_hmac_key(), _record_body(rec), hashlib.sha256).hexdigest()


def _sig_ok(rec: dict) -> bool:
    sig = str(rec.get("sig") or "")
    if not sig:
        return False
    if int(rec.get("v") or 0) == 2:
        expect = hmac.new(_hmac_key_legacy(), _record_body(rec),
                          hashlib.sha256).hexdigest()
    else:
        expect = hmac.new(_hmac_key(), _record_body(rec),
                          hashlib.sha256).hexdigest()
    return hmac.compare_digest(expect, sig)


def _read_record(data_dir: Path) -> dict | None:
    path = license_path(data_dir)
    if not path.exists():
        return None
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return rec if isinstance(rec, dict) else None


# ── 多副本保存：防止删除 data 目录重置计时 ──────────────
REG_KEY = r"Software\webai2api\license"
SHADOW_FILE = ".activation.json"


def _shadow_path() -> Path | None:
    """影子文件：放在数据目录的上一级（删 data 不会带走）。"""
    try:
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        try:
            from .edition import is_lite
            name = "webai2api-lite" if is_lite() else "webai2api"
        except Exception:
            name = "webai2api"
        return Path(base) / name / SHADOW_FILE
    except Exception:
        return None


def _reg_read() -> dict | None:
    if os.name != "nt":
        return None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, REG_KEY) as k:
            raw, _ = winreg.QueryValueEx(k, "record")
        rec = json.loads(raw)
        return rec if isinstance(rec, dict) else None
    except Exception:
        return None


def _reg_write(rec: dict) -> None:
    if os.name != "nt":
        return
    try:
        import winreg
        with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, REG_KEY, 0,
                                winreg.KEY_SET_VALUE) as k:
            winreg.SetValueEx(k, "record", 0, winreg.REG_SZ,
                              json.dumps(rec, ensure_ascii=False))
    except Exception:
        pass


def _shadow_read() -> dict | None:
    p = _shadow_path()
    if p is None:
        return None
    try:
        if not p.exists():
            return None
        rec = json.loads(p.read_text(encoding="utf-8"))
        return rec if isinstance(rec, dict) else None
    except Exception:
        return None


def _shadow_write(rec: dict) -> None:
    p = _shadow_path()
    if p is None:
        return
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetFileAttributesW(str(p), 0x02)  # 隐藏
    except Exception:
        pass


def _record_ok(rec: dict | None) -> bool:
    """副本本身是否可信：签名正确且属于本机。

    v2（旧公式）仅在宽限期内可信——用于已发货客户透明迁移到 v3；
    宽限期后伪造 v2 续命这条路被彻底关闭。
    """
    if not isinstance(rec, dict):
        return False
    ver = int(rec.get("v") or 0)
    if ver not in SUPPORTED_VERSIONS:
        return False
    if ver == 2 and time.time() > V2_GRACE_TS:
        return False
    if not _sig_ok(rec):
        return False
    rec_machine = str(rec.get("machine") or "").upper()
    return not rec_machine or rec_machine == machine_code()


# ── 最早激活时间地板 ────────────────────────────────
# 双份存储（注册表 + 影子文件）：即使授权文件被删光重建，
# 同一码的 activated_at 也不可能被拨回比"见过的最早值"更早。
FLOOR_FILE = ".activation_floor.json"


def _floor_shadow_path() -> Path | None:
    sp = _shadow_path()
    return sp.parent / FLOOR_FILE if sp else None


def _floor_read() -> dict[str, float]:
    out: dict[str, float] = {}
    for raw in (_floor_reg_read(), _floor_shadow_read()):
        if isinstance(raw, dict):
            for k, v in raw.items():
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if fv > 0:
                    cur = out.get(str(k))
                    if cur is None or fv < cur:
                        out[str(k)] = fv
    return out


def _floor_reg_read() -> dict | None:
    if os.name != "nt":
        return None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, REG_KEY) as k:
            raw, _ = winreg.QueryValueEx(k, "floor")
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _floor_reg_write(data: dict) -> None:
    if os.name != "nt":
        return
    try:
        import winreg
        with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, REG_KEY, 0,
                                winreg.KEY_SET_VALUE) as k:
            winreg.SetValueEx(k, "floor", 0, winreg.REG_SZ,
                              json.dumps(data, ensure_ascii=False))
    except Exception:
        pass


def _floor_shadow_read() -> dict | None:
    p = _floor_shadow_path()
    if p is None:
        return None
    try:
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _floor_write(code_id: str, activated_at: float) -> None:
    if not code_id or activated_at <= 0:
        return
    cur = _floor_read()
    old = cur.get(code_id)
    if old is not None and old <= activated_at:
        return  # 已有更早的地板，不抬高
    cur[code_id] = activated_at
    # 只保留最近 8 个码
    items = sorted(cur.items(), key=lambda kv: kv[1])[-8:]
    data = dict(items)
    _floor_reg_write(data)
    p = _floor_shadow_path()
    if p is not None:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            import ctypes
            ctypes.windll.kernel32.SetFileAttributesW(str(p), 0x02)
        except Exception:
            pass


def _best_record(data_dir: Path) -> dict | None:
    """合并三份副本：同码取最早激活时间，续费取最新一波。

    - 先按码 ID 分组（不同码 = 续费或换码）
    - 同组内存在 v3 时，v2 候选一律无视（防拿旧公式伪造降级）
    - 选激活时间最晚的那一组（最近一次激活/续费）
    - 组内取最早的激活时间（防止靠删副本把时间往后挪）
    """
    candidates = [_read_record(data_dir), _shadow_read(), _reg_read()]
    groups: dict[str, list[dict]] = {}
    for rec in candidates:
        if not _record_ok(rec):
            continue
        try:
            cid = str(rec.get("code_id") or rec.get("code") or "")
            activated = float(rec.get("activated_at") or 0)
        except Exception:
            continue
        if not cid or activated <= 0:
            continue
        groups.setdefault(cid, []).append(rec)
    if not groups:
        return None
    best_cid, best_group = max(
        groups.items(),
        key=lambda kv: max(float(r.get("activated_at") or 0) for r in kv[1]),
    )
    v3 = [r for r in best_group if int(r.get("v") or 0) >= 3]
    pool = v3 or best_group
    return min(pool, key=lambda r: float(r.get("activated_at") or 0))


def _mirror(rec: dict) -> None:
    """把权威记录写回所有副本。"""
    for writer in (_shadow_write, _reg_write):
        try:
            writer(rec)
        except Exception:
            pass


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
    _mirror(rec)
    _LIC_CACHE["key"] = None
    return rec


_LIC_CACHE: dict = {"key": None, "doc": None}


def load_license(data_dir: Path) -> dict:
    """读取并校验授权，返回状态字典。

    三份副本（data / 影子文件 / 注册表）合并取权威记录，
    任一缺失会自动补回——删 data 无法重置计时。
    结果按文件 (mtime, size) 缓存：验签约 0.2s，不能每个请求都做。
    """
    path = license_path(data_dir)
    try:
        stt = path.stat()
        cache_key = (str(path), stt.st_mtime_ns, stt.st_size)
    except OSError:
        cache_key = None

    if cache_key is not None and _LIC_CACHE["key"] == cache_key:
        return _LIC_CACHE["doc"]

    rec = _best_record(data_dir)
    if rec is None:
        if not path.exists() and _shadow_path() and not _shadow_path().exists() \
                and _reg_read() is None:
            result = {"valid": False, "reason": "未授权，请在设置里输入授权码",
                      "code": "missing"}
        else:
            result = {"valid": False, "reason": "授权文件格式错误", "code": "bad_format"}
    else:
        result = _verify_record(rec)
        if result.get("valid"):
            # 记录「最早激活地板」，防止删档重建后把时间拨回
            try:
                _floor_write(str(result.get("code_id") or ""),
                             float(result.get("activated_at") or 0))
            except Exception:
                pass
        # v2→v3 透明升级：用权威激活时间重新签成 v3
        authoritative = rec
        if result.get("upgraded"):
            authoritative = {
                "v": LICENSE_VERSION,
                "code": rec.get("code", ""),
                "code_id": rec.get("code_id", ""),
                "days": rec.get("days", 0),
                "issued": rec.get("issued", ""),
                "activated_at": float(result.get("activated_at") or 0),
                "machine": machine_code(),
            }
            authoritative["sig"] = _sign_record(authoritative)
        # 自愈：把权威记录补回 data 与其它副本
        try:
            cur = _read_record(data_dir)
            if cur != authoritative:
                data_dir.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(authoritative, ensure_ascii=False, indent=2),
                               encoding="utf-8")
                tmp.replace(path)
        except Exception:
            pass
        _mirror(authoritative)
        result = {k: v for k, v in result.items() if k != "upgraded"}

    if cache_key is not None:
        _LIC_CACHE["key"] = cache_key
        _LIC_CACHE["doc"] = result
    return result


def _verify_record(rec: dict) -> dict:
    ver = int(rec.get("v") or 0)
    if ver not in SUPPORTED_VERSIONS:
        return {"valid": False, "reason": "授权文件版本过旧，请重新输入授权码",
                "code": "legacy"}
    if ver == 2 and time.time() > V2_GRACE_TS:
        return {"valid": False, "reason": "授权文件版本过旧，请重新输入授权码",
                "code": "legacy"}

    # 1) 机器码一致（授权码只能在首台机器上用）
    #    先查机器码，再验 HMAC：换机后 HMAC 必然对不上，要给出正确原因
    rec_machine = str(rec.get("machine") or "").upper()
    cur = machine_code()
    if rec_machine and rec_machine != cur:
        return {"valid": False, "code": "other_device",
                "reason": "此授权码已在其他设备激活", "machine": rec_machine}

    # 2) 本地记录没有被改过（v2 用旧公式、v3 用 pepper 公式）
    if not _sig_ok(rec):
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

    # 4) 最早激活时间地板：删光副本重新激活也拨不回更晚（防重置续命）。
    #    地板记录「见过的最早激活时间」，任何更晚的 activated_at 一律按地板算。
    floor_ts = _floor_read().get(info["id"]) or 0.0
    if 0 < floor_ts < activated:
        activated = floor_ts

    expires_ts = activated + days * 86400
    now = time.time()
    base = {
        "code_id": info["id"],
        "days": days,
        "activated": time.strftime("%Y-%m-%d %H:%M", time.localtime(activated)),
        "expires": time.strftime("%Y-%m-%d %H:%M", time.localtime(expires_ts)),
        "machine": cur,
        "activated_at": activated,
        "expires_ts": expires_ts,
        "seconds_left": max(0, int(expires_ts - now)),
    }

    if now >= expires_ts:
        return {**base, "valid": False, "code": "expired",
                "reason": f"授权已于 {base['expires']} 到期", "seconds_left": 0}

    left = max(0, int((expires_ts - now) // 86400))
    out = {**base, "valid": True, "code": "ok", "reason": "",
           "days_left": left}
    if ver < LICENSE_VERSION:
        out["upgraded"] = True   # 提示 load_license 把 v2 记录升级重写为 v3
    return out


# ── 激活 ────────────────────────────────────────────
def activate(data_dir: Path, code: str) -> dict:
    """首次激活或续费。返回 {ok, error, ...状态}。"""
    info = issuer.verify_code((code or "").strip(), public_key())
    if info is None:
        return {"ok": False, "error": "授权码无效，请确认复制完整"}

    # 从三份副本里取记录（data 被删也能拿到原来的激活时间）
    rec = _best_record(data_dir)
    if rec is not None:
        cur = machine_code()
        rec_machine = str(rec.get("machine") or "").upper()
        if rec_machine and rec_machine != cur:
            return {"ok": False, "error": "此授权码已在其他设备激活"}
        if str(rec.get("code_id") or "") == info["id"]:
            # 同一个码：沿用最早的激活时间，不允许重置
            write_record(data_dir, info,
                         activated_at=float(rec.get("activated_at") or 0) or None)
            st = load_license(data_dir)
            if not st.get("valid"):
                return {"ok": False, "error": st.get("reason") or "授权无效"}
            return {"ok": True, "renewed": False, **st}

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
        "activated_at": lic.get("activated_at", 0.0),
        "expires_ts": lic.get("expires_ts", 0.0),
        "seconds_left": lic.get("seconds_left", 0),
    }
