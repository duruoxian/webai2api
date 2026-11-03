"""opencode 配置读写（纯函数 + 纯 IO，自 admin_api 拆出）。

一键接入/断开 opencode 的全部配置操作：配置文件定位、JSONC 扫描与改写
（手写扫描器，逐条经 tests/test_opencode_config.py 锁定——不换第三方
jsonc 库：注释/字符串边界行为差异大，历史换库即回归）、供应商块生成与
插入、备份上限清理。admin_api 只 re-export 端点真用的 5 个名字，
测试直连本模块（2026-11-02 拆分）。
"""
from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

from .providers import DEFAULT_LIMITS, infer_provider, model_capabilities


def _prune_opencode_backups(path: Path, keep: int = 10) -> None:
    """opencode 配置备份只留最近 keep 份（此前每次接入/断开都新建，无限累积）。"""
    try:
        baks = sorted(path.parent.glob(f"{path.name}.bak_*"))
        for old in baks[:-max(1, int(keep))]:
            old.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass


# ── opencode 配置辅助 ──────────────────────────────
def _opencode_config_path() -> Path:
    """定位 opencode 配置文件。

    优先级：
    1. 环境变量 WEBAI2API_OPENCODE_CONFIG（测试/自定义）
    2. 项目级 .opencode/opencode.json(c)（若有）
    3. 已存在的用户级 opencode.jsonc / opencode.json
    4. 默认 ~/.config/opencode/opencode.jsonc（新建）
    """
    env = os.environ.get("WEBAI2API_OPENCODE_CONFIG", "")
    if env:
        return Path(env)
    base = Path.home() / ".config" / "opencode"
    jsonc = base / "opencode.jsonc"
    plain = base / "opencode.json"
    if jsonc.exists():
        return jsonc
    if plain.exists():
        return plain
    for proj in (Path.cwd() / ".opencode", Path.cwd()):
        for name in ("opencode.jsonc", "opencode.json"):
            p = proj / name
            if p.exists():
                return p
    return jsonc


def _read_current_port() -> int | None:
    """尝试读取 run.py 写下的 current_port.txt，返回端口或 None。"""
    try:
        import sys

        from .paths import user_data_dir

        # 写入端（run.py）落在日志目录：frozen=exe 目录，回退=数据目录本身，
        # 开发态=包目录——读端按候选逐一尝试（2026-11-02 复审对齐）
        cands = [user_data_dir().parent, user_data_dir()]
        if getattr(sys, "frozen", False):
            cands.insert(0, Path(sys.executable).parent)
        else:
            cands.insert(0, Path(__file__).resolve().parent)
        p = None
        for c in cands:
            f = c / "current_port.txt"
            if f.exists():
                p = f
                break
        if p is None:
            return None
        first = p.read_text(encoding="utf-8").strip().splitlines()[0]
        return int(first)
    except Exception:
        return None


def _opencode_status_io() -> dict:
    """阻塞 IO 全部放在线程池里跑：是否存在/是否安装/是否含 webai2api provider。"""
    path = _opencode_config_path()
    try:
        exists = path.exists()
    except Exception:
        exists = False
    try:
        installed = bool(shutil.which("opencode")) or exists
    except Exception:
        installed = exists
    has_provider = False
    if exists:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            has_provider = _provider_has_entry(
                text, "webai2api", section="providers"
            ) or _provider_has_entry(text, "webai2api", section="provider")
        except Exception:
            has_provider = False
    return {
        "installed": installed,
        "config_path": str(path),
        "has_provider": has_provider,
    }


def _build_opencode_snippet(host: str, key: str, models: list[str]) -> str:
    """生成 opencode V2 格式的 webai2api 供应商块。

    V2（opencode ≥2.x）与 V1 的差异：顶层键 providers；供应商用
    package（@opencode/ai/providers/openai-compatible）与 settings；
    模型能力用 capabilities（reasoning 是对象）；variants 是**数组**
    （每项 {"id": 档位名, "settings": {...}}）——对象 map 形式过不了
    V2 schema，整个 provider 会被静默丢弃（0.1.27 迁移时踩过）。
    """
    lines = []
    for m in models:
        mid = str(m)
        low = mid.lower()
        jmid = json.dumps(mid, ensure_ascii=False)
        prov = infer_provider(mid)
        # 能力声明与 /v1/models 元数据同源（providers.model_capabilities），
        # 防止两处各自维护漂移（2026-11-02）
        caps = model_capabilities(mid)
        reasoning = "true" if caps["reasoning"]["supported"] else "false"
        vision = json.dumps(caps["input"], separators=(",", ":"))
        lim = DEFAULT_LIMITS.get(prov, DEFAULT_LIMITS["deepseek"])
        ctx, out = lim["context"], lim["output"]
        tiers: list[tuple[str, str]] = []
        if prov == "chatglm":
            tiers = [("快速", "low"), ("深度", "medium"), ("极致", "max")]
        elif prov == "kimi":
            # Kimi 三档：标准 / 进阶 / 极致；K2.6（快速模型）无极致档
            tiers = (
                [("标准", "low"), ("进阶", "high")]
                if "k2.6" in low
                else [("标准", "low"), ("进阶", "high"), ("极致", "max")]
            )
        elif prov == "qwen":
            # Plus / Max：自动 / 思考 / 快速；Omni-Flash：只有 思考 / 快速
            tiers = (
                [("思考", "high"), ("快速", "low")]
                if "flash" in low
                else [("自动", "medium"), ("思考", "high"), ("快速", "low")]
            )
        variants = ""
        if tiers:
            items = ", ".join(
                f'{{ "id": {json.dumps(t, ensure_ascii=False)}, '
                f'"settings": {{ "reasoningEffort": "{effort}" }} }}'
                for t, effort in tiers
            )
            variants = f',\n      "variants": [{items}]'
        lines.append(
            f'    {jmid}: {{ "name": {jmid},\n'
            f'      "capabilities": {{ "input": {vision}, "output": ["text"],\n'
            f'        "reasoning": {{ "supported": {reasoning} }} }},\n'
            f'      "limit": {{ "context": {ctx}, "output": {out} }}{variants} }}'
        )
    return (
        '"webai2api": {\n'
        '  "package": "@opencode/ai/providers/openai-compatible",\n'
        '  "name": "webai2api",\n'
        '  "settings": {\n'
        f'    "baseURL": {json.dumps(host + "/v1", ensure_ascii=False)},\n'
        f'    "apiKey": {json.dumps(str(key), ensure_ascii=False)},\n'
        '    "maxRetries": 0\n'
        "  },\n"
        '  "compatibility": { "streamUsage": true },\n'
        '  "models": {\n' + ",\n".join(lines) + "\n  }\n}"
    )


def _strip_jsonc_comments(text: str) -> str:
    """逐字符移除 // 行注释与 /* */ 块注释；字符串字面量（含转义）内部不动。

    幂等：对已无注释的文本再次调用结果不变。
    """
    out: list[str] = []
    i = 0
    n = len(text)
    in_str = False
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if text.startswith("//", i):
            end = text.find("\n", i)
            if end < 0:
                break
            i = end  # 保留换行符
            continue
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end < 0 else end + 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _match_close(text: str, open_idx: int) -> int:
    """从 open_idx 处的 { 或 [ 开始做配对，返回配对闭合符下标；字符串/注释内的括号不计。

    找不到闭合返回 -1。
    """
    depth = 0
    i = open_idx
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            i += 1
            while i < n:
                c = text[i]
                if c == "\\":
                    i += 2
                    continue
                if c == '"':
                    break
                i += 1
            i += 1
            continue
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _find_provider_span(text: str, section: str = "provider") -> tuple[int, int] | None:
    """定位顶层 section（"provider"=V1 / "providers"=V2）对象的区间 (起 {, 止 } 含)。

    找不到返回 None。逐字符扫描：注释被跳过；字符串字面量内部的键名
    因其后不跟 : 而跳过。返回值下标是**原始 text** 中的下标，可直接切片。
    """
    target_key = '"' + section + '"'
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            j = i + 1
            while j < n:
                c = text[j]
                if c == "\\":
                    j += 2
                    continue
                if c == '"':
                    break
                j += 1
            literal = text[i : j + 1]
            k = _skip_ws_comment(text, j + 1)
            if literal == target_key and k < n and text[k] == ":":
                k = _skip_ws_comment(text, k + 1)
                if k < n and text[k] == "{":
                    close = _match_close(text, k)
                    if close < 0:
                        return None
                    return (k, close)
            i = j + 1
            continue
        if text.startswith("//", i):
            nl = text.find("\n", i)
            i = n if nl < 0 else nl + 1
            continue
        if text.startswith("/*", i):
            ce = text.find("*/", i + 2)
            i = n if ce < 0 else ce + 2
            continue
        i += 1
    return None


def _skip_ws_comment(text: str, i: int) -> int:
    """跳过空白与 //、/* */ 注释。"""
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in " \t\r\n":
            i += 1
            continue
        if text.startswith("//", i):
            nl = text.find("\n", i)
            i = n if nl < 0 else nl + 1
            continue
        if text.startswith("/*", i):
            ce = text.find("*/", i + 2)
            i = n if ce < 0 else ce + 2
            continue
        break
    return i


def _find_top_level_key(text: str, start: int, end: int, name: str) -> int:
    """在 provider 区间 (start, end) 内查找第一层键 "name": 的绝对下标，找不到返回 -1。

    start/end 为顶层 '{' '}' 的下标。只认深度 0 的键，避免命中兄弟 provider
    内层嵌套的同名键（如 real.models.webai2api）。跳过注释与字符串字面量。
    """
    i = start + 1
    depth = 0
    target = '"' + name + '"'
    while i < end:
        ch = text[i]
        if ch == '"':
            j = i + 1
            while j < end:
                c = text[j]
                if c == "\\":
                    j += 2
                    continue
                if c == '"':
                    break
                j += 1
            literal = text[i : j + 1]
            k = _skip_ws_comment(text, j + 1)
            if depth == 0 and literal == target and k < len(text) and text[k] == ":":
                return i
            i = j + 1
            continue
        if text.startswith("//", i):
            nl = text.find("\n", i)
            i = end if nl < 0 else nl + 1
            continue
        if text.startswith("/*", i):
            ce = text.find("*/", i + 2)
            i = end if ce < 0 else ce + 2
            continue
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        i += 1
    return -1


def _provider_has_entry(text: str, name: str, section: str = "provider") -> bool:
    """provider(s) 对象**第一层**是否存在键 "name":（嵌套同名不算）。"""
    span = _find_provider_span(text, section)
    if not span:
        return False
    start, end = span
    return _find_top_level_key(text, start, end, name) >= 0


def _remove_provider_block(text: str, name: str, section: str = "provider") -> str:
    """删掉 provider(s) 对象**第一层**的 "name": { ... }, 块（只在该区间内操作）。"""
    span = _find_provider_span(text, section)
    if not span:
        return text
    start, end = span
    key_idx = _find_top_level_key(text, start, end, name)
    if key_idx < 0:
        return text
    brace = text.find("{", key_idx)
    if brace < 0 or brace > end:
        return text
    close = _match_close(text, brace)
    if close < 0:
        return text
    # 连同其后的逗号一起删（若有）
    tail = close + 1
    while tail < len(text) and text[tail] in " \t\r\n":
        tail += 1
    if tail < len(text) and text[tail] == ",":
        tail += 1
        while tail < len(text) and text[tail] in " \t\r\n":
            tail += 1
    # 去掉 key 之前紧邻的缩进和残留逗号
    head = key_idx
    while head > start + 1 and text[head - 1] in " \t":
        head -= 1
    if head > start + 1 and text[head - 1] == ",":
        head -= 1
        while head > start + 1 and text[head - 1] in " \t":
            head -= 1
    out = text[:head] + text[tail:]
    # 重新配对清理区间内的尾逗号（, } / , ]），只作用于该 provider 区间
    new_span = _find_provider_span(out, section)
    if new_span:
        s2, e2 = new_span
        seg = out[s2 : e2 + 1]
        cleaned = re.sub(r",(\s*[}\]])", r"\1", seg)
        out = out[:s2] + cleaned + out[e2 + 1 :]
    return out


def _ensure_provider_section(text: str, section: str = "provider") -> str:
    """确保顶层存在 "section": { }，缺则插入；返回新文本。

    兼容：空文件、只有一个 {}、有其它顶层键、带 // 与 /* */ 注释。
    顶层不是对象（无法安全改写）时原样返回，由调用方报错。
    """
    stripped = _strip_jsonc_comments(text).strip()
    if not stripped:
        return (
            '{\n  "$schema": "https://opencode.ai/config.json",\n  "'
            + section
            + '": {\n  }\n}\n'
        )

    # 顶层必须是 { ... }
    first = _skip_ws_comment(text, 0)
    if first >= len(text) or text[first] != "{":
        return text
    close = _match_close(text, first)
    if close < 0:
        return text

    # 已有该 section 键（值不是对象也算存在，交给后续逻辑判断）
    body = text[first + 1 : close]
    existing = _find_provider_span(text, section)
    if existing is not None:
        return text

    inner = body.strip()
    insert_at = first + 1
    if inner:
        # 有其它键：追加 ", "section": {}"
        block = '\n  "' + section + '": {\n  },\n'
        return text[:insert_at] + block + body.lstrip("\r\n") + text[close:]
    return text[:insert_at] + '\n  "' + section + '": {\n  }\n' + text[close:]


def _remove_top_level_section(text: str, section: str) -> str:
    """整体移除顶层 "section": { ... }（含键名、冒号与相邻逗号）。段不存在时原样返回。"""
    span = _find_provider_span(text, section=section)
    if not span:
        return text
    brace_open, brace_close = span
    key_literal = '"' + section + '"'
    key_end = text.rfind(key_literal, 0, brace_open)
    start = brace_open
    if key_end >= 0 and text[key_end + len(key_literal) : brace_open].strip() == ":":
        start = key_end  # 键名 + 冒号一并移除
    tail = brace_close + 1
    while tail < len(text) and text[tail] in " \t\r\n":
        tail += 1
    if tail < len(text) and text[tail] == ",":
        tail += 1
        while tail < len(text) and text[tail] in " \t\r\n":
            tail += 1
    head = start
    while head > 0 and text[head - 1] in " \t":
        head -= 1
    if head > 0 and text[head - 1] == ",":
        head -= 1
        while head > 0 and text[head - 1] in " \t":
            head -= 1
    return text[:head] + text[tail:]


def _write_opencode_config(snippet: str) -> dict:
    """把 webai2api 供应商块写入 opencode 配置（V2 格式）。

    V2（opencode ≥2.x）：顶层键是 "providers"，供应商用
    package/settings/capabilities。同时清理历史遗留：
    - 旧 V1 "provider" 段里的 webai2api 块（若整段因此变空，连段一起移除）
    - disabled_providers 数组里的 webai2api 项
    """
    path = _opencode_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            '{\n  "$schema": "https://opencode.ai/config.json",\n'
            '  "providers": {\n  }\n}\n',
            encoding="utf-8",
        )

    text = path.read_text(encoding="utf-8")
    backup = path.with_name(
        f"{path.name}.bak_{datetime.now().strftime('%Y%m%d%H%M%S_%f')}"
    )
    backup.write_text(text, encoding="utf-8")
    _prune_opencode_backups(path)

    # V1 迁移：旧 "provider" 段里如果有 webai2api（V1 格式残留），先摘除；
    # 旧段若因此变空（只剩空白），整段连键名一起移除，不给 V2 配置留孤儿键
    if _provider_has_entry(text, "webai2api", section="provider"):
        text = _remove_provider_block(text, "webai2api", section="provider")
        legacy = _find_provider_span(text, section="provider")
        if legacy:
            inner = _strip_jsonc_comments(text[legacy[0] + 1 : legacy[1]]).strip()
            inner = re.sub(r",", "", inner).strip()
            if not inner:
                text = _remove_top_level_section(text, "provider")

    # 配置里没有 providers 段（全新安装的 opencode 往往如此）：自动补一个空段，
    # 而不是让用户手动粘贴——保证无论装在哪台机器都能一键导入。
    if _find_provider_span(text, section="providers") is None:
        text = _ensure_provider_section(text, section="providers")

    span = _find_provider_span(text, section="providers")
    if not span:
        raise RuntimeError("配置格式异常（顶层不是对象），请手动粘贴")
    _brace, close_pos = span
    # providers 块闭合后下一个非空白字符必须是 , 或 }（顶层对象结尾）
    tail = _strip_jsonc_comments(text[close_pos + 1 :]).lstrip(" \t\n\r")
    if tail and tail[0] not in ",}":
        raise RuntimeError("配置格式异常，请手动粘贴")

    # 移除旧的 webai2api 块（幂等：重复接入不产生重复条目）
    text = _remove_provider_block(text, "webai2api", section="providers")

    # 重新定位 providers 的 {（同样限定在 providers 对象内，避免注释/字符串干扰）
    span2 = _find_provider_span(text, section="providers")
    if not span2:
        raise RuntimeError("配置格式异常，请手动粘贴")
    orig_brace = span2[0]
    if orig_brace < 0 or orig_brace >= len(text) or text[orig_brace] != "{":
        raise RuntimeError("配置格式异常，请手动粘贴")

    indent = "    "
    block = snippet.replace("\n", "\n" + indent)
    after = text[orig_brace + 1 :]
    # 去掉 after 开头空白和已有的尾部逗号，避免双逗号
    after_stripped = after.lstrip(" \t\n\r")
    if after_stripped.startswith(","):
        after_stripped = after_stripped[1:].lstrip(" \t\n\r")
    # 把 block 插入到 { 之后，保留原来内容；确保后面有换行
    if after_stripped and not after_stripped.endswith("\n"):
        after_stripped += "\n"
    # provider 段原本为空（直接以 } 收尾）时不能加逗号，否则产生悬空 ",}"
    sep = "" if after_stripped.startswith("}") else ","
    out = text[: orig_brace + 1] + "\n" + indent + block + sep + "\n" + after_stripped

    # 同时清理 disabled_providers 里的 webai2api
    out = _remove_disabled_provider(out, "webai2api")

    path.write_text(out, encoding="utf-8")
    return {"ok": True, "path": str(path), "backup": str(backup)}


def _remove_disabled_provider(text: str, name: str) -> str:
    """从 disabled_providers 数组中移除指定项。"""
    key = '"disabled_providers"'
    idx = text.find(key)
    if idx < 0:
        return text
    arr_start = text.find("[", idx)
    if arr_start < 0:
        return text
    depth = 0
    arr_end = -1
    for i in range(arr_start, len(text)):
        ch = text[i]
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
            if depth == 0:
                arr_end = i
                break
    if arr_end < 0:
        return text
    arr_text = text[arr_start + 1 : arr_end]
    # 简单正则替换掉包含 name 的项
    pattern = re.compile(r'\s*"' + re.escape(name) + r'"\s*,?')
    new_arr = pattern.sub("", arr_text)
    # 清理末尾逗号
    new_arr = re.sub(r",\s*]", "]", "[" + new_arr + "]")[1:-1]
    return text[: arr_start + 1] + new_arr + text[arr_end:]


def _opencode_structure_ok(text: str) -> bool:
    """括号/字符串是否配平（注释内的括号忽略），用于判断能否安全改写。"""
    depth = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            i += 1
            while i < n:
                c = text[i]
                if c == "\\":
                    i += 2
                    continue
                if c == '"':
                    break
                i += 1
            i += 1
            continue
        if text.startswith("//", i):
            nl = text.find("\n", i)
            i = n if nl < 0 else nl + 1
            continue
        if text.startswith("/*", i):
            ce = text.find("*/", i + 2)
            i = n if ce < 0 else ce + 2
            continue
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth < 0:
                return False
        i += 1
    return depth == 0


def _remove_opencode_provider() -> dict:
    path = _opencode_config_path()
    if not path.exists():
        # 本来就没接入：幂等成功，不建文件
        return {"ok": True, "not_present": True, "path": str(path)}
    text = path.read_text(encoding="utf-8")

    in_providers = _find_provider_span(
        text, section="providers"
    ) is not None and _provider_has_entry(text, "webai2api", section="providers")
    in_legacy = _find_provider_span(
        text, section="provider"
    ) is not None and _provider_has_entry(text, "webai2api", section="provider")
    in_disabled = '"webai2api"' in _strip_jsonc_comments(text) and (
        _remove_disabled_provider(text, "webai2api") != text
    )
    if not in_providers and not in_legacy and not in_disabled:
        # providers 段不存在，或存在但没有 webai2api：幂等成功，不写文件不出备份
        return {"ok": True, "not_present": True, "path": str(path)}

    # 有残留需要删，但 JSONC 本身已损坏：改写可能把用户配置越改越坏，直接报错让用户手改
    if not _opencode_structure_ok(text):
        raise RuntimeError("配置文件 JSONC 格式异常（括号/引号不配对），无法安全移除")

    backup = path.with_name(
        f"{path.name}.bak_{datetime.now().strftime('%Y%m%d%H%M%S_%f')}"
    )
    backup.write_text(text, encoding="utf-8")
    _prune_opencode_backups(path)

    new_text = (
        _remove_provider_block(text, "webai2api", section="providers")
        if in_providers
        else text
    )
    new_text = _remove_provider_block(new_text, "webai2api", section="provider")
    # V1 段被摘空后整段移除，避免孤儿键
    legacy = _find_provider_span(new_text, section="provider")
    if legacy:
        inner = (
            _strip_jsonc_comments(new_text[legacy[0] + 1 : legacy[1]])
            .strip()
            .replace(",", "")
        )
        if not inner:
            new_text = _remove_top_level_section(new_text, "provider")
    new_text = _remove_disabled_provider(new_text, "webai2api")
    if new_text == text:
        return {
            "ok": True,
            "not_present": True,
            "path": str(path),
            "backup": str(backup),
        }
    path.write_text(new_text, encoding="utf-8")
    return {"ok": True, "path": str(path), "backup": str(backup)}
