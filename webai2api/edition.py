"""版本判定：pro（完整运营面板）/ lite（面向普通用户的客户端版）。

打包 lite 时在包内放一个 edition.txt（内容 lite），运行时读取；
源码调试可用环境变量 WEBAI2API_EDITION=lite 覆盖。
"""
from __future__ import annotations

import os

from .paths import resource_path, user_data_dir

_CACHE: str | None = None


def edition() -> str:
    global _CACHE
    if _CACHE:
        return _CACHE
    val = (os.environ.get("WEBAI2API_EDITION") or "").strip().lower()
    if val not in ("lite", "pro"):
        for p in (resource_path("edition.txt"), user_data_dir() / "edition.txt"):
            try:
                if p.exists():
                    val = p.read_text(encoding="utf-8").strip().lower()
                    break
            except Exception:
                pass
    _CACHE = "lite" if val == "lite" else "pro"
    return _CACHE


def is_lite() -> bool:
    return edition() == "lite"
