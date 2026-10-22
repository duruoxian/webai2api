"""版本判定：pro（完整运营面板）/ lite（面向普通用户的客户端版）。

打包 lite 时在包内放一个 edition.txt（内容 lite），运行时读取。
安全：打包态（frozen）只认随二进制内嵌的 edition.txt——
      忽略环境变量 WEBAI2API_EDITION 与用户数据目录里的 edition.txt，
      否则客户丢一个内容为 pro 的文件（或设个环境变量）就能绕过授权拦截。
      仅源码调试态允许环境变量 / 数据目录兜底。
"""
from __future__ import annotations

import os

from .paths import is_frozen, resource_path, user_data_dir

_CACHE: str | None = None


def edition() -> str:
    global _CACHE
    if _CACHE:
        return _CACHE
    frozen = is_frozen()
    val = ""
    if not frozen:
        # 仅源码调试态允许环境变量覆盖；打包版忽略，防绕过授权
        val = (os.environ.get("WEBAI2API_EDITION") or "").strip().lower()
    if val not in ("lite", "pro"):
        paths = [resource_path("edition.txt")]
        if not frozen:
            # 打包版绝不读用户可写的 data 目录；仅源码态兜底
            paths.append(user_data_dir() / "edition.txt")
        for p in paths:
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
