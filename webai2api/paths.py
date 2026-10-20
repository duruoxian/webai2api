"""路径工具：同时兼容源码运行与 PyInstaller 打包后的 exe。

- resource_path(...)  只读资源（面板 HTML、wasm 等）→ 打包后位于 _MEIPASS
- user_data_dir()     可写数据（数据库、浏览器 profile）→ 打包后位于 exe 同级 data/
"""
from __future__ import annotations

import sys
from pathlib import Path


def is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def _bundle_root() -> Path:
    """只读资源根目录。"""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    # 源码运行：项目根 = 本文件的上上级
    return Path(__file__).resolve().parent.parent


def resource_path(*parts: str) -> Path:
    return _bundle_root().joinpath(*parts)


def _exe_dir() -> Path:
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def user_data_dir() -> Path:
    d = _exe_dir() / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def default_wasm_path() -> str:
    """默认 PoW wasm 位置：优先 exe 同级目录（用户可替换），其次打包内。"""
    beside = _exe_dir() / "sha3_wasm_bg.wasm"
    if beside.exists():
        return str(beside)
    bundled = resource_path("sha3_wasm_bg.wasm")
    if bundled.exists():
        return str(bundled)
    return str(beside)
