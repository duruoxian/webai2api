# -*- mode: python ; coding: utf-8 -*-
"""客户端版（lite）打包配置。

打包命令：
    .venv\\Scripts\\pyinstaller webai2api-lite.spec --noconfirm --distpath dist-lite --workpath build-lite

产物：dist-lite/webai2api/webai2api.exe
与专业版的区别（同一代码库，运行期按 edition.txt 分派行为）：
- 打入 edition.txt（内容 lite）+ 双面板（index.html 专业版 / lite_panel.html 客户版）
- 应用名 webai2api-lite，图标同专业版
"""
from PyInstaller.utils.hooks import collect_all, collect_data_files
import os

# 生成 edition.txt（lite 标记），随包分发。
# 写进 gitignored 的 build/ 而不是源码根：源码根若残留 edition.txt，
# 会让源码态运行/测试被误判成 lite（v0.1.26 前的真实事故）。
_build_dir = os.path.join(os.path.dirname(os.path.abspath(SPEC)), "build")
os.makedirs(_build_dir, exist_ok=True)
_edition_file = os.path.join(_build_dir, "edition.txt")
with open(_edition_file, "w", encoding="utf-8") as f:
    f.write("lite")

datas = [
    ("webai2api/web/index.html", "webai2api/web"),
    ("webai2api/web/lite_panel.html", "webai2api/web"),
    ("webai2api/web/app.css", "webai2api/web"),
    ("webai2api/web/hyalite.js", "webai2api/web"),
    (_edition_file, "."),
]

if os.path.exists("sha3_wasm_bg.wasm"):
    datas.append(("sha3_wasm_bg.wasm", "."))

APP_ICON = "assets/webai2api-lite.ico"
if os.path.exists(APP_ICON):
    datas.append((APP_ICON, "assets"))
    _legacy_icon = "assets/webai2api.ico"
    if os.path.exists(_legacy_icon):
        datas.append((_legacy_icon, "assets"))

for pkg in ("uvicorn", "fastapi", "starlette", "anyio", "click", "h11", "websockets"):
    d, b, h = collect_all(pkg)
    datas += d

wt_datas, wt_bins, wt_hidden = collect_all("wasmtime")
datas += wt_datas

datas += collect_data_files("deepseek_pow")

# curl_cffi（浏览器 TLS 指纹传输层）
try:
    _cc_datas, _cc_bins, _cc_hidden = collect_all("curl_cffi")
    datas += _cc_datas
except Exception:
    _cc_hidden = []
    _cc_bins = []

_tray_datas, _tray_bins, _tray_hidden = [], [], []
for pkg in ("pystray", "PIL"):
    try:
        d, b, h = collect_all(pkg)
        _tray_datas += d
        _tray_bins += b
        _tray_hidden += h
    except Exception:
        pass
datas += _tray_datas

wv_datas, wv_bins, wv_hidden = collect_all("webview")
datas += wv_datas
wt_bins = wt_bins + wv_bins

hiddenimports = [
    "webai2api",
    "webai2api.edition",
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
    "webview",
    "webview.platforms.edgechromium",
    "clr_loader",
    "pythonnet",
    "socks",          # PySocks：socks5 代理（运行期动态 import）
    "sockshandler",
] + wt_hidden + wv_hidden + _tray_hidden + _cc_hidden

a = Analysis(
    ["run.py"],
    pathex=[],
    binaries=wt_bins + _tray_bins + _cc_bins,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "numpy", "pandas", "playwright"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="webai2api",
    icon=APP_ICON if os.path.exists(APP_ICON) else None,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="webai2api",
)
