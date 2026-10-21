# -*- mode: python ; coding: utf-8 -*-
"""客户端版（lite）打包配置。

打包命令：
    .venv\\Scripts\\pyinstaller webai2api-lite.spec --noconfirm --distpath dist-lite --workpath build-lite

产物：dist-lite/webai2api/webai2api.exe
与专业版的区别：
- 额外打入 web/lite.html、web/lite.css、edition.txt（内容 lite）
- 应用名 webai2api-lite，图标同专业版
"""
from PyInstaller.utils.hooks import collect_all, collect_data_files
import os

# 生成 edition.txt（lite 标记），随包分发
_edition_file = os.path.join(os.path.dirname(os.path.abspath(SPEC)), "edition.txt")
with open(_edition_file, "w", encoding="utf-8") as f:
    f.write("lite")

datas = [
    ("webai2api/web/index.html", "webai2api/web"),
    ("webai2api/web/app.css", "webai2api/web"),
    ("webai2api/web/hyalite.js", "webai2api/web"),
    ("webai2api/web/lite.html", "webai2api/web"),
    ("webai2api/web/lite.css", "webai2api/web"),
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
    "webai2api.lite_api",
    "webai2api.license",
    "webai2api.ed25519",
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
] + wt_hidden + wv_hidden + _tray_hidden

a = Analysis(
    ["run.py"],
    pathex=[],
    binaries=wt_bins + _tray_bins,
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
