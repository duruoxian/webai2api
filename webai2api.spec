# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置。

打包命令（在项目根目录）：
    .venv\\Scripts\\pyinstaller webai2api.spec --noconfirm

产物：dist/webai2api/webai2api.exe（onedir 模式，启动快）
"""
from PyInstaller.utils.hooks import collect_all, collect_data_files

datas = [
    ("webai2api/web/index.html", "webai2api/web"),
]

# 若根目录有 wasm，一并打包
import os
if os.path.exists("sha3_wasm_bg.wasm"):
    datas.append(("sha3_wasm_bg.wasm", "."))

# 应用图标：既作为 exe 文件图标，也供运行期窗口图标使用
APP_ICON = "assets/webai2api.ico"
if os.path.exists(APP_ICON):
    datas.append((APP_ICON, "assets"))

# 收集 uvicorn / fastapi / starlette 全部子模块，避免冻结后缺模块
for pkg in ("uvicorn", "fastapi", "starlette", "anyio", "click", "h11", "websockets"):
    d, b, h = collect_all(pkg)
    datas += d

# wasmtime 原生 DLL
wt_datas, wt_bins, wt_hidden = collect_all("wasmtime")
datas += wt_datas

# deepseek_pow 内置 wasm
datas += collect_data_files("deepseek_pow")

# pywebview 原生窗口（Windows 走 WebView2 + pythonnet）
wv_datas, wv_bins, wv_hidden = collect_all("webview")
datas += wv_datas
wt_bins = wt_bins + wv_bins

hiddenimports = [
    "webai2api",
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
    "webview",
    "webview.platforms.edgechromium",
    "clr_loader",
    "pythonnet",
] + wt_hidden + wv_hidden

a = Analysis(
    ["run.py"],
    pathex=[],
    binaries=wt_bins,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "numpy", "PIL", "pandas", "playwright"],
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
