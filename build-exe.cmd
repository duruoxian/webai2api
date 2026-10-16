@echo off
REM 一键打包 exe（产物在 dist\webai2api\webai2api.exe）
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [错误] 未找到 .venv，请先运行： py -3.12 -m venv .venv ^&^& .venv\Scripts\pip install -e .
  exit /b 1
)
echo 正在打包，请稍候...
.venv\Scripts\pip install pyinstaller -q
.venv\Scripts\pyinstaller webai2api.spec --noconfirm --distpath dist --workpath build
echo.
echo 打包完成： dist\webai2api\webai2api.exe
pause
