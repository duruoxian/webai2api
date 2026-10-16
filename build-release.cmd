@echo off
REM 一键打包"可分发版"：产出干净、可直接发给朋友的压缩包
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [错误] 未找到 .venv，请先： py -3.12 -m venv .venv ^&^& .venv\Scripts\pip install -e .
  exit /b 1
)

echo [1/4] 清理上一次产物...
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build

echo [2/4] 打包...
.venv\Scripts\pyinstaller webai2api.spec --noconfirm --distpath dist --workpath build
if errorlevel 1 (
  echo [错误] 打包失败
  exit /b 1
)

echo [3/4] 清理运行时残留（数据库/日志/浏览器profile）...
if exist "dist\webai2api\data" rmdir /s /q "dist\webai2api\data"
if exist "dist\webai2api\webai2api.log" del /q "dist\webai2api\webai2api.log"

echo [4/4] 生成压缩包...
powershell -NoProfile -Command "Compress-Archive -Path 'dist\webai2api' -DestinationPath 'dist\webai2api-win64.zip' -Force"
echo.
echo 完成！可分发文件： dist\webai2api-win64.zip
echo 朋友解压后双击 webai2api.exe 即可（首次会自动创建 data 目录）。
pause
