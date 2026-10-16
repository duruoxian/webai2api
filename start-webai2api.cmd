@echo off
cd /d "%~dp0"
start "" /b ".venv\Scripts\pythonw.exe" -m webai2api --port 8800
