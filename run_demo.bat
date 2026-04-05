@echo off
echo Killing any existing Python processes...
taskkill /F /IM python.exe >nul 2>&1
timeout /t 2 /nobreak >nul

echo Setting environment...
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
set GRADIO_ANALYTICS_ENABLED=False
set GRADIO_TELEMETRY_ENABLED=False
set HF_HUB_OFFLINE=1
set DO_NOT_TRACK=1

echo Starting TACNet Gradio Demo...
echo.
echo When you see the URL below, open it in Chrome or Edge:
echo.
d:\TACNet\venv\Scripts\python.exe d:\TACNet\app_gradio.py
pause
