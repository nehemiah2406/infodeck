@echo off
REM InfoDeck launcher - double-click to start. Close this window (or press Ctrl+C) to stop.
setlocal
cd /d "%~dp0"
title InfoDeck

if not exist ".venv\Scripts\python.exe" (
    echo [InfoDeck] The Python environment is missing. Run these commands once in this folder:
    echo     python -m venv .venv
    echo     .venv\Scripts\python -m pip install -r requirements.txt
    pause
    exit /b 1
)
if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo [InfoDeck] Created .env from .env.example ^(AI answers stay off until you add an API key^).
)

REM Find this laptop's IP address on the office network
set "IP="
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "(Get-NetIPConfiguration | Where-Object { $_.IPv4DefaultGateway -and $_.NetAdapter.Status -eq 'Up' } | Select-Object -First 1).IPv4Address.IPAddress"`) do set "IP=%%i"

echo.
echo  ============================================================
echo    InfoDeck is starting (the first start can take a minute)
echo.
echo    On this laptop:      http://localhost:8501
if defined IP (
echo    For colleagues:      http://%IP%:8501
) else (
echo    For colleagues:      http://^<this-laptop-ip^>:8501   ^(run ipconfig to find it^)
)
echo.
echo    Keep this window open while people use InfoDeck.
echo  ============================================================
echo.

REM Open the browser on this laptop after a short delay
start "" /b cmd /c "timeout /t 8 /nobreak >nul & start http://localhost:8501"

".venv\Scripts\python.exe" -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
echo.
echo [InfoDeck] stopped.
pause
