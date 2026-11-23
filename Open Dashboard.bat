@echo off
rem Starts the trading dashboard (if not already running) and opens it in the browser.
cd /d "%~dp0"

curl -s -o nul http://127.0.0.1:8050 && goto open

start "Trading dashboard - close this window to stop it" /min ".venv\Scripts\python.exe" code\app.py

rem Wait up to 5 minutes for it to come up (first start after a code change rebuilds caches).
set /a tries=0
:wait
timeout /t 3 /nobreak >nul
curl -s -o nul http://127.0.0.1:8050 && goto open
set /a tries+=1
if %tries% lss 100 goto wait
echo Dashboard did not start. Open the minimized "Trading dashboard" window to see the error.
pause
exit /b 1

:open
start "" http://127.0.0.1:8050
