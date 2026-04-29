@echo off
setlocal
echo [FakeGuard] Stopping frontend (port 5173)...
for /f "tokens=5" %%a in ('netstat -aon ^| find ":5173" ^| find "LISTENING"') do taskkill /F /PID %%a >nul 2>&1
echo Done.
endlocal
