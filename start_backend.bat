@echo off
setlocal
set "ROOT=%~dp0"
set "APP_DIR=%ROOT%backend"
set "VENV=%ROOT%.venv\Scripts\activate.bat"

start "FakeGuard Backend" cmd /k ""%VENV%" && cd /d "%APP_DIR%" && python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"
endlocal
