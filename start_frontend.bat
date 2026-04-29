@echo off
setlocal

echo [FakeGuard] Syncing source to clean dev path...
robocopy "D:\FakeGuardFrontend\src"          "D:\FakeGuardDev\src"          /E /MIR /NFL /NDL /NJH /NJS >nul 2>&1
robocopy "D:\FakeGuardFrontend\public"        "D:\FakeGuardDev\public"       /E /MIR /NFL /NDL /NJH /NJS >nul 2>&1
robocopy "D:\FakeGuardFrontend" "D:\FakeGuardDev" vite.config.ts tailwind.config.js tsconfig.json postcss.config.js index.html /NFL /NDL /NJH /NJS >nul 2>&1

echo [FakeGuard] Starting frontend at http://localhost:5173 ...
start "FakeGuard Frontend" cmd /k "cd /d D:\FakeGuardDev && npm run dev -- --port 5173"

endlocal
