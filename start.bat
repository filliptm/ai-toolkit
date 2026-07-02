@echo off
setlocal

set "UI_PORT=3000"

echo Checking port %UI_PORT%...
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":%UI_PORT% .*LISTENING"') do (
    echo Killing process %%P on port %UI_PORT%...
    taskkill /F /PID %%P >nul 2>&1
)

cd /d "%~dp0ui"
npm run dev
