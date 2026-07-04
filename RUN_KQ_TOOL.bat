@echo off
setlocal
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0RUN_WINDOWS.ps1"

echo.
echo KQ Quant Tool launcher finished.
echo If the server stopped because of an error, read the message above.
pause
