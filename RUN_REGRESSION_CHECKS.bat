@echo off
setlocal
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0RUN_REGRESSION_CHECKS.ps1"

echo.
echo KQ Quant Tool regression checks finished.
echo If any check failed, read the message above.
pause
