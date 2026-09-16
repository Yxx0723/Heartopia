@echo off
setlocal
cd /d "%~dp0"
python -m tools.soak_test --targets 20
if errorlevel 1 pause
endlocal
