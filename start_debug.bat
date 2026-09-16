@echo off
setlocal
cd /d "%~dp0"
python main.py --debug
if errorlevel 1 pause
endlocal
