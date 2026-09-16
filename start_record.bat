@echo off
setlocal
cd /d "%~dp0"
python main.py --record
if errorlevel 1 pause
endlocal
