@echo off
setlocal
cd /d "%~dp0"
python main.py --check --config config/default.yaml
if errorlevel 1 pause
endlocal
