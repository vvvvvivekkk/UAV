@echo off
REM Runs demo.py with the project's own Python (.venv). Run setup_windows.bat first.
cd /d "%~dp0"
".venv\Scripts\python.exe" demo.py
pause
