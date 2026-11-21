@echo off
REM ============================================================
REM  ONE-TIME SETUP for the drone exam (run on every new laptop)
REM  Needs normal internet - NOT the Tello Wi-Fi.
REM  Double-click this file, or run:  setup.bat
REM  Creates .venv2 with everything, then checks it all works.
REM ============================================================
cd /d "%~dp0"

set PY=py
where py >nul 2>&1 || set PY=python

if exist .venv2\Scripts\python.exe (
    .venv2\Scripts\python.exe -c "print('venv ok')" >nul 2>&1
    if errorlevel 1 (
        echo .venv2 is damaged - repairing it...
        %PY% -m venv --upgrade .venv2
    )
) else (
    echo Creating .venv2 ...
    %PY% -m venv .venv2
)

.venv2\Scripts\python.exe -m pip --version >nul 2>&1 || .venv2\Scripts\python.exe -m ensurepip
.venv2\Scripts\python.exe -m pip install --upgrade pip
.venv2\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo INSTALL FAILED - check the internet connection and run setup.bat again.
    pause
    exit /b 1
)

echo.
.venv2\Scripts\python.exe exam\check_setup.py
pause
