@echo off
REM One-click setup for Windows: double-click this file, or run it in a terminal.
REM Creates a .venv folder (a private Python just for this project) and installs
REM djitellopy + ultralytics (YOLO, which also brings PyTorch CPU and OpenCV).

cd /d "%~dp0"

echo [1/4] Creating virtual environment in .venv ...
if not exist ".venv\Scripts\python.exe" (
    py -3.13 -m venv .venv 2>nul || py -m venv .venv || python -m venv .venv
)
if not exist ".venv\Scripts\python.exe" (
    echo Could not create .venv. Is Python installed? Try: py --version
    pause
    exit /b 1
)

echo [2/4] Upgrading pip ...
".venv\Scripts\python.exe" -m pip install --upgrade pip

echo [3/4] Installing packages (this can take several minutes) ...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Install failed. Check your internet connection and run this again.
    pause
    exit /b 1
)

echo [4/4] Checking the install ...
".venv\Scripts\python.exe" -c "import ultralytics; ultralytics.checks()"

echo.
echo Done. Next: run_demo.bat  (or: .venv\Scripts\python.exe demo.py)
pause
