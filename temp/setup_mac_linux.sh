#!/usr/bin/env bash
# One-command setup for macOS / Linux:  bash setup_mac_linux.sh
set -e
cd "$(dirname "$0")"

echo "[1/4] Creating virtual environment in .venv ..."
[ -x .venv/bin/python ] || python3 -m venv .venv

echo "[2/4] Upgrading pip ..."
.venv/bin/python -m pip install --upgrade pip

echo "[3/4] Installing packages (this can take several minutes) ..."
.venv/bin/python -m pip install -r requirements.txt

echo "[4/4] Checking the install ..."
.venv/bin/python -c "import ultralytics; ultralytics.checks()"

echo
echo "Done. Next: .venv/bin/python demo.py"
