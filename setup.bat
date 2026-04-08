@echo off
title Schematic Comparator - Setup
cd /d "%~dp0"
echo.
echo  ============================================
echo   Schematic Comparator - First-time Setup
echo  ============================================
echo.
where uv >nul 2>nul
if %errorlevel%==0 (
    echo [uv] Creating virtual environment...
    uv venv .venv
    echo [uv] Installing packages...
    uv pip install -r requirements.txt --python .venv\Scripts\python.exe
) else (
    echo [pip] Creating virtual environment...
    python -m venv .venv
    echo [pip] Installing packages...
    .venv\Scripts\pip install -r requirements.txt
)
echo.
echo  Setup complete! Run start.bat to launch.
pause
