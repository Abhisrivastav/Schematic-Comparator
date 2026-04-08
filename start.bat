@echo off
title Schematic Comparator
cd /d "%~dp0"
echo.
echo  ============================================
echo   Schematic Comparator
echo   http://127.0.0.1:5000
echo  ============================================
echo.
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found.
    echo Please run setup.bat first.
    pause
    exit /b 1
)
echo  Starting server...
.venv\Scripts\python.exe app.py
pause
