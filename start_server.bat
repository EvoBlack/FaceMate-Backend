@echo off
echo ============================================================
echo Starting FaceMate Backend Server
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

REM Navigate to the script directory
cd /d "%~dp0"

REM Start the Flask server (which will handle ADB automatically)
python app.py

pause
