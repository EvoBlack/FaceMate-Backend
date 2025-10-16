@echo off
echo ========================================
echo Pushing FaceMate Backend to GitHub
echo ========================================
echo.

cd /d "%~dp0"

echo [1/5] Initializing Git repository...
git init
if errorlevel 1 (
    echo Error: Git initialization failed
    pause
    exit /b 1
)

echo.
echo [2/5] Adding remote repository...
git remote remove origin 2>nul
git remote add origin https://github.com/EvoBlack/FaceMate-Backend.git
if errorlevel 1 (
    echo Error: Failed to add remote
    pause
    exit /b 1
)

echo.
echo [3/5] Adding files...
git add .
if errorlevel 1 (
    echo Error: Failed to add files
    pause
    exit /b 1
)

echo.
echo [4/5] Creating commit...
git commit -m "Initial commit: FaceMate Backend API with face recognition"
if errorlevel 1 (
    echo Error: Failed to create commit
    pause
    exit /b 1
)

echo.
echo [5/5] Pushing to GitHub...
git branch -M main
git push -u origin main --force
if errorlevel 1 (
    echo Error: Failed to push to GitHub
    echo.
    echo Please check:
    echo - Your GitHub credentials
    echo - Repository exists: https://github.com/EvoBlack/FaceMate-Backend.git
    echo - You have write access to the repository
    pause
    exit /b 1
)

echo.
echo ========================================
echo SUCCESS! Backend pushed to GitHub
echo ========================================
echo.
echo Repository: https://github.com/EvoBlack/FaceMate-Backend.git
echo.
pause
