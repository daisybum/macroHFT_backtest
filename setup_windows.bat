@echo off
REM MacroHFT Bitcoin Backtesting - Windows Setup Script (CPU-only)
REM This script creates a virtual environment and installs all dependencies

echo ========================================
echo MacroHFT Windows Setup (CPU-only)
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo [1/5] Checking Python version...
python --version

echo.
echo [2/5] Creating virtual environment...
if exist venv (
    echo Virtual environment already exists. Removing old one...
    rmdir /s /q venv
)
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)
echo Virtual environment created successfully!

echo.
echo [3/5] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo.
echo [4/5] Upgrading pip, setuptools, and wheel...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo WARNING: Failed to upgrade pip
)

echo.
echo [5/5] Installing dependencies (this may take 5-10 minutes)...
echo This will install PyTorch CPU-only version (no GPU required)
echo.
pip install -r requirements_backtest.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    echo.
    echo Troubleshooting tips:
    echo 1. Make sure you have internet connection
    echo 2. Try running as Administrator
    echo 3. Check your antivirus settings
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Virtual environment: %CD%\venv
echo Python: venv\Scripts\python.exe
echo.
echo To activate the environment in the future:
echo   venv\Scripts\activate.bat
echo.
echo To verify installation:
echo   python verify_installation.py
echo.
echo To run backtest:
echo   python backtest\src\run_backtest.py
echo.
echo ========================================
pause

