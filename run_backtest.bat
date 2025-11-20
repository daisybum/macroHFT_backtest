@echo off
REM Quick start script for MacroHFT Bitcoin Backtesting (Windows)

echo ==================================
echo MacroHFT Bitcoin Backtest Runner
echo ==================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed
    exit /b 1
)

REM Check if config exists
if not exist "backtest\config\backtest_config.yaml" (
    echo Error: Config file not found at backtest\config\backtest_config.yaml
    exit /b 1
)

REM Create necessary directories
echo Creating directories...
if not exist "backtest\data\raw" mkdir backtest\data\raw
if not exist "backtest\data\processed" mkdir backtest\data\processed
if not exist "backtest\data\cache" mkdir backtest\data\cache
if not exist "backtest\results\trades" mkdir backtest\results\trades
if not exist "backtest\results\metrics" mkdir backtest\results\metrics
if not exist "backtest\results\plots" mkdir backtest\results\plots

REM Run the backtest
echo.
echo Starting backtest pipeline...
echo.

python backtest\src\run_backtest.py %*

echo.
echo ==================================
echo Backtest Complete!
echo ==================================
echo.
echo View results in: .\backtest\results\
pause

