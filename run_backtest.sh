#!/bin/bash
# Quick start script for MacroHFT Bitcoin Backtesting

echo "=================================="
echo "MacroHFT Bitcoin Backtest Runner"
echo "=================================="
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed"
    exit 1
fi

# Check if config exists
if [ ! -f "backtest/config/backtest_config.yaml" ]; then
    echo "Error: Config file not found at backtest/config/backtest_config.yaml"
    exit 1
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p backtest/data/raw
mkdir -p backtest/data/processed
mkdir -p backtest/data/cache
mkdir -p backtest/results/trades
mkdir -p backtest/results/metrics
mkdir -p backtest/results/plots

# Run the backtest
echo ""
echo "Starting backtest pipeline..."
echo ""

python backtest/src/run_backtest.py "$@"

echo ""
echo "=================================="
echo "Backtest Complete!"
echo "=================================="
echo ""
echo "View results in: ./backtest/results/"

