# Bitcoin MacroHFT Backtesting System

This backtesting system uses the MacroHFT hierarchical reinforcement learning framework to backtest trading strategies on Bitcoin using 1-minute candle data from Binance.

## Project Structure

```
backtest/
├── data/                      # Data storage
│   ├── raw/                   # Raw Binance data
│   ├── processed/             # Processed features
│   └── cache/                 # Cached computations
├── src/                       # Source code
│   ├── data_fetcher.py       # Binance API data acquisition
│   ├── feature_engineering.py # Technical indicators
│   ├── strategy.py           # MacroHFT strategy implementation
│   ├── backtester.py         # Core backtesting engine
│   └── analysis.py           # Performance analysis
├── models/                    # Pre-trained MacroHFT models
│   └── (loaded from ../MacroHFT/result/)
├── results/                   # Backtest results
│   ├── trades/               # Trade logs
│   ├── metrics/              # Performance metrics
│   └── plots/                # Visualization outputs
├── notebooks/                 # Jupyter notebooks for analysis
└── config/                    # Configuration files
    └── backtest_config.yaml  # Backtest parameters
```

## Usage

1. Install dependencies: `pip install -r requirements_backtest.txt`
2. Configure backtest parameters in `config/backtest_config.yaml`
3. Run data fetcher: `python backtest/src/data_fetcher.py`
4. Run backtest: `python backtest/src/run_backtest.py`
5. Analyze results: `python backtest/src/analysis.py`

## Configuration

Key parameters in `config/backtest_config.yaml`:
- Trading pair: BTCUSDT
- Timeframe: 1 minute
- Backtest period: Last 1 year
- Transaction costs: 0.02% (Binance standard)
- Position sizing: 0.01 BTC (matching MacroHFT training)

