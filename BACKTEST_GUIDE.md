# MacroHFT Bitcoin Backtesting Guide

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_backtest.txt
```

**Note**: If you encounter issues installing `ta-lib`, you can skip it for now. The backtester will work with the implemented technical indicators.

### 2. Run Complete Pipeline

The easiest way to run the entire backtest is using the main script:

```bash
python backtest/src/run_backtest.py
```

This will:
1. Download 1 year of Bitcoin 1-minute data from Binance
2. Compute all technical indicators
3. Load MacroHFT models and generate trading signals
4. Run the backtest simulation
5. Generate performance metrics and visualizations

### 3. View Results

Results are saved in `./backtest/results/`:
- `trades/trades.csv` - All executed trades
- `metrics/portfolio_history.csv` - Portfolio value over time
- `metrics/equity_curve.csv` - Equity curve data
- `metrics/performance_metrics.csv` - Summary metrics
- `plots/` - Visualization charts

## Configuration

Edit `backtest/config/backtest_config.yaml` to customize:

### Data Settings
```yaml
data:
  symbol: "BTCUSDT"
  interval: "1m"
  start_date: "2024-01-01"  # Or null for 1 year ago
```

### Trading Parameters
```yaml
trading:
  initial_capital: 10000.0
  max_holding_number: 0.01  # BTC position size
  transaction_cost: 0.0002  # 0.02% fee
  slippage: 0.0001  # 0.01%
```

### Model Settings
```yaml
model:
  device: "cuda:0"  # or "cpu"
  # Model paths (using pre-trained ETHUSDT models)
```

## Step-by-Step Execution

If you want to run steps individually:

### Step 1: Download Data

```bash
python backtest/src/data_fetcher.py --config backtest/config/backtest_config.yaml
```

Options:
- `--force`: Force re-download even if data exists

### Step 2: Engineer Features

```bash
python backtest/src/feature_engineering.py --config backtest/config/backtest_config.yaml
```

### Step 3: Generate Signals

```bash
python backtest/src/strategy.py \
    --config backtest/config/backtest_config.yaml \
    --input backtest/data/processed/BTCUSDT_1m_processed.feather \
    --output backtest/data/processed/BTCUSDT_with_signals.feather
```

### Step 4: Run Backtest

```bash
python backtest/src/backtester.py \
    --config backtest/config/backtest_config.yaml \
    --input backtest/data/processed/BTCUSDT_with_signals.feather
```

### Step 5: Analyze Results

```bash
python backtest/src/analysis.py \
    --config backtest/config/backtest_config.yaml \
    --price-data backtest/data/processed/BTCUSDT_with_signals.feather
```

## Advanced Usage

### Using Cached Data

To speed up subsequent runs, use skip flags:

```bash
python backtest/src/run_backtest.py \
    --skip-download \
    --skip-features \
    --skip-signals
```

This will use previously computed data and only re-run the backtest and analysis.

### Custom Date Range

Edit the config file:

```yaml
data:
  start_date: "2023-01-01"
  end_date: "2024-01-01"
```

### Different Initial Capital

```yaml
trading:
  initial_capital: 50000.0  # $50,000
```

### Disable Episodic Memory

If you want to test without the memory component:

```yaml
episodic_memory:
  enabled: false
```

### Skip Demonstration Q-table

To speed up signal generation (skip imitation learning):

```yaml
demonstration:
  compute: false
```

## Performance Metrics Explained

### Total Return
The overall percentage gain/loss of the strategy.

### Sharpe Ratio
Risk-adjusted return. Higher is better. Above 1.0 is good, above 2.0 is excellent.

Formula: `(Return - Risk-free rate) / Volatility`

### Max Drawdown
The largest peak-to-trough decline. Measures worst-case loss.

### Win Rate
Percentage of winning trades out of total trades.

### Profit Factor
Ratio of gross profits to gross losses. Above 1.0 means profitable.

### Calmar Ratio
Ratio of annualized return to maximum drawdown. Measures return vs. risk.

## Troubleshooting

### GPU/CUDA Issues

If you don't have CUDA or encounter GPU errors:

```yaml
model:
  device: "cpu"
```

### Memory Issues

If you run out of memory with large datasets:

1. Reduce the date range
2. Process data in chunks
3. Use a machine with more RAM

### Missing Models

The system expects pre-trained models in:
```
MacroHFT/result/low_level/ETHUSDT/best_model/
├── slope/
│   ├── 1/best_model.pkl
│   ├── 2/best_model.pkl
│   └── 3/best_model.pkl
└── vol/
    ├── 1/best_model.pkl
    ├── 2/best_model.pkl
    └── 3/best_model.pkl
```

If models are missing, you'll need to train them first using the MacroHFT training pipeline.

### Binance API Rate Limits

If you hit rate limits:
- The fetcher automatically sleeps between requests
- For very large downloads, run overnight
- Consider downloading data once and reusing it

## Understanding the Results

### Equity Curve Plot
Shows your portfolio value over time vs. Buy & Hold benchmark.

### Drawdown Plot
Shows periods when your strategy was underwater (below peak).

### Trades Plot
Visual representation of buy/sell signals on the price chart.

### Returns Distribution
Histogram showing the distribution of your returns. Should ideally be:
- Centered slightly above zero (positive average return)
- Not too fat-tailed (extreme moves are rare)

## Tips for Better Results

1. **Longer Backtest Period**: Test on at least 1 year of data
2. **Out-of-Sample Testing**: Keep some data for validation
3. **Transaction Costs**: Be realistic with fees and slippage
4. **Walk-Forward Analysis**: Test on rolling windows
5. **Parameter Sensitivity**: Try different position sizes
6. **Market Conditions**: Test in different market regimes (bull, bear, sideways)

## Next Steps

After running the initial backtest:

1. Review the performance metrics
2. Analyze the equity curve and drawdown
3. Inspect individual trades for patterns
4. Consider adjusting parameters
5. Test on different time periods
6. Compare with buy-and-hold benchmark

## Support

For issues or questions:
1. Check the error messages carefully
2. Review the configuration file
3. Ensure all dependencies are installed
4. Check that MacroHFT models are available

