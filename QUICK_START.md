# MacroHFT Bitcoin Backtesting - Quick Start Guide

## 🎉 Implementation Complete!

All backtesting code has been successfully implemented. You just need to install dependencies and run!

---

## ✅ What's Been Implemented

### Core System (100% Complete)
- ✅ **Data Acquisition**: Download 1-min BTC data from Binance
- ✅ **Feature Engineering**: Compute all technical indicators  
- ✅ **Strategy**: Full MacroHFT (6 sub-agents + hyperagent + episodic memory)
- ✅ **Backtester**: Event-driven engine with transaction costs
- ✅ **Analysis**: Performance metrics and visualizations
- ✅ **Orchestrator**: Complete pipeline automation

### Files Created
```
backtest/
├── config/backtest_config.yaml    ✅ Configuration
├── src/
│   ├── data_fetcher.py            ✅ Binance data download
│   ├── feature_engineering.py     ✅ Technical indicators
│   ├── strategy.py                ✅ MacroHFT integration
│   ├── backtester.py              ✅ Backtest engine
│   ├── analysis.py                ✅ Performance analysis
│   └── run_backtest.py            ✅ Main runner
requirements_backtest.txt          ✅ Dependencies list
verify_installation.py             ✅ System check script
run_backtest.sh                    ✅ Unix runner
run_backtest.bat                   ✅ Windows runner
BACKTEST_GUIDE.md                  ✅ User manual
IMPLEMENTATION_STATUS.md           ✅ Full documentation
```

---

## 🚀 Getting Started (3 Steps)

### Step 1: Install Dependencies

```bash
pip install -r requirements_backtest.txt
```

This will install:
- numpy, pandas, scipy
- pytorch (for models)
- python-binance (for data)
- matplotlib, seaborn (for plots)
- And more...

**Note**: Installation may take 5-10 minutes depending on your internet speed.

### Step 2: Verify Installation

```bash
python verify_installation.py
```

You should see:
```
[SUCCESS] All checks passed!
Your system is ready to run MacroHFT Bitcoin backtesting.
```

### Step 3: Run Backtest

**Option A: One-Command (Recommended)**
```bash
# Windows
run_backtest.bat

# Linux/Mac
bash run_backtest.sh
```

**Option B: Direct Python**
```bash
python backtest/src/run_backtest.py
```

**Expected Time**: 30-60 minutes for first run (downloads data + computes features + runs backtest)

---

## 📊 What Will Happen

The system will automatically:

1. **Download Data** (5-10 min)
   - Fetch 1 year of BTC/USDT 1-minute candles from Binance
   - ~525,000 data points
   - Save to `backtest/data/raw/`

2. **Engineer Features** (5-10 min)
   - Compute technical indicators (RSI, MACD, Bollinger Bands, etc.)
   - Calculate slope_360 and vol_360 for classification
   - Save to `backtest/data/processed/`

3. **Generate Signals** (10-20 min)
   - Load 6 pre-trained MacroHFT models
   - Initialize episodic memory
   - Compute demonstration Q-table
   - Generate trading signals for each candle
   - Save signals to processed data

4. **Run Backtest** (5-10 min)
   - Simulate trading with transaction costs
   - Track portfolio performance
   - Record all trades
   - Save to `backtest/results/trades/`

5. **Analyze Results** (2-5 min)
   - Calculate performance metrics
   - Generate visualizations
   - Compare with Buy & Hold benchmark
   - Save to `backtest/results/metrics/` and `backtest/results/plots/`

---

## 📈 Results You'll Get

### Performance Metrics
Located in: `backtest/results/metrics/performance_metrics.csv`

```
Total Return: X%
Sharpe Ratio: X.XX
Max Drawdown: X.X%
Win Rate: XX.X%
Profit Factor: X.XX
Calmar Ratio: X.XX
Number of Trades: XXX
```

### Visualizations  
Located in: `backtest/results/plots/`

- **equity_curve.png**: Your strategy vs Buy & Hold
- **drawdown.png**: Drawdown over time
- **trades_on_price.png**: Buy/sell signals on price chart
- **returns_distribution.png**: Returns histogram
- **monthly_returns_heatmap.png**: Monthly performance

### Trade History
Located in: `backtest/results/trades/trades.csv`

Complete log of every trade executed:
- Timestamp
- Action (BUY/SELL)
- Price
- Quantity
- Fees
- Portfolio value

---

## ⚙️ Configuration

Edit `backtest/config/backtest_config.yaml` to customize:

### Date Range
```yaml
data:
  start_date: "2023-01-01"  # Change this
  end_date: null            # null = today
```

### Initial Capital
```yaml
trading:
  initial_capital: 10000.0  # Change to $50,000, $100,000, etc.
```

### Position Size
```yaml
trading:
  max_holding_number: 0.01  # BTC per trade (0.01 = $400-600 typically)
```

### Transaction Costs
```yaml
trading:
  transaction_cost: 0.0002  # 0.02% (Binance standard)
  slippage: 0.0001          # 0.01%
```

### GPU vs CPU
```yaml
model:
  device: "cuda:0"  # Use "cpu" if no GPU
```

---

## 🔄 Subsequent Runs (Fast Mode)

After the first run, use caching to skip completed steps:

```bash
python backtest/src/run_backtest.py --skip-download --skip-features
```

This will:
- ✅ Use existing data (no re-download)
- ✅ Use existing features (no re-computation)
- ⏩ Re-run signals, backtest, and analysis only

**Time**: 10-15 minutes instead of 30-60 minutes

---

## 📚 Documentation

- **BACKTEST_GUIDE.md**: Comprehensive user guide
- **IMPLEMENTATION_STATUS.md**: Technical documentation
- **MacroHFT_Analysis.md**: Algorithm explanation
- **backtest/README.md**: Project overview

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution**: Run `pip install -r requirements_backtest.txt`

### Issue: "CUDA out of memory"
**Solution**: Edit config, change `device: "cpu"`

### Issue: "Binance rate limit"
**Solution**: Wait 1 minute and try again. The fetcher has automatic retry logic.

### Issue: "Models not found"
**Solution**: Ensure `MacroHFT/result/low_level/ETHUSDT/best_model/` contains 6 .pkl files

### Issue: Slow performance
**Solutions**:
- Use GPU instead of CPU (much faster)
- Reduce date range (test with 1 month first)
- Use `--skip-download --skip-features` for subsequent runs

---

## 💡 Tips

1. **First Time**: Run with default settings to see how it works
2. **Test Short Period**: Try 1 month first (`start_date: "2024-10-01"`)
3. **Check Results**: Review metrics before running longer backtests
4. **Iterate**: Adjust parameters and re-run with caching enabled
5. **Document**: Keep notes on what settings work best

---

## 🎯 Next Steps

After your first successful backtest:

1. **Analyze Results**: Look at equity curve and metrics
2. **Compare Benchmark**: How does it compare to Buy & Hold?
3. **Test Different Periods**: Bull market, bear market, sideways
4. **Adjust Parameters**: Try different position sizes or costs
5. **Walk-Forward Analysis**: Test on multiple time windows

---

## ✨ Summary

You're all set! The complete Bitcoin MacroHFT backtesting system is ready to run.

**To begin**:
```bash
# 1. Install dependencies
pip install -r requirements_backtest.txt

# 2. Verify
python verify_installation.py

# 3. Run backtest
python backtest/src/run_backtest.py
```

**Expected Output**: Performance metrics, visualizations, and trade history in `backtest/results/`

---

## 📞 Support

If you encounter issues:
1. Check the error message carefully
2. Verify all dependencies are installed
3. Ensure models exist in `MacroHFT/result/low_level/ETHUSDT/`
4. Try with CPU mode if GPU issues occur
5. Test with a shorter date range first

---

**Good luck with your backtesting!** 🚀📈

*Last Updated: 2024-11-12*

