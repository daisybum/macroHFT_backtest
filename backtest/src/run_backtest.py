"""
Main Orchestration Script for MacroHFT Bitcoin Backtesting
Runs the complete pipeline from data fetching to analysis
"""

import os
import sys
import argparse
import yaml
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from data_fetcher import BinanceDataFetcher
from feature_engineering import FeatureEngineer
from strategy import MacroHFTStrategy
from backtester import MacroHFTBacktester
from analysis import PerformanceAnalyzer


def main():
    """Main execution pipeline"""
    parser = argparse.ArgumentParser(
        description='Run complete MacroHFT Bitcoin backtest pipeline'
    )
    parser.add_argument('--config', type=str, 
                       default='./backtest/config/backtest_config.yaml',
                       help='Path to config file')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip data download if data already exists')
    parser.add_argument('--skip-features', action='store_true',
                       help='Skip feature engineering if processed data exists')
    parser.add_argument('--skip-signals', action='store_true',
                       help='Skip signal generation if signals already exist')
    parser.add_argument('--force-redownload', action='store_true',
                       help='Force re-download of data')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n" + "="*70)
    print(" "*15 + "MacroHFT BITCOIN BACKTESTING SYSTEM")
    print("="*70)
    print(f"\nStart Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: {args.config}")
    print("="*70 + "\n")
    
    # =================================================================
    # STEP 1: DATA ACQUISITION
    # =================================================================
    print("\n" + "="*70)
    print("STEP 1: DATA ACQUISITION")
    print("="*70)
    
    fetcher = BinanceDataFetcher(config_path=args.config)
    
    raw_data_file = os.path.join(
        config['data']['data_dir'],
        f"{config['data']['symbol']}_{config['data']['interval']}_raw.feather"
    )
    
    if os.path.exists(raw_data_file) and args.skip_download and not args.force_redownload:
        print(f"\n[OK] Using existing data: {raw_data_file}")
        df_raw = fetcher.load_data()
    else:
        print("\n>> Fetching data from Binance API...")
        df_raw = fetcher.run(force_download=args.force_redownload)
    
    print(f"\n[OK] Raw data loaded: {len(df_raw):,} candles")
    print(f"  Date range: {df_raw.index[0]} to {df_raw.index[-1]}")
    
    # =================================================================
    # STEP 2: FEATURE ENGINEERING
    # =================================================================
    print("\n" + "="*70)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*70)
    
    engineer = FeatureEngineer(config_path=args.config)
    
    processed_data_file = os.path.join(
        config['data']['processed_dir'],
        "BTCUSDT_1m_processed.feather"
    )
    
    if os.path.exists(processed_data_file) and args.skip_features:
        print(f"\n[OK] Using existing processed data: {processed_data_file}")
        df_processed = engineer.load_processed_data()
    else:
        print("\n>> Computing technical indicators...")
        df_processed = engineer.compute_all_features(df_raw)
        engineer.save_processed_data(df_processed)
    
    print(f"\n[OK] Processed data ready: {len(df_processed):,} candles, {len(df_processed.columns)} features")
    
    # =================================================================
    # STEP 3: STRATEGY & SIGNAL GENERATION
    # =================================================================
    print("\n" + "="*70)
    print("STEP 3: SIGNAL GENERATION")
    print("="*70)
    
    strategy = MacroHFTStrategy(config_path=args.config)
    
    signals_data_file = os.path.join(
        config['data']['processed_dir'],
        "BTCUSDT_with_signals.feather"
    )
    
    if os.path.exists(signals_data_file) and args.skip_signals:
        print(f"\n[OK] Using existing signals: {signals_data_file}")
        import pandas as pd
        df_with_signals = pd.read_feather(signals_data_file)
        if 'timestamp' in df_with_signals.columns:
            df_with_signals.set_index('timestamp', inplace=True)
    else:
        print("\n>> Generating trading signals using MacroHFT models...")
        
        # Compute demonstration Q-table if configured
        if config['demonstration']['compute']:
            print("\n>> Computing demonstration Q-table (this may take a while)...")
            strategy.compute_demonstration_qtable(df_processed)
        
        # Generate signals
        df_with_signals = strategy.generate_signals(
            df_processed, 
            initial_action=config['backtest']['initial_position']
        )
        
        # Save signals
        df_with_signals.reset_index().to_feather(signals_data_file)
        print(f"\n[OK] Signals saved to: {signals_data_file}")
    
    # Signal statistics
    buy_signals = (df_with_signals['signal'] == 'BUY').sum()
    sell_signals = (df_with_signals['signal'] == 'SELL').sum()
    hold_signals = (df_with_signals['signal'] == 'HOLD').sum()
    
    print(f"\n[OK] Signal generation complete:")
    print(f"  Buy signals:  {buy_signals:,}")
    print(f"  Sell signals: {sell_signals:,}")
    print(f"  Hold signals: {hold_signals:,}")
    
    # =================================================================
    # STEP 4: BACKTESTING
    # =================================================================
    print("\n" + "="*70)
    print("STEP 4: BACKTESTING")
    print("="*70)
    
    backtester = MacroHFTBacktester(config_path=args.config)
    
    print("\n>> Running backtest...")
    results = backtester.run(df_with_signals, verbose=True)
    
    # Save results
    print("\n>> Saving backtest results...")
    backtester.save_results()
    
    print(f"\n[OK] Backtest complete:")
    print(f"  Initial capital: ${results['initial_capital']:,.2f}")
    print(f"  Final value:     ${results['final_value']:,.2f}")
    print(f"  Total return:    {results['total_return']*100:.2f}%")
    print(f"  Total trades:    {results['total_trades']:,}")
    
    # =================================================================
    # STEP 5: PERFORMANCE ANALYSIS
    # =================================================================
    print("\n" + "="*70)
    print("STEP 5: PERFORMANCE ANALYSIS")
    print("="*70)
    
    analyzer = PerformanceAnalyzer(config_path=args.config)
    
    print("\n>> Calculating performance metrics...")
    analyzer.generate_report(price_data=df_with_signals)
    
    # =================================================================
    # SUMMARY
    # =================================================================
    print("\n" + "="*70)
    print("BACKTEST PIPELINE COMPLETE")
    print("="*70)
    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults saved to: {config['output']['results_dir']}")
    print("\nGenerated files:")
    print(f"  - Trades:          {config['output']['results_dir']}/trades/trades.csv")
    print(f"  - Portfolio:       {config['output']['results_dir']}/metrics/portfolio_history.csv")
    print(f"  - Equity curve:    {config['output']['results_dir']}/metrics/equity_curve.csv")
    print(f"  - Metrics:         {config['output']['results_dir']}/metrics/performance_metrics.csv")
    print(f"  - Plots:           {config['output']['results_dir']}/plots/")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()

