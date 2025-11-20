"""
Binance Data Fetcher for Bitcoin 1-minute OHLCV Data
Downloads historical data from Binance API and stores as feather files
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Optional, List
import pandas as pd
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException
import time
from tqdm import tqdm
import yaml


class BinanceDataFetcher:
    """Fetch historical 1-minute candle data from Binance for Bitcoin"""
    
    def __init__(self, config_path: str = "./backtest/config/backtest_config.yaml"):
        """Initialize Binance data fetcher"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize Binance client (public API, no keys needed for historical data)
        self.client = Client("", "")  # Empty keys for public endpoints
        
        self.symbol = self.config['data']['symbol']
        self.interval = self.config['data']['interval']
        self.data_dir = self.config['data']['data_dir']
        
        # Create data directory
        os.makedirs(self.data_dir, exist_ok=True)
        
    def _get_date_range(self) -> tuple:
        """Calculate date range for data fetching"""
        end_date = datetime.now()
        
        # Get start date from config or default to 1 year ago
        start_date_str = self.config['data'].get('start_date')
        if start_date_str:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        else:
            start_date = end_date - timedelta(days=365)
        
        return start_date, end_date
    
    def fetch_klines(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Fetch klines (OHLCV) data from Binance
        
        Args:
            start_time: Start datetime
            end_time: End datetime
            
        Returns:
            DataFrame with OHLCV data
        """
        print(f"Fetching {self.symbol} {self.interval} data from {start_time} to {end_time}")
        
        # Convert to milliseconds timestamp
        start_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)
        
        all_klines = []
        current_ts = start_ts
        
        # Binance API limit: 1000 klines per request
        limit = 1000
        
        with tqdm(total=(end_ts - start_ts) // (60 * 1000), desc="Downloading") as pbar:
            while current_ts < end_ts:
                try:
                    # Fetch klines
                    klines = self.client.get_klines(
                        symbol=self.symbol,
                        interval=self.interval,
                        startTime=current_ts,
                        endTime=end_ts,
                        limit=limit
                    )
                    
                    if not klines:
                        break
                    
                    all_klines.extend(klines)
                    
                    # Update timestamp for next batch
                    current_ts = klines[-1][0] + 60000  # Add 1 minute in milliseconds
                    
                    pbar.update(len(klines))
                    
                    # Rate limiting: sleep to avoid hitting API limits
                    time.sleep(0.1)
                    
                except BinanceAPIException as e:
                    print(f"Binance API Error: {e}")
                    time.sleep(5)
                    continue
                except Exception as e:
                    print(f"Error: {e}")
                    time.sleep(5)
                    continue
        
        # Convert to DataFrame
        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                          'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Sort by timestamp
        df.sort_index(inplace=True)
        
        print(f"Downloaded {len(df)} candles")
        
        return df
    
    def check_data_quality(self, df: pd.DataFrame) -> dict:
        """Check data quality and report issues"""
        print("\n=== Data Quality Check ===")
        
        quality_report = {
            'total_rows': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'date_range': (df.index.min(), df.index.max()),
            'duplicates': df.index.duplicated().sum(),
        }
        
        # Check for missing timestamps (gaps)
        time_diff = df.index.to_series().diff()
        expected_diff = pd.Timedelta(minutes=1)
        gaps = time_diff[time_diff > expected_diff]
        quality_report['gaps'] = len(gaps)
        
        if len(gaps) > 0:
            print(f"WARNING: Found {len(gaps)} gaps in data")
            print(f"Largest gap: {gaps.max()}")
        
        # Check for outliers (basic check)
        price_change = df['close'].pct_change()
        extreme_moves = price_change[abs(price_change) > 0.1]  # >10% move in 1 minute
        quality_report['extreme_moves'] = len(extreme_moves)
        
        if len(extreme_moves) > 0:
            print(f"WARNING: Found {len(extreme_moves)} extreme price moves (>10%)")
        
        print(f"Total rows: {quality_report['total_rows']}")
        print(f"Date range: {quality_report['date_range'][0]} to {quality_report['date_range'][1]}")
        print(f"Missing values: {sum(quality_report['missing_values'].values())}")
        print(f"Duplicates: {quality_report['duplicates']}")
        
        return quality_report
    
    def save_data(self, df: pd.DataFrame, filename: str = None):
        """Save data to feather format"""
        if filename is None:
            filename = f"{self.symbol}_{self.interval}_raw.feather"
        
        filepath = os.path.join(self.data_dir, filename)
        
        # Reset index to save timestamp as column
        df_to_save = df.reset_index()
        df_to_save.to_feather(filepath)
        
        print(f"\nData saved to: {filepath}")
        print(f"File size: {os.path.getsize(filepath) / (1024*1024):.2f} MB")
        
        return filepath
    
    def load_data(self, filename: str = None) -> pd.DataFrame:
        """Load data from feather format"""
        if filename is None:
            filename = f"{self.symbol}_{self.interval}_raw.feather"
        
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_feather(filepath)
        df.set_index('timestamp', inplace=True)
        
        print(f"Loaded data from: {filepath}")
        print(f"Shape: {df.shape}")
        
        return df
    
    def run(self, force_download: bool = False):
        """Main execution: fetch and save data"""
        filename = f"{self.symbol}_{self.interval}_raw.feather"
        filepath = os.path.join(self.data_dir, filename)
        
        # Check if data already exists
        if os.path.exists(filepath) and not force_download:
            print(f"Data file already exists: {filepath}")
            print("Loading existing data...")
            df = self.load_data(filename)
            self.check_data_quality(df)
            return df
        
        # Fetch new data
        start_date, end_date = self._get_date_range()
        df = self.fetch_klines(start_date, end_date)
        
        # Quality check
        quality_report = self.check_data_quality(df)
        
        # Save data
        self.save_data(df, filename)
        
        return df


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch Bitcoin data from Binance')
    parser.add_argument('--config', type=str, default='./backtest/config/backtest_config.yaml',
                       help='Path to config file')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download even if data exists')
    
    args = parser.parse_args()
    
    # Initialize and run fetcher
    fetcher = BinanceDataFetcher(config_path=args.config)
    df = fetcher.run(force_download=args.force)
    
    print("\n=== Data Fetching Complete ===")
    print(f"Total candles: {len(df)}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")


if __name__ == "__main__":
    main()

