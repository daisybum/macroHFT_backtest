"""
Feature Engineering for MacroHFT Bitcoin Backtesting
Computes technical indicators matching MacroHFT's training features
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.linear_model import LinearRegression
import yaml
from typing import List, Optional


# Add MacroHFT to path for importing feature lists
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))


class FeatureEngineer:
    """Compute technical indicators for MacroHFT strategy"""
    
    def __init__(self, config_path: str = "./backtest/config/backtest_config.yaml"):
        """Initialize feature engineer"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load feature lists from MacroHFT
        feature_list_path = self.config['model']['feature_list_path']
        
        try:
            self.single_features = np.load(
                os.path.join(feature_list_path, 'single_features.npy'),
                allow_pickle=True
            ).tolist()
        except:
            # Default features if file not loadable
            self.single_features = self._get_default_single_features()
        
        try:
            self.trend_features = np.load(
                os.path.join(feature_list_path, 'trend_features.npy'),
                allow_pickle=True
            ).tolist()
        except:
            # Default features if file not loadable
            self.trend_features = self._get_default_trend_features()
        
        print(f"Loaded {len(self.single_features)} single features")
        print(f"Loaded {len(self.trend_features)} trend features")
        
    def _get_default_single_features(self) -> List[str]:
        """Default single state features"""
        return ['close', 'volume', 'high', 'low', 'open']
    
    def _get_default_trend_features(self) -> List[str]:
        """Default trend state features"""
        return ['close_ma_20', 'volume_ma_20', 'rsi_14', 'macd']
    
    def smooth_data(self, data: np.ndarray, N: int = 1, Wn: float = 0.05) -> np.ndarray:
        """
        Apply Butterworth filter for smoothing
        
        Args:
            data: Input time series
            N: Filter order
            Wn: Cutoff frequency
            
        Returns:
            Smoothed data
        """
        b, a = butter(N, Wn, btype='low')
        return filtfilt(b, a, data)
    
    def get_slope(self, data: np.ndarray) -> float:
        """
        Calculate slope using linear regression
        
        Args:
            data: Input time series
            
        Returns:
            Slope coefficient
        """
        X = np.arange(len(data)).reshape(-1, 1)
        model = LinearRegression().fit(X, data)
        return model.coef_[0]
    
    def get_slope_window(self, window: pd.Series) -> float:
        """
        Calculate slope for a rolling window
        
        Args:
            window: Pandas Series window
            
        Returns:
            Slope coefficient
        """
        if len(window) < 2:
            return 0.0
        
        try:
            # Smooth the window
            y = self.smooth_data(window.values)
            
            # Calculate slope
            X = np.arange(len(y)).reshape(-1, 1)
            model = LinearRegression().fit(X, y)
            return model.coef_[0]
        except:
            return 0.0
    
    def compute_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute basic technical indicators
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added features
        """
        df = df.copy()
        
        print("Computing basic features...")
        
        # Returns
        df['returns'] = df['close'].pct_change()
        
        # Log returns
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price changes
        df['price_change'] = df['close'] - df['open']
        df['price_change_pct'] = (df['close'] - df['open']) / df['open']
        
        # High-Low range
        df['hl_range'] = df['high'] - df['low']
        df['hl_range_pct'] = (df['high'] - df['low']) / df['close']
        
        # Volume features
        df['volume_change'] = df['volume'].pct_change()
        
        # VWAP
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        return df
    
    def compute_lob_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features based on Limit Order Book (LOB) data if available.
        
        Args:
            df: DataFrame potentially containing LOB columns (bid1_p, bid1_q, etc.)
            
        Returns:
            DataFrame with LOB features added
        """
        df = df.copy()
        
        # Check if LOB columns exist (at least level 1)
        if 'bid1_p' not in df.columns or 'ask1_p' not in df.columns:
            print("LOB columns not found. Skipping LOB feature computation.")
            return df
            
        print("Computing LOB microstructure features...")
        
        # 1. Weighted Average Price (WAP)
        # WAP = (BidPrice * AskQty + AskPrice * BidQty) / (BidQty + AskQty)
        # Using level 1
        df['wap1'] = (df['bid1_p'] * df['ask1_q'] + df['ask1_p'] * df['bid1_q']) / (df['bid1_q'] + df['ask1_q'])
        
        # 2. Price Spread
        df['spread'] = df['ask1_p'] - df['bid1_p']
        df['mid_price'] = (df['ask1_p'] + df['bid1_p']) / 2
        
        # 3. Volume Imbalance
        # (BidQty - AskQty) / (BidQty + AskQty)
        # Often called Order Book Imbalance (OBI)
        df['volume_imbalance'] = (df['bid1_q'] - df['ask1_q']) / (df['bid1_q'] + df['ask1_q'])
        
        # 4. Total Depth (Level 1-5)
        # Sum of quantities for top 5 levels
        bid_q_cols = [c for c in df.columns if 'bid' in c and '_q' in c]
        ask_q_cols = [c for c in df.columns if 'ask' in c and '_q' in c]
        
        df['total_bid_depth'] = df[bid_q_cols].sum(axis=1)
        df['total_ask_depth'] = df[ask_q_cols].sum(axis=1)
        df['total_depth'] = df['total_bid_depth'] + df['total_ask_depth']
        df['depth_imbalance'] = (df['total_bid_depth'] - df['total_ask_depth']) / df['total_depth']
        
        # 5. Micro-price (weighted by volume imbalance)
        # P_micro = P_mid + Imbalance * (Spread / 2)
        df['micro_price'] = df['mid_price'] + df['volume_imbalance'] * (df['spread'] / 2)
        
        # Fill potential NaNs (e.g. zero volume)
        df = df.fillna(method='ffill').fillna(0)
        
        return df

    def compute_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute moving average features"""
        df = df.copy()
        
        print("Computing moving averages...")
        
        windows = [5, 10, 20, 50, 100, 200]
        
        for window in windows:
            # Price MAs
            df[f'close_ma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'close_std_{window}'] = df['close'].rolling(window=window).std()
            
            # Volume MAs
            df[f'volume_ma_{window}'] = df['volume'].rolling(window=window).mean()
            
            # Price relative to MA
            df[f'close_to_ma_{window}'] = (df['close'] - df[f'close_ma_{window}']) / df[f'close_ma_{window}']
        
        # Exponential moving averages
        ema_windows = [12, 26, 50]
        for window in ema_windows:
            df[f'close_ema_{window}'] = df['close'].ewm(span=window).mean()
        
        return df
    
    def compute_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute momentum indicators"""
        df = df.copy()
        
        print("Computing momentum indicators...")
        
        # RSI
        def compute_rsi(data, window=14):
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        df['rsi_14'] = compute_rsi(df['close'], 14)
        df['rsi_28'] = compute_rsi(df['close'], 28)
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Stochastic Oscillator
        def compute_stochastic(high, low, close, window=14):
            lowest_low = low.rolling(window=window).min()
            highest_high = high.rolling(window=window).max()
            stoch = 100 * (close - lowest_low) / (highest_high - lowest_low)
            return stoch
        
        df['stoch_k'] = compute_stochastic(df['high'], df['low'], df['close'], 14)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # Rate of Change
        df['roc_10'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
        
        # Williams %R
        def compute_williams_r(high, low, close, window=14):
            highest_high = high.rolling(window=window).max()
            lowest_low = low.rolling(window=window).min()
            wr = -100 * (highest_high - close) / (highest_high - lowest_low)
            return wr
        
        df['williams_r'] = compute_williams_r(df['high'], df['low'], df['close'], 14)
        
        return df
    
    def compute_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volatility features"""
        df = df.copy()
        
        print("Computing volatility features...")
        
        windows = [10, 20, 50]
        
        for window in windows:
            # Historical volatility
            df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
            
            # Parkinson volatility (uses high-low)
            df[f'parkinson_vol_{window}'] = np.sqrt(
                (1 / (4 * np.log(2))) * 
                (np.log(df['high'] / df['low']) ** 2).rolling(window=window).mean()
            )
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR (Average True Range)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr_14'] = df['tr'].rolling(window=14).mean()
        
        return df
    
    def compute_macrohft_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute MacroHFT-specific features (slope_360, vol_360)
        These are critical for the hierarchical agent system
        """
        df = df.copy()
        
        print("Computing MacroHFT-specific features...")
        
        window_size = 360
        
        # Slope feature (trend detection)
        df['slope_360'] = df['close'].rolling(window=window_size).apply(
            self.get_slope_window, raw=False
        )
        
        # Volatility feature
        df['return'] = df['close'].pct_change().fillna(0)
        df['vol_360'] = df['return'].rolling(window=window_size).std()
        
        print(f"Computed slope_360 and vol_360 features")
        
        return df
    
    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features needed for MacroHFT
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all computed features
        """
        print("\n=== Computing All Features ===")
        
        df = self.compute_basic_features(df)
        
        # Add LOB features if columns exist
        df = self.compute_lob_features(df)
        
        df = self.compute_moving_averages(df)
        df = self.compute_momentum_indicators(df)
        df = self.compute_volatility_features(df)
        df = self.compute_macrohft_specific_features(df)
        
        # Drop rows with NaN (due to rolling windows)
        initial_rows = len(df)
        df = df.dropna()
        final_rows = len(df)
        
        print(f"\nDropped {initial_rows - final_rows} rows with NaN values")
        print(f"Final dataset: {final_rows} rows, {len(df.columns)} features")
        
        return df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = "BTCUSDT_1m_processed.feather"):
        """Save processed data with features"""
        processed_dir = self.config['data']['processed_dir']
        os.makedirs(processed_dir, exist_ok=True)
        
        filepath = os.path.join(processed_dir, filename)
        
        # Reset index to save timestamp
        df_to_save = df.reset_index()
        df_to_save.to_feather(filepath)
        
        print(f"\nProcessed data saved to: {filepath}")
        print(f"File size: {os.path.getsize(filepath) / (1024*1024):.2f} MB")
        
        return filepath
    
    def load_processed_data(self, filename: str = "BTCUSDT_1m_processed.feather") -> pd.DataFrame:
        """Load processed data"""
        processed_dir = self.config['data']['processed_dir']
        filepath = os.path.join(processed_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Processed data not found: {filepath}")
        
        df = pd.read_feather(filepath)
        
        # Set timestamp as index if it exists
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
        elif 'index' in df.columns:
            df.set_index('index', inplace=True)
        
        print(f"Loaded processed data from: {filepath}")
        print(f"Shape: {df.shape}")
        
        return df


def main():
    """Main execution"""
    import argparse
    from data_fetcher import BinanceDataFetcher
    
    parser = argparse.ArgumentParser(description='Engineer features for Bitcoin backtesting')
    parser.add_argument('--config', type=str, default='./backtest/config/backtest_config.yaml',
                       help='Path to config file')
    parser.add_argument('--input', type=str, default=None,
                       help='Input raw data file')
    parser.add_argument('--output', type=str, default='BTCUSDT_1m_processed.feather',
                       help='Output processed data file')
    
    args = parser.parse_args()
    
    # Initialize feature engineer
    engineer = FeatureEngineer(config_path=args.config)
    
    # Load raw data
    if args.input:
        print(f"Loading data from {args.input}")
        df = pd.read_feather(args.input)
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
    else:
        # Load from default location
        # Try to load merged LOB data first, else fallback to raw
        lob_path = "./MacroHFT/data/BTCUSDT/whole/df_whole.feather"
        if os.path.exists(lob_path):
            print(f"Loading merged LOB data from {lob_path}")
            df = pd.read_feather(lob_path)
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
        else:
            print("LOB data not found. Loading standard OHLCV data...")
            fetcher = BinanceDataFetcher(config_path=args.config)
            df = fetcher.load_data()
    
    # Compute features
    df_processed = engineer.compute_all_features(df)
    
    # Save processed data
    engineer.save_processed_data(df_processed, args.output)
    
    print("\n=== Feature Engineering Complete ===")
    print(f"Total features: {len(df_processed.columns)}")


if __name__ == "__main__":
    main()
