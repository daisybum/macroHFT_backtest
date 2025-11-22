import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
RAW_DATA_DIR = "./backtest/data/raw"
LOB_DATA_DIR = "./backtest/data/processed_lob"
OUTPUT_BASE_DIR = "./MacroHFT/data/BTCUSDT"

def load_klines():
    print("Loading Klines...")
    zips = glob.glob(os.path.join(RAW_DATA_DIR, "*1m*.zip"))
    zips.sort()
    
    dfs = []
    for zip_path in zips:
        # Skip depthUpdate files if glob was too loose (it shouldn't be with *1m*)
        if "depthUpdate" in zip_path: 
            continue
            
        print(f"  Reading {os.path.basename(zip_path)}...")
        # Binance Kline CSV has no header
        df = pd.read_csv(
            zip_path, 
            compression='zip', 
            header=None,
            usecols=[0, 1, 2, 3, 4, 5],
            names=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        dfs.append(df)
        
    if not dfs:
        print("No Kline data found!")
        return None
        
    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.sort_values('timestamp').reset_index(drop=True)
    
    # Ensure timestamp is int/datetime consistent
    # Convert to datetime for merging if needed, or keep as ms int
    # MacroHFT usually expects timestamp as datetime or specific format.
    # In prepare_btc_data.py it used pd.to_datetime(unit='ms')
    
    return full_df

def load_lob():
    print("Loading LOB Snapshots...")
    files = glob.glob(os.path.join(LOB_DATA_DIR, "lob_snapshot_*.feather"))
    files.sort()
    
    if not files:
        print("No LOB data found!")
        return None
        
    dfs = []
    for f in files:
        print(f"  Reading {os.path.basename(f)}...")
        df = pd.read_feather(f)
        dfs.append(df)
        
    full_lob = pd.concat(dfs, ignore_index=True)
    full_lob = full_lob.sort_values('timestamp').reset_index(drop=True)
    return full_lob

def main():
    # 1. Load Data
    df_klines = load_klines()
    df_lob = load_lob()
    
    if df_klines is None:
        return

    # 2. Merge
    print("\nMerging Data...")
    # Ensure timestamps are same type (int64 usually for ms)
    df_klines['timestamp'] = df_klines['timestamp'].astype(np.int64)
    
    if df_lob is not None:
        df_lob['timestamp'] = df_lob['timestamp'].astype(np.int64)
        
        # Merge on timestamp (Left join to keep all candles)
        df_merged = pd.merge_asof(
            df_klines, 
            df_lob, 
            on='timestamp', 
            direction='backward', # Use latest available snapshot
            tolerance=60000 # 1 minute tolerance
        )
        
        # Forward fill missing LOB data just in case
        lob_cols = [c for c in df_merged.columns if 'bid' in c or 'ask' in c]
        df_merged[lob_cols] = df_merged[lob_cols].ffill().fillna(0)
        
    else:
        print("Warning: LOB data missing. Filling with zeros (NOT RECOMMENDED for MacroHFT).")
        df_merged = df_klines
        # Add dummy columns
        for i in range(1, 6):
            df_merged[f'bid{i}_p'] = 0.0
            df_merged[f'bid{i}_q'] = 0.0
            df_merged[f'ask{i}_p'] = 0.0
            df_merged[f'ask{i}_q'] = 0.0
            
    # 3. Convert Timestamp to Datetime for MacroHFT
    df_merged['timestamp'] = pd.to_datetime(df_merged['timestamp'], unit='ms')
    
    # 4. Split Train/Val/Test
    print("\nSplitting Train/Val/Test (70/15/15)...")
    n = len(df_merged)
    train_size = int(n * 0.70)
    val_size = int(n * 0.15)
    
    df_train = df_merged.iloc[:train_size].reset_index(drop=True)
    df_val = df_merged.iloc[train_size:train_size+val_size].reset_index(drop=True)
    df_test = df_merged.iloc[train_size+val_size:].reset_index(drop=True)
    
    print(f"  Total: {n}")
    print(f"  Train: {len(df_train)}")
    print(f"  Val:   {len(df_val)}")
    print(f"  Test:  {len(df_test)}")
    
    # 5. Save
    print("\nSaving to MacroHFT/data/BTCUSDT...")
    base_path = Path(OUTPUT_BASE_DIR)
    
    for split in ['train', 'val', 'test']:
        (base_path / split).mkdir(parents=True, exist_ok=True)
        
    df_train.to_feather(base_path / "train" / "df_train.feather")
    df_val.to_feather(base_path / "val" / "df_val.feather")
    df_test.to_feather(base_path / "test" / "df_test.feather")
    
    # Also save whole for reference
    (base_path / "whole").mkdir(parents=True, exist_ok=True)
    df_merged.to_feather(base_path / "whole" / "df_whole.feather")
    
    print("[OK] Data Preparation Complete!")

if __name__ == "__main__":
    main()

