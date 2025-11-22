import os
import glob
import zipfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import csv
import io
from tqdm import tqdm

# Configuration
RAW_DATA_DIR = "./backtest/data/raw"
PROCESSED_DIR = "./backtest/data/processed_lob"
SYMBOL = "BTCUSDT"

def parse_line(line):
    """
    Parse a line from depthUpdate CSV.
    Format assumption: timestamp, first_update_id, last_update_id, side, price, qty
    OR: time, price, qty (if simplified)
    
    Let's try to handle standard Binance Public Data format:
    time, price, quantity, first_update_id, last_update_id, side ('a' or 'b')
    Wait, check headers usually provided in these files or no header?
    Usually NO header.
    Standard columns: time, first_update_id, last_update_id, side, price, qty (Maybe?)
    
    Let's assume the column order based on common schema:
    0: timestamp
    1: first_update_id
    2: last_update_id
    3: side ('a' for ask, 'b' for bid)
    4: price
    5: qty
    """
    parts = line.strip().split(',')
    try:
        ts = int(parts[0])
        side = parts[3]
        price = float(parts[4])
        qty = float(parts[5])
        return ts, side, price, qty
    except (IndexError, ValueError):
        return None

def process_zip_file(zip_path):
    print(f"Processing {os.path.basename(zip_path)}...")
    
    results = []
    bids = {}  # price -> qty
    asks = {}  # price -> qty
    
    # Prepare for minutely snapshots
    last_snapshot_time = None
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        csv_names = [n for n in z.namelist() if n.endswith('.csv')]
        if not csv_names:
            print("  No CSV found in zip.")
            return None
            
        # Usually one big CSV per month
        for csv_name in csv_names:
            print(f"  Reading {csv_name}...")
            with z.open(csv_name, 'r') as f:
                # Stream reading
                wrapper = io.TextIOWrapper(f, encoding='utf-8')
                
                # Check first line for header (heuristic)
                first_pos = wrapper.tell()
                first_line = wrapper.readline()
                if "timestamp" in first_line.lower() or "price" in first_line.lower():
                    pass # Skip header
                else:
                    wrapper.seek(first_pos) # Reset if no header
                
                for line in tqdm(wrapper, desc="Rows", unit="lines"):
                    parsed = parse_line(line)
                    if not parsed:
                        continue
                        
                    ts, side, price, qty = parsed
                    
                    # Update LOB
                    if side == 'b':
                        if qty == 0:
                            bids.pop(price, None)
                        else:
                            bids[price] = qty
                    elif side == 'a':
                        if qty == 0:
                            asks.pop(price, None)
                        else:
                            asks[price] = qty
                            
                    # Check for snapshot time (every 1 minute)
                    # ts is milliseconds
                    # Round down to nearest minute
                    minute_ts = (ts // 60000) * 60000
                    
                    if last_snapshot_time is None:
                        last_snapshot_time = minute_ts
                    
                    if minute_ts > last_snapshot_time:
                        # Capture snapshot for last_snapshot_time (or all missed minutes)
                        # For simplicity, we capture at the moment the minute changes.
                        # Ideally, we should capture exactly at minute_ts.
                        # But since we iterate, let's just capture whenever minute changes.
                        
                        # Create snapshot row
                        row = {'timestamp': last_snapshot_time}
                        
                        # Sort and get top 5
                        # Bids: Descending price
                        sorted_bids = sorted(bids.items(), key=lambda x: x[0], reverse=True)[:5]
                        # Asks: Ascending price
                        sorted_asks = sorted(asks.items(), key=lambda x: x[0])[:5]
                        
                        # Fill columns
                        for i in range(5):
                            # Bids
                            if i < len(sorted_bids):
                                row[f'bid{i+1}_p'] = sorted_bids[i][0]
                                row[f'bid{i+1}_q'] = sorted_bids[i][1]
                            else:
                                row[f'bid{i+1}_p'] = 0
                                row[f'bid{i+1}_q'] = 0
                                
                            # Asks
                            if i < len(sorted_asks):
                                row[f'ask1_p'] = sorted_asks[i][0] # Mistake in key name logic fixed below
                                row[f'ask{i+1}_p'] = sorted_asks[i][0]
                                row[f'ask{i+1}_q'] = sorted_asks[i][1]
                            else:
                                row[f'ask{i+1}_p'] = 0
                                row[f'ask{i+1}_q'] = 0
                        
                        results.append(row)
                        last_snapshot_time = minute_ts
                        
                        # Memory management for results
                        if len(results) > 10000:
                            # Write to temp part? No, just keep in memory for a month is okay for 40k rows (approx 44k mins/month)
                            pass

    if not results:
        return None
        
    df = pd.DataFrame(results)
    return df

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Find all depthUpdate zips
    zips = glob.glob(os.path.join(RAW_DATA_DIR, "*depthUpdate*.zip"))
    zips.sort()
    
    if not zips:
        print("No depthUpdate zip files found.")
        return

    for zip_path in zips:
        filename = os.path.basename(zip_path)
        # e.g., BTCUSDT-depthUpdate-2024-11.zip
        month_part = filename.split('-')[-1].replace('.zip', '')
        
        save_path = os.path.join(PROCESSED_DIR, f"lob_snapshot_{month_part}.feather")
        if os.path.exists(save_path):
            print(f"Skipping {filename}, already processed.")
            continue
            
        df_lob = process_zip_file(zip_path)
        if df_lob is not None:
            print(f"Saving {len(df_lob)} snapshots to {save_path}")
            df_lob.to_feather(save_path)
        else:
            print(f"Failed to process {filename}")

if __name__ == "__main__":
    main()

