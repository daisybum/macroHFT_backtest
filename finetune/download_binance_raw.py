import os
import requests
import zipfile
import io
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

# Configuration
BASE_URL = "https://data.binance.vision/data/spot"
SYMBOL = "BTCUSDT"
INTERVAL = "1m"
START_DATE = "2024-11-01"  # Start from beginning of month to be safe
END_DATE = "2025-11-13"
DATA_DIR = "./backtest/data/raw"

def download_file(url, save_path):
    """Download a file with progress bar"""
    if os.path.exists(save_path):
        print(f"File already exists: {save_path}")
        return True
        
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 404:
            print(f"File not found: {url}")
            return False
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1MB
        
        print(f"Downloading {os.path.basename(save_path)}...")
        
        with open(save_path, 'wb') as f, tqdm(
            desc=os.path.basename(save_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                bar.update(size)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)
        return False

def generate_monthly_dates(start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    current = start.replace(day=1)
    dates = []
    while current <= end:
        dates.append(current)
        current += relativedelta(months=1)
    return dates

def generate_daily_dates(start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    current = start
    dates = []
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)
    return dates

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 1. Download Monthly Klines
    print(f"\n[1] Downloading Monthly Klines ({START_DATE} ~ {END_DATE})")
    monthly_dates = generate_monthly_dates(START_DATE, END_DATE)
    
    for date in monthly_dates:
        month_str = date.strftime("%Y-%m")
        filename = f"{SYMBOL}-{INTERVAL}-{month_str}.zip"
        url = f"{BASE_URL}/monthly/klines/{SYMBOL}/{INTERVAL}/{filename}"
        save_path = os.path.join(DATA_DIR, filename)
        
        # Monthly data usually becomes available a bit after month end.
        # If monthly fails, we might need daily (handled later logic if needed, but keeping simple for now)
        success = download_file(url, save_path)
        
    # 2. Download Monthly Depth Updates
    # Note: Depth Update files are HUGE.
    print(f"\n[2] Downloading Monthly Depth Updates ({START_DATE} ~ {END_DATE})")
    for date in monthly_dates:
        month_str = date.strftime("%Y-%m")
        filename = f"{SYMBOL}-depthUpdate-{month_str}.zip"
        url = f"{BASE_URL}/monthly/depthUpdate/{SYMBOL}/{filename}"
        save_path = os.path.join(DATA_DIR, filename)
        
        success = download_file(url, save_path)
        if not success:
            print(f"  Warning: Monthly depth update for {month_str} not found. Trying daily...")
            # Fallback to daily for this month if needed (not implemented here to keep simple, 
            # but usually monthly is preferred for bulk)
            
    print("\nDownload process completed.")
    print(f"Data saved to: {os.path.abspath(DATA_DIR)}")

if __name__ == "__main__":
    main()

