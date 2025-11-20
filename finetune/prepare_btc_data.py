"""
Bitcoin 데이터를 MacroHFT 형식으로 준비
OHLCV 데이터만 사용 (orderbook 데이터 없음)
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def prepare_btc_data():
    """
    백테스트용 BTC 데이터를 MacroHFT 학습 형식으로 변환
    """
    print("="*70)
    print("Bitcoin 데이터 준비 (MacroHFT 형식)")
    print("="*70)
    
    # 데이터 로드
    print("\n[1/5] 데이터 로딩...")
    raw_data_path = "./backtest/data/raw/BTCUSDT_1m_raw.feather"
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"데이터 파일이 없습니다: {raw_data_path}")
    
    df = pd.read_feather(raw_data_path)
    print(f"  총 {len(df):,}개 캔들 로드")
    print(f"  기간: {pd.to_datetime(df['timestamp'].iloc[0], unit='ms')} ~ "
          f"{pd.to_datetime(df['timestamp'].iloc[-1], unit='ms')}")
    
    # MacroHFT에 필요한 컬럼만 선택
    print("\n[2/5] 데이터 포맷 변환...")
    df_macrohft = pd.DataFrame({
        'timestamp': pd.to_datetime(df['timestamp'], unit='ms'),
        'open': df['open'],
        'high': df['high'],
        'low': df['low'],
        'close': df['close'],
        'volume': df['volume']
    })
    
    # Train/Val/Test 분할 (70%/15%/15%)
    print("\n[3/5] Train/Val/Test 분할...")
    n = len(df_macrohft)
    train_size = int(n * 0.70)
    val_size = int(n * 0.15)
    
    df_train = df_macrohft[:train_size].reset_index(drop=True)
    df_val = df_macrohft[train_size:train_size + val_size].reset_index(drop=True)
    df_test = df_macrohft[train_size + val_size:].reset_index(drop=True)
    
    print(f"  Train: {len(df_train):,} 캔들")
    print(f"  Val:   {len(df_val):,} 캔들")
    print(f"  Test:  {len(df_test):,} 캔들")
    
    # 디렉토리 생성
    print("\n[4/5] 디렉토리 생성...")
    base_path = Path("./MacroHFT/data/BTCUSDT")
    for split in ['train', 'val', 'test', 'whole']:
        (base_path / split).mkdir(parents=True, exist_ok=True)
    
    # 데이터 저장
    print("\n[5/5] 데이터 저장...")
    df_train.to_feather(base_path / "df_train.feather")
    df_val.to_feather(base_path / "df_val.feather")
    df_test.to_feather(base_path / "df_test.feather")
    
    print(f"\n[OK] 데이터 준비 완료!")
    print(f"  저장 위치: {base_path}")
    print(f"  파일: df_train.feather, df_val.feather, df_test.feather")
    
    return df_train, df_val, df_test

if __name__ == "__main__":
    df_train, df_val, df_test = prepare_btc_data()
    
    print("\n" + "="*70)
    print("데이터 샘플:")
    print("="*70)
    print(df_train.head())
    print("\n" + "="*70)

