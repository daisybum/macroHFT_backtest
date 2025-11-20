"""
Bitcoin 데이터 분해 및 라벨링
MacroHFT preprocess/decomposition.py를 BTCUSDT에 맞게 수정
"""

import numpy as np
import pandas as pd
import os
import pickle
from scipy.signal import butter, filtfilt
from sklearn.linear_model import LinearRegression
from pathlib import Path

def smooth_data(data):
    """Butterworth 필터로 데이터 스무딩"""
    N, Wn = 1, 0.05
    b, a = butter(N, Wn, btype='low')
    return filtfilt(b, a, data)

def get_slope(smoothed_data):
    """선형 회귀로 slope 계산"""
    X = np.arange(len(smoothed_data)).reshape(-1, 1)
    model = LinearRegression().fit(X, smoothed_data)
    return model.coef_[0]

def get_slope_window(window):
    """윈도우 단위로 slope 계산"""
    N, Wn = 1, 0.05
    b, a = butter(N, Wn, btype='low')
    y = filtfilt(b, a, window.values)
    X = np.arange(len(y)).reshape(-1, 1)   
    model = LinearRegression().fit(X, y)
    return model.coef_[0]

def chunk(df_train, df_val, df_test, dataset='BTCUSDT'):
    """데이터를 chunk_size 단위로 분할하여 저장"""
    chunk_size = 4320  # MacroHFT 기본값 (3일치 1분 데이터)
    base_path = Path(f'./MacroHFT/data/{dataset}')
    
    print(f"\n  Chunking 데이터 (chunk_size={chunk_size})...")
    
    # Train chunks
    n_train_chunks = int(len(df_train) / chunk_size)
    for i in range(n_train_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        df_chunk = df_train[start:end].reset_index(drop=True)
        df_chunk.to_feather(base_path / 'train' / f'df_{i}.feather')
    print(f"    Train: {n_train_chunks} chunks")

    # Val chunks
    n_val_chunks = int(len(df_val) / chunk_size)
    for i in range(n_val_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        df_chunk = df_val[start:end].reset_index(drop=True)
        df_chunk.to_feather(base_path / 'val' / f'df_{i}.feather')
    print(f"    Val:   {n_val_chunks} chunks")

    # Test chunks
    n_test_chunks = int(len(df_test) / chunk_size)
    for i in range(n_test_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        df_chunk = df_test[start:end].reset_index(drop=True)
        df_chunk.to_feather(base_path / 'test' / f'df_{i}.feather')
    print(f"    Test:  {n_test_chunks} chunks")

def label_slope(df_train, df_val, df_test, dataset='BTCUSDT'):
    """Slope 기반으로 데이터 라벨링"""
    chunk_size = 4320
    base_path = Path(f'./MacroHFT/data/{dataset}')
    
    print(f"\n  Slope 라벨링...")
    
    # Train slopes
    slopes_train = []
    for i in range(0, int(len(df_train) / chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk = df_train['close'][start:end].values
        smoothed_chunk = smooth_data(chunk)
        slope = get_slope(smoothed_chunk)
        slopes_train.append(slope)

    # Val slopes
    slopes_val = []
    for i in range(0, int(len(df_val) / chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk = df_val['close'][start:end].values
        smoothed_chunk = smooth_data(chunk)
        slope = get_slope(smoothed_chunk)
        slopes_val.append(slope)

    # Test slopes
    slopes_test = []
    for i in range(0, int(len(df_test) / chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk = df_test['close'][start:end].values
        smoothed_chunk = smooth_data(chunk)
        slope = get_slope(smoothed_chunk)
        slopes_test.append(slope)

    # Quantile-based labeling (5 labels: 0, 1, 2, 3, 4)
    quantiles = [0, 0.05, 0.35, 0.65, 0.95, 1]
    slope_labels_train, bins = pd.qcut(slopes_train, q=quantiles, retbins=True, labels=False)

    # Train indices
    train_indices = [[] for _ in range(5)]
    for index, label in enumerate(slope_labels_train):
        train_indices[label].append(index)
    with open(base_path / 'train' / 'slope_labels.pkl', 'wb') as file:
        pickle.dump(train_indices, file)

    # Val/Test labels using train bins
    bins[0] = -100
    bins[-1] = 100
    slope_labels_val = pd.cut(slopes_val, bins=bins, labels=False, include_lowest=True)
    slope_labels_val = [1 if element == 0 else element for element in slope_labels_val]
    slope_labels_val = [3 if element == 4 else element for element in slope_labels_val]
    slope_labels_test = pd.cut(slopes_test, bins=bins, labels=False, include_lowest=True)
    slope_labels_test = [1 if element == 0 else element for element in slope_labels_test]
    slope_labels_test = [3 if element == 4 else element for element in slope_labels_test]

    # Val indices
    val_indices = [[] for _ in range(5)]
    for index, label in enumerate(slope_labels_val):
        val_indices[label].append(index)
    with open(base_path / 'val' / 'slope_labels.pkl', 'wb') as file:
        pickle.dump(val_indices, file)
    
    # Test indices
    test_indices = [[] for _ in range(5)]
    for index, label in enumerate(slope_labels_test):
        test_indices[label].append(index)
    with open(base_path / 'test' / 'slope_labels.pkl', 'wb') as file:
        pickle.dump(test_indices, file)
    
    print(f"    Train labels: {[len(x) for x in train_indices]}")
    print(f"    Val labels:   {[len(x) for x in val_indices]}")
    print(f"    Test labels:  {[len(x) for x in test_indices]}")

def label_volatility(df_train, df_val, df_test, dataset='BTCUSDT'):
    """Volatility 기반으로 데이터 라벨링"""
    chunk_size = 4320
    base_path = Path(f'./MacroHFT/data/{dataset}')
    
    print(f"\n  Volatility 라벨링...")
    
    # Train volatilities
    volatilities_train = []
    for i in range(0, int(len(df_train) / chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk = df_train[start:end].copy()
        chunk['return'] = chunk['close'].pct_change().fillna(0)
        volatility = chunk['return'].std()
        volatilities_train.append(volatility)

    # Val volatilities
    volatilities_val = []
    for i in range(0, int(len(df_val) / chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk = df_val[start:end].copy()
        chunk['return'] = chunk['close'].pct_change().fillna(0)
        volatility = chunk['return'].std()
        volatilities_val.append(volatility)
    
    # Test volatilities
    volatilities_test = []
    for i in range(0, int(len(df_test) / chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk = df_test[start:end].copy()
        chunk['return'] = chunk['close'].pct_change().fillna(0)
        volatility = chunk['return'].std()
        volatilities_test.append(volatility)

    # Quantile-based labeling
    quantiles = [0, 0.05, 0.35, 0.65, 0.95, 1]
    vol_labels_train, bins = pd.qcut(volatilities_train, q=quantiles, retbins=True, labels=False)

    # Train indices
    train_indices = [[] for _ in range(5)]
    for index, label in enumerate(vol_labels_train):
        train_indices[label].append(index)
    with open(base_path / 'train' / 'vol_labels.pkl', 'wb') as file:
        pickle.dump(train_indices, file)

    # Val/Test labels using train bins
    bins[0] = 0
    bins[-1] = 1
    vol_labels_val = pd.cut(volatilities_val, bins=bins, labels=False, include_lowest=True)
    vol_labels_val = [1 if element == 0 else element for element in vol_labels_val]
    vol_labels_val = [3 if element == 4 else element for element in vol_labels_val]
    vol_labels_test = pd.cut(volatilities_test, bins=bins, labels=False, include_lowest=True)
    vol_labels_test = [1 if element == 0 else element for element in vol_labels_test]
    vol_labels_test = [3 if element == 4 else element for element in vol_labels_test]

    # Val indices
    val_indices = [[] for _ in range(5)]
    for index, label in enumerate(vol_labels_val):
        val_indices[label].append(index)
    with open(base_path / 'val' / 'vol_labels.pkl', 'wb') as file:
        pickle.dump(val_indices, file)
    
    # Test indices
    test_indices = [[] for _ in range(5)]
    for index, label in enumerate(vol_labels_test):
        test_indices[label].append(index)
    with open(base_path / 'test' / 'vol_labels.pkl', 'wb') as file:
        pickle.dump(test_indices, file)
    
    print(f"    Train labels: {[len(x) for x in train_indices]}")
    print(f"    Val labels:   {[len(x) for x in val_indices]}")
    print(f"    Test labels:  {[len(x) for x in test_indices]}")

def label_whole(df):
    """전체 데이터에 slope_360, vol_360 feature 추가"""
    window_size_list = [360]
    for i in range(len(window_size_list)):
        window_size = window_size_list[i]
        df['slope_{}'.format(window_size)] = df['close'].rolling(window=window_size).apply(get_slope_window)
        df['return'] = df['close'].pct_change().fillna(0)
        df['vol_{}'.format(window_size)] = df['return'].rolling(window=window_size).std()
    return df

def decompose_btc():
    """Bitcoin 데이터 분해 메인 함수"""
    print("="*70)
    print("Bitcoin 데이터 분해 및 라벨링")
    print("="*70)
    
    dataset = 'BTCUSDT'
    base_path = Path(f'./MacroHFT/data/{dataset}')
    
    # 데이터 로드
    print("\n[1/4] 데이터 로딩...")
    df_train = pd.read_feather(base_path / 'df_train.feather')
    df_val = pd.read_feather(base_path / 'df_val.feather')
    df_test = pd.read_feather(base_path / 'df_test.feather')
    print(f"  Train: {len(df_train):,} rows")
    print(f"  Val:   {len(df_val):,} rows")
    print(f"  Test:  {len(df_test):,} rows")
    
    # Chunking
    print("\n[2/4] 데이터 Chunking...")
    chunk(df_train, df_val, df_test, dataset)
    
    # Slope labeling
    print("\n[3/4] Slope & Volatility 라벨링...")
    label_slope(df_train, df_val, df_test, dataset)
    label_volatility(df_train, df_val, df_test, dataset)
    
    # Whole dataset with features
    print("\n[4/4] 전체 데이터에 feature 추가...")
    df_train = label_whole(df_train).dropna().reset_index(drop=True).iloc[1:].reset_index(drop=True)
    df_val = label_whole(df_val).dropna().reset_index(drop=True).iloc[1:].reset_index(drop=True)
    df_test = label_whole(df_test).dropna().reset_index(drop=True).iloc[1:].reset_index(drop=True)
    
    df_train.to_feather(base_path / 'whole' / 'train.feather')
    df_val.to_feather(base_path / 'whole' / 'val.feather')
    df_test.to_feather(base_path / 'whole' / 'test.feather')
    
    print(f"  Train: {len(df_train):,} rows (after dropna)")
    print(f"  Val:   {len(df_val):,} rows (after dropna)")
    print(f"  Test:  {len(df_test):,} rows (after dropna)")
    
    print("\n[OK] 데이터 분해 완료!")
    print(f"  저장 위치: {base_path}")
    print("  생성된 파일:")
    print("    - train/val/test: df_*.feather (chunks)")
    print("    - train/val/test: slope_labels.pkl, vol_labels.pkl")
    print("    - whole: train.feather, val.feather, test.feather")
    
    return df_train, df_val, df_test

if __name__ == "__main__":
    df_train, df_val, df_test = decompose_btc()
    
    print("\n" + "="*70)
    print("데이터 샘플 (with features):")
    print("="*70)
    print(df_train.head())
    print("\n" + "="*70)

