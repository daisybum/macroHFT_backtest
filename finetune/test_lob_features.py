import pandas as pd
import numpy as np
import sys
import os

# Add src path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../backtest/src'))

from feature_engineering import FeatureEngineer

def test_lob_features():
    print("Testing LOB Feature Engineering...")
    
    # Create dummy dataframe with LOB columns
    dates = pd.date_range(start='2024-01-01', periods=10, freq='1min')
    
    data = {
        'timestamp': dates,
        'open': [50000 + i*10 for i in range(10)],
        'high': [50050 + i*10 for i in range(10)],
        'low': [49950 + i*10 for i in range(10)],
        'close': [50020 + i*10 for i in range(10)],
        'volume': [100 + i for i in range(10)],
        # LOB Data (Level 1)
        'bid1_p': [49990 + i*10 for i in range(10)],
        'bid1_q': [1.5 + i*0.1 for i in range(10)],
        'ask1_p': [50010 + i*10 for i in range(10)],
        'ask1_q': [2.0 + i*0.1 for i in range(10)],
        # Level 2-5 (Simplified)
    }
    
    # Add levels 2-5
    for lvl in range(2, 6):
        data[f'bid{lvl}_p'] = [49990 - lvl + i*10 for i in range(10)]
        data[f'bid{lvl}_q'] = [1.0 for i in range(10)]
        data[f'ask{lvl}_p'] = [50010 + lvl + i*10 for i in range(10)]
        data[f'ask{lvl}_q'] = [1.0 for i in range(10)]
        
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    print("Input Columns:", df.columns.tolist())
    
    # Initialize Engineer
    engineer = FeatureEngineer()
    
    # Compute Features
    # Note: compute_all_features calls compute_lob_features
    try:
        df_processed = engineer.compute_lob_features(df)
        
        expected_cols = ['wap1', 'spread', 'volume_imbalance', 'micro_price', 'total_depth']
        missing = [c for c in expected_cols if c not in df_processed.columns]
        
        if missing:
            print(f"[FAIL] Missing LOB features: {missing}")
        else:
            print(f"[PASS] All LOB features computed.")
            print("Sample Output:")
            print(df_processed[['wap1', 'spread', 'micro_price']].head())
            
    except Exception as e:
        print(f"[ERROR] Feature computation failed: {e}")
        raise e

if __name__ == "__main__":
    test_lob_features()

