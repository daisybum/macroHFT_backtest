"""
MacroHFT Strategy Implementation for Bitcoin Backtesting
Loads pre-trained models and generates trading signals
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import yaml
from typing import Dict, Tuple, Optional

# Add MacroHFT to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from MacroHFT.model.net import subagent, hyperagent
from MacroHFT.RL.util.memory import episodicmemory
from MacroHFT.tools.demonstration import make_q_table_reward


class MacroHFTStrategy:
    """
    MacroHFT hierarchical trading strategy
    Uses 6 sub-agents and hyperagent with episodic memory
    """
    
    def __init__(self, config_path: str = "./backtest/config/backtest_config.yaml"):
        """Initialize MacroHFT strategy"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Device configuration
        device_str = self.config['model']['device']
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load feature lists
        self._load_feature_lists()
        
        # Model dimensions
        self.n_action = 2  # Binary: 0=no position, 1=full position
        self.n_state_1 = len(self.tech_indicator_list)
        self.n_state_2 = len(self.tech_indicator_list_trend)
        
        # Initialize models
        self._initialize_models()
        
        # Initialize episodic memory if enabled
        if self.config['episodic_memory']['enabled']:
            self._initialize_memory()
        else:
            self.memory = None
        
        # Demonstration Q-table (will be computed if needed)
        self.q_table = None
        
        print("MacroHFT Strategy initialized successfully")
    
    def _load_feature_lists(self):
        """Load feature lists from MacroHFT"""
        feature_list_path = self.config['model']['feature_list_path']
        
        try:
            self.tech_indicator_list = np.load(
                os.path.join(feature_list_path, 'single_features.npy'),
                allow_pickle=True
            ).tolist()
        except:
            print("Warning: Could not load single_features.npy, using defaults")
            self.tech_indicator_list = ['close', 'volume', 'high', 'low', 'open']
        
        try:
            self.tech_indicator_list_trend = np.load(
                os.path.join(feature_list_path, 'trend_features.npy'),
                allow_pickle=True
            ).tolist()
        except:
            print("Warning: Could not load trend_features.npy, using defaults")
            self.tech_indicator_list_trend = ['close_ma_20', 'volume_ma_20']
        
        self.clf_list = ['slope_360', 'vol_360']
        
        print(f"Loaded {len(self.tech_indicator_list)} single features")
        print(f"Loaded {len(self.tech_indicator_list_trend)} trend features")
    
    def _initialize_models(self):
        """Initialize all sub-agents and hyperagent"""
        print("\nInitializing models...")
        
        hidden_dim = 64
        
        # Initialize 6 sub-agents
        self.slope_1 = subagent(self.n_state_1, self.n_state_2, self.n_action, hidden_dim).to(self.device)
        self.slope_2 = subagent(self.n_state_1, self.n_state_2, self.n_action, hidden_dim).to(self.device)
        self.slope_3 = subagent(self.n_state_1, self.n_state_2, self.n_action, hidden_dim).to(self.device)
        self.vol_1 = subagent(self.n_state_1, self.n_state_2, self.n_action, hidden_dim).to(self.device)
        self.vol_2 = subagent(self.n_state_1, self.n_state_2, self.n_action, hidden_dim).to(self.device)
        self.vol_3 = subagent(self.n_state_1, self.n_state_2, self.n_action, hidden_dim).to(self.device)
        
        # Load pre-trained weights from ETHUSDT models
        model_paths = {
            'slope_1': self.config['model']['slope_models'][0],
            'slope_2': self.config['model']['slope_models'][1],
            'slope_3': self.config['model']['slope_models'][2],
            'vol_1': self.config['model']['vol_models'][0],
            'vol_2': self.config['model']['vol_models'][1],
            'vol_3': self.config['model']['vol_models'][2],
        }
        
        try:
            self.slope_1.load_state_dict(torch.load(model_paths['slope_1'], map_location=self.device))
            self.slope_2.load_state_dict(torch.load(model_paths['slope_2'], map_location=self.device))
            self.slope_3.load_state_dict(torch.load(model_paths['slope_3'], map_location=self.device))
            self.vol_1.load_state_dict(torch.load(model_paths['vol_1'], map_location=self.device))
            self.vol_2.load_state_dict(torch.load(model_paths['vol_2'], map_location=self.device))
            self.vol_3.load_state_dict(torch.load(model_paths['vol_3'], map_location=self.device))
            print("[OK] Loaded all 6 sub-agent models")
        except Exception as e:
            print(f"Warning: Could not load sub-agent models: {e}")
            print("Models will use random initialization")
        
        # Set sub-agents to eval mode
        self.slope_1.eval()
        self.slope_2.eval()
        self.slope_3.eval()
        self.vol_1.eval()
        self.vol_2.eval()
        self.vol_3.eval()
        
        # Create agent dictionaries for easy access
        self.slope_agents = {
            0: self.slope_1,
            1: self.slope_2,
            2: self.slope_3
        }
        self.vol_agents = {
            0: self.vol_1,
            1: self.vol_2,
            2: self.vol_3
        }
        
        # Initialize hyperagent
        hyperagent_hidden_dim = 32
        self.hyperagent = hyperagent(
            self.n_state_1, self.n_state_2, self.n_action, hyperagent_hidden_dim
        ).to(self.device)
        self.hyperagent.eval()
        
        print("[OK] Initialized hyperagent")
    
    def _initialize_memory(self):
        """Initialize episodic memory"""
        capacity = self.config['episodic_memory']['capacity']
        k = self.config['episodic_memory']['k_neighbors']
        
        self.memory = episodicmemory(
            capacity=capacity,
            k=k,
            state_dim=self.n_state_1,
            state_dim_2=self.n_state_2,
            hidden_dim=64,
            device=self.device
        )
        
        print(f"[OK] Initialized episodic memory (capacity={capacity}, k={k})")
    
    def compute_demonstration_qtable(self, df: pd.DataFrame):
        """
        Compute demonstration Q-table for the entire dataset
        This provides imitation learning guidance
        """
        if not self.config['demonstration']['compute']:
            print("Skipping demonstration Q-table computation (disabled in config)")
            return
        
        print("\nComputing demonstration Q-table...")
        print("This may take a while for large datasets...")
        
        try:
            self.q_table = make_q_table_reward(
                df=df,
                num_action=self.n_action,
                max_holding=self.config['trading']['max_holding_number'],
                reward_scale=self.config['demonstration']['reward_scale'],
                gamma=self.config['demonstration']['gamma'],
                commission_fee=self.config['demonstration']['commission_fee'],
                max_punish=self.config['demonstration']['max_punish']
            )
            print(f"[OK] Computed Q-table with shape: {self.q_table.shape}")
        except Exception as e:
            print(f"Warning: Could not compute Q-table: {e}")
            self.q_table = None
    
    def calculate_q(self, w: torch.Tensor, qs: list) -> torch.Tensor:
        """
        Calculate weighted Q-values from sub-agents
        
        Args:
            w: Weight tensor (batch, 6)
            qs: List of Q-tensors from 6 sub-agents
            
        Returns:
            Combined Q-tensor (batch, n_actions)
        """
        q_tensor = torch.stack(qs)  # (6, batch, n_actions)
        q_tensor = q_tensor.permute(1, 0, 2)  # (batch, 6, n_actions)
        weights_reshaped = w.view(-1, 1, 6)  # (batch, 1, 6)
        combined_q = torch.bmm(weights_reshaped, q_tensor).squeeze(1)  # (batch, n_actions)
        
        return combined_q
    
    def get_action(self, state: np.ndarray, state_trend: np.ndarray, 
                   state_clf: np.ndarray, previous_action: int) -> int:
        """
        Get trading action from MacroHFT strategy
        
        Args:
            state: Single state features
            state_trend: Trend state features
            state_clf: Classification features [slope_360, vol_360]
            previous_action: Previous action taken
            
        Returns:
            Action (0 or 1)
        """
        with torch.no_grad():
            # Convert to tensors
            x1 = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            x2 = torch.FloatTensor(state_trend).unsqueeze(0).to(self.device)
            x3 = torch.FloatTensor(state_clf).unsqueeze(0).to(self.device)
            prev_a = torch.LongTensor([previous_action]).to(self.device)
            
            # Get Q-values from all sub-agents
            qs = [
                self.slope_agents[0](x1, x2, prev_a),
                self.slope_agents[1](x1, x2, prev_a),
                self.slope_agents[2](x1, x2, prev_a),
                self.vol_agents[0](x1, x2, prev_a),
                self.vol_agents[1](x1, x2, prev_a),
                self.vol_agents[2](x1, x2, prev_a)
            ]
            
            # Get weights from hyperagent
            w = self.hyperagent(x1, x2, x3, prev_a)
            
            # Calculate combined Q-values
            q_values = self.calculate_q(w, qs)
            
            # Select action with highest Q-value
            action = torch.argmax(q_values, dim=1).item()
            
        return action
    
    def get_signal(self, row: pd.Series, previous_action: int) -> Dict:
        """
        Generate trading signal for a single timestep
        
        Args:
            row: DataFrame row with all features
            previous_action: Previous action
            
        Returns:
            Dictionary with action and metadata
        """
        # Extract state features
        try:
            state = row[self.tech_indicator_list].values.astype(np.float32)
            state_trend = row[self.tech_indicator_list_trend].values.astype(np.float32)
            state_clf = row[self.clf_list].values.astype(np.float32)
        except KeyError as e:
            print(f"Missing features: {e}")
            # Return hold signal if features missing
            return {
                'action': previous_action,
                'signal': 'HOLD',
                'error': str(e)
            }
        
        # Check for NaN values
        if np.any(np.isnan(state)) or np.any(np.isnan(state_trend)) or np.any(np.isnan(state_clf)):
            return {
                'action': previous_action,
                'signal': 'HOLD',
                'error': 'NaN in features'
            }
        
        # Get action from strategy
        action = self.get_action(state, state_trend, state_clf, previous_action)
        
        # Translate to signal
        if action == 1 and previous_action == 0:
            signal = 'BUY'
        elif action == 0 and previous_action == 1:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        return {
            'action': action,
            'signal': signal,
            'previous_action': previous_action
        }
    
    def generate_signals(self, df: pd.DataFrame, initial_action: int = 0) -> pd.DataFrame:
        """
        Generate trading signals for entire dataset
        
        Args:
            df: DataFrame with all features
            initial_action: Starting position
            
        Returns:
            DataFrame with added signal columns
        """
        print("\n=== Generating Trading Signals ===")
        
        df = df.copy()
        
        # Initialize signal columns
        df['action'] = 0
        df['signal'] = 'HOLD'
        
        previous_action = initial_action
        
        # Compute demonstration Q-table if configured
        if self.config['demonstration']['compute'] and self.q_table is None:
            self.compute_demonstration_qtable(df)
        
        # Generate signals row by row
        print(f"Processing {len(df)} timesteps...")
        
        for idx in range(len(df)):
            if idx % 10000 == 0:
                print(f"Progress: {idx}/{len(df)} ({100*idx/len(df):.1f}%)")
            
            row = df.iloc[idx]
            signal_dict = self.get_signal(row, previous_action)
            
            df.loc[df.index[idx], 'action'] = signal_dict['action']
            df.loc[df.index[idx], 'signal'] = signal_dict['signal']
            
            previous_action = signal_dict['action']
        
        # Signal statistics
        signal_counts = df['signal'].value_counts()
        print("\n=== Signal Statistics ===")
        print(signal_counts)
        print(f"\nBuy signals: {signal_counts.get('BUY', 0)}")
        print(f"Sell signals: {signal_counts.get('SELL', 0)}")
        print(f"Hold signals: {signal_counts.get('HOLD', 0)}")
        
        return df


def main():
    """Main execution for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate MacroHFT trading signals')
    parser.add_argument('--config', type=str, default='./backtest/config/backtest_config.yaml',
                       help='Path to config file')
    parser.add_argument('--input', type=str, required=True,
                       help='Input processed data file')
    parser.add_argument('--output', type=str, default='BTCUSDT_with_signals.feather',
                       help='Output file with signals')
    
    args = parser.parse_args()
    
    # Initialize strategy
    strategy = MacroHFTStrategy(config_path=args.config)
    
    # Load processed data
    df = pd.read_feather(args.input)
    if 'timestamp' in df.columns:
        df.set_index('timestamp', inplace=True)
    
    # Generate signals
    df_with_signals = strategy.generate_signals(df)
    
    # Save results
    output_dir = os.path.dirname(args.output) or './backtest/data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    df_with_signals.reset_index().to_feather(args.output)
    print(f"\nSignals saved to: {args.output}")


if __name__ == "__main__":
    main()

