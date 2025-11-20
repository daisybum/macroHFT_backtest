"""
Backtesting Engine for MacroHFT Bitcoin Strategy
Event-driven backtester with transaction cost modeling
"""

import os
import sys
import numpy as np
import pandas as pd
import yaml
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class Trade:
    """Represents a single trade"""
    timestamp: datetime
    action: str  # 'BUY' or 'SELL'
    price: float
    quantity: float
    value: float
    commission: float
    slippage: float
    total_cost: float
    position_after: float
    cash_after: float
    portfolio_value: float


@dataclass
class PortfolioState:
    """Current portfolio state"""
    cash: float
    position: float  # BTC holdings
    position_value: float
    total_value: float
    returns: float
    trades: int


class MacroHFTBacktester:
    """
    Event-driven backtester for MacroHFT strategy
    """
    
    def __init__(self, config_path: str = "./backtest/config/backtest_config.yaml"):
        """Initialize backtester"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Trading parameters
        self.initial_capital = self.config['trading']['initial_capital']
        self.max_holding = self.config['trading']['max_holding_number']
        self.transaction_cost = self.config['trading']['transaction_cost']
        self.slippage = self.config['trading']['slippage']
        
        # Portfolio state
        self.cash = self.initial_capital
        self.position = 0.0  # BTC holdings
        self.position_value = 0.0
        self.total_value = self.initial_capital
        
        # Trade history
        self.trades: List[Trade] = []
        self.portfolio_history: List[PortfolioState] = []
        
        # Performance tracking
        self.equity_curve = []
        self.returns = []
        
        print(f"Backtester initialized with ${self.initial_capital:,.2f} capital")
    
    def calculate_position_size(self, action: int, current_price: float) -> float:
        """
        Calculate position size based on action
        
        Args:
            action: 0 (no position) or 1 (full position)
            current_price: Current BTC price
            
        Returns:
            Target position in BTC
        """
        if action == 0:
            return 0.0
        elif action == 1:
            # Use max_holding as position size
            return self.max_holding
        else:
            raise ValueError(f"Invalid action: {action}")
    
    def calculate_trade_cost(self, quantity: float, price: float, side: str) -> Tuple[float, float, float]:
        """
        Calculate total trade cost including commission and slippage
        
        Args:
            quantity: Trade quantity in BTC
            price: Trade price
            side: 'BUY' or 'SELL'
            
        Returns:
            (commission, slippage_cost, total_cost)
        """
        trade_value = quantity * price
        
        # Commission (percentage of trade value)
        commission = trade_value * self.transaction_cost
        
        # Slippage (assume adverse price movement)
        if side == 'BUY':
            slippage_cost = trade_value * self.slippage
        else:  # SELL
            slippage_cost = trade_value * self.slippage
        
        total_cost = commission + slippage_cost
        
        return commission, slippage_cost, total_cost
    
    def execute_trade(self, timestamp: datetime, action: int, price: float, 
                     signal: str) -> Optional[Trade]:
        """
        Execute a trade
        
        Args:
            timestamp: Trade timestamp
            action: Target action (0 or 1)
            price: Current price
            signal: Trading signal ('BUY', 'SELL', 'HOLD')
            
        Returns:
            Trade object if trade executed, None if hold
        """
        if signal == 'HOLD':
            # Update position value but don't trade
            self.position_value = self.position * price
            self.total_value = self.cash + self.position_value
            return None
        
        target_position = self.calculate_position_size(action, price)
        position_change = target_position - self.position
        
        if abs(position_change) < 1e-8:
            # No meaningful change
            return None
        
        # Determine trade side
        if position_change > 0:
            # BUY
            side = 'BUY'
            quantity = position_change
            
            # Calculate costs
            commission, slippage_cost, total_cost = self.calculate_trade_cost(
                quantity, price, side
            )
            
            # Total cash needed
            total_needed = (quantity * price) + total_cost
            
            # Check if we have enough cash
            if total_needed > self.cash:
                print(f"Warning: Insufficient cash for trade. Need ${total_needed:.2f}, have ${self.cash:.2f}")
                return None
            
            # Execute buy
            self.cash -= total_needed
            self.position += quantity
            
        else:  # position_change < 0
            # SELL
            side = 'SELL'
            quantity = abs(position_change)
            
            # Check if we have enough position
            if quantity > self.position:
                print(f"Warning: Trying to sell {quantity:.8f} BTC but only have {self.position:.8f} BTC")
                quantity = self.position
            
            # Calculate costs
            commission, slippage_cost, total_cost = self.calculate_trade_cost(
                quantity, price, side
            )
            
            # Execute sell
            cash_received = (quantity * price) - total_cost
            self.cash += cash_received
            self.position -= quantity
        
        # Update portfolio value
        self.position_value = self.position * price
        self.total_value = self.cash + self.position_value
        
        # Create trade record
        trade = Trade(
            timestamp=timestamp,
            action=side,
            price=price,
            quantity=quantity,
            value=quantity * price,
            commission=commission,
            slippage=slippage_cost,
            total_cost=total_cost,
            position_after=self.position,
            cash_after=self.cash,
            portfolio_value=self.total_value
        )
        
        self.trades.append(trade)
        
        return trade
    
    def update_portfolio_state(self, timestamp: datetime, price: float):
        """Update and record portfolio state"""
        self.position_value = self.position * price
        self.total_value = self.cash + self.position_value
        
        returns = (self.total_value / self.initial_capital) - 1
        
        state = PortfolioState(
            cash=self.cash,
            position=self.position,
            position_value=self.position_value,
            total_value=self.total_value,
            returns=returns,
            trades=len(self.trades)
        )
        
        self.portfolio_history.append(state)
        self.equity_curve.append(self.total_value)
    
    def run(self, df: pd.DataFrame, verbose: bool = True) -> Dict:
        """
        Run backtest on data with signals
        
        Args:
            df: DataFrame with signals and price data
            verbose: Print progress
            
        Returns:
            Backtest results dictionary
        """
        print("\n" + "="*60)
        print("STARTING BACKTEST")
        print("="*60)
        print(f"Period: {df.index[0]} to {df.index[-1]}")
        print(f"Total candles: {len(df):,}")
        print(f"Initial capital: ${self.initial_capital:,.2f}")
        print("="*60 + "\n")
        
        # Reset state
        self.cash = self.initial_capital
        self.position = 0.0
        self.position_value = 0.0
        self.total_value = self.initial_capital
        self.trades = []
        self.portfolio_history = []
        self.equity_curve = []
        
        # Track previous action
        previous_action = self.config['backtest']['initial_position']
        
        # Run backtest
        for idx in range(len(df)):
            if verbose and idx % 50000 == 0 and idx > 0:
                progress = 100 * idx / len(df)
                print(f"Progress: {idx:,}/{len(df):,} ({progress:.1f}%) | "
                      f"Portfolio: ${self.total_value:,.2f} | "
                      f"Trades: {len(self.trades)}")
            
            row = df.iloc[idx]
            timestamp = df.index[idx]
            price = row['close']
            action = int(row['action'])
            signal = row['signal']
            
            # Execute trade if signal
            trade = self.execute_trade(timestamp, action, price, signal)
            
            # Update portfolio state
            self.update_portfolio_state(timestamp, price)
            
            previous_action = action
        
        # Final statistics
        print("\n" + "="*60)
        print("BACKTEST COMPLETE")
        print("="*60)
        print(f"Final portfolio value: ${self.total_value:,.2f}")
        print(f"Total return: {((self.total_value/self.initial_capital - 1)*100):.2f}%")
        print(f"Total trades: {len(self.trades)}")
        print(f"Final position: {self.position:.8f} BTC")
        print(f"Final cash: ${self.cash:,.2f}")
        print("="*60 + "\n")
        
        # Create results dictionary
        results = {
            'initial_capital': self.initial_capital,
            'final_value': self.total_value,
            'total_return': (self.total_value / self.initial_capital) - 1,
            'total_trades': len(self.trades),
            'trades': self.trades,
            'portfolio_history': self.portfolio_history,
            'equity_curve': self.equity_curve,
            'final_position': self.position,
            'final_cash': self.cash
        }
        
        return results
    
    def get_trades_df(self) -> pd.DataFrame:
        """Convert trades to DataFrame"""
        if not self.trades:
            return pd.DataFrame()
        
        trades_data = [asdict(trade) for trade in self.trades]
        df = pd.DataFrame(trades_data)
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def get_portfolio_df(self) -> pd.DataFrame:
        """Convert portfolio history to DataFrame"""
        if not self.portfolio_history:
            return pd.DataFrame()
        
        portfolio_data = [asdict(state) for state in self.portfolio_history]
        df = pd.DataFrame(portfolio_data)
        
        return df
    
    def save_results(self, results_dir: str = None):
        """Save backtest results"""
        if results_dir is None:
            results_dir = self.config['output']['results_dir']
        
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'trades'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'metrics'), exist_ok=True)
        
        # Save trades
        if self.config['output']['save_trades'] and self.trades:
            trades_df = self.get_trades_df()
            trades_file = os.path.join(results_dir, 'trades', 'trades.csv')
            trades_df.to_csv(trades_file)
            print(f"Saved trades to: {trades_file}")
        
        # Save portfolio history
        portfolio_df = self.get_portfolio_df()
        portfolio_file = os.path.join(results_dir, 'metrics', 'portfolio_history.csv')
        portfolio_df.to_csv(portfolio_file)
        print(f"Saved portfolio history to: {portfolio_file}")
        
        # Save equity curve
        equity_df = pd.DataFrame({
            'equity': self.equity_curve
        })
        equity_file = os.path.join(results_dir, 'metrics', 'equity_curve.csv')
        equity_df.to_csv(equity_file)
        print(f"Saved equity curve to: {equity_file}")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run MacroHFT backtest')
    parser.add_argument('--config', type=str, default='./backtest/config/backtest_config.yaml',
                       help='Path to config file')
    parser.add_argument('--input', type=str, required=True,
                       help='Input data with signals')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize backtester
    backtester = MacroHFTBacktester(config_path=args.config)
    
    # Load data
    df = pd.read_feather(args.input)
    if 'timestamp' in df.columns:
        df.set_index('timestamp', inplace=True)
    
    # Run backtest
    results = backtester.run(df, verbose=True)
    
    # Save results
    backtester.save_results(args.output_dir)
    
    print("\n=== Backtest Complete ===")


if __name__ == "__main__":
    main()

