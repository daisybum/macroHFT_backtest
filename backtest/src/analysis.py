"""
Performance Analysis for MacroHFT Backtest
Calculates metrics and generates visualizations
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from typing import Dict, List, Tuple
from datetime import datetime


# Set plotting style
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (14, 8)


class PerformanceAnalyzer:
    """Analyze backtest performance and generate reports"""
    
    def __init__(self, config_path: str = "./backtest/config/backtest_config.yaml"):
        """Initialize analyzer"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.results_dir = self.config['output']['results_dir']
        self.plots_dir = os.path.join(self.results_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def load_results(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load backtest results"""
        trades_file = os.path.join(self.results_dir, 'trades', 'trades.csv')
        portfolio_file = os.path.join(self.results_dir, 'metrics', 'portfolio_history.csv')
        equity_file = os.path.join(self.results_dir, 'metrics', 'equity_curve.csv')
        
        trades_df = pd.read_csv(trades_file, index_col=0, parse_dates=True) if os.path.exists(trades_file) else pd.DataFrame()
        portfolio_df = pd.read_csv(portfolio_file, index_col=0) if os.path.exists(portfolio_file) else pd.DataFrame()
        equity_df = pd.read_csv(equity_file, index_col=0) if os.path.exists(equity_file) else pd.DataFrame()
        
        return trades_df, portfolio_df, equity_df
    
    def calculate_returns(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate returns from equity curve"""
        returns = equity_curve.pct_change().fillna(0)
        return returns
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate annualized Sharpe ratio
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        # Annualization factor (1-minute data: 365 * 24 * 60 minutes per year)
        periods_per_year = 365 * 24 * 60
        
        excess_returns = returns - (risk_free_rate / periods_per_year)
        sharpe = np.sqrt(periods_per_year) * (excess_returns.mean() / returns.std())
        
        return sharpe
    
    def calculate_max_drawdown(self, equity_curve: pd.Series) -> Tuple[float, int]:
        """
        Calculate maximum drawdown and duration
        
        Args:
            equity_curve: Equity curve series
            
        Returns:
            (max_drawdown, max_drawdown_duration)
        """
        if len(equity_curve) == 0:
            return 0.0, 0
        
        # Calculate running maximum
        running_max = equity_curve.expanding().max()
        
        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max
        
        max_drawdown = drawdown.min()
        
        # Calculate drawdown duration
        is_drawdown = drawdown < 0
        drawdown_periods = []
        current_period = 0
        
        for in_dd in is_drawdown:
            if in_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        max_dd_duration = max(drawdown_periods) if drawdown_periods else 0
        
        return max_drawdown, max_dd_duration
    
    def calculate_win_rate(self, trades_df: pd.DataFrame) -> float:
        """Calculate win rate from trades"""
        if len(trades_df) == 0:
            return 0.0
        
        # Calculate PnL per trade (consecutive BUY-SELL pairs)
        buy_trades = trades_df[trades_df['action'] == 'BUY']
        sell_trades = trades_df[trades_df['action'] == 'SELL']
        
        if len(buy_trades) == 0 or len(sell_trades) == 0:
            return 0.0
        
        # Match buys with sells
        wins = 0
        total_pairs = min(len(buy_trades), len(sell_trades))
        
        for i in range(total_pairs):
            buy_price = buy_trades.iloc[i]['price']
            sell_price = sell_trades.iloc[i]['price']
            
            if sell_price > buy_price:
                wins += 1
        
        win_rate = wins / total_pairs if total_pairs > 0 else 0.0
        
        return win_rate
    
    def calculate_profit_factor(self, trades_df: pd.DataFrame) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        if len(trades_df) == 0:
            return 0.0
        
        buy_trades = trades_df[trades_df['action'] == 'BUY']
        sell_trades = trades_df[trades_df['action'] == 'SELL']
        
        if len(buy_trades) == 0 or len(sell_trades) == 0:
            return 0.0
        
        total_pairs = min(len(buy_trades), len(sell_trades))
        
        gross_profit = 0.0
        gross_loss = 0.0
        
        for i in range(total_pairs):
            buy_price = buy_trades.iloc[i]['price']
            buy_quantity = buy_trades.iloc[i]['quantity']
            sell_price = sell_trades.iloc[i]['price']
            sell_quantity = sell_trades.iloc[i]['quantity']
            
            quantity = min(buy_quantity, sell_quantity)
            pnl = (sell_price - buy_price) * quantity
            
            if pnl > 0:
                gross_profit += pnl
            else:
                gross_loss += abs(pnl)
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        return profit_factor
    
    def calculate_calmar_ratio(self, total_return: float, max_drawdown: float, years: float) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown)"""
        if max_drawdown == 0:
            return 0.0
        
        annualized_return = (1 + total_return) ** (1 / years) - 1
        calmar = annualized_return / abs(max_drawdown)
        
        return calmar
    
    def calculate_all_metrics(self, trades_df: pd.DataFrame, portfolio_df: pd.DataFrame, 
                             equity_df: pd.DataFrame, price_data: pd.DataFrame = None) -> Dict:
        """Calculate all performance metrics"""
        print("\n=== Calculating Performance Metrics ===")
        
        metrics = {}
        
        # Basic metrics
        if len(portfolio_df) > 0:
            initial_capital = portfolio_df.iloc[0]['total_value']
            final_value = portfolio_df.iloc[-1]['total_value']
            total_return = (final_value / initial_capital) - 1
            
            metrics['initial_capital'] = initial_capital
            metrics['final_value'] = final_value
            metrics['total_return'] = total_return
            metrics['total_return_pct'] = total_return * 100
        
        # Trade statistics
        metrics['total_trades'] = len(trades_df)
        if len(trades_df) > 0:
            metrics['total_commission'] = trades_df['commission'].sum()
            metrics['total_slippage'] = trades_df['slippage'].sum()
            metrics['total_costs'] = trades_df['total_cost'].sum()
        
        # Returns and risk metrics
        if len(equity_df) > 0:
            equity_curve = equity_df['equity']
            returns = self.calculate_returns(equity_curve)
            
            metrics['sharpe_ratio'] = self.calculate_sharpe_ratio(returns)
            metrics['max_drawdown'], metrics['max_drawdown_duration'] = self.calculate_max_drawdown(equity_curve)
            metrics['max_drawdown_pct'] = metrics['max_drawdown'] * 100
            
            # Volatility (annualized)
            periods_per_year = 365 * 24 * 60
            metrics['volatility'] = returns.std() * np.sqrt(periods_per_year)
        
        # Trade performance
        metrics['win_rate'] = self.calculate_win_rate(trades_df)
        metrics['win_rate_pct'] = metrics['win_rate'] * 100
        metrics['profit_factor'] = self.calculate_profit_factor(trades_df)
        
        # Calmar ratio
        if 'total_return' in metrics and 'max_drawdown' in metrics:
            # Approximate years (assuming 1 year of data)
            years = len(equity_df) / (365 * 24 * 60) if len(equity_df) > 0 else 1
            metrics['calmar_ratio'] = self.calculate_calmar_ratio(
                metrics['total_return'], metrics['max_drawdown'], years
            )
        
        # Buy & Hold comparison
        if price_data is not None and len(price_data) > 0:
            initial_price = price_data.iloc[0]['close']
            final_price = price_data.iloc[-1]['close']
            bh_return = (final_price / initial_price) - 1
            metrics['buy_hold_return'] = bh_return
            metrics['buy_hold_return_pct'] = bh_return * 100
            metrics['excess_return'] = total_return - bh_return
            metrics['excess_return_pct'] = (total_return - bh_return) * 100
        
        return metrics
    
    def print_metrics(self, metrics: Dict):
        """Print metrics in formatted table"""
        print("\n" + "="*60)
        print("PERFORMANCE METRICS")
        print("="*60)
        
        print("\n--- Portfolio ---")
        if 'initial_capital' in metrics:
            print(f"Initial Capital:      ${metrics['initial_capital']:,.2f}")
            print(f"Final Value:          ${metrics['final_value']:,.2f}")
            print(f"Total Return:         {metrics['total_return_pct']:.2f}%")
        
        print("\n--- Risk Metrics ---")
        if 'sharpe_ratio' in metrics:
            print(f"Sharpe Ratio:         {metrics['sharpe_ratio']:.3f}")
        if 'max_drawdown_pct' in metrics:
            print(f"Max Drawdown:         {metrics['max_drawdown_pct']:.2f}%")
            print(f"Max DD Duration:      {metrics['max_drawdown_duration']:,} periods")
        if 'volatility' in metrics:
            print(f"Volatility (Annual):  {metrics['volatility']*100:.2f}%")
        if 'calmar_ratio' in metrics:
            print(f"Calmar Ratio:         {metrics['calmar_ratio']:.3f}")
        
        print("\n--- Trading ---")
        print(f"Total Trades:         {metrics['total_trades']:,}")
        if 'win_rate_pct' in metrics:
            print(f"Win Rate:             {metrics['win_rate_pct']:.2f}%")
        if 'profit_factor' in metrics:
            print(f"Profit Factor:        {metrics['profit_factor']:.3f}")
        if 'total_costs' in metrics:
            print(f"Total Costs:          ${metrics['total_costs']:,.2f}")
        
        print("\n--- Benchmark Comparison ---")
        if 'buy_hold_return_pct' in metrics:
            print(f"Buy & Hold Return:    {metrics['buy_hold_return_pct']:.2f}%")
            print(f"Excess Return:        {metrics['excess_return_pct']:.2f}%")
        
        print("="*60 + "\n")
    
    def plot_equity_curve(self, equity_df: pd.DataFrame, price_data: pd.DataFrame = None):
        """Plot equity curve"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Equity curve
        equity_curve = equity_df['equity']
        ax1.plot(equity_curve.values, label='Strategy', linewidth=2)
        
        # Buy & Hold comparison
        if price_data is not None:
            initial_capital = equity_curve.iloc[0]
            initial_price = price_data.iloc[0]['close']
            bh_equity = initial_capital * (price_data['close'] / initial_price)
            ax1.plot(bh_equity.values, label='Buy & Hold', linewidth=2, alpha=0.7)
        
        ax1.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        running_max = pd.Series(equity_curve.values).expanding().max()
        drawdown = (equity_curve.values - running_max) / running_max * 100
        
        ax2.fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.3)
        ax2.plot(drawdown, color='red', linewidth=1)
        ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time (periods)', fontsize=12)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(self.plots_dir, 'equity_curve.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Saved equity curve plot: {plot_file}")
        
        plt.close()
    
    def plot_trades(self, trades_df: pd.DataFrame, price_data: pd.DataFrame):
        """Plot trades on price chart"""
        if len(trades_df) == 0:
            print("No trades to plot")
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot price
        ax.plot(price_data.index, price_data['close'], label='BTC Price', linewidth=1, alpha=0.7)
        
        # Plot trades
        buy_trades = trades_df[trades_df['action'] == 'BUY']
        sell_trades = trades_df[trades_df['action'] == 'SELL']
        
        ax.scatter(buy_trades.index, buy_trades['price'], 
                  color='green', marker='^', s=100, label='Buy', zorder=5)
        ax.scatter(sell_trades.index, sell_trades['price'], 
                  color='red', marker='v', s=100, label='Sell', zorder=5)
        
        ax.set_title('Trading Activity', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('BTC Price ($)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(self.plots_dir, 'trades.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Saved trades plot: {plot_file}")
        
        plt.close()
    
    def plot_returns_distribution(self, equity_df: pd.DataFrame):
        """Plot returns distribution"""
        equity_curve = equity_df['equity']
        returns = self.calculate_returns(equity_curve)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax1.hist(returns * 100, bins=100, edgecolor='black', alpha=0.7)
        ax1.axvline(returns.mean() * 100, color='red', linestyle='--', linewidth=2, label=f'Mean: {returns.mean()*100:.4f}%')
        ax1.set_title('Returns Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Returns (%)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(returns, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(self.plots_dir, 'returns_distribution.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Saved returns distribution plot: {plot_file}")
        
        plt.close()
    
    def save_metrics(self, metrics: Dict):
        """Save metrics to file"""
        metrics_file = os.path.join(self.results_dir, 'metrics', 'performance_metrics.csv')
        
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(metrics_file, index=False)
        
        print(f"Saved metrics to: {metrics_file}")
    
    def generate_report(self, price_data: pd.DataFrame = None):
        """Generate complete analysis report"""
        print("\n" + "="*60)
        print("GENERATING ANALYSIS REPORT")
        print("="*60)
        
        # Load results
        trades_df, portfolio_df, equity_df = self.load_results()
        
        # Calculate metrics
        metrics = self.calculate_all_metrics(trades_df, portfolio_df, equity_df, price_data)
        
        # Print metrics
        self.print_metrics(metrics)
        
        # Save metrics
        self.save_metrics(metrics)
        
        # Generate plots
        if self.config['analysis']['plot_equity_curve']:
            self.plot_equity_curve(equity_df, price_data)
        
        if self.config['analysis']['plot_drawdown']:
            # Already included in equity curve plot
            pass
        
        if self.config['analysis']['plot_trades'] and price_data is not None:
            self.plot_trades(trades_df, price_data)
        
        self.plot_returns_distribution(equity_df)
        
        print("\n=== Analysis Complete ===")
        print(f"Results saved to: {self.results_dir}")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze backtest results')
    parser.add_argument('--config', type=str, default='./backtest/config/backtest_config.yaml',
                       help='Path to config file')
    parser.add_argument('--price-data', type=str, default=None,
                       help='Path to price data for comparison plots')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = PerformanceAnalyzer(config_path=args.config)
    
    # Load price data if provided
    price_data = None
    if args.price_data:
        price_data = pd.read_feather(args.price_data)
        if 'timestamp' in price_data.columns:
            price_data.set_index('timestamp', inplace=True)
    
    # Generate report
    analyzer.generate_report(price_data)


if __name__ == "__main__":
    main()

