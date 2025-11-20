"""
MacroHFT Bitcoin Backtesting System
"""

__version__ = "1.0.0"
__author__ = "MacroHFT Backtest"

from .data_fetcher import BinanceDataFetcher
from .feature_engineering import FeatureEngineer
from .strategy import MacroHFTStrategy
from .backtester import MacroHFTBacktester
from .analysis import PerformanceAnalyzer

__all__ = [
    'BinanceDataFetcher',
    'FeatureEngineer',
    'MacroHFTStrategy',
    'MacroHFTBacktester',
    'PerformanceAnalyzer',
]

