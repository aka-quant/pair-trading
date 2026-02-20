"""
Модуль бэктестера парного трейдинга BTC/ETH
"""

from .backtest_engine import BacktestEngine
from .strategy import PairTradingStrategy
from . import config

__all__ = ['BacktestEngine', 'PairTradingStrategy', 'config']
