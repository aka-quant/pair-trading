"""
Главный скрипт для запуска бэктестирования стратегии парной торговли
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from pathlib import Path

from config import StrategyConfig, DataConfig, BacktestConfig
from src.backtest import Backtester


def main():
    """Основная функция запуска бэктеста"""
    
    print("=" * 80)
    print("ЗАПУСК БЭКТЕСТИРОВАНИЯ СТРАТЕГИИ ПАРНОЙ ТОРГОВЛИ")
    print("=" * 80)
    
    # ========================================================================
    # КОНФИГУРАЦИЯ
    # ========================================================================
    
    # Параметры стратегии
    strategy_config = StrategyConfig(
        base_symbol="BTC/USDT:USDT",
        quote_symbol="ETH/USDT:USDT",
        len_long=100,
        len_short=10,
        delta_window=1000,
        entry_threshold=2.0,
        exit_threshold=0.0,
        position_size=1.0,
        maker_fee=0.0002,
        taker_fee=0.0005
    )
    
    # Параметры данных
    data_config = DataConfig(
        exchange="okx",
        timeframe="1h",
        start_date="2024-01-01 00:00:00",
        end_date="2024-12-31 23:59:59",
        cache_dir="data/cache",
        use_cache=True
    )
    
    # Параметры бэктеста
    backtest_config = BacktestConfig(
        initial_capital=10000.0,
        benchmark="BTC/USDT:USDT",
        risk_free_rate=0.02,
        save_plots=True,
        plots_dir="results/plots",
        save_reports=True,
        reports_dir="results/reports"
    )
    
    # ========================================================================
    # СОЗДАНИЕ И ЗАПУСК БЭКТЕСТЕРА
    # ========================================================================
    
    backtester = Backtester(
        strategy_config=strategy_config,
        data_config=data_config,
        backtest_config=backtest_config
    )
    
    # Запуск полного цикла бэктестирования
    metrics = backtester.run_full_backtest(force_reload=False)
    
    # ========================================================================
    # ИТОГИ
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("✅ БЭКТЕСТ ЗАВЕРШЕН УСПЕШНО!")
    print("=" * 80)
    
    print(f"\n📊 Основные результаты:")
    print(f"  Общая доходность: {metrics['total_return']:.2f}%")
    print(f"  Конечный капитал: ${metrics['final_equity']:,.2f}")
    print(f"  Максимальная просадка: {metrics['max_drawdown']:.2f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Всего сделок: {metrics['total_trades']}")
    print(f"  Винрейт: {metrics['win_rate']:.2f}%")
    
    print(f"\n📁 Результаты сохранены:")
    print(f"  Графики: {backtest_config.plots_dir}/")
    print(f"  Отчеты: {backtest_config.reports_dir}/")
    print(f"  CSV файлы: results/exports/")
    
    return backtester, metrics


if __name__ == "__main__":
    backtester, metrics = main()
