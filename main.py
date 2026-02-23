"""
Главный файл для запуска бэктестирования стратегии "Парная торговля 2.0"
"""

from src import config
from src.backtest_engine import BacktestEngine
from src.strategy import PairTradingStrategy


def main():
    """
    Основная функция для запуска бэктеста
    """
    print("="*70)
    print(" " * 10 + "БЭКТЕСТ: ПАРНАЯ ТОРГОВЛЯ 2.0")
    print(" " * 5 + "smart-lab.ru/mobile/topic/1265846/")
    print("="*70)
    print()
    
    # Инициализация движка бэктестирования
    engine = BacktestEngine(config)
    
    # Загрузка данных
    btc_1h, eth_1h = engine.load_data()
    
    # Сохраняем данные для использования в расчетах
    engine.btc_data = btc_1h
    engine.eth_data = eth_1h
    
    # Расчет индикаторов (Beta, Spread, Z-score, Volatility filter)
    data = engine.calculate_indicators(btc_1h, eth_1h)
    engine.data = data  # Сохраняем для визуализации
    
    # Инициализация стратегии
    strategy = PairTradingStrategy(data, config)
    
    # Запуск бэктеста
    trades, equity_curve = engine.run_backtest(strategy)
    
    # Вывод статистики
    engine.print_statistics()
    
    # Сохранение результатов
    engine.save_results()
    
    # Визуализация
    engine.plot_results()
    
    print(f"\n{'='*70}")
    print(" " * 20 + "БЭКТЕСТ ЗАВЕРШЕН!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
