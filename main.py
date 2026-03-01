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
    
    # Новый красивый вывод параметров стратегии
    engine = BacktestEngine(config)
    engine.print_backtest_settings()
    
    # Загрузка данных
    first_leg, second_leg = engine.load_data()
    engine.first_leg_data = first_leg
    engine.second_leg_data = second_leg
    # Расчет индикаторов (Beta, Spread, Z-score, Volatility filter)
    data = engine.calculate_indicators(first_leg, second_leg)
    engine.data = data  # Сохраняем для визуализации


    # Инициализация стратегии
    strategy = PairTradingStrategy(data, config)

    # Запуск бэктеста
    trades, equity_curve = engine.run_backtest(strategy)

    # Сравнение: сколько сигналов ENTRY сработало при can_trade==True
    entry_signals = [s for s in trades if 'ENTRY' in s.get('type','')]
    print(f"\nENTRY сигналов: {len(entry_signals)}")
    
    # Вывод статистики
    engine.print_statistics()
    
    # Сохранение результатов
    engine.save_results()
    
    # Визуализация
    # Визуализация: оба дашборда вызываются через plot_results()
    engine.plot_results()
    
    print(f"\n{'='*70}")
    print(" " * 20 + "БЭКТЕСТ ЗАВЕРШЕН!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
