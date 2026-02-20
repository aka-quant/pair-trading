"""
Главный файл для запуска бэктестирования торговой стратегии
"""

from src import config
from src.backtest_engine import BacktestEngine
from src.strategy import PairTradingStrategy


def main():
    """
    Основная функция для запуска бэктеста
    """
    print("="*60)
    print("БЭКТЕСТИРОВАНИЕ СТРАТЕГИИ ПАРНОГО ТРЕЙДИНГА BTC/ETH")
    print("="*60)
    
    # Инициализация движка бэктестирования
    engine = BacktestEngine(config)
    
    # Загрузка данных
    btc_15m, btc_1h, eth_15m, eth_1h = engine.load_data()
    
    # Расчет индикаторов
    data = engine.calculate_indicators(btc_15m, btc_1h, eth_15m, eth_1h)
    engine.data = data  # Сохраняем данные в движке для визуализации
    
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
    
    print(f"\n{'='*60}")
    print("БЭКТЕСТ ЗАВЕРШЕН!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
