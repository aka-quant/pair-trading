"""
Конфигурация параметров для бэктестирования стратегии парной торговли
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class StrategyConfig:
    """Параметры торговой стратегии"""
    
    # Торговые пары
    base_symbol: str = "BTC/USDT:USDT"  # Базовый актив
    quote_symbol: str = "ETH/USDT:USDT"  # Котируемый актив
    
    # Параметры скользящих средних
    len_long: int = 100   # Период длинной MA
    len_short: int = 10   # Период короткой MA
    
    # Параметры для расчета дельты
    delta_window: int = 1000  # Окно для расчета медианы и стандартного отклонения
    
    # Сигналы входа/выхода
    entry_threshold: float = 2.0  # Порог входа в позицию (количество сигм)
    exit_threshold: float = 0.0   # Порог выхода из позиции (возврат к медиане)
    
    # Управление рисками
    position_size: float = 1.0    # Размер позиции (доля капитала)
    stop_loss: Optional[float] = None  # Стоп-лосс в процентах (если нужен)
    take_profit: Optional[float] = None  # Тейк-профит в процентах (если нужен)
    
    # Комиссии
    maker_fee: float = 0.0002  # 0.02% комиссия мейкера на OKX
    taker_fee: float = 0.0005  # 0.05% комиссия тейкера на OKX


@dataclass
class DataConfig:
    """Параметры загрузки данных"""
    
    exchange: str = "okx"
    timeframe: str = "1h"  # Таймфрейм: 1m, 5m, 15m, 1h, 4h, 1d
    
    # Период данных для бэктеста
    start_date: str = "2024-01-01 00:00:00"
    end_date: str = "2024-12-31 23:59:59"
    
    # Путь для кэширования данных
    cache_dir: str = "data/cache"
    use_cache: bool = True  # Использовать кэш для ускорения


@dataclass
class BacktestConfig:
    """Параметры бэктестирования"""
    
    initial_capital: float = 10000.0  # Начальный капитал в USDT
    
    # Параметры отчетности
    benchmark: Optional[str] = "BTC/USDT:USDT"  # Бенчмарк для сравнения
    risk_free_rate: float = 0.02  # Безрисковая ставка (2% годовых)
    
    # Параметры визуализации
    save_plots: bool = True
    plots_dir: str = "results/plots"
    
    # Параметры отчетов
    save_reports: bool = True
    reports_dir: str = "results/reports"


# Создание конфигураций по умолчанию
DEFAULT_STRATEGY_CONFIG = StrategyConfig()
DEFAULT_DATA_CONFIG = DataConfig()
DEFAULT_BACKTEST_CONFIG = BacktestConfig()
