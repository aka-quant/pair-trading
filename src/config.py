"""
Конфигурационный файл с параметрами стратегии и бэктеста
Стратегия "Парная торговля 2.0" на основе статьи smart-lab.ru/mobile/topic/1265846/
"""

# === Параметры данных ===
BTC_1H_FILE = 'data/BTC-USDT-SWAP_2021-01-01_2025-12-31_1H.csv'
ETH_1H_FILE = 'data/ETH-USDT-SWAP_2021-01-01_2025-12-31_1H.csv'

# === Параметры стратегии "Парная торговля 2.0" ===

# --- Параметры Beta (hedge ratio) ---
BETA_LOOKBACK = 60  # Окно для расчета beta через OLS регрессию (60 баров)
USE_ROLLING_BETA = True  # Использовать ли скользящую beta (адаптация к изменению корреляции)

# --- Параметры спреда и Z-score ---
SPREAD_LOOKBACK = 30  # Окно для расчета среднего и std спреда (30 баров)
Z_SCORE_ENTRY = 1.5  # Порог Z-score для входа (±1.5σ вместо классических ±2.0σ)
Z_SCORE_EXIT = 0.0   # Порог Z-score для выхода (0 = возврат к среднему)

# --- Режим выхода из позиций ---
EXIT_MODE = 1  # 1 = Выход при возврате к 0 (консервативный)
               # 2 = Выход при пересечении противоположного порога (агрессивный, always in market)

# --- Фильтр волатильности ---
USE_VOLATILITY_FILTER = True  # Включить фильтр волатильности
VOLATILITY_THRESHOLD = 0.8  # Входим только если σ_current >= 0.8 × σ_long
VOLATILITY_LONG_LOOKBACK = 60  # Окно для расчета долгосрочной волатильности

# === Параметры коинтеграции и стационарности ===
# "Cointegration beats correlation" - из статьи Amberdata

# ADF тест (Augmented Dickey-Fuller) - проверка стационарности спреда
USE_ADF_TEST = True  # Использовать ADF тест для проверки коинтеграции
ADF_PVALUE_THRESHOLD = 0.05  # Порог p-value для ADF теста (< 0.05 = стационарен)
ADF_LOOKBACK = 60  # Окно данных для ADF теста

# Hurst exponent - проверка mean reversion свойств
USE_HURST_FILTER = True  # Использовать Hurst exponent для проверки mean reversion
HURST_MAX_THRESHOLD = 0.5  # Максимальный Hurst (< 0.5 = mean reverting, идеально < 0.4)
HURST_LOOKBACK = 60  # Окно данных для расчета Hurst

# Динамическая проверка коинтеграции
COINTEGRATION_CHECK_ENABLED = False  # ВРЕМЕННО ОТКЛЮЧЕНО для тестирования базовой логики

# === Параметры бэктеста ===
INITIAL_CAPITAL = 10000  # Начальный капитал в долларах
COMMISSION_RATE = 0.0    # Комиссия за сделку в процентах (0.0 = без комиссии, 0.1 = 0.1%)

# === Параметры визуализации ===
SAVE_TRADES_LOG = True   # Сохранять ли лог сделок
TRADES_LOG_FILE = 'trades_log.xlsx'  # Формат Excel для корректного отображения столбцов
SAVE_CHART = False       # Сохранять ли график результатов
CHART_FILE = 'backtest_results.png'
CHART_DPI = 150          # Разрешение графика
SHOW_CHART = True        # Показывать ли график после завершения
