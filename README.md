# Бэктестирование стратегии парной торговли

Модульная инфраструктура для бэктестирования стратегии парной торговли на криптовалютном рынке (биржа OKX).

## 📋 Описание

Проект реализует стратегию парной торговли, основанную на:
- Расчете дельты между двумя активами (например, BTC/ETH)
- Приведении дельты к стационарности через скользящие средние
- Использовании z-score для определения точек входа/выхода
- Пороговых значениях ±2σ для открытия позиций

## 🏗️ Структура проекта

```
pair-trading/
├── previous/              # Предыдущие наработки (индикаторы TradingView)
│   └── tradingview.ipynb
├── src/                   # Исходный код модулей
│   ├── __init__.py
│   ├── data_loader.py     # Загрузка данных с OKX
│   ├── indicators.py      # Расчет индикаторов
│   ├── strategy.py        # Логика торговой стратегии
│   └── backtest.py        # Движок бэктестирования
├── data/                  # Кэш данных
│   └── cache/
├── results/               # Результаты бэктестов
│   ├── plots/            # Графики
│   ├── reports/          # HTML отчеты
│   └── exports/          # CSV файлы
├── config.py              # Конфигурация параметров
├── backtest_notebook.ipynb # Главный ноутбук для запуска
├── requirements.txt       # Зависимости
├── .env.example          # Пример файла с API ключами
└── README.md             # Этот файл
```

## 🚀 Быстрый старт

### 1. Установка зависимостей

```powershell
pip install -r requirements.txt
```

### 2. Настройка API ключей (опционально)

Для загрузки публичных данных API ключи не требуются. Если нужны приватные данные:

```powershell
cp .env.example .env
# Отредактируйте .env файл, добавив ваши ключи API OKX
```

### 3. Запуск бэктеста

Откройте `backtest_notebook.ipynb` в Jupyter и выполните все ячейки.

Или запустите программно:

```python
from config import StrategyConfig, DataConfig, BacktestConfig
from src.backtest import Backtester

# Создание конфигураций
strategy_config = StrategyConfig()
data_config = DataConfig()
backtest_config = BacktestConfig()

# Инициализация и запуск
backtester = Backtester(strategy_config, data_config, backtest_config)
metrics = backtester.run_full_backtest()
```

## ⚙️ Конфигурация

### Параметры стратегии (`StrategyConfig`)

```python
base_symbol = "BTC/USDT:USDT"      # Базовый актив
quote_symbol = "ETH/USDT:USDT"     # Котируемый актив
len_long = 100                      # Период длинной MA
len_short = 10                      # Период короткой MA
delta_window = 1000                 # Окно для расчета статистики
entry_threshold = 2.0               # Порог входа (σ)
exit_threshold = 0.0                # Порог выхода
position_size = 1.0                 # Размер позиции
maker_fee = 0.0002                  # Комиссия мейкера (0.02%)
taker_fee = 0.0005                  # Комиссия тейкера (0.05%)
```

### Параметры данных (`DataConfig`)

```python
exchange = "okx"
timeframe = "1h"                    # 1m, 5m, 15m, 1h, 4h, 1d
start_date = "2024-01-01 00:00:00"
end_date = "2024-12-31 23:59:59"
cache_dir = "data/cache"
use_cache = True
```

### Параметры бэктеста (`BacktestConfig`)

```python
initial_capital = 10000.0           # Начальный капитал (USDT)
benchmark = "BTC/USDT:USDT"         # Бенчмарк
risk_free_rate = 0.02               # Безрисковая ставка (2%)
save_plots = True
save_reports = True
```

## 📊 Метрики

Проект использует библиотеку **QuantStats** для расчета всех ключевых метрик:

- **Доходность**: Total Return, CAGR, Daily/Monthly/Yearly Returns
- **Риск**: Volatility, Max Drawdown, Sharpe Ratio, Sortino Ratio, Calmar Ratio
- **Сделки**: Total Trades, Win Rate, Avg Win/Loss, Profit Factor
- **И многое другое**: полный набор метрик для профессионального анализа

## 📈 Результаты

После запуска бэктеста вы получите:

1. **Графики** (`results/plots/`):
   - Изменение капитала
   - Просадка
   - Z-score с торговыми сигналами
   - Позиции

2. **HTML отчет** (`results/reports/quantstats_report.html`):
   - Полный отчет QuantStats с метриками и графиками

3. **CSV файлы** (`results/exports/`):
   - `backtest_results.csv` - полные результаты
   - `trades.csv` - список всех сделок
   - `metrics.csv` - сводка метрик

## 🔧 Расширение функционала

### Добавление новых индикаторов

Отредактируйте `src/indicators.py`:

```python
@staticmethod
def calculate_my_indicator(data: pd.Series) -> pd.Series:
    # Ваш код
    return result
```

### Модификация логики стратегии

Отредактируйте `src/strategy.py`:

```python
def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
    # Измените логику входа/выхода
    pass
```

### Оптимизация параметров

Используйте раздел 6 в ноутбуке для grid search или другой оптимизации.

## 📚 Используемые технологии

- **pandas** - обработка данных
- **numpy** - численные вычисления
- **ccxt** - подключение к биржам
- **quantstats** - расчет метрик и отчетов
- **matplotlib/seaborn** - визуализация
- **jupyter** - интерактивная разработка

## ⚠️ Важные замечания

1. **Это только бэктест** - результаты прошлого не гарантируют будущую доходность
2. **Комиссии учтены** - используются реальные комиссии OKX
3. **Проскальзывание не учтено** - в реальной торговле результаты могут отличаться
4. **Данные кэшируются** - для ускорения повторных запусков

## 🎯 Следующие шаги

После успешного бэктеста:

1. Оптимизация параметров
2. Walk-forward анализ
3. Тестирование на разных парах
4. Добавление управления рисками
5. Алгоритмизация для live trading

## 📞 Поддержка

При возникновении проблем:
- Проверьте установку зависимостей
- Убедитесь в наличии интернет-соединения для загрузки данных
- Проверьте правильность формата торговых пар для OKX

## 📄 Лицензия

MIT License - используйте свободно для ваших проектов.

---

**Удачи в торговле! 🚀📈**
