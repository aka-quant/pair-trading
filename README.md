# Pair Trading 2.0: бэктест BTC/ETH

Mean-reversion стратегия на спреде двух криптоактивов (OKX perpetual swap, 1H, 2021–2025).
Идея: [«Парная торговля 2.0»](https://smart-lab.ru/mobile/topic/1265846/).

## Суть стратегии

1. **Beta** — скользящий hedge ratio через OLS по логарифмам цен (`BETA_LOOKBACK = 200`).
2. **Спред** = `log(BTC) − β·log(ETH)`, **Z-score** по окну `SPREAD_LOOKBACK = 200`.
3. **Фильтр входа** — ADF-тест стационарности спреда (`p < 0.15`); опционально Hurst и фильтр волатильности.
4. **Вход**: `Z < −1.5` → long спред (Buy BTC / Sell ETH), `Z > +1.5` → short спред.
5. **Выход** (`EXIT_MODE`): `1` — возврат Z к 0; `2` — пересечение противоположного порога (переворот, всегда в рынке).
6. Одна позиция одновременно, плечо `2x`, комиссия `0.05%` на ногу.

## Запуск

```bash
pip install -r requirements.txt
python main.py                      # бэктест: статистика в терминале, trades_log.xlsx, графики
python scripts/quantstats_report.py # (опц.) HTML-отчёт QuantStats по trades_log.xlsx
```

Данные берутся из `data/` (CSV, в git не хранятся; пути — `FIRST_LEG` / `SECOND_LEG` в `src/config.py`).

## Точки входа

| Файл | Назначение |
|---|---|
| [main.py](main.py) | запуск бэктеста: загрузка → индикаторы → сигналы → статистика → графики |
| [src/config.py](src/config.py) | все параметры: данные, окна, пороги, фильтры, капитал, плечо, комиссия |
| [src/strategy.py](src/strategy.py) | `PairTradingStrategy`: условия входа/выхода, генерация сигналов |
| [src/backtest_engine.py](src/backtest_engine.py) | `BacktestEngine`: индикаторы, симуляция, метрики, графики |
| [src/parser_okx.py](src/parser_okx.py) | скачивание свечей с OKX (`python src/parser_okx.py`, параметры в начале файла) |
| [src/data_cleaner.py](src/data_cleaner.py) | очистка выбросов в ценовых рядах |
| [scripts/quantstats_report.py](scripts/quantstats_report.py) | отчёт QuantStats → `backtest_results/quantstats_report.html` |

## Метрики в терминале

`engine.print_statistics()` выводит блоками:

- **Финансовые**: капитал начало/конец, PnL, Total return, CAGR, макс. просадка (USD и %), кол-во сделок.
- **По ногам**: средний вес, PnL (%, $) BTC и ETH.
- **Сделки**: Win rate, Profit factor, Avg win / Avg loss, Expectancy, Gross profit/loss.
- **Риск**: Sharpe, Sortino, Calmar, Recovery factor.
- **Доходность по годам** и первые 5 сделок.

Пример результата с параметрами по умолчанию (BTC/ETH 1H, 2021-01 … 2025-12):

| Метрика | Значение |
|---|---|
| Капитал | 10 000 → 20 568 $ (+105.7%) |
| CAGR | 15.85% |
| Сделок / Win rate | 104 / 56.7% |
| Profit factor | 1.28 |
| Sharpe / Sortino / Calmar | 1.39 / 1.95 / 0.30 |

## Результаты

- `trades_log.xlsx` — лог всех сделок
- `backtest_results/equity_curve.png`, `backtest_results/dashboard.png` — графики
- `backtest_results/quantstats_report.html` — отчёт QuantStats

Графики показываются интерактивно; отключить — `SHOW_CHART = False` в `src/config.py`.