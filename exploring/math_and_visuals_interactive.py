"""
Интерактивный справочник по математике и визуализации ключевых блоков парного трейдинга.

Блоки:
1. OLS-регрессия (β-коэффициент)
2. ADF-тест (статистическая стационарность)
3. Экспонента Херста (характер ряда)
4. Корреляция и коинтеграция
5. Расчет Z-score и генерация сигналов
6. Фильтр волатильности

Пользователь выбирает номер блока — программа выводит объяснение, формулы и пример визуализации.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
from hurst import compute_Hc


def show_ols():
    print("""
1. OLS-регрессия (β-коэффициент)
-------------------------------
Математика:
  β = Cov(log(PA), log(PB)) / Var(log(PB))
  Используется для расчета хедж-коэффициента между активами.
Визуализация:
  График log(PA) и β*log(PB) на одном поле, график скорректированного спреда.
""")
    # Пример визуализации
    t = np.arange(100)
    pa = np.cumsum(np.random.normal(0, 1, 100)) + 100
    pb = np.cumsum(np.random.normal(0, 1, 100)) + 100
    X = np.log(pb).reshape(-1, 1)
    y = np.log(pa)
    reg = LinearRegression().fit(X, y)
    beta = reg.coef_[0]
    spread = np.log(pa) - beta * np.log(pb)
    plt.figure(figsize=(10, 4))
    plt.plot(np.log(pa), label='log(PA)')
    plt.plot(beta * np.log(pb), label='beta*log(PB)')
    plt.legend(); plt.title('OLS-регрессия'); plt.show()
    plt.figure(figsize=(10, 4))
    plt.plot(spread, label='Спред')
    plt.axhline(spread.mean(), color='r', linestyle='--', label='Среднее')
    plt.legend(); plt.title('Скорректированный спред'); plt.show()


def show_adf():
    print("""
2. ADF-тест (Augmented Dickey-Fuller)
-------------------------------------
Математика:
  Проверяет стационарность спреда. p-value < 0.05 — спред стационарен.
Визуализация:
  График спреда с линией среднего, вывод p-value.
""")
    t = np.arange(100)
    spread = np.cumsum(np.random.normal(0, 1, 100))
    plt.figure(figsize=(10, 4))
    plt.plot(spread, label='Спред')
    plt.axhline(np.mean(spread), color='r', linestyle='--', label='Среднее')
    plt.legend(); plt.title('Спред для ADF-теста'); plt.show()
    adf_result = adfuller(spread)
    print(f"ADF p-value: {adf_result[1]:.4f}")


def show_hurst():
    print("""
3. Экспонента Херста
--------------------
Математика:
  H < 0.5 — средне-возвратный, H ≈ 0.5 — random walk, H > 0.5 — трендовый.
Визуализация:
  График спреда, расчет Hurst exponent.
""")
    t = np.arange(100)
    spread = np.cumsum(np.random.normal(0, 1, 100))
    plt.figure(figsize=(10, 4))
    plt.plot(spread, label='Спред')
    plt.legend(); plt.title('Спред для Hurst'); plt.show()
    H, c, data = compute_Hc(spread, kind='price', simplified=True)
    print(f"Hurst exponent: {H:.3f}")


def show_corr_cointegration():
    print("""
4. Корреляция и коинтеграция
----------------------------
Математика:
  Корреляция — мера линейной связи (Pearson). Коинтеграция — наличие долгосрочной связи.
  Для парного трейдинга важна высокая корреляция и подтвержденная коинтеграция.
Визуализация:
  Графики двух активов, расчет корреляции, визуализация совместного движения.
""")
    t = np.arange(100)
    pa = np.cumsum(np.random.normal(0, 1, 100)) + 100
    pb = pa + np.random.normal(0, 2, 100)
    plt.figure(figsize=(10, 4))
    plt.plot(pa, label='PA')
    plt.plot(pb, label='PB')
    plt.legend(); plt.title('Корреляция активов'); plt.show()
    corr = np.corrcoef(pa, pb)[0, 1]
    print(f"Корреляция: {corr:.3f}")


def show_zscore_signals():
    print("""
5. Z-score и генерация сигналов
-------------------------------
Математика:
  Z = (Спред_t - μ) / σ
  Сигналы: Z < -1.5 — long, Z > 1.5 — short.
Визуализация:
  График Z-score, выделение зон входа/выхода.
""")
    spread = np.cumsum(np.random.normal(0, 1, 100))
    window = 30
    zscore = (spread - pd.Series(spread).rolling(window).mean()) / pd.Series(spread).rolling(window).std()
    plt.figure(figsize=(10, 4))
    plt.plot(zscore, label='Z-score')
    plt.axhline(-1.5, color='g', linestyle='--', label='Long signal')
    plt.axhline(1.5, color='r', linestyle='--', label='Short signal')
    plt.legend(); plt.title('Z-score и сигналы'); plt.show()


def show_volatility_filter():
    print("""
6. Фильтр волатильности
------------------------
Математика:
  Вход только если σ_текущая > 0.8 * σ_долгосрочная (например, за 60 баров).
Визуализация:
  График текущей и долгосрочной волатильности, выделение зон допуска.
""")
    spread = np.cumsum(np.random.normal(0, 1, 100))
    current_vol = pd.Series(spread).rolling(10).std()
    long_vol = pd.Series(spread).rolling(60).std()
    plt.figure(figsize=(10, 4))
    plt.plot(current_vol, label='Текущая σ (10)')
    plt.plot(long_vol, label='Долгосрочная σ (60)')
    plt.axhline(0.8, color='r', linestyle='--', label='Порог 0.8')
    plt.legend(); plt.title('Фильтр волатильности'); plt.show()


blocks = {
    '1': show_ols,
    '2': show_adf,
    '3': show_hurst,
    '4': show_corr_cointegration,
    '5': show_zscore_signals,
    '6': show_volatility_filter
}

if __name__ == "__main__":
    print("Выберите блок для изучения:")
    print("1 — OLS-регрессия (β-коэффициент)")
    print("2 — ADF-тест (статистическая стационарность)")
    print("3 — Экспонента Херста")
    print("4 — Корреляция и коинтеграция")
    print("5 — Z-score и генерация сигналов")
    print("6 — Фильтр волатильности")
    choice = input("Введите номер блока: ").strip()
    if choice in blocks:
        blocks[choice]()
    else:
        print("Некорректный выбор.")
