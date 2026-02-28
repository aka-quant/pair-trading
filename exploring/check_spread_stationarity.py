import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# Импортируем spread из основного файла
from rolling_linear_regression_pair_trading import spread, lookback, n

# Проверка на стационарность (ADF)
adf_result = adfuller(spread)
print('ADF statistic:', adf_result[0])
print('ADF p-value:', adf_result[1])

# Тест на экспоненту Херста (mean reversion)
def hurst_exponent(ts):
    lags = range(2, 20)
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]

hurst = hurst_exponent(np.array(spread))
print(f'Hurst exponent: {hurst:.4f}')
if hurst < 0.5:
    print('Hurst < 0.5: ряд mean-reverting (возврат к среднему)')
elif hurst > 0.5:
    print('Hurst > 0.5: ряд трендовый (persistent)')
else:
    print('Hurst ≈ 0.5: случайное блуждание (random walk)')

# Визуализация спреда и его mean reversion
plt.figure(figsize=(12, 4))
plt.plot(range(lookback, n), spread, label='Spread (y - (alpha + beta*x))')
plt.axhline(np.mean(spread), color='red', linestyle='--', label='Mean')
plt.title('Проверка стационарности и mean reversion спреда')
plt.xlabel('Индекс')
plt.ylabel('Spread')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- Rolling ADF и Hurst ---
window = 60
rolling_adf = []
rolling_hurst = []
for i in range(window, len(spread)):
    window_spread = np.array(spread[i-window:i])
    # ADF
    try:
        adf_p = adfuller(window_spread)[1]
    except:
        adf_p = np.nan
    rolling_adf.append(adf_p)
    # Hurst
    try:
        h = hurst_exponent(window_spread)
    except:
        h = np.nan
    rolling_hurst.append(h)

# Исправление оси X для rolling-графиков
x_idx = range(window, window + len(rolling_adf))

# Визуализация rolling ADF и Hurst
fig, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(x_idx, rolling_adf, label='Rolling ADF p-value', color='brown')
ax1.axhline(0.05, color='red', linestyle='--', label='ADF 0.05 threshold')
ax1.set_ylabel('ADF p-value')
ax1.set_xlabel('Индекс')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(x_idx, rolling_hurst, label='Rolling Hurst', color='purple', alpha=0.6)
ax2.axhline(0.5, color='gray', linestyle='--', label='Hurst 0.5 threshold')
ax2.set_ylabel('Hurst exponent')
ax2.legend(loc='upper right')

plt.title('Rolling ADF p-value и Hurst exponent (окно=60)')
plt.tight_layout()
plt.show()
