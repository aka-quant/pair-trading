import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

# Генерация синтетических данных для двух активов
np.random.seed(42)
n = 200
x = np.cumsum(np.random.normal(0, 1, n))  # базовый актив (например, log(futures))
beta_true = 1.5
alpha_true = 0.7
y = alpha_true + beta_true * x + np.random.normal(0, 1, n)  # второй актив (например, log(spot))

df = pd.DataFrame({'x': x, 'y': y})

# Визуализация исходных рядов
plt.figure(figsize=(12, 4))
plt.plot(df['x'], label='x (log(futures))')
plt.plot(df['y'], label='y (log(spot))')
plt.title('Синтетические ряды для парной торговли')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Обычная линейная регрессия (по всей серии)
X = add_constant(df['x'])
model = OLS(df['y'], X).fit()
print('Результаты линейной регрессии по всей серии:')
print(model.summary())

# Визуализация регрессии
plt.figure(figsize=(8, 5))
plt.scatter(df['x'], df['y'], alpha=0.5, label='Данные')
plt.plot(df['x'], model.predict(X), color='red', label='Регрессия')
plt.xlabel('x (log(futures))')
plt.ylabel('y (log(spot))')
plt.title('Линейная регрессия: y = alpha + beta * x')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Rolling beta (скользящая регрессия)
lookback = 40
rolling_betas = []
rolling_alphas = []
for i in range(lookback, n):
    X_window = add_constant(df['x'].iloc[i-lookback:i])
    y_window = df['y'].iloc[i-lookback:i]
    model_window = OLS(y_window, X_window).fit()
    rolling_betas.append(model_window.params['x'])
    rolling_alphas.append(model_window.params['const'])

# Визуализация rolling beta
plt.figure(figsize=(12, 4))
plt.plot(range(lookback, n), rolling_betas, label='Rolling beta')
plt.axhline(beta_true, color='red', linestyle='--', label='Истинное beta')
plt.title(f'Rolling beta (окно={lookback})')
plt.xlabel('Индекс')
plt.ylabel('Beta')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Расчёт спреда с rolling beta
spread = []
for i, idx in enumerate(range(lookback, n)):
    beta = rolling_betas[i]
    alpha = rolling_alphas[i]
    s = df['y'].iloc[idx] - (alpha + beta * df['x'].iloc[idx])
    spread.append(s)

plt.figure(figsize=(12, 4))
plt.plot(range(lookback, n), spread, label='Spread (y - (alpha + beta*x))')
plt.title('Скользящий спред для парной торговли')
plt.xlabel('Индекс')
plt.ylabel('Spread')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Теперь вы можете экспериментировать с lookback, шумом, beta_true и alpha_true,
# чтобы увидеть, как меняется rolling beta и спред.
