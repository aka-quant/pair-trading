import pandas as pd
import matplotlib.pyplot as plt

# Загрузите свои файлы (укажите актуальные пути к данным)
spot_path = 'data/BTC-USDT_2022-05-27_2025-12-31_1H.csv'  # путь к спотовым данным
futures_path = 'data/BTC-USDT-SWAP_2022-05-27_2025-12-31_1H.csv'  # путь к фьючерсным данным

# Загрузка данных
spot = pd.read_csv(spot_path, parse_dates=['timestamp'])
futures = pd.read_csv(futures_path, parse_dates=['timestamp'])

# Индекс по времени
spot.set_index('timestamp', inplace=True)
futures.set_index('timestamp', inplace=True)

# Совместим по времени (inner join)
df = spot[['close']].rename(columns={'close': 'spot_close'}).join(
    futures[['close']].rename(columns={'close': 'futures_close'}),
    how='inner'
)

# Дельта
# (цена закрытия спота - цена закрытия фьючерса)
df['delta'] = df['spot_close'] - df['futures_close']

# Визуализация
df['delta'].plot(figsize=(16, 6), title='Delta: Spot Close - Futures Close', grid=True)
plt.xlabel('Time')
plt.ylabel('Delta (USD)')
plt.tight_layout()
plt.show()
