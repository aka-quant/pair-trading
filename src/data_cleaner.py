import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, filepath, timestamp_col='timestamp', value_col='close', window=21, z_thresh=4.0):
        self.filepath = filepath
        self.timestamp_col = timestamp_col
        self.value_col = value_col
        self.window = window
        self.z_thresh = z_thresh
        self.outliers = []
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.filepath, parse_dates=[self.timestamp_col])
        self.df.set_index(self.timestamp_col, inplace=True)

    def detect_and_replace_outliers(self):
        values = self.df[self.value_col]
        rolling_median = values.rolling(window=self.window, center=True, min_periods=1).median()
        rolling_std = values.rolling(window=self.window, center=True, min_periods=1).std()
        z_scores = (values - rolling_median) / (rolling_std.replace(0, np.nan))
        outlier_mask = np.abs(z_scores) > self.z_thresh
        for idx in self.df.index[outlier_mask]:
            old_value = self.df.at[idx, self.value_col]
            median_value = rolling_median.at[idx]
            self.df.at[idx, self.value_col] = median_value
            self.outliers.append({
                'timestamp': idx,
                'old_value': old_value,
                'new_value': median_value
            })

    def save_cleaned(self):
        import os
        base, ext = os.path.splitext(self.filepath)
        out_path = f"{base}_cleaned{ext}"
        self.df.reset_index().to_csv(out_path, index=False)
        print(f"Очищенный файл сохранён как: {out_path}")

    def print_outliers(self):
        print(f"\nЗаменённые выбросы (timestamp, old_value, new_value):")
        for o in self.outliers:
            print(f"{o['timestamp']}: {o['old_value']} → {o['new_value']}")

if __name__ == '__main__':
    # Пример использования
    # Замените путь и столбцы на свои
    cleaner = DataCleaner(
        filepath='data/BTC-USDT_2022-05-27_2025-12-31_1H.csv',
        timestamp_col='timestamp',
        value_col='close',
        window=21,
        z_thresh=4.0
    )
    cleaner.load_data()
    cleaner.detect_and_replace_outliers()
    cleaner.save_cleaned()
    cleaner.print_outliers()
