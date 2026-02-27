"""
Скрипт для парсинга рыночных данных с OKX (спот), используя python-okx и метод get_history_candlesticks.
"""

import okx.MarketData as MarketData
import pandas as pd
import time
from datetime import datetime, timezone

# Параметры запроса
INST_ID = "BTC-USDT"
BAR = "1H"  # 1-часовой таймфрейм
LIMIT = 300  # Максимум 300 свечей за запрос
START_DATE = "2022-05-27 09:00:00"
END_DATE = "2025-12-31 23:59:59"

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

marketDataAPI = MarketData.MarketAPI()

def iso_to_ts(iso_str):
    """Преобразует строку даты в миллисекунды с начала эпохи (UTC)."""
    dt = datetime.strptime(iso_str, DATE_FORMAT).replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def ts_to_iso(ts):
    """Преобразует миллисекунды с начала эпохи в строку даты."""
    return datetime.utcfromtimestamp(ts / 1000).strftime(DATE_FORMAT)

def fetch_all_candles(inst_id, bar, start_date, end_date, limit=300):
    after = iso_to_ts(END_DATE)    # конец периода
    before = iso_to_ts(START_DATE) # начало периода
    all_candles = []
    while True:
        result = marketDataAPI.get_history_candlesticks(
            instId=inst_id,
            bar=bar,
            after=str(after),
            before=str(before),
            limit=str(limit)
        )
        candles = result.get('data', [])
        if not candles:
            print("API вернул пустой ответ. Завершаем.")
            break
        for candle in candles:
            candle_ts = int(candle[0])
            if candle_ts < before:
                print(f"Достигнута начальная дата. Загружено всего {len(all_candles)} свечей.")
                return all_candles
            all_candles.append(candle)
            if len(all_candles) % 300 == 0:
                print(f"Загружено {len(all_candles)} свечей...")
                print(f"Последняя загруженная дата: {ts_to_iso(candle_ts)}")
        after = int(candles[-1][0]) - 1
        time.sleep(1)
    return all_candles

def main():
    candles = fetch_all_candles(INST_ID, BAR, START_DATE, END_DATE, LIMIT)
    if not candles:
        print("Нет данных!")
        return
    df = pd.DataFrame(candles, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote', 'confirm'
    ])
    df['timestamp'] = df['timestamp'].astype(int)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[(df['timestamp'] >= START_DATE) & (df['timestamp'] <= END_DATE)]
    df.sort_values('timestamp', inplace=True)
    df.set_index('timestamp', inplace=True)
    df.to_csv(f"data/{INST_ID}_{START_DATE[:10]}_{END_DATE[:10]}_{BAR}.csv")
    print(f"Сохранено {len(df)} свечей в data/{INST_ID}_{START_DATE[:10]}_{END_DATE[:10]}_{BAR}.csv")

if __name__ == "__main__":
    main()