"""
Модуль для загрузки данных с биржи OKX
"""

import os
import ccxt
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
from dotenv import load_dotenv

load_dotenv()


class DataLoader:
    """Класс для загрузки и кэширования данных с биржи OKX"""
    
    def __init__(self, cache_dir: str = "data/cache", use_cache: bool = True):
        """
        Инициализация загрузчика данных
        
        Args:
            cache_dir: Директория для кэширования данных
            use_cache: Использовать ли кэшированные данные
        """
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Инициализация подключения к OKX
        self.exchange = ccxt.okx({
            'apiKey': os.getenv('OKX_API_KEY', ''),
            'secret': os.getenv('OKX_SECRET', ''),
            'password': os.getenv('OKX_PASSWORD', ''),
            'enableRateLimit': True,
        })
    
    def _get_cache_filename(self, symbol: str, timeframe: str, 
                           start_date: str, end_date: str) -> Path:
        """Генерация имени файла кэша"""
        symbol_clean = symbol.replace('/', '_').replace(':', '_')
        start_clean = start_date.replace(' ', '_').replace(':', '-')
        end_clean = end_date.replace(' ', '_').replace(':', '-')
        filename = f"{symbol_clean}_{timeframe}_{start_clean}_{end_clean}.csv"
        return self.cache_dir / filename
    
    def fetch_ohlcv(self, symbol: str, timeframe: str, 
                     start_date: str, end_date: str,
                     force_reload: bool = False) -> pd.DataFrame:
        """
        Загрузка OHLCV данных с биржи OKX
        
        Args:
            symbol: Торговая пара (например, "BTC/USDT:USDT")
            timeframe: Таймфрейм (например, "1h")
            start_date: Дата начала в формате "YYYY-MM-DD HH:MM:SS"
            end_date: Дата окончания в формате "YYYY-MM-DD HH:MM:SS"
            force_reload: Принудительная загрузка без использования кэша
            
        Returns:
            DataFrame с колонками: timestamp, open, high, low, close, volume
        """
        cache_file = self._get_cache_filename(symbol, timeframe, start_date, end_date)
        
        # Проверка кэша
        if self.use_cache and not force_reload and cache_file.exists():
            print(f"Загрузка данных из кэша: {cache_file}")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return df
        
        # Конвертация дат в timestamp
        since = int(datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
        until = int(datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
        
        print(f"Загрузка данных {symbol} с {start_date} по {end_date}...")
        
        # Загрузка данных порциями
        all_ohlcv = []
        current_since = since
        
        while current_since < until:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=1000  # Максимальный лимит за один запрос
                )
                
                if not ohlcv:
                    break
                
                all_ohlcv.extend(ohlcv)
                current_since = ohlcv[-1][0] + 1
                
                # Если достигли конца периода
                if ohlcv[-1][0] >= until:
                    break
                    
                print(f"Загружено {len(all_ohlcv)} свечей...")
                
            except Exception as e:
                print(f"Ошибка при загрузке данных: {e}")
                break
        
        # Создание DataFrame
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Фильтрация по датам
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        # Сохранение в кэш
        if self.use_cache:
            df.to_csv(cache_file)
            print(f"Данные сохранены в кэш: {cache_file}")
        
        print(f"Загружено {len(df)} свечей для {symbol}")
        return df
    
    def fetch_pair_data(self, base_symbol: str, quote_symbol: str,
                       timeframe: str, start_date: str, end_date: str,
                       force_reload: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Загрузка данных для пары активов
        
        Args:
            base_symbol: Базовый актив
            quote_symbol: Котируемый актив
            timeframe: Таймфрейм
            start_date: Дата начала
            end_date: Дата окончания
            force_reload: Принудительная перезагрузка
            
        Returns:
            Tuple с DataFrame для базового и котируемого актива
        """
        base_df = self.fetch_ohlcv(base_symbol, timeframe, start_date, end_date, force_reload)
        quote_df = self.fetch_ohlcv(quote_symbol, timeframe, start_date, end_date, force_reload)
        
        # Синхронизация по времени (inner join)
        base_df = base_df.add_prefix('base_')
        quote_df = quote_df.add_prefix('quote_')
        
        merged = pd.concat([base_df, quote_df], axis=1).dropna()
        
        print(f"\nСинхронизированные данные: {len(merged)} свечей")
        
        return merged
