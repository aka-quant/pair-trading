"""
Модуль расчета индикаторов для парной торговли
"""

import pandas as pd
import numpy as np
from typing import Tuple


class PairTradingIndicators:
    """Класс для расчета индикаторов стратегии парной торговли"""
    
    @staticmethod
    def calculate_delta(base_close: pd.Series, quote_close: pd.Series) -> pd.Series:
        """
        Расчет абсолютной дельты между активами
        
        Args:
            base_close: Цены закрытия базового актива
            quote_close: Цены закрытия котируемого актива
            
        Returns:
            Серия значений дельты (base/quote)
        """
        return base_close / quote_close
    
    @staticmethod
    def calculate_delta_percent(delta: pd.Series, len_long: int) -> pd.Series:
        """
        Расчет процентного отклонения дельты от долгосрочной MA
        
        Args:
            delta: Абсолютная дельта
            len_long: Период длинной скользящей средней
            
        Returns:
            Серия процентных отклонений дельты
        """
        sma_ref = delta.rolling(window=len_long).mean()
        delta_pct = (delta - sma_ref) / sma_ref * 100
        return delta_pct
    
    @staticmethod
    def calculate_moving_averages(delta_pct: pd.Series, 
                                  len_long: int, 
                                  len_short: int) -> Tuple[pd.Series, pd.Series]:
        """
        Расчет скользящих средних для дельты
        
        Args:
            delta_pct: Процентное отклонение дельты
            len_long: Период длинной MA
            len_short: Период короткой MA
            
        Returns:
            Tuple (long_ma, short_ma)
        """
        long_ma = delta_pct.rolling(window=len_long).mean()
        short_ma = delta_pct.rolling(window=len_short).mean()
        return long_ma, short_ma
    
    @staticmethod
    def calculate_bands(delta_pct: pd.Series, window: int = 1000) -> dict:
        """
        Расчет полос (медиана ± стандартное отклонение)
        
        Args:
            delta_pct: Процентное отклонение дельты
            window: Окно для расчета медианы и std
            
        Returns:
            Словарь с ключами: median, std, upper_1sigma, lower_1sigma, 
                               upper_2sigma, lower_2sigma
        """
        median = delta_pct.rolling(window=window).median()
        std = delta_pct.rolling(window=window).std()
        
        return {
            'median': median,
            'std': std,
            'upper_1sigma': median + std,
            'lower_1sigma': median - std,
            'upper_2sigma': median + 2 * std,
            'lower_2sigma': median - 2 * std,
        }
    
    @staticmethod
    def calculate_z_score(delta_pct: pd.Series, window: int = 1000) -> pd.Series:
        """
        Расчет z-score для дельты
        
        Args:
            delta_pct: Процентное отклонение дельты
            window: Окно для расчета
            
        Returns:
            Серия z-score значений
        """
        median = delta_pct.rolling(window=window).median()
        std = delta_pct.rolling(window=window).std()
        z_score = (delta_pct - median) / std
        return z_score
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame, 
                                 len_long: int = 100,
                                 len_short: int = 10,
                                 delta_window: int = 1000) -> pd.DataFrame:
        """
        Расчет всех индикаторов для датафрейма с данными пары
        
        Args:
            df: DataFrame с колонками base_close и quote_close
            len_long: Период длинной MA
            len_short: Период короткой MA
            delta_window: Окно для расчета статистики
            
        Returns:
            DataFrame с добавленными индикаторами
        """
        result = df.copy()
        
        # Абсолютная дельта
        result['delta'] = PairTradingIndicators.calculate_delta(
            df['base_close'], df['quote_close']
        )
        
        # Процентное отклонение дельты
        result['delta_pct'] = PairTradingIndicators.calculate_delta_percent(
            result['delta'], len_long
        )
        
        # Скользящие средние
        long_ma, short_ma = PairTradingIndicators.calculate_moving_averages(
            result['delta_pct'], len_long, len_short
        )
        result['ma_long'] = long_ma
        result['ma_short'] = short_ma
        
        # Полосы
        bands = PairTradingIndicators.calculate_bands(
            result['delta_pct'], delta_window
        )
        for key, value in bands.items():
            result[key] = value
        
        # Z-score
        result['z_score'] = PairTradingIndicators.calculate_z_score(
            result['delta_pct'], delta_window
        )
        
        return result
