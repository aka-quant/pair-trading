"""
Торговая стратегия парного трейдинга BTC/ETH
Реализация стратегии "Парная торговля 2.0" из статьи smart-lab.ru/mobile/topic/1265846/
"""

import numpy as np
import pandas as pd


class PairTradingStrategy:
    """
    Стратегия "Парная торговля 2.0" - mean reversion на основе Z-score спреда
    
    МАТЕМАТИКА СТРАТЕГИИ:
    1. Beta (β) = Cov(log(PA), log(PB)) / Var(log(PB)) - hedge ratio через OLS
    2. Спред = log(PA) - β × log(PB) - нормализованный спред
    3. Z-score = (Спред - μ) / σ - отклонение от среднего в сигмах
    
    ЛОГИКА ВХОДА:
    - Long спред (Buy BTC, Sell ETH): Z < -1.5 (актив BTC недооценен)
    - Short спред (Sell BTC, Buy ETH): Z > +1.5 (актив BTC переоценен)
    - Фильтр волатильности: σ_current >= 0.8 × σ_long
    
    ЛОГИКА ВЫХОДА:
    - Возврат Z-score к нулю (возврат к среднему)
    
    ПАРАМЕТРЫ:
    - Таймфрейм: 1H (адаптация от 4H из статьи)
    - Z-score порог: 1.5 (вместо классических 2.0)
    - Lookback для спреда: 30 баров
    - Rolling beta: 60 баров (опционально)
    """
    
    def __init__(self, data, config):
        """
        Инициализация стратегии
        
        Args:
            data: DataFrame с колонками:
                - btc_close, eth_close (цены закрытия)
                - spread (логарифмический спред)
                - z_score (Z-score спреда)
                - sigma_current, sigma_long (волатильности для фильтра)
            config: объект конфигурации с параметрами
        """
        self.data = data
        self.config = config
        self.position = 0  # 0 = нет позиции, 1 = long спред, -1 = short спред
        self.entry_price = 0
        self.entry_z_score = 0
        self.trades = []
        
    def check_long_spread_entry(self, i):
        """
        Проверка условия входа в LONG СПРЕД (Buy BTC, Sell ETH):
        - Z-score < -Z_SCORE_ENTRY (спред ниже среднего, BTC недооценен относительно ETH)
        - Фильтр волатильности: σ_current >= VOLATILITY_THRESHOLD × σ_long
        
        Реальные позиции: LONG BTC + SHORT ETH
        """
        if i == 0:
            return False
            
        current = self.data.iloc[i]
        
        # Проверка Z-score (спред слишком низкий, ожидаем возврат вверх)
        z_condition = current['z_score'] < -self.config.Z_SCORE_ENTRY
        
        # Проверка фильтра волатильности (входим только в "нервном" рынке)
        if self.config.USE_VOLATILITY_FILTER:
            vola_condition = current['sigma_current'] >= (
                self.config.VOLATILITY_THRESHOLD * current['sigma_long']
            )
        else:
            vola_condition = True
        
        return z_condition and vola_condition
    
    def check_short_spread_entry(self, i):
        """
        Проверка условия входа в SHORT СПРЕД (Sell BTC, Buy ETH):
        - Z-score > +Z_SCORE_ENTRY (спред выше среднего, BTC переоценен относительно ETH)
        - Фильтр волатильности: σ_current >= VOLATILITY_THRESHOLD × σ_long
        
        Реальные позиции: SHORT BTC + LONG ETH
        """
        if i == 0:
            return False
            
        current = self.data.iloc[i]
        
        # Проверка Z-score (спред слишком высокий, ожидаем возврат вниз)
        z_condition = current['z_score'] > self.config.Z_SCORE_ENTRY
        
        # Проверка фильтра волатильности
        if self.config.USE_VOLATILITY_FILTER:
            vola_condition = current['sigma_current'] >= (
                self.config.VOLATILITY_THRESHOLD * current['sigma_long']
            )
        else:
            vola_condition = True
        
        return z_condition and vola_condition
    
    def check_long_spread_exit(self, i):
        """
        Проверка условия выхода из LONG СПРЕДА:
        - Z-score возвращается к нулю или пересекает его (mean reversion завершен)
        - Альтернативно: Z-score пересекает противоположный порог (reversal)
        """
        if i == 0:
            return False
            
        current = self.data.iloc[i]
        previous = self.data.iloc[i-1]
        
        # Выход 1: Z-score вернулся к нулю (пересек снизу вверх)
        mean_reversion = (previous['z_score'] < self.config.Z_SCORE_EXIT and 
                         current['z_score'] >= self.config.Z_SCORE_EXIT)
        
        # Выход 2: Z-score пересек противоположный порог (слишком далеко ушел в другую сторону)
        opposite_threshold = current['z_score'] > self.config.Z_SCORE_ENTRY
        
        return mean_reversion or opposite_threshold
    
    def check_short_spread_exit(self, i):
        """
        Проверка условия выхода из SHORT СПРЕДА:
        - Z-score возвращается к нулю или пересекает его (mean reversion завершен)
        - Альтернативно: Z-score пересекает противоположный порог (reversal)
        """
        if i == 0:
            return False
            
        current = self.data.iloc[i]
        previous = self.data.iloc[i-1]
        
        # Выход 1: Z-score вернулся к нулю (пересек сверху вниз)
        mean_reversion = (previous['z_score'] > self.config.Z_SCORE_EXIT and 
                         current['z_score'] <= self.config.Z_SCORE_EXIT)
        
        # Выход 2: Z-score пересек противоположный порог
        opposite_threshold = current['z_score'] < -self.config.Z_SCORE_ENTRY
        
        return mean_reversion or opposite_threshold
    
    def generate_signals(self):
        """
        Генерирует торговые сигналы на всей истории данных
        
        Returns:
            list: список сигналов с информацией о типе, Z-score, спреде и индексе
        """
        signals = []
        
        for i in range(len(self.data)):
            current = self.data.iloc[i]
            
            # Если нет позиции, проверяем входы
            if self.position == 0:
                if self.check_long_spread_entry(i):
                    signals.append({
                        'index': i,
                        'timestamp': self.data.index[i],
                        'type': 'LONG_SPREAD_ENTRY',
                        'z_score': current['z_score'],
                        'spread': current['spread'],
                        'btc_close': current['btc_close'],
                        'eth_close': current['eth_close'],
                        'beta': current.get('beta', None)
                    })
                    self.position = 1
                    self.entry_z_score = current['z_score']
                    
                elif self.check_short_spread_entry(i):
                    signals.append({
                        'index': i,
                        'timestamp': self.data.index[i],
                        'type': 'SHORT_SPREAD_ENTRY',
                        'z_score': current['z_score'],
                        'spread': current['spread'],
                        'btc_close': current['btc_close'],
                        'eth_close': current['eth_close'],
                        'beta': current.get('beta', None)
                    })
                    self.position = -1
                    self.entry_z_score = current['z_score']
            
            # Если есть long spread позиция, проверяем выход
            elif self.position == 1:
                if self.check_long_spread_exit(i):
                    signals.append({
                        'index': i,
                        'timestamp': self.data.index[i],
                        'type': 'LONG_SPREAD_EXIT',
                        'z_score': current['z_score'],
                        'spread': current['spread'],
                        'btc_close': current['btc_close'],
                        'eth_close': current['eth_close'],
                        'entry_z_score': self.entry_z_score,
                        'beta': current.get('beta', None)
                    })
                    self.position = 0
                    self.entry_z_score = 0
            
            # Если есть short spread позиция, проверяем выход
            elif self.position == -1:
                if self.check_short_spread_exit(i):
                    signals.append({
                        'index': i,
                        'timestamp': self.data.index[i],
                        'type': 'SHORT_SPREAD_EXIT',
                        'z_score': current['z_score'],
                        'spread': current['spread'],
                        'btc_close': current['btc_close'],
                        'eth_close': current['eth_close'],
                        'entry_z_score': self.entry_z_score,
                        'beta': current.get('beta', None)
                    })
                    self.position = 0
                    self.entry_z_score = 0
        
        return signals
