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
                - first_leg_close, second_leg_close (цены закрытия)
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
        
        Примечание: Фильтры коинтеграции, Hurst и волатильности проверяются 
        в generate_signals() перед вызовом этой функции
        """
        if i == 0:
            return False
            
        current = self.data.iloc[i]
        
        # Z-score (спред слишком низкий, ожидаем возврат вверх)
        return current['z_score'] < -self.config.Z_SCORE_ENTRY
    
    def check_short_spread_entry(self, i):
        """
        Проверка условия входа в SHORT СПРЕД (Sell BTC, Buy ETH):
        - Z-score > +Z_SCORE_ENTRY (спред выше среднего, BTC переоценен относительно ETH)
        - Фильтр волатильности: σ_current >= VOLATILITY_THRESHOLD × σ_long
        
        Реальные позиции: SHORT BTC + LONG ETH
        
        Примечание: Фильтры коинтеграции, Hurst и волатильности проверяются 
        в generate_signals() перед вызовом этой функции
        """
        if i == 0:
            return False
            
        current = self.data.iloc[i]
        
        # Z-score (спред слишком высокий, ожидаем возврат вниз)
        return current['z_score'] > self.config.Z_SCORE_ENTRY
    
    def check_long_spread_exit(self, i):
        """
        Проверка условия выхода из LONG СПРЕДА:
        
        Режим 1 (EXIT_MODE = 1): Выход при возврате к нулю
        - Z-score пересекает 0 снизу вверх
        - Консервативный подход, есть периоды без позиций
        
        Режим 2 (EXIT_MODE = 2): Выход при пересечении противоположного порога
        - Z-score пересекает +Z_SCORE_ENTRY (например, +1.5) снизу вверх
        - Агрессивный подход, переворот в SHORT
        """
        if i == 0:
            return False
            
        current = self.data.iloc[i]
        previous = self.data.iloc[i-1]
        
        if self.config.EXIT_MODE == 1:
            # Режим 1: Выход при возврате к нулю
            return (previous['z_score'] < self.config.Z_SCORE_EXIT and 
                    current['z_score'] >= self.config.Z_SCORE_EXIT)
        
        elif self.config.EXIT_MODE == 2:
            # Режим 2: Выход при пересечении противоположного порога
            # LONG закрываем когда Z-score пересекает +1.5 СНИЗУ ВВЕРХ
            return (previous['z_score'] <= self.config.Z_SCORE_ENTRY and 
                    current['z_score'] > self.config.Z_SCORE_ENTRY)
        
        return False
    
    def check_short_spread_exit(self, i):
        """
        Проверка условия выхода из SHORT СПРЕДА:
        
        Режим 1 (EXIT_MODE = 1): Выход при возврате к нулю
        - Z-score пересекает 0 сверху вниз
        - Консервативный подход, есть периоды без позиций
        
        Режим 2 (EXIT_MODE = 2): Выход при пересечении противоположного порога
        - Z-score пересекает -Z_SCORE_ENTRY (например, -1.5) сверху вниз
        - Агрессивный подход, переворот в LONG
        """
        if i == 0:
            return False
            
        current = self.data.iloc[i]
        previous = self.data.iloc[i-1]
        
        if self.config.EXIT_MODE == 1:
            # Режим 1: Выход при возврате к нулю
            return (previous['z_score'] > self.config.Z_SCORE_EXIT and 
                    current['z_score'] <= self.config.Z_SCORE_EXIT)
        
        elif self.config.EXIT_MODE == 2:
            # Режим 2: Выход при пересечении противоположного порога
            # SHORT закрываем когда Z-score пересекает -1.5 СВЕРХУ ВНИЗ
            return (previous['z_score'] >= -self.config.Z_SCORE_ENTRY and 
                    current['z_score'] < -self.config.Z_SCORE_ENTRY)
        
        return False
    
    def generate_signals(self):
        """
        Генерирует торговые сигналы на всей истории данных с учётом всех фильтров из конфига
        """
        signals = []
        # Сброс состояния позиции перед генерацией сигналов
        self.position = 0
        self.entry_z_score = 0

        for i in range(len(self.data)):
            current = self.data.iloc[i]

            # Если нет позиции, проверяем входы (применяем фильтры)
            if self.position == 0:
                # Универсальная проверка всех фильтров (ТОЛЬКО ДЛЯ ВХОДОВ!)
                if 'can_trade' in current and not current['can_trade']:
                    continue
                if getattr(self.config, 'USE_ADF_TEST', False) and ('is_cointegrated' in current and not current['is_cointegrated']):
                    continue
                if getattr(self.config, 'USE_HURST_FILTER', False) and ('is_mean_reverting' in current and not current['is_mean_reverting']):
                    continue
                if getattr(self.config, 'USE_VOLATILITY_FILTER', False):
                    if 'sigma_current' in current and 'sigma_long' in current:
                        if current['sigma_current'] < self.config.VOLATILITY_THRESHOLD * current['sigma_long']:
                            continue
                if self.check_long_spread_entry(i):
                    signals.append({
                        'index': i,
                        'timestamp': self.data.index[i],
                        'type': 'LONG_SPREAD_ENTRY',
                        'z_score': current['z_score'],
                        'spread': current['spread'],
                        'first_leg_close': current['first_leg_close'],
                        'second_leg_close': current['second_leg_close'],
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
                        'first_leg_close': current['first_leg_close'],
                        'second_leg_close': current['second_leg_close'],
                        'beta': current.get('beta', None)
                    })
                    self.position = -1
                    self.entry_z_score = current['z_score']

            # Если есть long spread позиция, проверяем выход
            elif self.position == 1:
                # ЗАЩИТА: Принудительный выход если спред потерял коинтеграцию
                if (self.config.USE_COINTEGRATION_EXIT and 
                    'can_trade' in current and not current['can_trade']):
                    signals.append({
                        'index': i,
                        'timestamp': self.data.index[i],
                        'type': 'LONG_SPREAD_EXIT',
                        'z_score': current['z_score'],
                        'spread': current['spread'],
                        'first_leg_close': current['first_leg_close'],
                        'second_leg_close': current['second_leg_close'],
                        'entry_z_score': self.entry_z_score,
                        'beta': current.get('beta', None),
                        'exit_reason': 'COINTEGRATION_LOST'
                    })
                    self.position = 0
                    self.entry_z_score = 0
                # Обычный выход по условиям стратегии
                elif self.check_long_spread_exit(i):
                    signals.append({
                        'index': i,
                        'timestamp': self.data.index[i],
                        'type': 'LONG_SPREAD_EXIT',
                        'z_score': current['z_score'],
                        'spread': current['spread'],
                        'first_leg_close': current['first_leg_close'],
                        'second_leg_close': current['second_leg_close'],
                        'entry_z_score': self.entry_z_score,
                        'beta': current.get('beta', None),
                        'exit_reason': 'NORMAL'
                    })
                    
                    # EXIT_MODE = 2: Переворот позиции (LONG → SHORT)
                    # НО ТОЛЬКО если все фильтры пройдены!
                    if (self.config.EXIT_MODE == 2 and 
                        current['z_score'] > self.config.Z_SCORE_ENTRY and
                        current.get('can_trade', True)):  # Проверяем can_trade перед переворотом
                        # Сразу входим в SHORT после выхода из LONG
                        signals.append({
                            'index': i,
                            'timestamp': self.data.index[i],
                            'type': 'SHORT_SPREAD_ENTRY',
                            'z_score': current['z_score'],
                            'spread': current['spread'],
                            'first_leg_close': current['first_leg_close'],
                            'second_leg_close': current['second_leg_close'],
                            'beta': current.get('beta', None)
                        })
                        self.position = -1
                        self.entry_z_score = current['z_score']
                    else:
                        # EXIT_MODE = 1 или фильтры не пройдены: Просто закрываем позицию
                        self.position = 0
                        self.entry_z_score = 0

            # Если есть short spread позиция, проверяем выход
            elif self.position == -1:
                # ЗАЩИТА: Принудительный выход если спред потерял коинтеграцию
                if (self.config.USE_COINTEGRATION_EXIT and 
                    'can_trade' in current and not current['can_trade']):
                    signals.append({
                        'index': i,
                        'timestamp': self.data.index[i],
                        'type': 'SHORT_SPREAD_EXIT',
                        'z_score': current['z_score'],
                        'spread': current['spread'],
                        'first_leg_close': current['first_leg_close'],
                        'second_leg_close': current['second_leg_close'],
                        'entry_z_score': self.entry_z_score,
                        'beta': current.get('beta', None),
                        'exit_reason': 'COINTEGRATION_LOST'
                    })
                    self.position = 0
                    self.entry_z_score = 0
                # Обычный выход по условиям стратегии
                elif self.check_short_spread_exit(i):
                    signals.append({
                        'index': i,
                        'timestamp': self.data.index[i],
                        'type': 'SHORT_SPREAD_EXIT',
                        'z_score': current['z_score'],
                        'spread': current['spread'],
                        'first_leg_close': current['first_leg_close'],
                        'second_leg_close': current['second_leg_close'],
                        'entry_z_score': self.entry_z_score,
                        'beta': current.get('beta', None),
                        'exit_reason': 'NORMAL'
                    })
                    
                    # EXIT_MODE = 2: Переворот позиции (SHORT → LONG)
                    # НО ТОЛЬКО если все фильтры пройдены!
                    if (self.config.EXIT_MODE == 2 and 
                        current['z_score'] < -self.config.Z_SCORE_ENTRY and
                        current.get('can_trade', True)):  # Проверяем can_trade перед переворотом
                        # Сразу входим в LONG после выхода из SHORT
                        signals.append({
                            'index': i,
                            'timestamp': self.data.index[i],
                            'type': 'LONG_SPREAD_ENTRY',
                            'z_score': current['z_score'],
                            'spread': current['spread'],
                            'first_leg_close': current['first_leg_close'],
                            'second_leg_close': current['second_leg_close'],
                            'beta': current.get('beta', None)
                        })
                        self.position = 1
                        self.entry_z_score = current['z_score']
                    else:
                        # EXIT_MODE = 1 или фильтры не пройдены: Просто закрываем позицию
                        self.position = 0
                        self.entry_z_score = 0

        return signals
