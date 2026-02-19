"""
Модуль торговой стратегии парной торговли
"""

import pandas as pd
import numpy as np
from typing import Optional
from enum import Enum


class Position(Enum):
    """Типы позиций"""
    NONE = 0      # Нет позиции
    LONG = 1      # Длинная позиция (base актив переоценен, short base / long quote)
    SHORT = -1    # Короткая позиция (base актив недооценен, long base / short quote)


class PairTradingStrategy:
    """Стратегия парной торговли на основе z-score"""
    
    def __init__(self, 
                 entry_threshold: float = 2.0,
                 exit_threshold: float = 0.0,
                 position_size: float = 1.0,
                 maker_fee: float = 0.0002,
                 taker_fee: float = 0.0005):
        """
        Инициализация стратегии
        
        Args:
            entry_threshold: Порог входа (количество сигм)
            exit_threshold: Порог выхода
            position_size: Размер позиции (доля капитала)
            maker_fee: Комиссия мейкера
            taker_fee: Комиссия тейкера
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.position_size = position_size
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Генерация торговых сигналов на основе z-score
        
        Args:
            df: DataFrame с индикаторами (должен содержать колонку z_score)
            
        Returns:
            DataFrame с добавленными сигналами
        """
        result = df.copy()
        
        # Инициализация
        result['position'] = Position.NONE.value
        result['signal'] = 0  # 1 = открытие, -1 = закрытие
        
        current_position = Position.NONE
        
        for i in range(len(result)):
            z_score = result['z_score'].iloc[i]
            
            # Пропускаем NaN значения
            if pd.isna(z_score):
                result.loc[result.index[i], 'position'] = current_position.value
                continue
            
            # Логика входа в позицию
            if current_position == Position.NONE:
                # Вход в SHORT позицию (base переоценен)
                if z_score > self.entry_threshold:
                    current_position = Position.SHORT
                    result.loc[result.index[i], 'signal'] = 1
                
                # Вход в LONG позицию (base недооценен)
                elif z_score < -self.entry_threshold:
                    current_position = Position.LONG
                    result.loc[result.index[i], 'signal'] = 1
            
            # Логика выхода из позиции
            else:
                # Выход из SHORT позиции
                if current_position == Position.SHORT and z_score <= self.exit_threshold:
                    result.loc[result.index[i], 'signal'] = -1
                    current_position = Position.NONE
                
                # Выход из LONG позиции
                elif current_position == Position.LONG and z_score >= self.exit_threshold:
                    result.loc[result.index[i], 'signal'] = -1
                    current_position = Position.NONE
                
                # Реверс позиции (опционально - можно включить позже)
                # elif current_position == Position.SHORT and z_score < -self.entry_threshold:
                #     result.loc[result.index[i], 'signal'] = -1  # закрыть SHORT
                #     current_position = Position.LONG
                #     # На следующей итерации откроем LONG
            
            result.loc[result.index[i], 'position'] = current_position.value
        
        return result
    
    def calculate_returns(self, df: pd.DataFrame, initial_capital: float = 10000.0) -> pd.DataFrame:
        """
        Расчет доходности стратегии
        
        Args:
            df: DataFrame с сигналами и позициями
            initial_capital: Начальный капитал
            
        Returns:
            DataFrame с расчетами доходности
        """
        result = df.copy()
        
        # Расчет изменения цен
        result['base_return'] = result['base_close'].pct_change()
        result['quote_return'] = result['quote_close'].pct_change()
        
        # Расчет доходности пары (спред)
        # При LONG позиции: покупаем base, продаем quote
        # При SHORT позиции: продаем base, покупаем quote
        result['pair_return'] = result['base_return'] - result['quote_return']
        
        # Доходность стратегии
        result['strategy_return'] = result['position'].shift(1) * result['pair_return']
        
        # Учет комиссий при смене позиции
        result['commission'] = 0.0
        position_changes = result['signal'] != 0
        result.loc[position_changes, 'commission'] = self.taker_fee * 2  # Открытие обеих ног
        
        # Чистая доходность
        result['net_return'] = result['strategy_return'] - result['commission']
        
        # Кумулятивная доходность
        result['cumulative_return'] = (1 + result['net_return'].fillna(0)).cumprod()
        
        # Капитал
        result['equity'] = initial_capital * result['cumulative_return']
        
        # Drawdown
        result['peak'] = result['equity'].cummax()
        result['drawdown'] = (result['equity'] - result['peak']) / result['peak']
        
        return result
    
    def get_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Извлечение списка всех сделок
        
        Args:
            df: DataFrame с сигналами
            
        Returns:
            DataFrame со сделками
        """
        trades = []
        entry_idx = None
        entry_position = None
        
        for i in range(len(df)):
            signal = df['signal'].iloc[i]
            position = df['position'].iloc[i]
            
            # Открытие позиции
            if signal == 1 and entry_idx is None:
                entry_idx = i
                entry_position = position
            
            # Закрытие позиции
            elif signal == -1 and entry_idx is not None:
                exit_idx = i
                
                trade = {
                    'entry_time': df.index[entry_idx],
                    'exit_time': df.index[exit_idx],
                    'entry_z_score': df['z_score'].iloc[entry_idx],
                    'exit_z_score': df['z_score'].iloc[exit_idx],
                    'position': entry_position,
                    'base_entry_price': df['base_close'].iloc[entry_idx],
                    'base_exit_price': df['base_close'].iloc[exit_idx],
                    'quote_entry_price': df['quote_close'].iloc[entry_idx],
                    'quote_exit_price': df['quote_close'].iloc[exit_idx],
                    'duration': df.index[exit_idx] - df.index[entry_idx],
                }
                
                # Расчет P&L
                if entry_position == Position.LONG.value:
                    # LONG: покупка base, продажа quote
                    trade['return'] = (
                        (trade['base_exit_price'] / trade['base_entry_price'] - 1) -
                        (trade['quote_exit_price'] / trade['quote_entry_price'] - 1)
                    )
                elif entry_position == Position.SHORT.value:
                    # SHORT: продажа base, покупка quote
                    trade['return'] = (
                        (trade['quote_exit_price'] / trade['quote_entry_price'] - 1) -
                        (trade['base_exit_price'] / trade['base_entry_price'] - 1)
                    )
                
                # Учет комиссий
                trade['net_return'] = trade['return'] - (self.taker_fee * 2)
                
                trades.append(trade)
                
                entry_idx = None
                entry_position = None
        
        return pd.DataFrame(trades)
