"""
Торговая стратегия парного трейдинга BTC/ETH
"""

class PairTradingStrategy:
    """
    Стратегия парного трейдинга BTC/ETH с mean reversion
    
    ЗАЩИТА ОТ МНОЖЕСТВЕННЫХ ПОЗИЦИЙ:
    - self.position отслеживает текущую позицию (0 = нет, 1 = лонг, -1 = шорт)
    - Новые позиции открываются ТОЛЬКО когда self.position == 0
    - Одновременно может быть открыта только ОДНА позиция до момента закрытия
    
    ЛОГИКА ВХОДА/ВЫХОДА:
    - Вход: пересечение ±2σ на 15m (точная точка входа) + подтверждение на 1H
    - Выход: пересечение ±2σ на 1H (основной сигнал выхода) + подтверждение на 15m
    
    Позиции по инструментам:
    - LONG по стратегии = SHORT BTC + LONG ETH (ставка на снижение Delta)
    - SHORT по стратегии = LONG BTC + SHORT ETH (ставка на рост Delta)
    """
    
    def __init__(self, data, config):
        """
        Инициализация стратегии
        
        Args:
            data: DataFrame с колонками delta_pct_15m, upper_2sigma_15m, lower_2sigma_15m,
                  delta_pct_1h, upper_2sigma_1h, lower_2sigma_1h
            config: объект конфигурации с параметрами
        """
        self.data = data
        self.config = config
        self.position = 0  # 0 = нет позиции, 1 = лонг, -1 = шорт
        self.entry_price = 0
        self.trades = []
        
    def check_long_entry(self, i):
        """
        Проверка условия входа в лонг:
        - Delta_pct пересекает +2σ на 15m (снизу вверх) - для точки входа
        - Delta_pct находится выше +2σ на 1H - подтверждение тренда
        
        Реальные позиции: SHORT BTC + LONG ETH
        """
        if i == 0:
            return False
            
        current = self.data.iloc[i]
        previous = self.data.iloc[i-1]
        
        # Пересечение +2σ на 15m (снизу вверх) - точка входа
        cross_up_15m = (previous['delta_pct_15m'] <= previous['upper_2sigma_15m'] and 
                        current['delta_pct_15m'] > current['upper_2sigma_15m'])
        
        # Выше +2σ на 1H - подтверждение
        above_1h = current['delta_pct_1h'] > current['upper_2sigma_1h']
        
        return cross_up_15m and above_1h
    
    def check_short_entry(self, i):
        """
        Проверка условия входа в шорт:
        - Delta_pct пересекает -2σ на 15m (сверху вниз) - для точки входа
        - Delta_pct находится ниже -2σ на 1H - подтверждение тренда
        
        Реальные позиции: LONG BTC + SHORT ETH
        """
        if i == 0:
            return False
            
        current = self.data.iloc[i]
        previous = self.data.iloc[i-1]
        
        # Пересечение -2σ на 15m (сверху вниз) - точка входа
        cross_down_15m = (previous['delta_pct_15m'] >= previous['lower_2sigma_15m'] and 
                          current['delta_pct_15m'] < current['lower_2sigma_15m'])
        
        # Ниже -2σ на 1H - подтверждение
        below_1h = current['delta_pct_1h'] < current['lower_2sigma_1h']
        
        return cross_down_15m and below_1h
    
    def check_long_exit(self, i):
        """
        Проверка условия выхода из лонга:
        - Delta_pct пересекает -2σ на 1H (сверху вниз) - выход по часовому ТФ
        - Delta_pct находится ниже -2σ на 15m - подтверждение
        """
        if i == 0:
            return False
            
        current = self.data.iloc[i]
        previous = self.data.iloc[i-1]
        
        # Пересечение -2σ на 1H (сверху вниз) - основной сигнал выхода
        cross_down_1h = (previous['delta_pct_1h'] >= previous['lower_2sigma_1h'] and 
                         current['delta_pct_1h'] < current['lower_2sigma_1h'])
        
        # Ниже -2σ на 15m - подтверждение
        below_15m = current['delta_pct_15m'] < current['lower_2sigma_15m']
        
        return cross_down_1h and below_15m
    
    def check_short_exit(self, i):
        """
        Проверка условия выхода из шорта:
        - Delta_pct пересекает +2σ на 1H (снизу вверх) - выход по часовому ТФ
        - Delta_pct находится выше +2σ на 15m - подтверждение
        """
        if i == 0:
            return False
            
        current = self.data.iloc[i]
        previous = self.data.iloc[i-1]
        
        # Пересечение +2σ на 1H (снизу вверх) - основной сигнал выхода
        cross_up_1h = (previous['delta_pct_1h'] <= previous['upper_2sigma_1h'] and 
                       current['delta_pct_1h'] > current['upper_2sigma_1h'])
        
        # Выше +2σ на 15m - подтверждение
        above_15m = current['delta_pct_15m'] > current['upper_2sigma_15m']
        
        return cross_up_1h and above_15m
    
    def generate_signals(self):
        """
        Генерирует торговые сигналы на всей истории данных
        
        Returns:
            list: список сигналов с информацией о типе, цене и индексе
        """
        signals = []
        
        for i in range(len(self.data)):
            # Если нет позиции, проверяем входы
            if self.position == 0:
                if self.check_long_entry(i):
                    signals.append({
                        'index': i,
                        'timestamp': self.data.index[i],
                        'type': 'LONG ENTRY',
                        'price': self.data.iloc[i]['delta_pct_15m']
                    })
                    self.position = 1
                    self.entry_price = self.data.iloc[i]['delta_pct_15m']
                    
                elif self.check_short_entry(i):
                    signals.append({
                        'index': i,
                        'timestamp': self.data.index[i],
                        'type': 'SHORT ENTRY',
                        'price': self.data.iloc[i]['delta_pct_15m']
                    })
                    self.position = -1
                    self.entry_price = self.data.iloc[i]['delta_pct_15m']
            
            # Если есть лонг позиция, проверяем выход
            elif self.position == 1:
                if self.check_long_exit(i):
                    signals.append({
                        'index': i,
                        'timestamp': self.data.index[i],
                        'type': 'LONG EXIT',
                        'price': self.data.iloc[i]['delta_pct_15m'],
                        'entry_price': self.entry_price
                    })
                    self.position = 0
                    self.entry_price = 0
            
            # Если есть шорт позиция, проверяем выход
            elif self.position == -1:
                if self.check_short_exit(i):
                    signals.append({
                        'index': i,
                        'timestamp': self.data.index[i],
                        'type': 'SHORT EXIT',
                        'price': self.data.iloc[i]['delta_pct_15m'],
                        'entry_price': self.entry_price
                    })
                    self.position = 0
                    self.entry_price = 0
        
        return signals
