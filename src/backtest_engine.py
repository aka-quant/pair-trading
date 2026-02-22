"""
Движок бэктестирования торговых стратегий
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
import matplotlib.dates as mdates


class BacktestEngine:
    """
    Универсальный движок для бэктестирования торговых стратегий
    """
    
    def __init__(self, config):
        """
        Инициализация движка бэктестирования
        
        Args:
            config: объект конфигурации с параметрами
        """
        self.config = config
        self.data = None
        self.btc_15m = None
        self.eth_15m = None
        self.trades = []
        self.equity_curve = []
        
    def load_data(self):
        """Загрузка и подготовка данных"""
        print("Загрузка данных...")
        
        # BTC данные
        btc_15m = pd.read_csv(self.config.BTC_15M_FILE)
        btc_1h = pd.read_csv(self.config.BTC_1H_FILE)
        
        # ETH данные
        eth_15m = pd.read_csv(self.config.ETH_15M_FILE)
        eth_1h = pd.read_csv(self.config.ETH_1H_FILE)
        
        # Конвертируем timestamp в datetime
        for df in [btc_15m, btc_1h, eth_15m, eth_1h]:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        print(f"BTC 15m: {len(btc_15m)} свечей")
        print(f"BTC 1H: {len(btc_1h)} свечей")
        print(f"ETH 15m: {len(eth_15m)} свечей")
        print(f"ETH 1H: {len(eth_1h)} свечей")
        
        return btc_15m, btc_1h, eth_15m, eth_1h
    
    def calculate_indicators(self, btc_15m, btc_1h, eth_15m, eth_1h):
        """
        Расчет индикаторов для обоих таймфреймов
        
        Returns:
            DataFrame с индикаторами и данными для бэктеста
        """
        print("\nРасчет индикаторов...")
        
        # 15-минутный таймфрейм
        indicators_15m = self._calculate_delta_indicators(
            btc_15m, eth_15m,
            self.config.LEN_LONG,
            self.config.LEN_SHORT,
            self.config.MEDIAN_PERIOD
        )
        
        # Часовой таймфрейм
        indicators_1h = self._calculate_delta_indicators(
            btc_1h, eth_1h,
            self.config.LEN_LONG,
            self.config.LEN_SHORT,
            self.config.MEDIAN_PERIOD
        )
        
        # Объединяем данные 1H в 15m (forward fill)
        indicators_1h_resampled = indicators_1h.reindex(indicators_15m.index, method='ffill')
        
        # Объединяем в один датафрейм
        data = pd.DataFrame(index=indicators_15m.index)
        
        # Абсолютная дельта (BTC/ETH)
        data['delta_abs_15m'] = indicators_15m['delta']
        data['delta_abs_1h'] = indicators_1h_resampled['delta']
        
        # Относительная дельта (%)
        data['delta_pct_15m'] = indicators_15m['delta_pct']
        data['upper_2sigma_15m'] = indicators_15m['upper_2sigma']
        data['lower_2sigma_15m'] = indicators_15m['lower_2sigma']
        
        data['delta_pct_1h'] = indicators_1h_resampled['delta_pct']
        data['upper_2sigma_1h'] = indicators_1h_resampled['upper_2sigma']
        data['lower_2sigma_1h'] = indicators_1h_resampled['lower_2sigma']
        
        # Удаляем NaN значения
        data = data.dropna()
        
        print(f"Данные для бэктеста: {len(data)} свечей")
        print(f"Период: с {data.index[0]} по {data.index[-1]}")
        
        return data
    
    def _calculate_delta_indicators(self, btc_df, eth_df, len_long, len_short, median_period):
        """
        Рассчитывает delta % и полосы ±2σ на основе индикаторов из TradingView
        """
        df = pd.DataFrame(index=btc_df.index)
        
        # Delta Calculation (абсолютная)
        delta = btc_df['close'] / eth_df['close']
        
        # SMA Reference (долгосрочная MA для дельты)
        sma_ref = delta.rolling(window=len_long, min_periods=len_long).mean()
        
        # Delta % (относительное отклонение от долгосрочной MA)
        delta_pct = (delta - sma_ref) / sma_ref * 100
        
        # Moving Averages на delta_pct
        sma_long = delta_pct.rolling(window=len_long, min_periods=len_long).mean()
        sma_short = delta_pct.rolling(window=len_short, min_periods=len_short).mean()
        
        # Median and StdDev
        median_val = delta_pct.rolling(window=median_period, min_periods=median_period).median()
        std_dev_val = delta_pct.rolling(window=median_period, min_periods=median_period).std()
        
        # Bands ±2σ
        upper_band_2 = median_val + self.config.SIGMA_THRESHOLD * std_dev_val
        lower_band_2 = median_val - self.config.SIGMA_THRESHOLD * std_dev_val
        
        # Сохраняем результаты
        df['delta'] = delta
        df['delta_pct'] = delta_pct
        df['sma_long'] = sma_long
        df['sma_short'] = sma_short
        df['median'] = median_val
        df['std_dev'] = std_dev_val
        df['upper_2sigma'] = upper_band_2
        df['lower_2sigma'] = lower_band_2
        
        return df
    
    def run_backtest(self, strategy):
        """
        Запуск бэктеста с заданной стратегией
        
        Args:
            strategy: объект стратегии с методом generate_signals()
        """
        print("\nЗапуск бэктеста...")
        
        # Генерируем сигналы
        signals = strategy.generate_signals()
        
        # Симулируем торговлю
        equity = self.config.INITIAL_CAPITAL
        position_open = False
        entry_data = None
        
        for signal in signals:
            # Защита от отрицательного капитала
            if equity <= 0:
                print(f"\n⚠️ ВНИМАНИЕ: Капитал достиг нуля или отрицательного значения!")
                print(f"Эмуляция остановлена на {signal['timestamp']}")
                break
            
            if 'ENTRY' in signal['type']:
                # Открытие позиции
                timestamp = signal['timestamp']
                
                # Получаем цены BTC и ETH на момент входа
                btc_price_entry = self.btc_15m.loc[timestamp, 'close']
                eth_price_entry = self.eth_15m.loc[timestamp, 'close']
                
                # Размер позиции = весь текущий капитал, делим пополам на каждую ногу
                position_size_usd = equity
                
                # Сохраняем данные входа
                entry_data = {
                    'timestamp_entry': timestamp,
                    'type_entry': signal['type'],
                    'delta_pct_entry': signal['price'],
                    'btc_price_entry': btc_price_entry,
                    'eth_price_entry': eth_price_entry,
                    'position_size_usd': position_size_usd,
                    'equity_before': equity
                }
                position_open = True
                
            elif 'EXIT' in signal['type'] and position_open:
                # Закрытие позиции
                timestamp = signal['timestamp']
                
                # Получаем цены BTC и ETH на момент выхода
                btc_price_exit = self.btc_15m.loc[timestamp, 'close']
                eth_price_exit = self.eth_15m.loc[timestamp, 'close']
                
                # Определяем тип сделки
                is_long = 'LONG' in signal['type']
                
                if is_long:
                    # LONG по стратегии = SHORT BTC + LONG ETH
                    btc_pnl_pct = (entry_data['btc_price_entry'] - btc_price_exit) / entry_data['btc_price_entry'] * 100  # SHORT BTC
                    eth_pnl_pct = (eth_price_exit - entry_data['eth_price_entry']) / entry_data['eth_price_entry'] * 100  # LONG ETH
                else:  # SHORT
                    # SHORT по стратегии = LONG BTC + SHORT ETH
                    btc_pnl_pct = (btc_price_exit - entry_data['btc_price_entry']) / entry_data['btc_price_entry'] * 100  # LONG BTC
                    eth_pnl_pct = (entry_data['eth_price_entry'] - eth_price_exit) / entry_data['eth_price_entry'] * 100  # SHORT ETH
                
                # Общий P&L в процентах (среднее от двух ног, так как капитал делится пополам)
                total_pnl_pct = btc_pnl_pct + eth_pnl_pct
                
                # Комиссия в процентах (применяется к общему P&L)
                commission_pct = abs(total_pnl_pct) * self.config.COMMISSION_RATE / 100
                
                # Чистый P&L в процентах после комиссии
                net_pnl_pct = total_pnl_pct - commission_pct
                
                # P&L в долларах (применяем к размеру позиции)
                pnl_usd = entry_data['position_size_usd'] * net_pnl_pct / 100
                
                # Обновляем капитал
                equity = entry_data['equity_before'] + pnl_usd
                
                # Записываем сделку о ВХОДЕ (с timestamp входа) - без результатов
                entry_record = {
                    'timestamp': entry_data['timestamp_entry'],  # Время ВХОДА
                    'type': 'LONG_ENTRY' if is_long else 'SHORT_ENTRY',
                    'delta_pct': entry_data['delta_pct_entry'],
                    'btc_price': entry_data['btc_price_entry'],
                    'eth_price': entry_data['eth_price_entry'],
                    'position_size_usd': entry_data['position_size_usd'],
                    'equity_before': entry_data['equity_before'],
                    'btc_pnl_%': None,
                    'eth_pnl_%': None,
                    'total_pnl_%': None,
                    'commission_%': None,
                    'net_pnl_%': None,
                    'pnl_usd': None,
                    'equity_after': None
                }
                
                # Добавляем запись о входе
                self.trades.append(entry_record)
                
                # Добавляем запись о ВЫХОДЕ (с timestamp выхода) - С РЕЗУЛЬТАТАМИ
                exit_record = {
                    'timestamp': timestamp,  # Время ВЫХОДА
                    'type': 'LONG_EXIT' if is_long else 'SHORT_EXIT',
                    'delta_pct': signal['price'],
                    'btc_price': btc_price_exit,
                    'eth_price': eth_price_exit,
                    'position_size_usd': entry_data['position_size_usd'],
                    'equity_before': entry_data['equity_before'],
                    'btc_pnl_%': btc_pnl_pct,
                    'eth_pnl_%': eth_pnl_pct,
                    'total_pnl_%': total_pnl_pct,
                    'commission_%': commission_pct,
                    'net_pnl_%': net_pnl_pct,
                    'pnl_usd': pnl_usd,
                    'equity_after': equity
                }
                self.trades.append(exit_record)
                
                position_open = False
                entry_data = None
            
            # Сохраняем equity после каждого сигнала
            self.equity_curve.append({
                'timestamp': signal['timestamp'],
                'equity': equity
            })
        
        return self.trades, self.equity_curve
    
    def print_statistics(self):
        """Печать статистики бэктеста"""
        print(f"\n{'='*60}")
        print("РЕЗУЛЬТАТЫ БЭКТЕСТА")
        print(f"{'='*60}")
        
        if len(self.trades) == 0:
            print("Сделок не найдено!")
            return
        
        trades_df = pd.DataFrame(self.trades)
        
        # Фильтруем только записи ВЫХОДОВ (они содержат полную информацию о результатах сделки)
        exit_trades = trades_df[trades_df['type'].str.contains('EXIT')].copy()
        
        if len(exit_trades) == 0:
            print("Закрытых сделок не найдено!")
            return
        
        print(f"\nВсего закрытых сделок: {len(exit_trades)}")
        print(f"Лонг сделок: {len(exit_trades[exit_trades['type'] == 'LONG_EXIT'])}")
        print(f"Шорт сделок: {len(exit_trades[exit_trades['type'] == 'SHORT_EXIT'])}")
        
        # Статистика по сделкам
        profitable = len(exit_trades[exit_trades['net_pnl_%'] > 0])
        losing = len(exit_trades[exit_trades['net_pnl_%'] < 0])
        
        print(f"\nПрибыльных: {profitable}")
        print(f"Убыточных: {losing}")
        print(f"Винрейт: {profitable / len(exit_trades) * 100:.2f}%")
        
        print(f"\n{'='*60}")
        print("СТАТИСТИКА ПО НОГАМ:")
        print(f"{'='*60}")
        
        print(f"\nBTC нога:")
        print(f"  Средний P&L: {exit_trades['btc_pnl_%'].mean():.4f}%")
        print(f"  Максимальный профит: {exit_trades['btc_pnl_%'].max():.4f}%")
        print(f"  Максимальный убыток: {exit_trades['btc_pnl_%'].min():.4f}%")
        
        print(f"\nETH нога:")
        print(f"  Средний P&L: {exit_trades['eth_pnl_%'].mean():.4f}%")
        print(f"  Максимальный профит: {exit_trades['eth_pnl_%'].max():.4f}%")
        print(f"  Максимальный убыток: {exit_trades['eth_pnl_%'].min():.4f}%")
        
        print(f"\nОбщая статистика:")
        print(f"  Средний общий P&L: {exit_trades['total_pnl_%'].mean():.4f}%")
        print(f"  Средний чистый P&L: {exit_trades['net_pnl_%'].mean():.4f}%")
        print(f"  Максимальный профит: {exit_trades['net_pnl_%'].max():.4f}%")
        print(f"  Максимальный убыток: {exit_trades['net_pnl_%'].min():.4f}%")
        
        if self.config.COMMISSION_RATE > 0:
            total_commission = exit_trades['commission_%'].sum()
            print(f"\n  Общая комиссия: {total_commission:.4f}%")
        
        print(f"\n  Средний P&L в USD: {exit_trades['pnl_usd'].mean():.2f}")
        print(f"  Общий P&L в USD: {exit_trades['pnl_usd'].sum():.2f}")
        
        equity_curve_df = pd.DataFrame(self.equity_curve)
        final_equity = equity_curve_df['equity'].iloc[-1]
        
        print(f"\n{'='*60}")
        print("КАПИТАЛ:")
        print(f"{'='*60}")
        print(f"Начальный капитал: ${self.config.INITIAL_CAPITAL:,.2f}")
        print(f"Финальный капитал: ${final_equity:,.2f}")
        print(f"Общий доход: ${final_equity - self.config.INITIAL_CAPITAL:,.2f}")
        print(f"Доходность: {(final_equity - self.config.INITIAL_CAPITAL) / self.config.INITIAL_CAPITAL * 100:.2f}%")
        
        # Максимальная просадка
        equity_array = equity_curve_df['equity'].values
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        print(f"Максимальная просадка: {max_drawdown:.2f}%")
        
        # Первые 5 сделок
        print(f"\n{'='*60}")
        print("ПЕРВЫЕ 5 СДЕЛОК:")
        print(f"{'='*60}")
        # Выводим только ключевые столбцы для читаемости
        display_cols = ['timestamp', 'type', 'btc_pnl_%', 'eth_pnl_%', 'net_pnl_%', 'pnl_usd', 'equity_after']
        print(exit_trades[display_cols].head(5).to_string(index=False))
    
    def save_results(self):
        """Сохранение результатов в файлы"""
        if self.config.SAVE_TRADES_LOG and len(self.trades) > 0:
            trades_df = pd.DataFrame(self.trades)
            # Сохраняем в формате Excel - данные будут сразу в столбцах при открытии
            trades_df.to_excel(self.config.TRADES_LOG_FILE, index=False, engine='openpyxl')
            print(f"\nСделки сохранены в '{self.config.TRADES_LOG_FILE}'")
    
    def plot_results(self):
        """Визуализация результатов бэктеста"""
        if len(self.trades) == 0:
            print("Нет данных для визуализации")
            return
        
        print("\nСоздание графиков...")
        
        # Создаем фигуру с общей осью X и дополнительным пространством для виджетов
        fig = plt.figure(figsize=(18, 20))
        
        # Основные графики (увеличено до 6 графиков)
        gs = fig.add_gridspec(7, 1, height_ratios=[1, 1, 1, 1, 1, 1, 0.3], hspace=0.3, top=0.95, bottom=0.08)
        ax1 = fig.add_subplot(gs[0])  # Delta Absolute 15m
        ax2 = fig.add_subplot(gs[1], sharex=ax1)  # Delta Absolute 1H
        ax3 = fig.add_subplot(gs[2], sharex=ax1)  # Delta % 15m
        ax4 = fig.add_subplot(gs[3], sharex=ax1)  # Delta % 1H
        ax5 = fig.add_subplot(gs[4], sharex=ax1)  # Equity curve
        ax6 = fig.add_subplot(gs[5], sharex=ax1)  # Reserved
        
        axes = [ax1, ax2, ax3, ax4, ax5, ax6]
        
        # Сохраняем исходные границы для кнопки Reset
        self.original_xlim = None
        
        # График 1: Абсолютная Delta 15m (с MA)
        ax1.plot(self.data.index, self.data['delta_abs_15m'], 
                label='Delta Absolute (15m)', linewidth=1.5, color='purple', alpha=0.8)
        
        # Добавляем скользящие средние для 15m
        sma_short_15m = self.data['delta_abs_15m'].rolling(window=self.config.LEN_SHORT, min_periods=self.config.LEN_SHORT).mean()
        sma_long_15m = self.data['delta_abs_15m'].rolling(window=self.config.LEN_LONG, min_periods=self.config.LEN_LONG).mean()
        
        ax1.plot(self.data.index, sma_short_15m, 
                label=f'SMA {self.config.LEN_SHORT} (15m)', linewidth=1.5, color='red', alpha=0.7)
        ax1.plot(self.data.index, sma_long_15m, 
                label=f'SMA {self.config.LEN_LONG} (15m)', linewidth=1.5, color='blue', alpha=0.7)
        
        ax1.set_title('Абсолютная Delta (15 минут) с скользящими средними', fontsize=13, fontweight='bold', pad=12)
        ax1.set_ylabel('Delta (BTC/ETH)', fontsize=10)
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='y', labelsize=9)
        
        # График 2: Абсолютная Delta 1H (с MA)
        ax2.plot(self.data.index, self.data['delta_abs_1h'], 
                label='Delta Absolute (1H)', linewidth=1.5, color='darkblue', alpha=0.8)
        
        # Добавляем скользящие средние для 1H
        sma_short_1h = self.data['delta_abs_1h'].rolling(window=self.config.LEN_SHORT, min_periods=self.config.LEN_SHORT).mean()
        sma_long_1h = self.data['delta_abs_1h'].rolling(window=self.config.LEN_LONG, min_periods=self.config.LEN_LONG).mean()
        
        ax2.plot(self.data.index, sma_short_1h, 
                label=f'SMA {self.config.LEN_SHORT} (1H)', linewidth=1.5, color='red', alpha=0.7)
        ax2.plot(self.data.index, sma_long_1h, 
                label=f'SMA {self.config.LEN_LONG} (1H)', linewidth=1.5, color='blue', alpha=0.7)
        
        ax2.set_title('Абсолютная Delta (1 час) с скользящими средними', fontsize=13, fontweight='bold', pad=12)
        ax2.set_ylabel('Delta (BTC/ETH)', fontsize=10)
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='y', labelsize=9)
        
        # График 3: Delta % и полосы на 15m (ВСЕ точки входа И выхода)
        ax3.plot(self.data.index, self.data['delta_pct_15m'], 
                label='Delta % (15m)', linewidth=1, color='orange')
        ax3.plot(self.data.index, self.data['upper_2sigma_15m'], 
                label='+1σ (15m)', linewidth=1, color='green', linestyle='--')
        ax3.plot(self.data.index, self.data['lower_2sigma_15m'], 
                label='-1σ (15m)', linewidth=1, color='red', linestyle='--')
        ax3.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        
        # Отмечаем ВСЕ сделки на 15m графике
        trades_df = pd.DataFrame(self.trades)
        
        # Отладочный вывод
        print(f"\nОтладка визуализации:")
        print(f"Всего записей в trades: {len(trades_df)}")
        print(f"Уникальные типы сделок: {trades_df['type'].unique()}")
        print(f"Количество по типам:")
        print(trades_df['type'].value_counts())
        
        # Разделяем на входы и выходы
        entry_trades = trades_df[trades_df['type'].str.contains('ENTRY')].copy()
        exit_trades = trades_df[trades_df['type'].str.contains('EXIT')].copy()
        
        print(f"\nВходов (ENTRY): {len(entry_trades)}")
        print(f"Выходов (EXIT): {len(exit_trades)}")
        
        # Разделяем на лонг и шорт
        long_entries = entry_trades[entry_trades['type'] == 'LONG_ENTRY']
        short_entries = entry_trades[entry_trades['type'] == 'SHORT_ENTRY']
        long_exits = exit_trades[exit_trades['type'] == 'LONG_EXIT']
        short_exits = exit_trades[exit_trades['type'] == 'SHORT_EXIT']
        
        print(f"Long entries: {len(long_entries)}")
        print(f"Short entries: {len(short_entries)}")
        print(f"Long exits: {len(long_exits)}")
        print(f"Short exits: {len(short_exits)}")
        
        # Входы и выходы для лонг позиций
        if len(long_entries) > 0:
            # Для входов в лонг: входим когда delta_pct ВЫШЕ верхней полосы (пересекает сверху)
            ax3.scatter(long_entries['timestamp'], long_entries['delta_pct'], 
                       color='green', marker='^', s=100, label='Long Entry', zorder=5, edgecolors='darkgreen', linewidth=1.5)
        if len(long_exits) > 0:
            # Для выходов из лонга: выходим когда delta_pct НИЖЕ нижней полосы
            ax3.scatter(long_exits['timestamp'], long_exits['delta_pct'], 
                       color='blue', marker='v', s=100, label='Long Exit', zorder=5, edgecolors='darkblue', linewidth=1.5)
        
        # Входы и выходы для шорт позиций
        if len(short_entries) > 0:
            # Для входов в шорт: входим когда delta_pct НИЖЕ нижней полосы (пересекает снизу)
            ax3.scatter(short_entries['timestamp'], short_entries['delta_pct'], 
                       color='red', marker='v', s=100, label='Short Entry', zorder=5, edgecolors='darkred', linewidth=1.5)
        if len(short_exits) > 0:
            # Для выходов из шорта: выходим когда delta_pct ВЫШЕ верхней полосы
            ax3.scatter(short_exits['timestamp'], short_exits['delta_pct'], 
                       color='orange', marker='^', s=100, label='Short Exit', zorder=5, edgecolors='darkorange', linewidth=1.5)
        
        ax3.set_title('Delta % и полосы ±1σ (15 минут) - Все точки входа и выхода', fontsize=13, fontweight='bold', pad=12)
        ax3.set_ylabel('Delta %', fontsize=10)
        ax3.legend(loc='best', fontsize=8, ncol=2)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='y', labelsize=9)
        
        # График 4: Delta % на 1H (для наглядности тренда)
        ax4.plot(self.data.index, self.data['delta_pct_1h'], 
                label='Delta % (1H)', linewidth=1, color='blue')
        ax4.plot(self.data.index, self.data['upper_2sigma_1h'], 
                label='+1σ (1H)', linewidth=1, color='green', linestyle='--')
        ax4.plot(self.data.index, self.data['lower_2sigma_1h'], 
                label='-1σ (1H)', linewidth=1, color='red', linestyle='--')
        ax4.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        ax4.set_title('Delta % и полосы ±1σ (1 час) - Подтверждение тренда', fontsize=13, fontweight='bold', pad=12)
        ax4.set_ylabel('Delta %', fontsize=10)
        ax4.legend(loc='best', fontsize=9)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='y', labelsize=9)
        
        # График 5: Equity curve
        equity_curve_df = pd.DataFrame(self.equity_curve)
        ax5.plot(equity_curve_df['timestamp'], equity_curve_df['equity'], 
                label='Equity', linewidth=2, color='purple')
        ax5.axhline(y=self.config.INITIAL_CAPITAL, color='gray', 
                   linestyle='--', linewidth=1, label='Starting Capital')
        ax5.set_title('Кривая капитала', fontsize=13, fontweight='bold', pad=12)
        ax5.set_xlabel('Дата', fontsize=10)
        ax5.set_ylabel('Капитал', fontsize=10)
        ax5.legend(loc='best', fontsize=9)
        ax5.grid(True, alpha=0.3)
        ax5.tick_params(axis='both', labelsize=9)
        
        # График 6: Оставляем пустым (можно использовать для дополнительной информации)
        ax6.axis('off')  # Скрываем пока
        
        # Форматирование оси X (только на последнем видимом графике)
        ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax5.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # Сохраняем исходные границы
        if self.original_xlim is None:
            self.original_xlim = ax5.get_xlim()
            self.original_ylim_ax1 = ax1.get_ylim()
        
        # === ИНТЕРАКТИВНЫЕ ВИДЖЕТЫ ДЛЯ ВЫБОРА ДИАПАЗОНА ДАТ ===
        
        # Создаем область для виджетов внизу
        widget_area = fig.add_subplot(gs[6])
        widget_area.axis('off')
        
        # Получаем диапазон дат для подсказок
        min_date = self.data.index.min()
        max_date = self.data.index.max()
        date_format = '%Y-%m-%d %H:%M:%S'
        
        # Текстовое поле для Start Date
        ax_start = plt.axes([0.15, 0.04, 0.15, 0.03])
        text_start = TextBox(ax_start, 'Start Date: ', 
                            initial=min_date.strftime(date_format),
                            color='lightgoldenrodyellow', hovercolor='lightyellow')
        
        # Текстовое поле для End Date
        ax_end = plt.axes([0.40, 0.04, 0.15, 0.03])
        text_end = TextBox(ax_end, 'End Date: ', 
                          initial=max_date.strftime(date_format),
                          color='lightgoldenrodyellow', hovercolor='lightyellow')
        
        # Кнопка Apply
        ax_apply = plt.axes([0.58, 0.04, 0.08, 0.03])
        btn_apply = Button(ax_apply, 'Apply Range', color='lightblue', hovercolor='skyblue')
        
        # Кнопка Reset
        ax_reset = plt.axes([0.68, 0.04, 0.08, 0.03])
        btn_reset = Button(ax_reset, 'Reset', color='lightcoral', hovercolor='salmon')
        
        # Кнопка Fit to Trades
        ax_fit = plt.axes([0.78, 0.04, 0.10, 0.03])
        btn_fit = Button(ax_fit, 'Fit to Trades', color='lightgreen', hovercolor='lightseagreen')
        
        # Функция применения диапазона
        def apply_range(event):
            try:
                start_str = text_start.text.strip()
                end_str = text_end.text.strip()
                
                # Пробуем разные форматы дат
                formats = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y/%m/%d %H:%M:%S', '%Y/%m/%d']
                
                start_date = None
                end_date = None
                
                for fmt in formats:
                    try:
                        start_date = pd.to_datetime(start_str, format=fmt)
                        break
                    except:
                        continue
                
                for fmt in formats:
                    try:
                        end_date = pd.to_datetime(end_str, format=fmt)
                        break
                    except:
                        continue
                
                if start_date is None or end_date is None:
                    print(f"❌ Ошибка: неверный формат даты. Используйте: YYYY-MM-DD HH:MM:SS или YYYY-MM-DD")
                    return
                
                if start_date >= end_date:
                    print(f"❌ Ошибка: начальная дата должна быть меньше конечной")
                    return
                
                # Применяем новый диапазон ко всем графикам
                for ax in axes:
                    ax.set_xlim(mdates.date2num(start_date), mdates.date2num(end_date))
                
                fig.canvas.draw_idle()
                print(f"✓ Применен диапазон: {start_date} - {end_date}")
                
            except Exception as e:
                print(f"❌ Ошибка при применении диапазона: {e}")
        
        # Функция сброса к исходному диапазону
        def reset_range(event):
            for ax in axes:
                ax.set_xlim(self.original_xlim)
            
            text_start.set_val(min_date.strftime(date_format))
            text_end.set_val(max_date.strftime(date_format))
            fig.canvas.draw_idle()
            print("✓ Диапазон сброшен к исходному")
        
        # Функция подгонки к сделкам
        def fit_to_trades(event):
            trades_df = pd.DataFrame(self.trades)
            if len(trades_df) > 0:
                trade_start = trades_df['timestamp'].min()
                trade_end = trades_df['timestamp'].max()
                
                # Добавляем небольшой отступ (5% от диапазона)
                time_range = trade_end - trade_start
                margin = time_range * 0.05
                
                fit_start = trade_start - margin
                fit_end = trade_end + margin
                
                for ax in axes:
                    ax.set_xlim(mdates.date2num(fit_start), mdates.date2num(fit_end))
                
                text_start.set_val(fit_start.strftime(date_format))
                text_end.set_val(fit_end.strftime(date_format))
                fig.canvas.draw_idle()
                print(f"✓ Подогнано к сделкам: {fit_start} - {fit_end}")
        
        # Привязываем функции к кнопкам
        btn_apply.on_clicked(apply_range)
        btn_reset.on_clicked(reset_range)
        btn_fit.on_clicked(fit_to_trades)
        
        # Инструкции для пользователя
        fig.text(0.01, 0.01, 
                'Формат даты: YYYY-MM-DD HH:MM:SS или YYYY-MM-DD | '
                'Также доступны: Zoom (лупа), Pan (крестик), Home в панели инструментов',
                fontsize=8, style='italic', color='gray')
        
        if self.config.SAVE_CHART:
            plt.savefig(self.config.CHART_FILE, dpi=self.config.CHART_DPI, bbox_inches='tight')
            print(f"График сохранен в '{self.config.CHART_FILE}'")
        
        if self.config.SHOW_CHART:
            plt.show()
        else:
            plt.close()
