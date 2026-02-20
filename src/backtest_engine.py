"""
Движок бэктестирования торговых стратегий
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
        
        for signal in signals:
            if 'ENTRY' in signal['type']:
                # Открытие позиции
                trade = {
                    'timestamp': signal['timestamp'],
                    'type': signal['type'],
                    'price': signal['price'],
                    'equity': equity
                }
                self.trades.append(trade)
                
            elif 'EXIT' in signal['type']:
                # Закрытие позиции
                entry_price = signal['entry_price']
                exit_price = signal['price']
                
                # Рассчитываем P&L
                if 'LONG' in signal['type']:
                    pnl = exit_price - entry_price
                else:  # SHORT
                    pnl = entry_price - exit_price
                
                # Применяем комиссию
                commission = abs(pnl) * self.config.COMMISSION_RATE / 100
                pnl -= commission
                
                # Обновляем капитал
                equity += pnl * self.config.POSITION_SIZE_MULTIPLIER
                
                trade = {
                    'timestamp': signal['timestamp'],
                    'type': signal['type'],
                    'price': exit_price,
                    'pnl': pnl,
                    'commission': commission,
                    'equity': equity
                }
                self.trades.append(trade)
            
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
        
        print(f"\nВсего сделок: {len(trades_df)}")
        print(f"Входы в лонг: {len(trades_df[trades_df['type'] == 'LONG ENTRY'])}")
        print(f"Выходы из лонга: {len(trades_df[trades_df['type'] == 'LONG EXIT'])}")
        print(f"Входы в шорт: {len(trades_df[trades_df['type'] == 'SHORT ENTRY'])}")
        print(f"Выходы из шорта: {len(trades_df[trades_df['type'] == 'SHORT EXIT'])}")
        
        # Статистика по закрытым сделкам
        closed_trades = trades_df[trades_df['type'].str.contains('EXIT')].copy()
        
        if len(closed_trades) > 0:
            print(f"\nЗакрытых сделок: {len(closed_trades)}")
            print(f"Прибыльных: {len(closed_trades[closed_trades['pnl'] > 0])}")
            print(f"Убыточных: {len(closed_trades[closed_trades['pnl'] < 0])}")
            print(f"Винрейт: {len(closed_trades[closed_trades['pnl'] > 0]) / len(closed_trades) * 100:.2f}%")
            print(f"\nСредний P&L: {closed_trades['pnl'].mean():.4f}")
            print(f"Максимальный профит: {closed_trades['pnl'].max():.4f}")
            print(f"Максимальный убыток: {closed_trades['pnl'].min():.4f}")
            
            if self.config.COMMISSION_RATE > 0:
                total_commission = closed_trades['commission'].sum()
                print(f"Общая комиссия: {total_commission:.2f}")
            
            equity_curve_df = pd.DataFrame(self.equity_curve)
            final_equity = equity_curve_df['equity'].iloc[-1]
            
            print(f"\nНачальный капитал: {self.config.INITIAL_CAPITAL}")
            print(f"Финальный капитал: {final_equity:.2f}")
            print(f"Общий доход: {final_equity - self.config.INITIAL_CAPITAL:.2f}")
            print(f"Доходность: {(final_equity - self.config.INITIAL_CAPITAL) / self.config.INITIAL_CAPITAL * 100:.2f}%")
            
            # Максимальная просадка
            equity_array = equity_curve_df['equity'].values
            running_max = np.maximum.accumulate(equity_array)
            drawdown = (equity_array - running_max) / running_max * 100
            max_drawdown = drawdown.min()
            print(f"Максимальная просадка: {max_drawdown:.2f}%")
            
            # Первые 10 сделок
            print(f"\n{'='*60}")
            print("ПЕРВЫЕ 10 СДЕЛОК:")
            print(f"{'='*60}")
            print(trades_df.head(10).to_string(index=False))
    
    def save_results(self):
        """Сохранение результатов в файлы"""
        if self.config.SAVE_TRADES_LOG and len(self.trades) > 0:
            trades_df = pd.DataFrame(self.trades)
            trades_df.to_csv(self.config.TRADES_LOG_FILE, index=False)
            print(f"\nСделки сохранены в '{self.config.TRADES_LOG_FILE}'")
    
    def plot_results(self):
        """Визуализация результатов бэктеста"""
        if len(self.trades) == 0:
            print("Нет данных для визуализации")
            return
        
        print("\nСоздание графиков...")
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 16))
        
        # График 1: Абсолютная Delta (BTC/ETH)
        ax1 = axes[0]
        ax1.plot(self.data.index, self.data['delta_abs_15m'], 
                label='Delta Absolute (15m)', linewidth=1, color='purple', alpha=0.7)
        ax1.plot(self.data.index, self.data['delta_abs_1h'], 
                label='Delta Absolute (1H)', linewidth=2, color='blue')
        ax1.set_title('Абсолютная Delta (BTC/ETH)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Delta (BTC/ETH)')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # График 2: Delta % и полосы на 15m
        ax2 = axes[1]
        ax2.plot(self.data.index, self.data['delta_pct_15m'], 
                label='Delta % (15m)', linewidth=1, color='orange')
        ax2.plot(self.data.index, self.data['upper_2sigma_15m'], 
                label='+2σ (15m)', linewidth=1, color='green', linestyle='--')
        ax2.plot(self.data.index, self.data['lower_2sigma_15m'], 
                label='-2σ (15m)', linewidth=1, color='red', linestyle='--')
        ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        
        # Отмечаем сделки
        trades_df = pd.DataFrame(self.trades)
        long_entries = trades_df[trades_df['type'] == 'LONG ENTRY']
        long_exits = trades_df[trades_df['type'] == 'LONG EXIT']
        short_entries = trades_df[trades_df['type'] == 'SHORT ENTRY']
        short_exits = trades_df[trades_df['type'] == 'SHORT EXIT']
        
        ax2.scatter(long_entries['timestamp'], long_entries['price'], 
                   color='green', marker='^', s=100, label='Long Entry', zorder=5)
        ax2.scatter(long_exits['timestamp'], long_exits['price'], 
                   color='blue', marker='v', s=100, label='Long Exit', zorder=5)
        ax2.scatter(short_entries['timestamp'], short_entries['price'], 
                   color='red', marker='v', s=100, label='Short Entry', zorder=5)
        ax2.scatter(short_exits['timestamp'], short_exits['price'], 
                   color='orange', marker='^', s=100, label='Short Exit', zorder=5)
        
        ax2.set_title('Delta % и полосы ±2σ (15 минут) - Точки входа', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Delta %')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # График 3: Delta % на 1H
        ax3 = axes[2]
        ax3.plot(self.data.index, self.data['delta_pct_1h'], 
                label='Delta % (1H)', linewidth=1, color='blue')
        ax3.plot(self.data.index, self.data['upper_2sigma_1h'], 
                label='+2σ (1H)', linewidth=1, color='green', linestyle='--')
        ax3.plot(self.data.index, self.data['lower_2sigma_1h'], 
                label='-2σ (1H)', linewidth=1, color='red', linestyle='--')
        ax3.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        ax3.set_title('Delta % и полосы ±2σ (1 час) - Точки выхода', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Delta %')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        
        # График 4: Equity curve
        ax4 = axes[3]
        equity_curve_df = pd.DataFrame(self.equity_curve)
        ax4.plot(equity_curve_df['timestamp'], equity_curve_df['equity'], 
                label='Equity', linewidth=2, color='purple')
        ax4.axhline(y=self.config.INITIAL_CAPITAL, color='gray', 
                   linestyle='--', linewidth=1, label='Starting Capital')
        ax4.set_title('Кривая капитала', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Дата')
        ax4.set_ylabel('Капитал')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.config.SAVE_CHART:
            plt.savefig(self.config.CHART_FILE, dpi=self.config.CHART_DPI, bbox_inches='tight')
            print(f"График сохранен в '{self.config.CHART_FILE}'")
        
        if self.config.SHOW_CHART:
            plt.show()
        else:
            plt.close()
