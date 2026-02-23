"""
Движок бэктестирования для стратегии "Парная торговля 2.0"
Реализация на основе статьи smart-lab.ru/mobile/topic/1265846/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
import matplotlib.dates as mdates
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller  # ADF тест для коинтеграции


class BacktestEngine:
    """
    Движок для бэктестирования стратегии парного трейдинга с Z-score
    """
    
    def __init__(self, config):
        """
        Инициализация движка бэктестирования
        
        Args:
            config: объект конфигурации с параметрами
        """
        self.config = config
        self.data = None
        self.btc_data = None
        self.eth_data = None
        self.trades = []
        self.equity_curve = []
        
    def load_data(self):
        """Загрузка и подготовка данных"""
        print("Загрузка данных...")
        
        # Загрузка данных
        btc_data = pd.read_csv(self.config.BTC_1H_FILE)
        eth_data = pd.read_csv(self.config.ETH_1H_FILE)
        
        # Конвертируем timestamp в datetime
        btc_data['timestamp'] = pd.to_datetime(btc_data['timestamp'])
        eth_data['timestamp'] = pd.to_datetime(eth_data['timestamp'])
        
        btc_data.set_index('timestamp', inplace=True)
        eth_data.set_index('timestamp', inplace=True)
        
        print(f"BTC 1H: {len(btc_data)} свечей")
        print(f"ETH 1H: {len(eth_data)} свечей")
        print(f"Период: с {btc_data.index[0]} по {btc_data.index[-1]}")
        
        # Сохраняем для расчета PnL
        self.btc_data = btc_data
        self.eth_data = eth_data
        
        return btc_data, eth_data
    
    def calculate_beta(self, log_btc, log_eth, lookback):
        """
        Расчет Beta через OLS регрессию: β = Cov(log(PA), log(PB)) / Var(log(PB))
        
        Args:
            log_btc: логарифмы цен BTC
            log_eth: логарифмы цен ETH
            lookback: окно для расчета
            
        Returns:
            Series с beta коэффициентами
        """
        beta_series = pd.Series(index=log_btc.index, dtype=float)
        
        for i in range(lookback, len(log_btc)):
            # Берем окно данных
            y = log_btc.iloc[i-lookback:i].values
            x = log_eth.iloc[i-lookback:i].values
            
            # OLS регрессия
            X = add_constant(x)
            model = OLS(y, X)
            results = model.fit()
            
            # Beta - это коэффициент при x (индекс 1)
            beta_series.iloc[i] = results.params[1]
        
        return beta_series
    
    def calculate_adf_test(self, spread, lookback):
        """
        ADF тест (Augmented Dickey-Fuller) - проверка стационарности спреда
        
        Проверяет коинтеграцию пары: если спред стационарен, то пара коинтегрирована.
        "Cointegration beats correlation" - из статьи Amberdata
        
        Args:
            spread: временной ряд спреда
            lookback: окно данных для теста
            
        Returns:
            Series с p-value ADF теста (< 0.05 = стационарен = коинтегрирован)
        """
        adf_pvalue = pd.Series(index=spread.index, dtype=float)
        
        for i in range(lookback, len(spread)):
            # Берем окно данных
            window_data = spread.iloc[i-lookback:i].values
            
            try:
                # ADF тест: H0 = есть единичный корень (нестационарен)
                # Если p-value < 0.05, отвергаем H0 → ряд стационарен
                adf_result = adfuller(window_data, maxlag=1, regression='c', autolag=None)
                adf_pvalue.iloc[i] = adf_result[1]  # p-value
            except:
                adf_pvalue.iloc[i] = 1.0  # В случае ошибки считаем нестационарным
        
        return adf_pvalue
    
    def calculate_hurst_exponent(self, spread, lookback):
        """
        Hurst exponent - проверка mean reversion свойств спреда
        
        Интерпретация:
        - H < 0.5: mean reverting (идеально для парной торговли, лучше < 0.4)
        - H = 0.5: случайное блуждание (не подходит)
        - H > 0.5: трендовый (не подходит для mean reversion)
        
        Args:
            spread: временной ряд спреда
            lookback: окно данных для расчета
            
        Returns:
            Series с Hurst exponent
        """
        hurst_series = pd.Series(index=spread.index, dtype=float)
        
        for i in range(lookback, len(spread)):
            # Берем окно данных
            ts = spread.iloc[i-lookback:i].values
            
            try:
                # Метод R/S (Rescaled Range)
                lags = range(2, min(20, len(ts) // 2))
                tau = []
                
                for lag in lags:
                    # Делим на подпериоды
                    n_subseries = len(ts) // lag
                    rs_values = []
                    
                    for j in range(n_subseries):
                        subseries = ts[j*lag:(j+1)*lag]
                        
                        # Mean
                        mean = np.mean(subseries)
                        
                        # Cumulative deviations
                        deviations = subseries - mean
                        cumdev = np.cumsum(deviations)
                        
                        # Range
                        R = np.max(cumdev) - np.min(cumdev)
                        
                        # Standard deviation
                        S = np.std(subseries, ddof=1)
                        
                        if S > 0:
                            rs_values.append(R / S)
                    
                    if len(rs_values) > 0:
                        tau.append(np.mean(rs_values))
                
                # Linear regression log(tau) vs log(lags)
                if len(tau) > 2:
                    log_lags = np.log(list(lags))
                    log_tau = np.log(tau)
                    
                    # Убираем inf и nan
                    valid_idx = np.isfinite(log_lags) & np.isfinite(log_tau)
                    if np.sum(valid_idx) > 2:
                        poly = np.polyfit(log_lags[valid_idx], log_tau[valid_idx], 1)
                        hurst = poly[0]  # Наклон = Hurst exponent
                        hurst_series.iloc[i] = hurst
                    else:
                        hurst_series.iloc[i] = 0.5
                else:
                    hurst_series.iloc[i] = 0.5
            except:
                hurst_series.iloc[i] = 0.5  # В случае ошибки - нейтральное значение
        
        return hurst_series
    
    def calculate_indicators(self, btc_data, eth_data):
        """
        Расчет всех индикаторов для стратегии:
        1. Beta (β) через OLS регрессию
        2. Логарифмический спред = log(BTC) - β × log(ETH)
        3. Z-score спреда
        4. Волатильности для фильтра
        
        Returns:
            DataFrame с индикаторами
        """
        print("\nРасчет индикаторов стратегии...")
        
        # Создаем DataFrame для индикаторов
        df = pd.DataFrame(index=btc_data.index)
        
        # Цены закрытия
        df['btc_close'] = btc_data['close']
        df['eth_close'] = eth_data['close']
        
        # === ШАГ 1: Расчет Beta через OLS ===
        print("  - Расчет Beta (hedge ratio) через OLS регрессию...")
        log_btc = np.log(btc_data['close'])
        log_eth = np.log(eth_data['close'])
        
        if self.config.USE_ROLLING_BETA:
            # Rolling beta - адаптируется к изменению корреляции
            df['beta'] = self.calculate_beta(log_btc, log_eth, self.config.BETA_LOOKBACK)
        else:
            # Фиксированная beta на всем периоде
            X = add_constant(log_eth.values)
            model = OLS(log_btc.values, X)
            results = model.fit()
            fixed_beta = results.params[1]
            df['beta'] = fixed_beta
            print(f"    Фиксированная Beta = {fixed_beta:.4f}")
        
        # === ШАГ 2: Расчет спреда ===
        print("  - Расчет логарифмического спреда...")
        # Спред = log(PA) - β × log(PB)
        df['log_btc'] = log_btc
        df['log_eth'] = log_eth
        df['spread'] = log_btc - df['beta'] * log_eth
        
        # === ШАГ 3: Расчет статистик спреда (μ, σ) на скользящем окне ===
        print("  - Расчет среднего и стандартного отклонения спреда...")
        df['spread_mean'] = df['spread'].rolling(
            window=self.config.SPREAD_LOOKBACK, 
            min_periods=self.config.SPREAD_LOOKBACK
        ).mean()
        
        df['spread_std'] = df['spread'].rolling(
            window=self.config.SPREAD_LOOKBACK, 
            min_periods=self.config.SPREAD_LOOKBACK
        ).std()
        
        # === ШАГ 4: Расчет Z-score ===
        print("  - Расчет Z-score...")
        # Z = (Спред - μ) / σ
        df['z_score'] = (df['spread'] - df['spread_mean']) / df['spread_std']
        
        # === ШАГ 5: Фильтр волатильности ===
        if self.config.USE_VOLATILITY_FILTER:
            print("  - Расчет фильтра волатильности...")
            # Текущая волатильность (короткое окно)
            df['sigma_current'] = df['spread'].rolling(
                window=self.config.SPREAD_LOOKBACK, 
                min_periods=self.config.SPREAD_LOOKBACK
            ).std()
            
            # Долгосрочная волатильность
            df['sigma_long'] = df['spread'].rolling(
                window=self.config.VOLATILITY_LONG_LOOKBACK, 
                min_periods=self.config.VOLATILITY_LONG_LOOKBACK
            ).std()
        else:
            df['sigma_current'] = 1.0
            df['sigma_long'] = 1.0
        
        # Корреляция для информации
        print("  - Расчет корреляции...")
        df['correlation'] = log_btc.rolling(
            window=self.config.BETA_LOOKBACK, 
            min_periods=self.config.BETA_LOOKBACK
        ).corr(log_eth)
        
        # === ШАГ 6: ADF тест - проверка коинтеграции ===
        # "Cointegration beats correlation" - Amberdata
        if self.config.USE_ADF_TEST:
            print("  - ADF тест (проверка стационарности спреда / коинтеграции)...")
            df['adf_pvalue'] = self.calculate_adf_test(
                df['spread'], 
                self.config.ADF_LOOKBACK
            )
            # is_cointegrated = True если p-value < порога (спред стационарен)
            df['is_cointegrated'] = df['adf_pvalue'] < self.config.ADF_PVALUE_THRESHOLD
        else:
            df['adf_pvalue'] = 0.0
            df['is_cointegrated'] = True
        
        # === ШАГ 7: Hurst exponent - проверка mean reversion ===
        if self.config.USE_HURST_FILTER:
            print("  - Hurst exponent (проверка mean reversion свойств)...")
            df['hurst'] = self.calculate_hurst_exponent(
                df['spread'], 
                self.config.HURST_LOOKBACK
            )
            # is_mean_reverting = True если Hurst < порога
            df['is_mean_reverting'] = df['hurst'] < self.config.HURST_MAX_THRESHOLD
        else:
            df['hurst'] = 0.5
            df['is_mean_reverting'] = True
        
        # === КОМБИНИРОВАННЫЙ ФИЛЬТР: Торговать только если коинтегрирован И mean reverting ===
        if self.config.COINTEGRATION_CHECK_ENABLED:
            df['can_trade'] = df['is_cointegrated'] & df['is_mean_reverting']
        else:
            df['can_trade'] = True
        
        # Удаляем NaN
        df = df.dropna()
        
        print(f"\nДанные готовы для бэктеста: {len(df)} свечей")
        print(f"Период: с {df.index[0]} по {df.index[-1]}")
        
        # Статистика
        print(f"\n=== СТАТИСТИКА ИНДИКАТОРОВ ===")
        if self.config.USE_ROLLING_BETA:
            print(f"Beta (среднее): {df['beta'].mean():.4f}")
            print(f"Beta (мин/макс): {df['beta'].min():.4f} / {df['beta'].max():.4f}")
        print(f"Корреляция (среднее): {df['correlation'].mean():.4f}")
        print(f"Z-score (мин/макс): {df['z_score'].min():.2f} / {df['z_score'].max():.2f}")
        print(f"Спред (среднее): {df['spread'].mean():.6f}")
        print(f"Спред (std): {df['spread'].std():.6f}")
        
        # Статистика коинтеграции
        if self.config.USE_ADF_TEST:
            cointegrated_pct = df['is_cointegrated'].sum() / len(df) * 100
            print(f"\n=== КОИНТЕГРАЦИЯ (ADF тест) ===")
            print(f"ADF p-value (среднее): {df['adf_pvalue'].mean():.4f}")
            print(f"Коинтегрирован: {cointegrated_pct:.1f}% времени")
            print(f"  (p-value < {self.config.ADF_PVALUE_THRESHOLD} = стационарен)")
        
        # Статистика Hurst
        if self.config.USE_HURST_FILTER:
            mean_reverting_pct = df['is_mean_reverting'].sum() / len(df) * 100
            print(f"\n=== MEAN REVERSION (Hurst exponent) ===")
            print(f"Hurst (среднее): {df['hurst'].mean():.4f}")
            print(f"Hurst (мин/макс): {df['hurst'].min():.4f} / {df['hurst'].max():.4f}")
            print(f"Mean reverting: {mean_reverting_pct:.1f}% времени")
            print(f"  (Hurst < {self.config.HURST_MAX_THRESHOLD} = mean reverting)")
        
        # Общая статистика торговли
        if self.config.COINTEGRATION_CHECK_ENABLED:
            can_trade_pct = df['can_trade'].sum() / len(df) * 100
            print(f"\n=== РАЗРЕШЕНА ТОРГОВЛЯ ===")
            print(f"Можно торговать: {can_trade_pct:.1f}% времени")
            print(f"  (коинтегрирован И mean reverting)")
        
        return df
    
    def run_backtest(self, strategy):
        """
        Запуск бэктеста с заданной стратегией
        
        Args:
            strategy: объект стратегии с методом generate_signals()
        """
        print("\n" + "="*60)
        print("ЗАПУСК БЭКТЕСТА")
        print("="*60)
        
        # Генерируем сигналы
        signals = strategy.generate_signals()
        
        print(f"Сгенерировано сигналов: {len(signals)}")
        
        # Симулируем торговлю
        equity = self.config.INITIAL_CAPITAL
        position_open = False
        entry_data = None
        
        for signal in signals:
            # Защита от отрицательного капитала
            if equity <= 0:
                print(f"\n⚠️ ВНИМАНИЕ: Капитал достиг нуля!")
                print(f"Остановка на {signal['timestamp']}")
                break
            
            if 'ENTRY' in signal['type']:
                # Открытие позиции
                timestamp = signal['timestamp']
                
                # Получаем цены на момент входа
                btc_price_entry = self.btc_data.loc[timestamp, 'close']
                eth_price_entry = self.eth_data.loc[timestamp, 'close']
                
                # Размер позиции = весь капитал
                position_size_usd = equity
                
                # Сохраняем данные входа
                entry_data = {
                    'timestamp_entry': timestamp,
                    'type_entry': signal['type'],
                    'spread_entry': signal['spread'],
                    'z_score_entry': signal['z_score'],
                    'beta': signal['beta'],
                    'btc_price_entry': btc_price_entry,
                    'eth_price_entry': eth_price_entry,
                    'position_size_usd': position_size_usd,
                    'equity_before': equity
                }
                position_open = True
                
            elif 'EXIT' in signal['type'] and position_open:
                # Закрытие позиции
                timestamp = signal['timestamp']
                
                # Получаем цены на момент выхода
                btc_price_exit = self.btc_data.loc[timestamp, 'close']
                eth_price_exit = self.eth_data.loc[timestamp, 'close']
                
                # Определяем тип сделки
                is_long_spread = 'LONG' in signal['type']
                
                if is_long_spread:
                    # LONG SPREAD = BUY BTC + SELL ETH
                    btc_pnl_pct = (btc_price_exit - entry_data['btc_price_entry']) / entry_data['btc_price_entry'] * 100
                    eth_pnl_pct = (entry_data['eth_price_entry'] - eth_price_exit) / entry_data['eth_price_entry'] * 100
                else:
                    # SHORT SPREAD = SELL BTC + BUY ETH
                    btc_pnl_pct = (entry_data['btc_price_entry'] - btc_price_exit) / entry_data['btc_price_entry'] * 100
                    eth_pnl_pct = (eth_price_exit - entry_data['eth_price_entry']) / entry_data['eth_price_entry'] * 100
                
                # Общий P&L (среднее от двух ног)
                total_pnl_pct = btc_pnl_pct + eth_pnl_pct
                
                # Комиссия
                commission_pct = abs(total_pnl_pct) * self.config.COMMISSION_RATE / 100
                net_pnl_pct = total_pnl_pct - commission_pct
                
                # P&L в долларах
                pnl_usd = entry_data['position_size_usd'] * net_pnl_pct / 100
                
                # Обновляем капитал
                equity = entry_data['equity_before'] + pnl_usd
                
                # Записываем вход
                entry_record = {
                    'timestamp': entry_data['timestamp_entry'],
                    'type': entry_data['type_entry'],
                    'spread': entry_data['spread_entry'],
                    'z_score': entry_data['z_score_entry'],
                    'beta': entry_data['beta'],
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
                self.trades.append(entry_record)
                
                # Записываем выход с результатами
                exit_record = {
                    'timestamp': timestamp,
                    'type': signal['type'],
                    'spread': signal['spread'],
                    'z_score': signal['z_score'],
                    'beta': signal['beta'],
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
            
            # Сохраняем equity
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
        
        # Только выходы
        exit_trades = trades_df[trades_df['type'].str.contains('EXIT')].copy()
        
        if len(exit_trades) == 0:
            print("Закрытых сделок не найдено!")
            return
        
        print(f"\nВсего закрытых сделок: {len(exit_trades)}")
        print(f"Long спред сделок: {len(exit_trades[exit_trades['type'] == 'LONG_SPREAD_EXIT'])}")
        print(f"Short спред сделок: {len(exit_trades[exit_trades['type'] == 'SHORT_SPREAD_EXIT'])}")
        
        # Статистика
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
        print(f"  Макс профит: {exit_trades['btc_pnl_%'].max():.4f}%")
        print(f"  Макс убыток: {exit_trades['btc_pnl_%'].min():.4f}%")
        
        print(f"\nETH нога:")
        print(f"  Средний P&L: {exit_trades['eth_pnl_%'].mean():.4f}%")
        print(f"  Макс профит: {exit_trades['eth_pnl_%'].max():.4f}%")
        print(f"  Макс убыток: {exit_trades['eth_pnl_%'].min():.4f}%")
        
        print(f"\nОбщая статистика:")
        print(f"  Средний общий P&L: {exit_trades['total_pnl_%'].mean():.4f}%")
        print(f"  Средний чистый P&L: {exit_trades['net_pnl_%'].mean():.4f}%")
        print(f"  Макс профит: {exit_trades['net_pnl_%'].max():.4f}%")
        print(f"  Макс убыток: {exit_trades['net_pnl_%'].min():.4f}%")
        
        if self.config.COMMISSION_RATE > 0:
            print(f"\n  Общая комиссия: {exit_trades['commission_%'].sum():.4f}%")
        
        print(f"\n  Средний P&L в USD: {exit_trades['pnl_usd'].mean():.2f}")
        print(f"  Общий P&L в USD: {exit_trades['pnl_usd'].sum():.2f}")
        
        # Капитал
        if len(self.equity_curve) > 0:
            equity_curve_df = pd.DataFrame(self.equity_curve)
            final_equity = equity_curve_df['equity'].iloc[-1]
            
            print(f"\n{'='*60}")
            print("КАПИТАЛ:")
            print(f"{'='*60}")
            print(f"Начальный капитал: ${self.config.INITIAL_CAPITAL:,.2f}")
            print(f"Финальный капитал: ${final_equity:,.2f}")
            print(f"Общий доход: ${final_equity - self.config.INITIAL_CAPITAL:,.2f}")
            print(f"Доходность: {(final_equity - self.config.INITIAL_CAPITAL) / self.config.INITIAL_CAPITAL * 100:.2f}%")
            
            # Просадка
            equity_array = equity_curve_df['equity'].values
            running_max = np.maximum.accumulate(equity_array)
            drawdown = (equity_array - running_max) / running_max * 100
            max_drawdown = drawdown.min()
            print(f"Макс просадка: {max_drawdown:.2f}%")
        
        # Первые 5 сделок
        print(f"\n{'='*60}")
        print("ПЕРВЫЕ 5 СДЕЛОК:")
        print(f"{'='*60}")
        display_cols = ['timestamp', 'type', 'z_score', 'btc_pnl_%', 'eth_pnl_%', 'net_pnl_%', 'pnl_usd', 'equity_after']
        print(exit_trades[display_cols].head(5).to_string(index=False))
    
    def save_results(self):
        """Сохранение результатов"""
        if self.config.SAVE_TRADES_LOG and len(self.trades) > 0:
            trades_df = pd.DataFrame(self.trades)
            trades_df.to_excel(self.config.TRADES_LOG_FILE, index=False, engine='openpyxl')
            print(f"\nСделки сохранены в '{self.config.TRADES_LOG_FILE}'")
    
    def plot_results(self):
        """Визуализация результатов"""
        if len(self.trades) == 0:
            print("Нет данных для визуализации")
            return
        
        print("\nСоздание графиков...")
        
        fig = plt.figure(figsize=(18, 16))
        gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 1, 1, 0.2], hspace=0.3, top=0.95, bottom=0.08)
        
        ax1 = fig.add_subplot(gs[0])  # Beta
        ax2 = fig.add_subplot(gs[1], sharex=ax1)  # Spread
        ax3 = fig.add_subplot(gs[2], sharex=ax1)  # Z-score + сигналы
        ax4 = fig.add_subplot(gs[3], sharex=ax1)  # Equity
        
        axes = [ax1, ax2, ax3, ax4]
        
        # График 1: Beta
        ax1.plot(self.data.index, self.data['beta'], label='Beta (hedge ratio)', linewidth=1.5, color='blue')
        ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax1.set_title('Beta коэффициент (OLS)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Beta', fontsize=10)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # График 2: Spread
        ax2.plot(self.data.index, self.data['spread'], label='Log Spread', linewidth=1, color='purple')
        ax2.plot(self.data.index, self.data['spread_mean'], label='Mean', linewidth=1.5, color='orange', linestyle='--')
        ax2.fill_between(self.data.index, 
                         self.data['spread_mean'] - self.data['spread_std'],
                         self.data['spread_mean'] + self.data['spread_std'],
                         alpha=0.2, color='orange', label='±1σ')
        ax2.set_title('Логарифмический спред', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Spread', fontsize=10)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # График 3: Z-score + сигналы
        ax3.plot(self.data.index, self.data['z_score'], label='Z-score', linewidth=1, color='darkblue')
        ax3.axhline(y=self.config.Z_SCORE_ENTRY, color='green', linestyle='--', linewidth=1, label=f'+{self.config.Z_SCORE_ENTRY}σ')
        ax3.axhline(y=-self.config.Z_SCORE_ENTRY, color='red', linestyle='--', linewidth=1, label=f'-{self.config.Z_SCORE_ENTRY}σ')
        ax3.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        
        # Отмечаем сделки
        trades_df = pd.DataFrame(self.trades)
        entry_trades = trades_df[trades_df['type'].str.contains('ENTRY')].copy()
        exit_trades = trades_df[trades_df['type'].str.contains('EXIT')].copy()
        
        long_entries = entry_trades[entry_trades['type'] == 'LONG_SPREAD_ENTRY']
        short_entries = entry_trades[entry_trades['type'] == 'SHORT_SPREAD_ENTRY']
        long_exits = exit_trades[exit_trades['type'] == 'LONG_SPREAD_EXIT']
        short_exits = exit_trades[exit_trades['type'] == 'SHORT_SPREAD_EXIT']
        
        if len(long_entries) > 0:
            ax3.scatter(long_entries['timestamp'], long_entries['z_score'], 
                       color='green', marker='^', s=100, label='Long Entry', zorder=5, edgecolors='darkgreen', linewidth=1.5)
        if len(long_exits) > 0:
            ax3.scatter(long_exits['timestamp'], long_exits['z_score'], 
                       color='blue', marker='v', s=100, label='Long Exit', zorder=5, edgecolors='darkblue', linewidth=1.5)
        if len(short_entries) > 0:
            ax3.scatter(short_entries['timestamp'], short_entries['z_score'], 
                       color='red', marker='v', s=100, label='Short Entry', zorder=5, edgecolors='darkred', linewidth=1.5)
        if len(short_exits) > 0:
            ax3.scatter(short_exits['timestamp'], short_exits['z_score'], 
                       color='orange', marker='^', s=100, label='Short Exit', zorder=5, edgecolors='darkorange', linewidth=1.5)
        
        ax3.set_title('Z-score и торговые сигналы', fontsize=13, fontweight='bold')
        ax3.set_ylabel('Z-score', fontsize=10)
        ax3.legend(loc='best', fontsize=8, ncol=2)
        ax3.grid(True, alpha=0.3)
        
        # График 4: Equity
        equity_curve_df = pd.DataFrame(self.equity_curve)
        ax4.plot(equity_curve_df['timestamp'], equity_curve_df['equity'], 
                label='Equity', linewidth=2, color='purple')
        ax4.axhline(y=self.config.INITIAL_CAPITAL, color='gray', 
                   linestyle='--', linewidth=1, label='Starting Capital')
        ax4.set_title('Кривая капитала', fontsize=13, fontweight='bold')
        ax4.set_xlabel('Дата', fontsize=10)
        ax4.set_ylabel('Капитал ($)', fontsize=10)
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)
        
        # Форматирование оси X
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax4.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        plt.tight_layout()
        
        if self.config.SAVE_CHART:
            plt.savefig(self.config.CHART_FILE, dpi=self.config.CHART_DPI, bbox_inches='tight')
            print(f"График сохранен в '{self.config.CHART_FILE}'")
        
        if self.config.SHOW_CHART:
            plt.show()
        else:
            plt.close()
