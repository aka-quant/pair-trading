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
    def print_backtest_settings(self):
        print("\n================= STRATEGY SETTINGS =================")
        cfg = self.config
        print(f"Режим расчёта прибыли      : {'Простой' if cfg.SIMPLE_PROFIT_MODE else 'Сложный'}")
        print(f"Beta lookback              : {cfg.BETA_LOOKBACK}")
        print(f"Rolling beta               : {cfg.USE_ROLLING_BETA}")
        print(f"Spread lookback            : {cfg.SPREAD_LOOKBACK}")
        print(f"Z-score entry/exit         : {cfg.Z_SCORE_ENTRY} / {cfg.Z_SCORE_EXIT}")
        print(f"Exit mode                  : {cfg.EXIT_MODE} ({'Возврат к 0' if cfg.EXIT_MODE==1 else 'Переворот/порог'})")
        print(f"Volatility filter          : {cfg.USE_VOLATILITY_FILTER} (threshold={cfg.VOLATILITY_THRESHOLD}, long_lookback={cfg.VOLATILITY_LONG_LOOKBACK})")
        print(f"ADF test                   : {cfg.USE_ADF_TEST} (p<{cfg.ADF_PVALUE_THRESHOLD}, lookback={cfg.ADF_LOOKBACK})")
        print(f"Hurst filter               : {cfg.USE_HURST_FILTER} (max={cfg.HURST_MAX_THRESHOLD}, lookback={cfg.HURST_LOOKBACK})")
        print(f"Cointegration filter       : {cfg.COINTEGRATION_CHECK_ENABLED}")
        print(f"Initial capital            : {cfg.INITIAL_CAPITAL}")
        print(f"Commission                 : {cfg.COMMISSION_RATE}%")
        print(f"Save trades log            : {cfg.SAVE_TRADES_LOG}")
        print(f"Save chart                 : {cfg.SAVE_CHART}")
        print(f"Show chart                 : {cfg.SHOW_CHART}")
        print("====================================================\n")

    def calculate_adf_test(self, spread, lookback):
        """
        Расчет p-value ADF теста по скользящему окну
        Args:
            spread: Series спреда
            lookback: окно для расчета
        Returns:
            Series с p-value ADF теста
        """
        adf_pvalues = pd.Series(index=spread.index, dtype=float)
        for i in range(lookback, len(spread)):
            window = spread.iloc[i-lookback:i].values
            try:
                adf_result = adfuller(window, autolag='AIC')
                adf_pvalues.iloc[i] = adf_result[1]
            except:
                adf_pvalues.iloc[i] = 1.0
        return adf_pvalues
    def plot_stats_dashboard(self):
        """Визуализация статистических метрик и дельты между ногами (строится всегда)"""
        import matplotlib.pyplot as plt
        import numpy as np
        print("\nСоздание дашборда статистических метрик...")
        fig2, axes = plt.subplots(5, 1, figsize=(18, 18), sharex=True, gridspec_kw={'hspace': 0.25})
        # 1. График Hurst
        if 'hurst' in self.data.columns:
            axes[0].plot(self.data.index, self.data['hurst'], label='Hurst exponent', color='purple')
            axes[0].axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            axes[0].set_ylabel('Hurst')
            axes[0].set_title('Hurst exponent (mean reversion)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        # 2. График ADF p-value
        if 'adf_pvalue' in self.data.columns:
            axes[1].plot(self.data.index, self.data['adf_pvalue'], label='ADF p-value', color='brown')
            axes[1].axhline(0.05, color='red', linestyle='--', linewidth=1, alpha=0.5, label='0.05 threshold')
            axes[1].set_ylabel('ADF p-value')
            axes[1].set_title('ADF test p-value (cointegration)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        # 3. Волатильность (sigma_current и sigma_long)
        if 'sigma_current' in self.data.columns and 'sigma_long' in self.data.columns:
            axes[2].plot(self.data.index, self.data['sigma_current'], label='Short-term volatility', color='orange')
            axes[2].plot(self.data.index, self.data['sigma_long'], label='Long-term volatility', color='blue', alpha=0.7)
            axes[2].set_ylabel('Volatility')
            axes[2].set_title('Volatility (short-term & long-term)')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        # 4. Дельта между первой и второй ногой
        if 'first_leg_close' in self.data.columns and 'second_leg_close' in self.data.columns:
            delta = self.data['first_leg_close'] - self.data['second_leg_close']
            axes[3].plot(self.data.index, delta, label='Delta (First leg - Second leg)', color='green')
            axes[3].set_ylabel('Delta')
            axes[3].set_title('Delta between First and Second Leg')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
        # 5. Spread (для полноты)
        if 'spread' in self.data.columns:
            axes[4].plot(self.data.index, self.data['spread'], label='Spread', color='black')
            axes[4].set_ylabel('Spread')
            axes[4].set_title('Logarithmic Spread')
            axes[4].legend()
            axes[4].grid(True, alpha=0.3)
        axes[-1].set_xlabel('Time')
        plt.suptitle('Статистические метрики и дельта между ногами', fontsize=16, fontweight='bold')
        plt.show()
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
        self.first_leg_data = None
        self.second_leg_data = None
        self.trades = []
        self.equity_curve = []
        
    def load_data(self):
        """Загрузка и подготовка данных"""
        print("Загрузка данных...")
        first_leg_data = pd.read_csv(self.config.FIRST_LEG)
        second_leg_data = pd.read_csv(self.config.SECOND_LEG)
        first_leg_data['timestamp'] = pd.to_datetime(first_leg_data['timestamp'])
        second_leg_data['timestamp'] = pd.to_datetime(second_leg_data['timestamp'])
        first_leg_data.set_index('timestamp', inplace=True)
        second_leg_data.set_index('timestamp', inplace=True)
        print(f"FIRST_LEG: {len(first_leg_data)} свечей")
        print(f"SECOND_LEG: {len(second_leg_data)} свечей")
        print(f"Период: с {first_leg_data.index[0]} по {first_leg_data.index[-1]}")
        self.first_leg_data = first_leg_data
        self.second_leg_data = second_leg_data
        return first_leg_data, second_leg_data
    
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
            y = log_btc.iloc[i-lookback:i].values
            x = log_eth.iloc[i-lookback:i].values
            X = add_constant(x)
            model = OLS(y, X)
            results = model.fit()
            beta_series.iloc[i] = results.params[1]
        return beta_series

    def calculate_hurst_exponent(self, spread, lookback):
        """
        Расчет Hurst exponent методом R/S (Rescaled Range)
        Args:
            spread: Series спреда
            lookback: окно для расчета
        Returns:
            Series с hurst коэффициентами
        """
        hurst_series = pd.Series(index=spread.index, dtype=float)
        for i in range(lookback, len(spread)):
            ts = spread.iloc[i-lookback:i].values
            try:
                lags = range(2, min(20, len(ts) // 2))
                tau = []
                for lag in lags:
                    n_subseries = len(ts) // lag
                    rs_values = []
                    for j in range(n_subseries):
                        subseries = ts[j*lag:(j+1)*lag]
                        mean = np.mean(subseries)
                        deviations = subseries - mean
                        cumdev = np.cumsum(deviations)
                        R = np.max(cumdev) - np.min(cumdev)
                        S = np.std(subseries, ddof=1)
                        if S > 0:
                            rs_values.append(R / S)
                    if len(rs_values) > 0:
                        tau.append(np.mean(rs_values))
                if len(tau) > 2:
                    log_lags = np.log(list(lags))
                    log_tau = np.log(tau)
                    valid_idx = np.isfinite(log_lags) & np.isfinite(log_tau)
                    if np.sum(valid_idx) > 2:
                        poly = np.polyfit(log_lags[valid_idx], log_tau[valid_idx], 1)
                        hurst = poly[0]
                        hurst_series.iloc[i] = hurst
                    else:
                        hurst_series.iloc[i] = 0.5
                else:
                    hurst_series.iloc[i] = 0.5
            except:
                hurst_series.iloc[i] = 0.5
        return hurst_series
    
    def calculate_indicators(self, first_leg_data, second_leg_data):
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
        df = pd.DataFrame(index=first_leg_data.index)
        df['first_leg_close'] = first_leg_data['close']
        df['second_leg_close'] = second_leg_data['close']
        print("  - Расчет Beta (hedge ratio) через OLS регрессию...")
        log_first = np.log(first_leg_data['close'])
        log_second = np.log(second_leg_data['close'])
        if self.config.USE_ROLLING_BETA:
            df['beta'] = self.calculate_beta(log_first, log_second, self.config.BETA_LOOKBACK)
        else:
            X = add_constant(log_second.values)
            model = OLS(log_first.values, X)
            results = model.fit()
            fixed_beta = results.params[1]
            df['beta'] = fixed_beta
            print(f"    Фиксированная Beta = {fixed_beta:.4f}")
        print("  - Расчет логарифмического спреда...")
        df['log_first'] = log_first
        df['log_second'] = log_second
        df['spread'] = log_first - df['beta'] * log_second
        
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
        df['correlation'] = log_first.rolling(
            window=self.config.BETA_LOOKBACK, 
            min_periods=self.config.BETA_LOOKBACK
        ).corr(log_second)
        
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
    # ...удалён блок статистики индикаторов, коинтеграции, Hurst, can_trade...
        
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
        fixed_capital = self.config.INITIAL_CAPITAL
        position_open = False
        position = None  # None, 'long', 'short'
        entry_data = None
        for signal in signals:
            if equity <= 0:
                print(f"\n⚠️ ВНИМАНИЕ: Капитал достиг нуля!")
                print(f"Остановка на {signal['timestamp']}")
                break

            # ENTRY LONG
            if signal['type'] == 'LONG_SPREAD_ENTRY' and position is None:
                timestamp = signal['timestamp']
                first_leg_price_entry = self.first_leg_data.loc[timestamp, 'close']
                second_leg_price_entry = self.second_leg_data.loc[timestamp, 'close']
                if getattr(self.config, 'SIMPLE_PROFIT_MODE', False):
                    position_size_usd = fixed_capital
                else:
                    position_size_usd = equity
                entry_data = {
                    'timestamp_entry': timestamp,
                    'type_entry': signal['type'],
                    'spread_entry': signal['spread'],
                    'z_score_entry': signal['z_score'],
                    'beta': signal['beta'],
                    'first_leg_price_entry': first_leg_price_entry,
                    'second_leg_price_entry': second_leg_price_entry,
                    'position_size_usd': position_size_usd,
                    'equity_before': equity
                }
                position = 'long'

            # ENTRY SHORT
            elif signal['type'] == 'SHORT_SPREAD_ENTRY' and position is None:
                timestamp = signal['timestamp']
                first_leg_price_entry = self.first_leg_data.loc[timestamp, 'close']
                second_leg_price_entry = self.second_leg_data.loc[timestamp, 'close']
                if getattr(self.config, 'SIMPLE_PROFIT_MODE', False):
                    position_size_usd = fixed_capital
                else:
                    position_size_usd = equity
                entry_data = {
                    'timestamp_entry': timestamp,
                    'type_entry': signal['type'],
                    'spread_entry': signal['spread'],
                    'z_score_entry': signal['z_score'],
                    'beta': signal['beta'],
                    'first_leg_price_entry': first_leg_price_entry,
                    'second_leg_price_entry': second_leg_price_entry,
                    'position_size_usd': position_size_usd,
                    'equity_before': equity
                }
                position = 'short'

            # EXIT LONG
            elif signal['type'] == 'LONG_SPREAD_EXIT' and position == 'long':
                timestamp = signal['timestamp']
                first_leg_price_exit = self.first_leg_data.loc[timestamp, 'close']
                second_leg_price_exit = self.second_leg_data.loc[timestamp, 'close']
                beta = entry_data['beta']
                total_weight = 1.0 + beta
                btc_weight = 1.0 / total_weight
                eth_weight = beta / total_weight
                first_leg_position_usd = entry_data['position_size_usd'] * btc_weight
                second_leg_position_usd = entry_data['position_size_usd'] * eth_weight
                first_leg_pnl_pct = (first_leg_price_exit - entry_data['first_leg_price_entry']) / entry_data['first_leg_price_entry'] * 100
                first_leg_pnl_usd = first_leg_position_usd * first_leg_pnl_pct / 100
                second_leg_pnl_pct = (entry_data['second_leg_price_entry'] - second_leg_price_exit) / entry_data['second_leg_price_entry'] * 100
                second_leg_pnl_usd = second_leg_position_usd * second_leg_pnl_pct / 100
                total_pnl_usd = first_leg_pnl_usd + second_leg_pnl_usd
                total_pnl_pct = (total_pnl_usd / entry_data['position_size_usd']) * 100
                commission_pct = abs(total_pnl_pct) * self.config.COMMISSION_RATE / 100
                net_pnl_pct = total_pnl_pct - commission_pct
                pnl_usd = entry_data['position_size_usd'] * net_pnl_pct / 100
                if getattr(self.config, 'SIMPLE_PROFIT_MODE', False):
                    equity = entry_data['equity_before'] + pnl_usd
                entry_record = {
                    'timestamp': entry_data['timestamp_entry'],
                    'type': entry_data['type_entry'],
                    'spread': entry_data['spread_entry'],
                    'z_score': entry_data['z_score_entry'],
                    'beta': entry_data['beta'],
                    'first_leg_price': entry_data['first_leg_price_entry'],
                    'second_leg_price': entry_data['second_leg_price_entry'],
                    'position_size_usd': entry_data['position_size_usd'],
                    'first_leg_weight': btc_weight,
                    'second_leg_weight': eth_weight,
                    'first_leg_position_usd': first_leg_position_usd,
                    'second_leg_position_usd': second_leg_position_usd,
                    'equity_before': entry_data['equity_before'],
                    'first_leg_pnl_%': None,
                    'second_leg_pnl_%': None,
                    'first_leg_pnl_usd': None,
                    'second_leg_pnl_usd': None,
                    'total_pnl_%': None,
                    'commission_%': None,
                    'net_pnl_%': None,
                    'pnl_usd': None,
                    'equity_after': None
                }
                self.trades.append(entry_record)
                exit_record = {
                    'timestamp': timestamp,
                    'type': signal['type'],
                    'spread': signal['spread'],
                    'z_score': signal['z_score'],
                    'beta': signal['beta'],
                    'first_leg_price': first_leg_price_exit,
                    'second_leg_price': second_leg_price_exit,
                    'position_size_usd': entry_data['position_size_usd'],
                    'first_leg_weight': btc_weight,
                    'second_leg_weight': eth_weight,
                    'first_leg_position_usd': first_leg_position_usd,
                    'second_leg_position_usd': second_leg_position_usd,
                    'equity_before': entry_data['equity_before'],
                    'first_leg_pnl_%': first_leg_pnl_pct,
                    'second_leg_pnl_%': second_leg_pnl_pct,
                    'first_leg_pnl_usd': first_leg_pnl_usd,
                    'second_leg_pnl_usd': second_leg_pnl_usd,
                    'total_pnl_%': total_pnl_pct,
                    'commission_%': commission_pct,
                    'net_pnl_%': net_pnl_pct,
                    'pnl_usd': pnl_usd,
                    'equity_after': equity
                }
                self.trades.append(exit_record)
                position = None
                entry_data = None

            # EXIT SHORT
            elif signal['type'] == 'SHORT_SPREAD_EXIT' and position == 'short':
                timestamp = signal['timestamp']
                first_leg_price_exit = self.first_leg_data.loc[timestamp, 'close']
                second_leg_price_exit = self.second_leg_data.loc[timestamp, 'close']
                beta = entry_data['beta']
                total_weight = 1.0 + beta
                btc_weight = 1.0 / total_weight
                eth_weight = beta / total_weight
                first_leg_position_usd = entry_data['position_size_usd'] * btc_weight
                second_leg_position_usd = entry_data['position_size_usd'] * eth_weight
                first_leg_pnl_pct = (entry_data['first_leg_price_entry'] - first_leg_price_exit) / entry_data['first_leg_price_entry'] * 100
                first_leg_pnl_usd = first_leg_position_usd * first_leg_pnl_pct / 100
                second_leg_pnl_pct = (second_leg_price_exit - entry_data['second_leg_price_entry']) / entry_data['second_leg_price_entry'] * 100
                second_leg_pnl_usd = second_leg_position_usd * second_leg_pnl_pct / 100
                total_pnl_usd = first_leg_pnl_usd + second_leg_pnl_usd
                total_pnl_pct = (total_pnl_usd / entry_data['position_size_usd']) * 100
                commission_pct = abs(total_pnl_pct) * self.config.COMMISSION_RATE / 100
                net_pnl_pct = total_pnl_pct - commission_pct
                pnl_usd = entry_data['position_size_usd'] * net_pnl_pct / 100
                if getattr(self.config, 'SIMPLE_PROFIT_MODE', False):
                    equity = entry_data['equity_before'] + pnl_usd
                entry_record = {
                    'timestamp': entry_data['timestamp_entry'],
                    'type': entry_data['type_entry'],
                    'spread': entry_data['spread_entry'],
                    'z_score': entry_data['z_score_entry'],
                    'beta': entry_data['beta'],
                    'first_leg_price': entry_data['first_leg_price_entry'],
                    'second_leg_price': entry_data['second_leg_price_entry'],
                    'position_size_usd': entry_data['position_size_usd'],
                    'first_leg_weight': btc_weight,
                    'second_leg_weight': eth_weight,
                    'first_leg_position_usd': first_leg_position_usd,
                    'second_leg_position_usd': second_leg_position_usd,
                    'equity_before': entry_data['equity_before'],
                    'first_leg_pnl_%': None,
                    'second_leg_pnl_%': None,
                    'first_leg_pnl_usd': None,
                    'second_leg_pnl_usd': None,
                    'total_pnl_%': None,
                    'commission_%': None,
                    'net_pnl_%': None,
                    'pnl_usd': None,
                    'equity_after': None
                }
                self.trades.append(entry_record)
                exit_record = {
                    'timestamp': timestamp,
                    'type': signal['type'],
                    'spread': signal['spread'],
                    'z_score': signal['z_score'],
                    'beta': signal['beta'],
                    'first_leg_price': first_leg_price_exit,
                    'second_leg_price': second_leg_price_exit,
                    'position_size_usd': entry_data['position_size_usd'],
                    'first_leg_weight': btc_weight,
                    'second_leg_weight': eth_weight,
                    'first_leg_position_usd': first_leg_position_usd,
                    'second_leg_position_usd': second_leg_position_usd,
                    'equity_before': entry_data['equity_before'],
                    'first_leg_pnl_%': first_leg_pnl_pct,
                    'second_leg_pnl_%': second_leg_pnl_pct,
                    'first_leg_pnl_usd': first_leg_pnl_usd,
                    'second_leg_pnl_usd': second_leg_pnl_usd,
                    'total_pnl_%': total_pnl_pct,
                    'commission_%': commission_pct,
                    'net_pnl_%': net_pnl_pct,
                    'pnl_usd': pnl_usd,
                    'equity_after': equity
                }
                self.trades.append(exit_record)
                position = None
                entry_data = None

            # Сохраняем equity
            self.equity_curve.append({
                'timestamp': signal['timestamp'],
                'equity': equity
            })
        
        return self.trades, self.equity_curve
    
    def print_statistics(self):
        print("\n=========================  BACKTEST RESULTS  =========================")
        trades_df = pd.DataFrame(self.trades)
        exit_trades = trades_df[trades_df['type'].str.contains('EXIT')].copy()
        if len(exit_trades) == 0:
            print("Нет закрытых сделок!")
            return

        # --- Финансовые метрики ---
        start_balance = self.config.INITIAL_CAPITAL
        final_balance = self.equity_curve[-1]['equity'] if self.equity_curve else start_balance
        total_pnl = final_balance - start_balance
        n_trades = len(exit_trades)
        gross_profit = exit_trades[exit_trades['pnl_usd'] > 0]['pnl_usd'].sum()
        gross_loss = exit_trades[exit_trades['pnl_usd'] < 0]['pnl_usd'].sum()
        total_return = (final_balance / start_balance - 1) * 100
        # Месячная доходность (CAGR)
        n_months = max(1, (pd.to_datetime(exit_trades['timestamp']).max() - pd.to_datetime(exit_trades['timestamp']).min()).days / 30.44)
        cagr = ((final_balance / start_balance) ** (1 / (n_months / 12)) - 1) * 100 if n_months > 0 else 0
        # Max drawdown
        equity_arr = np.array([x['equity'] for x in self.equity_curve])
        roll_max = np.maximum.accumulate(equity_arr)
        drawdown = roll_max - equity_arr
        max_drawdown = np.max(drawdown)
        max_drawdown_pct = max_drawdown / start_balance * 100

        print("------------------- Финансовые метрики -------------------")
        print(f"Капитал на начало      : {start_balance:12.2f}")
        print(f"Капитал на конец       : {final_balance:12.2f}")
        print(f"Суммарный PnL          : {total_pnl:12.2f}")
        print(f"Total return           : {total_return:8.2f} %")
        print(f"Месячная доходность    : {cagr:8.2f} % (CAGR)")
        print(f"Максимальная просадка  : {max_drawdown:12.2f} ({max_drawdown_pct:.2f} %)")
        print(f"Количество сделок      : {n_trades:6d}")
        print("----------------------------------------------------------")

        # --- Метрики по каждой ноге ---
        print("\n------------------- Метрики по каждой ноге -------------------")
        for leg, label in [("first_leg", "BTC/1st leg"), ("second_leg", "ETH/2nd leg")]:
            mean_weight = exit_trades[f'{leg}_weight'].mean()
            mean_pnl_pct = exit_trades[f'{leg}_pnl_%'].mean()
            max_pnl_pct = exit_trades[f'{leg}_pnl_%'].max()
            min_pnl_pct = exit_trades[f'{leg}_pnl_%'].min()
            mean_pnl_usd = exit_trades[f'{leg}_pnl_usd'].mean()
            print(f"{label:12}: Средний вес = {mean_weight:.4f}, Средний PnL = {mean_pnl_pct:.2f}%, Макс = {max_pnl_pct:.2f}%, Мин = {min_pnl_pct:.2f}%, Средний PnL $ = {mean_pnl_usd:.2f}")
        print("---------------------------------------------------------------")

        # --- Сделки, winrate, profit factor ---
        win_trades = exit_trades[exit_trades['net_pnl_%'] > 0]
        loss_trades = exit_trades[exit_trades['net_pnl_%'] < 0]
        win_rate = len(win_trades) / n_trades * 100 if n_trades else 0
        profit_factor = win_trades['pnl_usd'].sum() / abs(loss_trades['pnl_usd'].sum()) if len(loss_trades) else float('inf')
        avg_win = win_trades['pnl_usd'].mean() if len(win_trades) else 0
        avg_loss = loss_trades['pnl_usd'].mean() if len(loss_trades) else 0
        expectancy = exit_trades['pnl_usd'].mean() if n_trades else 0

        print("\n------------------- Метрики сделок -------------------")
        print(f"Win rate               : {win_rate:8.2f} %")
        print(f"Profit factor          : {profit_factor:8.2f}")
        print(f"Avg win / Avg loss     : {avg_win:8.2f} / {avg_loss:8.2f}")
        print(f"Expectancy (abs)       : {expectancy:8.2f}")
        print(f"Gross profit           : {gross_profit:12.2f}")
        print(f"Gross loss             : {gross_loss:12.2f}")
        print("-----------------------------------------------------")

        # --- Риск-метрики ---
        returns = exit_trades['pnl_usd'] / start_balance
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        sortino = returns.mean() / returns[returns < 0].std() * np.sqrt(252) if (returns < 0).std() > 0 else 0
        calmar = (total_pnl / max_drawdown) if max_drawdown > 0 else float('inf')
        recovery = (total_pnl / abs(gross_loss)) if gross_loss != 0 else float('inf')
        print("\n------------------- Метрики риска и волатильности -------------------")
        print(f"Sharpe ratio (trade)   : {sharpe:8.2f}")
        print(f"Sortino ratio          : {sortino:8.2f}")
        print(f"Calmar ratio           : {calmar:8.2f}")
        print(f"Recovery factor        : {recovery:8.2f}")
        print("--------------------------------------------------------------------")

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
        display_cols = ['timestamp', 'type', 'beta', 'z_score', 'first_leg_pnl_%', 'second_leg_pnl_%', 'net_pnl_%', 'pnl_usd', 'equity_after']
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
        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib.dates as mdates

        # --- 1. Дашборд по сделкам (equity, сигналы, сделки) ---
        print("\nСоздание дашборда по сделкам...")
        fig = plt.figure(figsize=(18, 16))
        gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 1, 1, 0.2], hspace=0.3, top=0.95, bottom=0.08)
        ax1 = fig.add_subplot(gs[0])  # Beta
        ax2 = fig.add_subplot(gs[1], sharex=ax1)  # Spread
        ax3 = fig.add_subplot(gs[2], sharex=ax1)  # Z-score + сигналы
        ax4 = fig.add_subplot(gs[3], sharex=ax1)  # Equity

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

        # --- 2. Статистический дашборд ---
        self.plot_stats_dashboard()
