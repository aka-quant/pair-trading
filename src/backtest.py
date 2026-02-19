"""
Модуль для бэктестирования стратегии
"""

import pandas as pd
import numpy as np
import quantstats as qs
from pathlib import Path
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

from config import StrategyConfig, DataConfig, BacktestConfig
from src.data_loader import DataLoader
from src.indicators import PairTradingIndicators
from src.strategy import PairTradingStrategy


class Backtester:
    """Класс для проведения бэктеста торговой стратегии"""
    
    def __init__(self,
                 strategy_config: StrategyConfig,
                 data_config: DataConfig,
                 backtest_config: BacktestConfig):
        """
        Инициализация бэктестера
        
        Args:
            strategy_config: Конфигурация стратегии
            data_config: Конфигурация данных
            backtest_config: Конфигурация бэктеста
        """
        self.strategy_config = strategy_config
        self.data_config = data_config
        self.backtest_config = backtest_config
        
        self.data_loader = DataLoader(
            cache_dir=data_config.cache_dir,
            use_cache=data_config.use_cache
        )
        
        self.strategy = PairTradingStrategy(
            entry_threshold=strategy_config.entry_threshold,
            exit_threshold=strategy_config.exit_threshold,
            position_size=strategy_config.position_size,
            maker_fee=strategy_config.maker_fee,
            taker_fee=strategy_config.taker_fee
        )
        
        self.data: Optional[pd.DataFrame] = None
        self.results: Optional[pd.DataFrame] = None
        self.trades: Optional[pd.DataFrame] = None
        self.metrics: Optional[Dict[str, Any]] = None
    
    def load_data(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Загрузка данных для бэктеста
        
        Args:
            force_reload: Принудительная перезагрузка данных
            
        Returns:
            DataFrame с данными пары
        """
        print("=" * 80)
        print("ЗАГРУЗКА ДАННЫХ")
        print("=" * 80)
        
        self.data = self.data_loader.fetch_pair_data(
            base_symbol=self.strategy_config.base_symbol,
            quote_symbol=self.strategy_config.quote_symbol,
            timeframe=self.data_config.timeframe,
            start_date=self.data_config.start_date,
            end_date=self.data_config.end_date,
            force_reload=force_reload
        )
        
        print(f"\nЗагружено данных: {len(self.data)} свечей")
        print(f"Период: с {self.data.index[0]} по {self.data.index[-1]}")
        
        return self.data
    
    def calculate_indicators(self) -> pd.DataFrame:
        """
        Расчет индикаторов
        
        Returns:
            DataFrame с индикаторами
        """
        if self.data is None:
            raise ValueError("Сначала загрузите данные с помощью load_data()")
        
        print("\n" + "=" * 80)
        print("РАСЧЕТ ИНДИКАТОРОВ")
        print("=" * 80)
        
        self.data = PairTradingIndicators.calculate_all_indicators(
            self.data,
            len_long=self.strategy_config.len_long,
            len_short=self.strategy_config.len_short,
            delta_window=self.strategy_config.delta_window
        )
        
        print(f"Индикаторы рассчитаны: {list(self.data.columns)}")
        
        return self.data
    
    def run_backtest(self) -> pd.DataFrame:
        """
        Запуск бэктеста
        
        Returns:
            DataFrame с результатами
        """
        if self.data is None or 'z_score' not in self.data.columns:
            raise ValueError("Сначала рассчитайте индикаторы с помощью calculate_indicators()")
        
        print("\n" + "=" * 80)
        print("ЗАПУСК БЭКТЕСТА")
        print("=" * 80)
        
        # Генерация сигналов
        self.results = self.strategy.generate_signals(self.data)
        
        # Расчет доходности
        self.results = self.strategy.calculate_returns(
            self.results,
            initial_capital=self.backtest_config.initial_capital
        )
        
        # Извлечение сделок
        self.trades = self.strategy.get_trades(self.results)
        
        print(f"\nВсего сделок: {len(self.trades)}")
        print(f"Начальный капитал: ${self.backtest_config.initial_capital:.2f}")
        print(f"Конечный капитал: ${self.results['equity'].iloc[-1]:.2f}")
        print(f"Общая доходность: {(self.results['cumulative_return'].iloc[-1] - 1) * 100:.2f}%")
        
        return self.results
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Расчет метрик стратегии
        
        Returns:
            Словарь с метриками
        """
        if self.results is None:
            raise ValueError("Сначала запустите бэктест с помощью run_backtest()")
        
        print("\n" + "=" * 80)
        print("РАСЧЕТ МЕТРИК")
        print("=" * 80)
        
        returns = self.results['net_return'].fillna(0)
        
        # Базовые метрики
        total_return = (self.results['cumulative_return'].iloc[-1] - 1) * 100
        
        # Метрики по сделкам
        if len(self.trades) > 0:
            win_rate = (self.trades['net_return'] > 0).sum() / len(self.trades) * 100
            avg_win = self.trades[self.trades['net_return'] > 0]['net_return'].mean() * 100
            avg_loss = self.trades[self.trades['net_return'] < 0]['net_return'].mean() * 100
            profit_factor = (
                abs(self.trades[self.trades['net_return'] > 0]['net_return'].sum()) /
                abs(self.trades[self.trades['net_return'] < 0]['net_return'].sum())
                if self.trades[self.trades['net_return'] < 0]['net_return'].sum() != 0 else np.inf
            )
            avg_trade_duration = self.trades['duration'].mean()
        else:
            win_rate = avg_win = avg_loss = profit_factor = avg_trade_duration = 0
        
        # Максимальная просадка
        max_drawdown = self.results['drawdown'].min() * 100
        
        # Коэффициент Шарпа (годовой)
        # Предполагаем, что данные почасовые
        if self.data_config.timeframe == '1h':
            periods_per_year = 365 * 24
        elif self.data_config.timeframe == '4h':
            periods_per_year = 365 * 6
        elif self.data_config.timeframe == '1d':
            periods_per_year = 365
        else:
            periods_per_year = 252  # По умолчанию для дневных данных
        
        sharpe_ratio = qs.stats.sharpe(returns, rf=self.backtest_config.risk_free_rate)
        sortino_ratio = qs.stats.sortino(returns, rf=self.backtest_config.risk_free_rate)
        calmar_ratio = qs.stats.calmar(returns)
        
        self.metrics = {
            'total_return': total_return,
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'avg_trade_duration': avg_trade_duration,
            'final_equity': self.results['equity'].iloc[-1],
        }
        
        # Вывод метрик
        print(f"\n{'Метрика':<30} {'Значение':>20}")
        print("-" * 52)
        for key, value in self.metrics.items():
            if isinstance(value, (int, np.integer)):
                print(f"{key:<30} {value:>20}")
            elif isinstance(value, pd.Timedelta):
                print(f"{key:<30} {str(value):>20}")
            else:
                print(f"{key:<30} {value:>20.2f}")
        
        return self.metrics
    
    def plot_results(self, save: bool = True):
        """
        Визуализация результатов бэктеста
        
        Args:
            save: Сохранить ли графики
        """
        if self.results is None:
            raise ValueError("Сначала запустите бэктест")
        
        print("\n" + "=" * 80)
        print("ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
        print("=" * 80)
        
        # Настройка стиля
        sns.set_style("darkgrid")
        
        # Создание фигуры с подграфиками
        fig, axes = plt.subplots(4, 1, figsize=(16, 12))
        
        # 1. График капитала
        axes[0].plot(self.results.index, self.results['equity'], 
                    label='Equity', linewidth=2, color='blue')
        axes[0].set_title('Изменение капитала', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Капитал (USDT)', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. График просадки
        axes[1].fill_between(self.results.index, self.results['drawdown'] * 100, 0,
                             alpha=0.3, color='red')
        axes[1].plot(self.results.index, self.results['drawdown'] * 100,
                    linewidth=2, color='red')
        axes[1].set_title('Просадка', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Drawdown (%)', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        # 3. График z-score с сигналами
        axes[2].plot(self.results.index, self.results['z_score'],
                    label='Z-Score', linewidth=1, color='black', alpha=0.7)
        axes[2].axhline(y=self.strategy_config.entry_threshold, color='red',
                       linestyle='--', label=f'Entry threshold: ±{self.strategy_config.entry_threshold}σ')
        axes[2].axhline(y=-self.strategy_config.entry_threshold, color='red',
                       linestyle='--')
        axes[2].axhline(y=0, color='green', linestyle='-', alpha=0.5, label='Exit threshold')
        
        # Отметки входа и выхода
        entries = self.results[self.results['signal'] == 1]
        exits = self.results[self.results['signal'] == -1]
        axes[2].scatter(entries.index, entries['z_score'],
                       color='green', marker='^', s=100, label='Entry', zorder=5)
        axes[2].scatter(exits.index, exits['z_score'],
                       color='red', marker='v', s=100, label='Exit', zorder=5)
        
        axes[2].set_title('Z-Score и торговые сигналы', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Z-Score', fontsize=12)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # 4. График позиций
        axes[3].fill_between(self.results.index, self.results['position'], 0,
                            alpha=0.3)
        axes[3].plot(self.results.index, self.results['position'],
                    linewidth=2, color='blue')
        axes[3].set_title('Позиции', fontsize=14, fontweight='bold')
        axes[3].set_ylabel('Position', fontsize=12)
        axes[3].set_xlabel('Дата', fontsize=12)
        axes[3].set_yticks([-1, 0, 1])
        axes[3].set_yticklabels(['SHORT', 'NONE', 'LONG'])
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save and self.backtest_config.save_plots:
            plots_dir = Path(self.backtest_config.plots_dir)
            plots_dir.mkdir(parents=True, exist_ok=True)
            filename = plots_dir / 'backtest_results.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"График сохранен: {filename}")
        
        plt.show()
    
    def generate_quantstats_report(self, save: bool = True):
        """
        Генерация отчета с помощью QuantStats
        
        Args:
            save: Сохранить ли HTML отчет
        """
        if self.results is None:
            raise ValueError("Сначала запустите бэктест")
        
        print("\n" + "=" * 80)
        print("ГЕНЕРАЦИЯ QUANTSTATS ОТЧЕТА")
        print("=" * 80)
        
        returns = self.results['net_return'].fillna(0)
        
        # Настройка QuantStats
        qs.extend_pandas()
        
        if save and self.backtest_config.save_reports:
            reports_dir = Path(self.backtest_config.reports_dir)
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            # HTML отчет
            html_file = reports_dir / 'quantstats_report.html'
            qs.reports.html(returns, output=str(html_file), title='Pair Trading Strategy Backtest')
            print(f"HTML отчет сохранен: {html_file}")
        
        # Вывод метрик в консоль
        print("\n" + "=" * 80)
        print("QUANTSTATS МЕТРИКИ")
        print("=" * 80)
        qs.reports.metrics(returns, mode='full')
    
    def run_full_backtest(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Полный цикл бэктестирования
        
        Args:
            force_reload: Принудительная перезагрузка данных
            
        Returns:
            Словарь с метриками
        """
        # 1. Загрузка данных
        self.load_data(force_reload=force_reload)
        
        # 2. Расчет индикаторов
        self.calculate_indicators()
        
        # 3. Запуск бэктеста
        self.run_backtest()
        
        # 4. Расчет метрик
        metrics = self.calculate_metrics()
        
        # 5. Визуализация
        self.plot_results(save=self.backtest_config.save_plots)
        
        # 6. QuantStats отчет
        self.generate_quantstats_report(save=self.backtest_config.save_reports)
        
        return metrics
