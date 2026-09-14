import os
import sys
import pandas as pd

try:
    import quantstats as qs
except Exception as e:
    print('MISSING_QUANTSTATS', e)
    print('Please run: pip install quantstats')
    sys.exit(2)


def load_equity(path='trades_log.xlsx'):
    df = pd.read_excel(path)
    # expect timestamp and equity_after
    ts_col = None
    for c in df.columns:
        if c.lower() in ('timestamp', 'time', 'date'):
            ts_col = c
            break
    eq_col = None
    for c in df.columns:
        if 'equity' in c.lower():
            eq_col = c
            break
    if ts_col is None or eq_col is None:
        raise ValueError('timestamp or equity column not found in trades_log.xlsx')
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.sort_values(ts_col)
    s = pd.Series(df[eq_col].values, index=df[ts_col])
    s = s.groupby(level=0).last()
    # hourly index 24/7
    idx = pd.date_range(start=s.index.min(), end=s.index.max(), freq='H')
    s_hour = s.reindex(idx, method='ffill').ffill()
    s_hour = s_hour.dropna()
    returns = s_hour.pct_change().dropna()
    return s_hour, returns


def main():
    out_dir = 'backtest_results'
    os.makedirs(out_dir, exist_ok=True)
    eq_hourly, returns = load_equity()
    print('periods per year used: 8760 (hourly 24/7)')
    ppy = 24 * 365
    # manual metrics
    mean_r = returns.mean()
    std_r = returns.std(ddof=1)
    sharpe = (mean_r / std_r) * (ppy ** 0.5) if std_r != 0 else float('nan')
    downside = returns[returns < 0]
    dd = downside.std(ddof=1) if len(downside) > 0 else float('nan')
    sortino = (mean_r / dd) * (ppy ** 0.5) if dd and dd != 0 else float('nan')
    total_return = eq_hourly.iloc[-1] / eq_hourly.iloc[0] - 1
    years = (eq_hourly.index[-1] - eq_hourly.index[0]).total_seconds() / (365.25 * 24 * 3600)
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else float('nan')
    # max drawdown
    cum = eq_hourly
    running_max = cum.cummax()
    drawdown = (cum / running_max) - 1
    max_dd = drawdown.min()
    vol = returns.std(ddof=1) * (ppy ** 0.5)
    print(f'SHARPE: {sharpe}')
    print(f'SORTINO: {sortino}')
    print(f'CAGR: {cagr}')
    print(f'TOTAL_RETURN: {total_return}')
    print(f'MAX_DRAWDOWN: {max_dd}')
    print(f'VOL_ANNUAL: {vol}')
    out_path = os.path.join(out_dir, 'quantstats_report.html')
    qs.reports.html(returns, output=out_path, title='Backtest QuantStats Report', periods_per_year=24*365)
    print('Saved report to', out_path)


if __name__ == '__main__':
    main()
