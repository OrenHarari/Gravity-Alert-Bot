import pandas as pd
import numpy as np
import itertools
import os
from eth_backtesting import build_dataset, atr, ema, bollinger_bands, rsi, macd, Engine
import warnings
warnings.filterwarnings('ignore')

def s_flash_crash_sweep(df, dev_atr, tp_ema_len):
    """
    Catches flash crashes. Deep discounts only.
    """
    d = df.copy()
    e200 = ema(d['Close'], 200)
    at = atr(d, 14)
    r = rsi(d['Close'], 14)
    
    lc = (d['Close'] < e200 - dev_atr * at) & (r < 25)
    sc = (d['Close'] > e200 + dev_atr * at) & (r > 75)
    
    e_tp = ema(d['Close'], tp_ema_len)
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc
    s['short_entry'] = sc
    # Exit when price normalizes back to a fast moving average
    s['long_exit'] = d['Close'] > e_tp
    s['short_exit'] = d['Close'] < e_tp
    
    # Very loose SL because we are already buying cheap
    s['sl'] = np.where(lc, d['Close'] - 5.0 * at, np.where(sc, d['Close'] + 5.0 * at, np.nan))
    s['tp'] = np.nan
    return s

def s_filtered_turtle(df, entry_len, exit_len, atr_mult, ema_filter):
    """
    Turtle Breakout but with strict regime filtering to avoid chops.
    """
    d = df.copy()
    high_max = d['High'].shift(1).rolling(entry_len).max()
    low_min = d['Low'].shift(1).rolling(entry_len).min()
    exit_high = d['High'].shift(1).rolling(exit_len).max()
    exit_low = d['Low'].shift(1).rolling(exit_len).min()
    
    e_trend = ema(d['Close'], ema_filter)
    at = atr(d, 20)
    
    lc = (d['Close'] > high_max) & (d['Close'] > e_trend)
    sc = (d['Close'] < low_min) & (d['Close'] < e_trend)
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc
    s['short_entry'] = sc
    s['long_exit'] = d['Close'] < exit_low
    s['short_exit'] = d['Close'] > exit_high
    s['sl'] = np.where(lc, d['Close'] - atr_mult * at, np.where(sc, d['Close'] + atr_mult * at, np.nan))
    s['tp'] = np.nan
    return s

def s_momentum_ignition(df, bb_window, bb_std, rsi_min):
    """
    Momentum Ignition: Enter after a volatility squeeze breaks out very strongly.
    """
    d = df.copy()
    sma, upper, lower = bollinger_bands(d['Close'], n=bb_window, std=bb_std)
    ml, _, h = macd(d['Close'], 12, 26, 9)
    r = rsi(d['Close'], 14)
    at = atr(d, 10)
    
    squeeze = ((upper - lower) / sma).rolling(50).quantile(0.1) # low volatility
    cur_width = (upper - lower) / sma
    
    # Ignition: we burst out of low volatility
    recent_squeeze = (cur_width < squeeze).rolling(10).max() > 0
    lc = recent_squeeze & (d['Close'] > upper) & (r > rsi_min) & (h > 0)
    sc = recent_squeeze & (d['Close'] < lower) & (r < (100-rsi_min)) & (h < 0)
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc
    s['short_entry'] = sc
    s['long_exit'] = r < 50
    s['short_exit'] = r > 50
    s['sl'] = np.where(lc, d['Close'] - 1.5 * at, np.where(sc, d['Close'] + 1.5 * at, np.nan))
    s['tp'] = np.nan # let it ride until RSI flips
    return s

def run_mission_150():
    df = build_dataset()
    phase_a = df[~df['is_oos']].reset_index(drop=True)
    
    results = []
    
    # 1. Filtered Turtle
    print("Optimization 1: Regime-Filtered Turtle...")
    grid1 = {'entry_len': [20, 40, 55, 80], 'exit_len': [10, 20], 'atr_mult': [2.0, 3.0], 'ema_filter': [50, 100, 200]}
    for p in [dict(zip(grid1.keys(), v)) for v in itertools.product(*grid1.values())]:
        sig = s_filtered_turtle(phase_a, **p)
        tr = Engine(phase_a).run(sig)
        if len(tr) > 15:
            raw_pnl = tr['pnl'].values; ret = (np.cumprod(1 + raw_pnl)[-1] - 1) * 100
            pf = raw_pnl[raw_pnl>0].sum() / abs(raw_pnl[raw_pnl<0].sum()) if (raw_pnl<0).any() else 99
            if ret > 100: results.append({'name': 'Filtered Turtle', 'ret': ret, 'wr': (raw_pnl>0).mean()*100, 'pf': pf, 'n': len(tr), 'p': p})

    # 2. Flash Crash Sweep
    print("Optimization 2: Flash Crash Sweeper...")
    grid2 = {'dev_atr': [3.0, 4.0, 5.0, 6.0], 'tp_ema_len': [20, 50]}
    for p in [dict(zip(grid2.keys(), v)) for v in itertools.product(*grid2.values())]:
        sig = s_flash_crash_sweep(phase_a, **p)
        tr = Engine(phase_a).run(sig)
        if len(tr) > 5:
            raw_pnl = tr['pnl'].values; ret = (np.cumprod(1 + raw_pnl)[-1] - 1) * 100
            pf = raw_pnl[raw_pnl>0].sum() / abs(raw_pnl[raw_pnl<0].sum()) if (raw_pnl<0).any() else 99
            if ret > 50: results.append({'name': 'Flash Crash Sweep', 'ret': ret, 'wr': (raw_pnl>0).mean()*100, 'pf': pf, 'n': len(tr), 'p': p})

    # 3. Momentum Ignition
    print("Optimization 3: Momentum Ignition Squeeze...")
    grid3 = {'bb_window': [20, 40], 'bb_std': [1.5, 2.0, 2.5], 'rsi_min': [60, 65, 70]}
    for p in [dict(zip(grid3.keys(), v)) for v in itertools.product(*grid3.values())]:
        sig = s_momentum_ignition(phase_a, **p)
        tr = Engine(phase_a).run(sig)
        if len(tr) > 15:
            raw_pnl = tr['pnl'].values; ret = (np.cumprod(1 + raw_pnl)[-1] - 1) * 100
            pf = raw_pnl[raw_pnl>0].sum() / abs(raw_pnl[raw_pnl<0].sum()) if (raw_pnl<0).any() else 99
            if ret > 80: results.append({'name': 'Momentum Ignition', 'ret': ret, 'wr': (raw_pnl>0).mean()*100, 'pf': pf, 'n': len(tr), 'p': p})

    results.sort(reverse=True, key=lambda x: x['ret'])
    print("\n🚀 MISSION 150% PNL SEARCH (UNLEVERAGED) 🚀")
    if not results: print("No combination cracked the vault.")
    for i, res in enumerate(results[:10]):
        print(f"Rank {i+1} [{res['name']}]: Return: +{res['ret']:.1f}% | WR: {res['wr']:.1f}% | PF: {res['pf']:.2f} | Trades: {res['n']}")
        print(f"Params: {res['p']}")

if __name__ == '__main__':
    run_mission_150()
