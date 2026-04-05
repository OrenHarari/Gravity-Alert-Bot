import pandas as pd
import numpy as np
import itertools
import os
from eth_backtesting import build_dataset, ema, atr, macd, bollinger_bands, rsi, Engine
import warnings
warnings.filterwarnings('ignore')

def s_avalanche_breakout(df, e_slow, e_fast, atr_len, sl_mult, tp_mult):
    """
    Out of the box: 'Avalanche Breakout'
    Instead of waiting for deep pullbacks, we BUY strength.
    When Fast EMA > Slow EMA and price breaks out of the Bollinger Band with momentum.
    """
    d = df.copy()
    e_s = ema(d['Close'], e_slow)
    e_f = ema(d['Close'], e_fast)
    
    # Momentum 
    ml, _, h = macd(d['Close'], 12, 26, 9)
    sma, upper, lower = bollinger_bands(d['Close'], n=20, std=1.5)
    at = atr(d, atr_len)
    
    bull = e_f > e_s
    bear = e_f < e_s
    
    # Entry: Strong push outside or touching the bands in direction of trend
    lc = bull & (d['Close'] > upper) & (h > 0) & (h > h.shift(1))
    sc = bear & (d['Close'] < lower) & (h < 0) & (h < h.shift(1))
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc
    s['short_entry'] = sc
    
    # Fast exit to protect profits on reversal
    s['long_exit'] = d['Close'] < e_f
    s['short_exit'] = d['Close'] > e_f
    
    s['sl'] = np.where(lc, d['Close'] - sl_mult * at, np.where(sc, d['Close'] + sl_mult * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + tp_mult * at, np.where(sc, d['Close'] - tp_mult * at, np.nan))
    return s

def s_mean_reversion_frenzy(df, rsi_ob, rsi_os, sl_mult, tp_mult):
    """
    High-Frequency Mean Reversion. Fade every extreme move.
    """
    d = df.copy()
    r = rsi(d['Close'], 14)
    at = atr(d, 10)
    sma, upper, lower = bollinger_bands(d['Close'], n=20, std=2.5)
    
    # Enter when price heavily deviates
    lc = (d['Close'] < lower) | (r < rsi_os)
    sc = (d['Close'] > upper) | (r > rsi_ob)
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc
    s['short_entry'] = sc
    s['long_exit'] = r > 50
    s['short_exit'] = r < 50
    
    s['sl'] = np.where(lc, d['Close'] - sl_mult * at, np.where(sc, d['Close'] + sl_mult * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + tp_mult * at, np.where(sc, d['Close'] - tp_mult * at, np.nan))
    return s

def run_unleveraged_search():
    df = build_dataset()
    phase_a = df[~df['is_oos']].reset_index(drop=True)
    
    # STRAT 1: AVALANCHE BREAKOUT
    grid1 = {
        'e_slow': [100, 200],
        'e_fast': [20, 50],
        'atr_len': [10, 14],
        'sl_mult': [1.5, 2.0, 3.0],
        'tp_mult': [4.0, 6.0, 8.0, 10.0]
    }
    keys1, values1 = zip(*grid1.items())
    perms1 = [dict(zip(keys1, v)) for v in itertools.product(*values1)]
    
    # STRAT 2: MR FRENZY
    grid2 = {
        'rsi_ob': [70, 75, 80],
        'rsi_os': [20, 25, 30],
        'sl_mult': [1.0, 1.5, 2.0],
        'tp_mult': [2.0, 3.0, 5.0]
    }
    keys2, values2 = zip(*grid2.items())
    perms2 = [dict(zip(keys2, v)) for v in itertools.product(*values2)]

    results = []
    
    print("Testing 'Avalanche Breakout' combinations...")
    for p in perms1:
        sig = s_avalanche_breakout(phase_a, **p)
        tr = Engine(phase_a, leverage=1.0).run(sig)
        if len(tr) < 20: continue
        
        raw_pnl = tr['pnl'].values
        eq = np.cumprod(1 + raw_pnl)
        ret = (eq[-1] - 1) * 100
        
        gp = raw_pnl[raw_pnl > 0].sum()
        gl = abs(raw_pnl[raw_pnl < 0].sum())
        pf = gp / gl if gl > 0 else 99
        wr = (raw_pnl > 0).mean() * 100
        
        if ret > 50:
            results.append({'name': 'Avalanche', 'ret': ret, 'wr': wr, 'pf': pf, 'n': len(tr), 'params': p})

    print("Testing 'Mean Reversion Frenzy' combinations...")
    for p in perms2:
        sig = s_mean_reversion_frenzy(phase_a, **p)
        tr = Engine(phase_a, leverage=1.0).run(sig)
        if len(tr) < 20: continue
        
        raw_pnl = tr['pnl'].values
        eq = np.cumprod(1 + raw_pnl)
        ret = (eq[-1] - 1) * 100
        
        gp = raw_pnl[raw_pnl > 0].sum()
        gl = abs(raw_pnl[raw_pnl < 0].sum())
        pf = gp / gl if gl > 0 else 99
        wr = (raw_pnl > 0).mean() * 100
        
        if ret > 50:
            results.append({'name': 'Mean Reversion', 'ret': ret, 'wr': wr, 'pf': pf, 'n': len(tr), 'params': p})

    results.sort(reverse=True, key=lambda x: x['ret'])
    
    print("\n🚀 UNLEVERAGED 100%+ PNL SEARCH 🚀")
    if not results:
        print("No strategy hit the lower threshold bounds.")
    for i, res in enumerate(results[:10]):
        print(f"Rank {i+1} [{res['name']}]: Return: +{res['ret']:.1f}% | WR: {res['wr']:.1f}% | PF: {res['pf']:.2f} | Trades: {res['n']}")
        print(f"Params: {res['params']}")

if __name__ == '__main__':
    run_unleveraged_search()
