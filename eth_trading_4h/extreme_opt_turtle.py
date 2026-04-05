import pandas as pd
import numpy as np
import itertools
import os
from eth_backtesting import build_dataset, atr, Engine
import warnings
warnings.filterwarnings('ignore')

def s_turtle_breakout(df, entry_len, exit_len, atr_mult):
    """
    Classic Turtle Trading / Donchian Breakout system.
    Very out of the box for crypto (we usually fade, here we follow breakouts).
    """
    d = df.copy()
    high_max = d['High'].shift(1).rolling(entry_len).max()
    low_min = d['Low'].shift(1).rolling(entry_len).min()
    
    exit_high = d['High'].shift(1).rolling(exit_len).max()
    exit_low = d['Low'].shift(1).rolling(exit_len).min()
    
    at = atr(d, 20)
    
    lc = d['Close'] > high_max
    sc = d['Close'] < low_min
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc
    s['short_entry'] = sc
    
    s['long_exit'] = d['Close'] < exit_low
    s['short_exit'] = d['Close'] > exit_high
    
    s['sl'] = np.where(lc, d['Close'] - atr_mult * at, np.where(sc, d['Close'] + atr_mult * at, np.nan))
    s['tp'] = np.nan # No TP, we let the exit rule act as a trailing stop
    return s

def run_turtle_search():
    df = build_dataset()
    phase_a = df[~df['is_oos']].reset_index(drop=True)
    
    grid = {
        'entry_len': [20, 40, 55],
        'exit_len': [10, 20],
        'atr_mult': [2.0, 3.0, 4.0]
    }
    keys, values = zip(*grid.items())
    perms = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    results = []
    
    print("Testing 'Turtle Breakout' combinations...")
    for p in perms:
        sig = s_turtle_breakout(phase_a, **p)
        tr = Engine(phase_a, leverage=1.0).run(sig)
        if len(tr) < 10: continue
        
        raw_pnl = tr['pnl'].values
        eq = np.cumprod(1 + raw_pnl)
        ret = (eq[-1] - 1) * 100
        
        gp = raw_pnl[raw_pnl > 0].sum()
        gl = abs(raw_pnl[raw_pnl < 0].sum())
        pf = gp / gl if gl > 0 else 99
        wr = (raw_pnl > 0).mean() * 100
        
        results.append({'name': 'Turtle', 'ret': ret, 'wr': wr, 'pf': pf, 'n': len(tr), 'params': p})

    results.sort(reverse=True, key=lambda x: x['ret'])
    
    print("\n🚀 UNLEVERAGED TURTLE SEARCH 🚀")
    if not results:
        print("No strategy hit the lower threshold bounds.")
    for i, res in enumerate(results[:5]):
        print(f"Rank {i+1} [{res['name']}]: Return: +{res['ret']:.1f}% | WR: {res['wr']:.1f}% | PF: {res['pf']:.2f} | Trades: {res['n']}")
        print(f"Params: {res['params']}")

if __name__ == '__main__':
    run_turtle_search()
