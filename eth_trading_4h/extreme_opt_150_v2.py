import pandas as pd
import numpy as np
import itertools
from eth_backtesting import build_dataset, atr, ema, Engine
import warnings
warnings.filterwarnings('ignore')

def s_filtered_turtle(df, entry_len, exit_len, atr_mult, ema_filter):
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

def run_mission_150():
    df = build_dataset()
    phase_a = df[~df['is_oos']].reset_index(drop=True)
    results = []
    
    # Very granular search to hit 150% on Turtle Breakout
    grid = {
        'entry_len': [10, 15, 20, 25, 30, 35, 40],
        'exit_len': [3, 5, 7, 10, 15],
        'atr_mult': [1.5, 2.0, 2.5, 3.0, 3.5],
        'ema_filter': [50, 100, 150, 200]
    }
    
    keys, values = zip(*grid.items())
    perms = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    for p in perms:
        sig = s_filtered_turtle(phase_a, **p)
        tr = Engine(phase_a).run(sig)
        if len(tr) > 20: # Ensure statistically significant
            raw_pnl = tr['pnl'].values
            ret = (np.cumprod(1 + raw_pnl)[-1] - 1) * 100
            if ret > 120:
                gp = raw_pnl[raw_pnl>0].sum()
                gl = abs(raw_pnl[raw_pnl<0].sum())
                pf = gp / gl if gl > 0 else 99
                wr = (raw_pnl > 0).mean() * 100
                results.append({'ret': ret, 'wr': wr, 'pf': pf, 'n': len(tr), 'p': p})

    results.sort(reverse=True, key=lambda x: x['ret'])
    print("\n🚀 MISSION 150% PNL ARCHIVE 🚀")
    for i, res in enumerate(results[:5]):
        print(f"Rank {i+1}: Return: +{res['ret']:.1f}% | WR: {res['wr']:.1f}% | PF: {res['pf']:.2f} | Trades: {res['n']}")
        print(f"Params: {res['p']}")

if __name__ == '__main__':
    run_mission_150()
