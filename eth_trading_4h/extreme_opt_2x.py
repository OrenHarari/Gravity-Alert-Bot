import pandas as pd
import numpy as np
import os
import itertools
from eth_backtesting import build_dataset, rsi, ema, bollinger_bands, macd, atr, Engine
from export_data import compute_metrics
import warnings
warnings.filterwarnings('ignore')

def s_aggressive_pullback(df, e200_len, rsi_limit, sl_atr, tp_atr, hold_period):
    """
    Looks for a solid trend, drops SL to survive turbulence, then lets runners fly.
    """
    d = df.copy()
    sma, upper, lower = bollinger_bands(d['Close'], n=20, std=2.0)
    ml, _, _ = macd(d['Close'], 12, 26, 9)
    r = rsi(d['Close'], 14)
    e_trend = ema(d['Close'], e200_len)
    at = atr(d, 10)
    
    bull_rc = (d['Close'] > e_trend) & (ml > 0)
    bear_rc = (d['Close'] < e_trend) & (ml < 0)
    
    lc = bull_rc & ((d['Close'] <= lower) | (r < rsi_limit))
    sc = bear_rc & ((d['Close'] >= upper) | (r > (100 - rsi_limit)))
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc
    s['short_entry'] = sc
    
    # We use very loose exits, meaning we rely on SL/TP mostly 
    s['long_exit'] = r > (50 + hold_period) # hold_period proxy
    s['short_exit'] = r < (50 - hold_period)
    
    s['sl'] = np.where(lc, d['Close'] - sl_atr * at, np.where(sc, d['Close'] + sl_atr * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + tp_atr * at, np.where(sc, d['Close'] - tp_atr * at, np.nan))
    return s

def run_extreme_optimizer():
    df = build_dataset()
    phase_a = df[~df['is_oos']].reset_index(drop=True)
    
    grid = {
        'e200_len': [50, 100, 200],
        'rsi_limit': [25, 30, 35],
        'sl_atr': [1.0, 1.5, 2.0, 3.0], 
        'tp_atr': [5.0, 6.0, 8.0, 10.0],
        'hold_period': [10, 15, 20] # exit RSI threshold
    }
    
    keys, values = zip(*grid.items())
    permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    results = []
    
    leverage = 2.0
    for i, p in enumerate(permutations):
        sig = s_aggressive_pullback(phase_a, p['e200_len'], p['rsi_limit'], p['sl_atr'], p['tp_atr'], p['hold_period'])
        tr = Engine(phase_a).run(sig)
        if len(tr) < 10: continue
        
        raw_pnl = tr['pnl'].values * leverage
        raw_pnl = np.clip(raw_pnl, -1.0, float('inf'))
        
        eq = np.cumprod(1 + raw_pnl)
        ret_pct = (eq[-1] - 1) * 100
        
        wins = max(1, (raw_pnl > 0).sum())
        losses = max(1, (raw_pnl < 0).sum())
        wr = wins / len(raw_pnl) * 100
        
        gp = raw_pnl[raw_pnl > 0].sum()
        gl = abs(raw_pnl[raw_pnl < 0].sum())
        pf = gp / gl if gl > 0 else 99.9
        
        results.append({
            'ret': ret_pct,
            'wr': wr,
            'pf': pf,
            'params': p,
            'n': len(tr)
        })
            
    results.sort(reverse=True, key=lambda x: x['ret'])
    
    print("\n🚀 2X LEVERAGE MAXIMUM PNL SEARCH 🚀")
    for i, res in enumerate(results[:3]):
        p = res['params']
        print(f"Rank {i+1}: Return: +{res['ret']:.1f}% | WR: {res['wr']:.1f}% | PF: {res['pf']:.2f} | Trades: {res['n']}")
        print(f"Params: EMA={p['e200_len']}, RSI={p['rsi_limit']}, SL={p['sl_atr']}x, TP={p['tp_atr']}x, HOLD={p['hold_period']}")

if __name__ == '__main__':
    run_extreme_optimizer()
