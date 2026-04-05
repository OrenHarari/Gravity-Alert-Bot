import pandas as pd
import numpy as np
import os
import itertools
from eth_backtesting import build_dataset, rsi, ema, bollinger_bands, macd, atr, Engine
from export_data import compute_metrics
import warnings
warnings.filterwarnings('ignore')

def s_aggressive_pullback(df, e200_len, bb_std, rsi_os, sl_atr, tp_atr):
    d = df.copy()
    sma, upper, lower = bollinger_bands(d['Close'], n=20, std=bb_std)
    ml, _, _ = macd(d['Close'], 12, 26, 9)
    r = rsi(d['Close'], 14); e_trend = ema(d['Close'], e200_len); at = atr(d, 10)
    
    bull_rc = (d['Close'] > e_trend) & (ml > 0)
    bear_rc = (d['Close'] < e_trend) & (ml < 0)
    
    lc = bull_rc & ((d['Close'] <= lower) | (r < rsi_os))
    sc = bear_rc & ((d['Close'] >= upper) | (r > (100-rsi_os)))
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc; s['short_entry'] = sc
    s['long_exit'] = r > 65
    s['short_exit'] = r < 35
    
    s['sl'] = np.where(lc, d['Close'] - sl_atr * at, np.where(sc, d['Close'] + sl_atr * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + tp_atr * at, np.where(sc, d['Close'] - tp_atr * at, np.nan))
    return s

def run_extreme_optimizer():
    df = build_dataset()
    phase_a = df[~df['is_oos']].reset_index(drop=True)
    
    grid = {
        'e200_len': [100, 200],
        'bb_std': [1.5, 2.0],
        'rsi_os': [25, 30],
        'sl_atr': [1.5, 2.0],
        'tp_atr': [4.0, 5.0, 6.0],
        'leverage': [1.0, 2.0, 3.0] # Let's see what happens with up to 3x isolated leverage
    }
    
    keys, values = zip(*grid.items())
    permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    results = []
    
    for i, p in enumerate(permutations):
        sig = s_aggressive_pullback(phase_a, p['e200_len'], p['bb_std'], p['rsi_os'], p['sl_atr'], p['tp_atr'])
        tr = Engine(phase_a).run(sig)
        if len(tr) < 10: continue
        
        raw_pnl = tr['pnl'].values * p['leverage']
        # If any trade hits -100% (liquidation), we cap it at -100% 
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
    
    print("\n🚀 EXTREME PNL OPTIMIZATION (Top 10 Results) 🚀")
    for i, res in enumerate(results[:10]):
        p = res['params']
        print(f"Rank {i+1}: Return: +{res['ret']:.1f}% | WR: {res['wr']:.1f}% | PF: {res['pf']:.2f} | Trades: {res['n']}")
        print(f"Params: EMA={p['e200_len']}, BB={p['bb_std']}, RSI={p['rsi_os']}, SL={p['sl_atr']}x, TP={p['tp_atr']}x, Leverage={p['leverage']}x")

if __name__ == '__main__':
    run_extreme_optimizer()
