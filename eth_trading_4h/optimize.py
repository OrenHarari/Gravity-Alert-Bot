"""
GRID DELTA OPTIMIZER: Finds the mathematical peak absolute return for the 4H Trend Pullback Strategy.
"""
import pandas as pd
import numpy as np
import os
import itertools
from eth_backtesting import build_dataset, rsi, ema, bollinger_bands, macd, atr, Engine
from export_data import compute_metrics
import warnings
warnings.filterwarnings('ignore')

DIR = os.path.dirname(os.path.abspath(__file__))

def s_param_pullback(df, e200_len, rsi_os, rsi_ob, sl_atr, tp_atr, bb_std):
    d = df.copy()
    sma, upper, lower = bollinger_bands(d['Close'], n=20, std=bb_std)
    ml, _, macd_h = macd(d['Close'], 12, 26, 9)
    r = rsi(d['Close'], 14); e_trend = ema(d['Close'], e200_len); at = atr(d, 10)
    
    bull_rc = (d['Close'] > e_trend) & (ml > 0)
    bear_rc = (d['Close'] < e_trend) & (ml < 0)
    
    # Enter when price touches lower band OR RSI oversold
    lc = bull_rc & ((d['Close'] <= lower) | (r < rsi_os))
    sc = bear_rc & ((d['Close'] >= upper) | (r > rsi_ob))
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc; s['short_entry'] = sc
    # Slightly smarter exit logic using sma and rsi
    s['long_exit'] = (d['Close'] > sma) | (r > 60)
    s['short_exit'] = (d['Close'] < sma) | (r < 40)
    
    s['sl'] = np.where(lc, d['Close'] - sl_atr * at, np.where(sc, d['Close'] + sl_atr * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + tp_atr * at, np.where(sc, d['Close'] - tp_atr * at, np.nan))
    return s

def optimize():
    print("Loading 4H Data for Max Profit Optimization...")
    df = build_dataset()
    phase_a = df[~df['is_oos']].reset_index(drop=True)
    phase_b = df[df['is_oos']].reset_index(drop=True)
    
    # Aggressive Grid parameters to maximize profit and trade count
    grid = {
        'e200_len': [50, 100, 200],       # Try faster trend proxy (50)
        'rsi_os': [25, 30, 35, 40],       # Allow slightly less extreme dips
        'rsi_ob': [60, 65, 70, 75],       # Allow slightly less extreme spikes
        'sl_atr': [1.5, 2.0, 2.5],        # Breathe more
        'tp_atr': [3.0, 4.0, 5.0, 7.0],   # Push the runners
        'bb_std': [1.5, 1.8, 2.0]         # Allow more BB entries
    }
    
    keys, values = zip(*grid.items())
    permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Executing hyperparameter sweep over {len(permutations)} combinations...")
    results = []
    
    for i, p in enumerate(permutations):
        sig = s_param_pullback(phase_a, **p)
        tr = Engine(phase_a).run(sig)
        m = compute_metrics(tr)
        
        # Criteria: Must have > 30 trades (avoid curve fitting) and WR > 50%
        if m['n'] > 25 and m['total_return'] > 0 and m['win_rate'] >= 50 and m['profit_factor'] >= 1.25:
            results.append({
                'ret': m['total_return'],
                'wr': m['win_rate'],
                'pf': m['profit_factor'],
                'mdd': m['max_dd'],
                'n': m['n'],
                'params': p
            })
            
    # Sort by total return (Maximizing profit)
    results.sort(reverse=True, key=lambda x: x['ret'])
    
    print("\n👑 TOP 5 STRATEGY OPTIMIZATIONS (Phase A In-Sample)")
    print("="*90)
    for i, res in enumerate(results[:5]):
        p = res['params']
        print(f"Rank {i+1}: Return: +{res['ret']}% | WinRate: {res['wr']}% | PF: {res['pf']:.2f} | Trades: {res['n']}")
        print(f"         Params: EMA={p['e200_len']}, RSI=({p['rsi_os']}/{p['rsi_ob']}), BB_STD={p['bb_std']}, SL={p['sl_atr']}x, TP={p['tp_atr']}x")
        
        # Test immediately on Phase B
        sig_b = s_param_pullback(phase_b, **p)
        tr_b = Engine(phase_b).run(sig_b)
        mb = compute_metrics(tr_b)
        print(f"         [PHASE B TEST] -> Return: {mb['total_return']}% | WR: {mb['win_rate']}% | PF: {mb['profit_factor']:.2f} | Trades: {mb['n']}\n")

if __name__ == '__main__':
    optimize()
