import pandas as pd
import numpy as np
import itertools
from eth_backtesting import build_dataset, atr, ema, rsi, macd, bollinger_bands, Engine
import warnings
warnings.filterwarnings('ignore')

def s_dual_momentum_pullback(df):
    d = df.copy()
    ml, _, hm = macd(d['Close'], 12, 26, 9)
    r = rsi(d['Close'], 14)
    at = atr(d, 10)
    sma, upper, lower = bollinger_bands(d['Close'], n=20, std=2.0)
    e = ema(d['Close'], 200)
    bull = (d['Close'] > e) & (ml > 0)
    bear = (d['Close'] < e) & (ml < 0)
    lc = bull & (r < 30) & (d['Close'] < lower)
    sc = bear & (r > 70) & (d['Close'] > upper)
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc
    s['short_entry'] = sc
    s['long_exit'] = r > 60
    s['short_exit'] = r < 40
    s['sl'] = np.where(lc, d['Close'] - 3.0 * at, np.where(sc, d['Close'] + 3.0 * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 5.0 * at, np.where(sc, d['Close'] - 5.0 * at, np.nan))
    return s

def run_holy_grail_v2():
    df = build_dataset()
    phase_a = df[~df['is_oos']].reset_index(drop=True)
    
    results = []
    print("Testing Trend Pullback with Kelly Scaling (Leverage)...")
    for lev in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        sig = s_dual_momentum_pullback(phase_a)
        tr = Engine(phase_a, leverage=lev).run(sig)
        
        if len(tr) > 5:
            raw_pnl = tr['pnl'].values
            eq = np.cumprod(1 + raw_pnl)
            ret = (eq[-1] - 1) * 100
            wr = (raw_pnl > 0).mean() * 100
            
            gp = raw_pnl[raw_pnl>0].sum()
            gl = abs(raw_pnl[raw_pnl<0].sum())
            pf = gp / gl if gl > 0 else 99
            results.append({'ret': ret, 'wr': wr, 'pf': pf, 'n': len(tr), 'lev': lev})

    results.sort(reverse=True, key=lambda x: x['ret'])
    print("\n🏆 HOLY GRAIL ARCHIVE 🏆")
    for i, res in enumerate(results[:5]):
        print(f"Rank {i+1}: Return: +{res['ret']:.1f}% | WR: {res['wr']:.1f}% | PF: {res['pf']:.2f} | Trades: {res['n']}")
        print(f"Leverage Used: {res['lev']}x")

if __name__ == '__main__':
    run_holy_grail_v2()
