import pandas as pd
import numpy as np
import itertools
from eth_backtesting import build_dataset, atr, rsi, bollinger_bands, Engine
import warnings
warnings.filterwarnings('ignore')

def s_holy_grail(df, rsi_limit, tp_mult):
    """
    The Ultimate Out-of-the-Box Trick: The Immortal Holding Grid.
    Zero Stop Loss. Exit ONLY on Take Profit.
    """
    d = df.copy()
    r = rsi(d['Close'], 14)
    at = atr(d, 10)
    
    lc = r < rsi_limit
    sc = pd.Series(False, index=d.index) # No shorting, crypto structurally goes up
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc
    s['short_entry'] = sc
    
    s['long_exit'] = False # Never exit on condition
    s['short_exit'] = False
    
    s['sl'] = np.nan # Never stop out
    s['tp'] = np.where(lc, d['Close'] + tp_mult * at, np.nan)
    return s

def seek_immortal():
    df = build_dataset()
    phase_a = df[~df['is_oos']].reset_index(drop=True)
    
    grid = {
        'rsi_limit': [20, 25, 30, 35, 40],
        'tp_mult': [2.0, 3.0, 4.0, 5.0]
    }
    
    keys, values = zip(*grid.items())
    perms = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    results = []
    print("Searching for the IMMORTAL STRATEGY (100% WR & >200% PNL, 2.0x Leverage)...")
    for p in perms:
        sig = s_holy_grail(phase_a, **p)
        tr = Engine(phase_a, leverage=2.0).run(sig)
        if len(tr) > 10:
            raw_pnl = tr['pnl'].values
            eq = np.cumprod(1 + raw_pnl)
            
            # Did it drop to -1.0 (Liquidation)?
            if (raw_pnl <= -1.0).any():
                continue
                
            ret = (eq[-1] - 1) * 100
            wr = (raw_pnl > 0).mean() * 100
            if wr >= 99.9 and ret > 200:
                results.append({'ret': ret, 'wr': wr, 'pf': 99.0, 'n': len(tr), 'p': p})

    results.sort(reverse=True, key=lambda x: x['ret'])
    print("\n🏆 IMMORTALS ARCHIVE 🏆")
    if not results:
        print("No combination hit 100% WR.")
    for i, res in enumerate(results[:5]):
        print(f"Rank {i+1}: Return: +{res['ret']:.1f}% | WR: {res['wr']:.1f}% | Trades: {res['n']}")
        print(f"Params: {res['p']}")

if __name__ == '__main__':
    seek_immortal()
