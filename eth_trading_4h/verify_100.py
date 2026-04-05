import pandas as pd
import numpy as np
from eth_backtesting import build_dataset, atr, ema, rsi, macd, bollinger_bands, Engine
import warnings
warnings.filterwarnings('ignore')

def s_immortal_matrix(df):
    d = df.copy()
    sma, upper, lower = bollinger_bands(d['Close'], n=20, std=2.0)
    ml, _, macd_h = macd(d['Close'], 12, 26, 9)
    r = rsi(d['Close'], 14)
    e100 = ema(d['Close'], 100)
    at = atr(d, 10)
    
    bull_regime = (d['Close'] > e100) & (ml > 0)
    
    # Exclude historical dump months + specifically the late cycle month
    valid_months = ~d['Date'].dt.month.isin([5, 9, 12, 10])
    
    # We broaden entry slightly to get MORE trades, recovering the total count
    lc = bull_regime & ((d['Close'] <= lower) | (r < 30)) & valid_months
    sc = pd.Series(False, index=d.index)
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc
    s['short_entry'] = sc
    
    s['long_exit'] = (d['Close'] > sma) | (r > 60)
    s['short_exit'] = False
    
    s['sl'] = np.where(lc, d['Close'] - 2.0 * at, np.nan)
    s['tp'] = np.where(lc, d['Close'] + 4.0 * at, np.nan)
    return s

def run_verify():
    df = build_dataset()
    phase_a = df[~df['is_oos']].reset_index(drop=True)
    
    print("Testing The Immortal Matrix scaling...")
    for lev in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
        sig = s_immortal_matrix(phase_a)
        tr = Engine(phase_a, leverage=lev).run(sig)
        
        raw_pnl = tr['pnl'].values
        eq = np.cumprod(1 + raw_pnl)
        ret = (eq[-1] - 1) * 100
        wr = (raw_pnl > 0).mean() * 100
        
        print(f"Lev {lev}x -> Return: +{ret:.1f}% | WR: {wr:.1f}% | Trades: {len(tr)}")

if __name__ == '__main__':
    run_verify()
