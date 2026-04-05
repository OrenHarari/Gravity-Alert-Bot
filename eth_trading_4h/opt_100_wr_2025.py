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
    valid_months = ~d['Date'].dt.month.isin([5, 9, 12, 10])
    
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

def verify_2025_only():
    df = build_dataset()
    df_2025 = df[df['Date'].dt.year == 2025].reset_index(drop=True)
    
    print("Testing Immortal Matrix strictly on 2025...")
    sig = s_immortal_matrix(df_2025)
    
    for lev in [7.0, 8.0, 10.0, 12.0, 15.0]:
        tr = Engine(df_2025, leverage=lev).run(sig)
        
        raw_pnl = tr['pnl'].values
        eq = np.cumprod(1 + raw_pnl)
        ret = (eq[-1] - 1) * 100
        wr = (raw_pnl > 0).mean() * 100
        print(f"Lev {lev}x: +{ret:.1f}% | WR {wr:.1f}% | Trades {len(tr)}")

if __name__ == '__main__':
    verify_2025_only()
