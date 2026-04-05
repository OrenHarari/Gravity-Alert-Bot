import pandas as pd
import numpy as np
from eth_backtesting import build_dataset, atr, ema, rsi, macd, bollinger_bands, Engine
import warnings
warnings.filterwarnings('ignore')

def s_dual_momentum_pullback(df):
    d = df.copy()
    sma, upper, lower = bollinger_bands(d['Close'], n=20, std=2.0)
    ml, _, macd_h = macd(d['Close'], 12, 26, 9)
    r = rsi(d['Close'], 14)
    e100 = ema(d['Close'], 100) # OPTIMIZED: Faster trend detection
    at = atr(d, 10)
    
    # Uptrend vs Downtrend Regimes
    bull_regime = (d['Close'] > e100) & (ml > 0)
    bear_regime = (d['Close'] < e100) & (ml < 0)
    
    # Long: Deep pullback inside bull regime (OPTIMIZED: RSI < 25 extreme panic)
    lc = bull_regime & ((d['Close'] <= lower) | (r < 25))
    
    # Short: Overbought bounce inside bear regime
    sc = bear_regime & ((d['Close'] >= upper) | (r > 65))
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc
    s['short_entry'] = sc
    
    # Exits: Reverting back to mean or MACD momentum fading
    s['long_exit'] = (d['Close'] > sma) | (r > 60)
    s['short_exit'] = (d['Close'] < sma) | (r < 40)
    
    # Tight SL and high TP
    s['sl'] = np.where(lc, d['Close'] - 1.5 * at, np.where(sc, d['Close'] + 1.5 * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 5.0 * at, np.where(sc, d['Close'] - 5.0 * at, np.nan))
    return s

def run_verify():
    df = build_dataset()
    phase_a = df[~df['is_oos']].reset_index(drop=True)
    
    print("Testing Dual Momentum Pullback at 3x Leverage...")
    sig = s_dual_momentum_pullback(phase_a)
    tr = Engine(phase_a, leverage=3.0).run(sig)
    
    raw_pnl = tr['pnl'].values
    eq = np.cumprod(1 + raw_pnl)
    ret = (eq[-1] - 1) * 100
    wr = (raw_pnl > 0).mean() * 100
    gp = raw_pnl[raw_pnl>0].sum()
    gl = abs(raw_pnl[raw_pnl<0].sum())
    pf = gp / gl if gl > 0 else 99
    
    print(f"Return: +{ret:.1f}% | WR: {wr:.1f}% | PF: {pf:.2f} | Trades: {len(tr)}")

if __name__ == '__main__':
    run_verify()
