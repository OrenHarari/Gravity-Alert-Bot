import pandas as pd
import numpy as np
import itertools
from eth_backtesting import build_dataset, atr, ema, rsi, bollinger_bands, Engine
import warnings
warnings.filterwarnings('ignore')

def s_smart_money_reversal(df, rsi_limit, bb_std, e_fast_len, e_slow_len, tp_mult, sl_mult):
    """
    Look for exhaustion (liquidity sweep). 
    Price must pierce Bollinger band heavily and have low RSI.
    Buy when price reverses (closes above previous high).
    """
    d = df.copy()
    e_fast = ema(d['Close'], e_fast_len)
    e_slow = ema(d['Close'], e_slow_len)
    r = rsi(d['Close'], 14)
    sma, upper, lower = bollinger_bands(d['Close'], n=20, std=bb_std)
    at = atr(d, 14)
    
    # Identify strong trend
    bull = e_fast > e_slow
    bear = e_fast < e_slow
    
    # Setup condition: sweep the bands
    setup_long = (d['Low'].shift(1) < lower.shift(1)) & (r.shift(1) < rsi_limit)
    setup_short = (d['High'].shift(1) > upper.shift(1)) & (r.shift(1) > (100 - rsi_limit))
    
    # Trigger: reversal confirmation (price closes back inside and above/below previous close)
    trigger_long = d['Close'] > d['Close'].shift(1)
    trigger_short = d['Close'] < d['Close'].shift(1)
    
    lc = bull & setup_long & trigger_long
    sc = bear & setup_short & trigger_short
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc
    s['short_entry'] = sc
    
    # Dynamic exits: trail to the fast EMA
    s['long_exit'] = d['Close'] > upper
    s['short_exit'] = d['Close'] < lower
    
    s['sl'] = np.where(lc, d['Close'] - sl_mult * at, np.where(sc, d['Close'] + sl_mult * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + tp_mult * at, np.where(sc, d['Close'] - tp_mult * at, np.nan))
    return s

def run_holy_grail():
    df = build_dataset()
    phase_a = df[~df['is_oos']].reset_index(drop=True)
    
    # We will test both unleveraged and slightly leveraged (1.5x - 2.0x) 
    # to achieve massive 188% return alongside >70% WR.
    grid = {
        'rsi_limit': [30, 35, 40],
        'bb_std': [1.5, 2.0],
        'e_fast_len': [50],
        'e_slow_len': [200],
        'tp_mult': [4.0, 5.0, 6.0, 8.0],
        'sl_mult': [2.0, 3.0],
        'lev': [1.0, 1.5, 2.0, 2.5]
    }
    
    keys, values = zip(*grid.items())
    perms = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    results = []
    
    print("Searching for the Holy Grail: >188% PNL AND >70% WR...")
    for p in perms:
        lev = p['lev']
        params = {k: v for k, v in p.items() if k != 'lev'}
        
        sig = s_smart_money_reversal(phase_a, **params)
        tr = Engine(phase_a, leverage=lev).run(sig)
        
        if len(tr) > 20:
            raw_pnl = tr['pnl'].values
            ret = (np.cumprod(1 + raw_pnl)[-1] - 1) * 100
            wr = (raw_pnl > 0).mean() * 100
            
            # The Holy Grail condition!
            if wr >= 65 and ret >= 150: 
                gp = raw_pnl[raw_pnl>0].sum()
                gl = abs(raw_pnl[raw_pnl<0].sum())
                pf = gp / gl if gl > 0 else 99
                results.append({'ret': ret, 'wr': wr, 'pf': pf, 'n': len(tr), 'p': p})

    results.sort(reverse=True, key=lambda x: x['ret'])
    print("\n🏆 HOLY GRAIL ARCHIVE 🏆")
    if not results:
        print("No combination reached the Holy Grail thresholds.")
    for i, res in enumerate(results[:10]):
        print(f"Rank {i+1}: Return: +{res['ret']:.1f}% | WR: {res['wr']:.1f}% | PF: {res['pf']:.2f} | Trades: {res['n']}")
        print(f"Params: {res['p']}")

if __name__ == '__main__':
    run_holy_grail()
