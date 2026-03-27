"""
================================================================================
BTC 4H — ROUND 2 OPTIMIZATION
================================================================================
Based on Round 1 findings:
- S7 (BB Mean Reversion) showed ✅ in Phase A — needs parameter tuning for B
- S5 (Golden Cross Sniper) had excellent Phase A returns — needs WR improvement  
- S8 (Triple Confirmation) — best absolute return in Phase A, close to passing
- S11 (Momentum Cascade) — ✅ Phase B, needs A fix

Round 2: Hybridize and refine the best concepts.
================================================================================
"""
import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(DIR, "BTC_4H_2Y.csv")
COMMISSION = 0.001

def build_dataset():
    df = pd.read_csv(DATA_FILE)
    df.columns = [c.strip().strip('"') for c in df.columns]
    date_col = 'Date' if 'Date' in df.columns else 'Datetime'
    df['Date'] = pd.to_datetime(df[date_col], format='mixed', utc=True).dt.tz_localize(None)
    for c in ['Open','High','Low','Close']:
        if df[c].dtype == object: df[c] = df[c].str.replace(',', '').astype(float)
    df = df.sort_values('Date').reset_index(drop=True)
    if 'Volume' not in df.columns: df['Volume'] = 1.0
    df = df[['Date','Open','High','Low','Close','Volume']]
    split_idx = int(len(df) * 0.8)
    df['is_oos'] = df.index >= split_idx
    if split_idx > 52:
        df.loc[split_idx-52:split_idx-1, 'is_oos'] = True
        df.loc[split_idx-52:split_idx-1, 'is_warmup'] = True
    else: df['is_warmup'] = False
    df['is_warmup'] = df.get('is_warmup', False).fillna(False)
    return df

ema = lambda s,n: s.ewm(span=n,adjust=False).mean()
def atr(df,n=14):
    h,l,c=df['High'],df['Low'],df['Close']; pc=c.shift(1)
    return pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1).ewm(com=n-1,min_periods=n).mean()
def macd(s,f=12,sl=26,sig=9):
    ml=ema(s,f)-ema(s,sl); return ml,ema(ml,sig),ml-ema(ml,sig)
def rsi(s, n=14):
    d = s.diff(); u = d.clip(lower=0).ewm(com=n-1, min_periods=n).mean()
    v = (-d.clip(upper=0)).ewm(com=n-1, min_periods=n).mean()
    return 100 - 100 / (1 + u / v.replace(0, np.nan))
def bollinger_bands(s, n=20, std=2):
    sma = s.rolling(window=n).mean(); roll_std = s.rolling(window=n).std()
    return sma, sma + (roll_std * std), sma - (roll_std * std)
def stochastic(df, k_period=14, d_period=3):
    low_min = df['Low'].rolling(k_period).min(); high_max = df['High'].rolling(k_period).max()
    k = 100 * (df['Close'] - low_min) / (high_max - low_min); d = k.rolling(d_period).mean()
    return k, d

class Engine:
    def __init__(self,df):
        self.df=df.copy().reset_index(drop=True)
        self.wm=df.get('is_warmup', pd.Series(False,index=df.index)).values
        self.trades=[]; self._p=None
    def run(self,sig):
        df=self.df; sig=sig.reset_index(drop=True)
        for i in range(len(df)):
            row=df.iloc[i]; s=sig.iloc[i]; wm=bool(self.wm[i])
            if self._p:
                p=self._p; hi,lo=row['High'],row['Low']; r=None
                if p['t']=='long':
                    if lo<=p['sl']:   r,ep='SL',p['sl']
                    elif hi>=p['tp']: r,ep='TP',p['tp']
                    elif s.get('long_exit',False) or s.get('short_entry',False): r,ep='sig',row['Close']
                else:
                    if hi>=p['sl']:   r,ep='SL',p['sl']
                    elif lo<=p['tp']: r,ep='TP',p['tp']
                    elif s.get('short_exit',False) or s.get('long_entry',False): r,ep='sig',row['Close']
                if r: self._rec(ep,row['Date'],r); self._p=None
            if not self._p and not wm:
                if s.get('long_entry',False) and not np.isnan(s.get('sl',np.nan)):
                    self._p=dict(t='long',entry=row['Close'],sl=s['sl'],tp=s['tp'],ed=row['Date'])
                elif s.get('short_entry',False) and not np.isnan(s.get('sl',np.nan)):
                    self._p=dict(t='short',entry=row['Close'],sl=s['sl'],tp=s['tp'],ed=row['Date'])
        if self._p:
            lr=df.iloc[-1]; self._rec(lr['Close'],lr['Date'],'end'); self._p=None
        return pd.DataFrame(self.trades)
    def _rec(self,ep,ed,r):
        p=self._p; e=p['entry']
        pnl=((ep-e)/e if p['t']=='long' else (e-ep)/e)-2*COMMISSION
        self.trades.append(dict(type=p['t'],entry_date=p['ed'],exit_date=ed,
                                entry=e,exit=ep,pnl=pnl,reason=r,win=pnl>0))

def rpt(tr,label):
    if tr is None or len(tr)==0:
        print(f"  [{label}]: 0 trades.")
        return dict(label=label,n=0,wr=0,pf=0,aw=0,al=0,ret=0,mdd=0,sh=0,ok=False)
    pnl=tr['pnl'].values; w=tr['win'].values
    n=len(pnl); nw=w.sum(); wr=nw/n
    gp=pnl[pnl>0].sum() if (pnl>0).any() else 0
    gl=abs(pnl[pnl<0].sum()) if (pnl<0).any() else 1e-9
    pf=gp/gl; aw=pnl[pnl>0].mean()*100 if (pnl>0).any() else 0; al=pnl[pnl<0].mean()*100 if (pnl<0).any() else 0
    eq=np.cumprod(1+pnl); ret=eq[-1]-1
    peak=np.maximum.accumulate(eq); mdd=((eq-peak)/peak).min()
    sh=pnl.mean()/pnl.std()*np.sqrt(52) if pnl.std()>0 else 0
    ok=wr>=0.50 and pf>=1.30
    print(f"  {label:60s} N={n:3d} WR={wr*100:4.0f}% PF={pf:5.2f} Ret={ret*100:+7.1f}% MDD={mdd*100:5.1f}% {'✅' if ok else '❌'}")
    return dict(label=label,n=n,wr=wr,pf=pf,aw=aw,al=al,ret=ret,mdd=mdd,sh=sh,ok=ok)

# ─── ROUND 2 STRATEGIES ─────────────────────────────────────────────────────

def s_hybrid_trend_reversion(df, rsi_enter=30, rsi_exit=55, bb_std=2.0, sl_mult=2.5, tp_mult=4.0):
    """
    HYBRID A: BB Mean Reversion (best WR from R1) + Triple Confirmation filter.
    Only buy BB dips when MACD and Stochastic also confirm.
    """
    d = df.copy()
    sma, upper, lower = bollinger_bands(d['Close'], n=20, std=bb_std)
    r = rsi(d['Close'], 14)
    ml, sg, h = macd(d['Close'], 12, 26, 9)
    k, d_line = stochastic(d, 14, 3)
    e100 = ema(d['Close'], 100)
    at = atr(d, 14)
    
    # Long: Below lower BB + RSI oversold + MACD recovering + Stochastic turning up
    lc = (d['Close'] < lower) & (r < rsi_enter) & (h > h.shift(1)) & (k > k.shift(1))
    # Short: Above upper BB + RSI overbought + MACD fading + Stochastic turning down
    sc = (d['Close'] > upper) & (r > (100-rsi_enter)) & (h < h.shift(1)) & (k < k.shift(1))
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc; s['short_entry'] = sc
    s['long_exit'] = (d['Close'] > sma) | (r > rsi_exit)
    s['short_exit'] = (d['Close'] < sma) | (r < (100-rsi_exit))
    s['sl'] = np.where(lc, d['Close'] - sl_mult * at, np.where(sc, d['Close'] + sl_mult * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + tp_mult * at, np.where(sc, d['Close'] - tp_mult * at, np.nan))
    return s

def s_adaptive_momentum(df):
    """
    HYBRID B: Adaptive Momentum — changes behavior based on volatility regime.
    High vol = mean reversion, Low vol = trend follow.
    """
    d = df.copy()
    at = atr(d, 14)
    at_slow = at.rolling(50).mean()
    high_vol = at > at_slow * 1.2
    
    r = rsi(d['Close'], 14)
    sma, upper, lower = bollinger_bands(d['Close'], n=20, std=2.0)
    ml, sg, h = macd(d['Close'], 12, 26, 9)
    e50 = ema(d['Close'], 50)
    
    # In high vol: mean revert
    lc_mr = high_vol & (d['Close'] < lower) & (r < 30) & (h > h.shift(1))
    sc_mr = high_vol & (d['Close'] > upper) & (r > 70) & (h < h.shift(1))
    
    # In low vol: trend follow (MACD cross + trend)
    h_cross_up = (h > 0) & (h.shift(1) <= 0)
    h_cross_dn = (h < 0) & (h.shift(1) >= 0)
    lc_tf = (~high_vol) & h_cross_up & (d['Close'] > e50) & (r > 45) & (r < 65)
    sc_tf = (~high_vol) & h_cross_dn & (d['Close'] < e50) & (r > 35) & (r < 55)
    
    lc = lc_mr | lc_tf
    sc = sc_mr | sc_tf
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc; s['short_entry'] = sc
    s['long_exit'] = np.where(high_vol, (d['Close'] > sma) | (r > 60), (h < 0) | (r > 75))
    s['short_exit'] = np.where(high_vol, (d['Close'] < sma) | (r < 40), (h > 0) | (r < 25))
    s['sl'] = np.where(lc, d['Close'] - 2.5 * at, np.where(sc, d['Close'] + 2.5 * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 4.0 * at, np.where(sc, d['Close'] - 4.0 * at, np.nan))
    return s

def s_golden_sniper_v2(df):
    """
    HYBRID C: Golden Cross Sniper V2.
    Uses EMA 50/200 for regime, then Stochastic for entry timing.
    Wider SL/TP for BTC's bigger moves.
    """
    d = df.copy()
    e50 = ema(d['Close'], 50)
    e200 = ema(d['Close'], 200)
    r = rsi(d['Close'], 14)
    k, d_line = stochastic(d, 14, 3)
    at = atr(d, 14)
    
    bull_regime = e50 > e200
    bear_regime = e50 < e200
    
    # In bull regime: buy stochastic oversold crosses
    stoch_buy = (k > d_line) & (k.shift(1) <= d_line.shift(1)) & (k < 25)
    stoch_sell = (k < d_line) & (k.shift(1) >= d_line.shift(1)) & (k > 75)
    
    lc = bull_regime & stoch_buy
    sc = bear_regime & stoch_sell
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc; s['short_entry'] = sc
    s['long_exit'] = (k > 80) | (~bull_regime)
    s['short_exit'] = (k < 20) | (~bear_regime)
    s['sl'] = np.where(lc, d['Close'] - 3.0 * at, np.where(sc, d['Close'] + 3.0 * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 6.0 * at, np.where(sc, d['Close'] - 6.0 * at, np.nan))
    return s

def s_volatility_regime_v2(df):
    """
    HYBRID D: Enhanced BB squeeze + regime.
    Detect the squeeze, wait for the expansion, ride it.
    """
    d = df.copy()
    sma, upper, lower = bollinger_bands(d['Close'], n=20, std=2.0)
    bb_width = (upper - lower) / sma
    bb_mean = bb_width.rolling(100).mean()
    
    # Tighter squeeze threshold
    squeezed = bb_width < bb_mean * 0.6
    was_squeezed = squeezed.rolling(6).max() > 0
    
    # Now detect the explosion
    r = rsi(d['Close'], 14)
    at = atr(d, 14)
    d['body'] = (d['Close'] - d['Open']).abs()
    d['avg_body'] = d['body'].rolling(20).mean()
    
    big_up = (d['Close'] > d['Open']) & (d['body'] > d['avg_body'] * 2) & was_squeezed
    big_dn = (d['Close'] < d['Open']) & (d['body'] > d['avg_body'] * 2) & was_squeezed
    
    lc = big_up & (r > 55)
    sc = big_dn & (r < 45)
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc; s['short_entry'] = sc
    s['long_exit'] = r > 75
    s['short_exit'] = r < 25
    s['sl'] = np.where(lc, d['Close'] - 1.5 * at, np.where(sc, d['Close'] + 1.5 * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 5.0 * at, np.where(sc, d['Close'] - 5.0 * at, np.nan))
    return s

def s_rsi_divergence_smart(df):
    """
    HYBRID E: Smart RSI — buys RSI dips, sells RSI pops, with price action confirmation.
    The key insight: exit quickly if the trade goes wrong (tight signal-based exit).
    """
    d = df.copy()
    r = rsi(d['Close'], 14)
    e50 = ema(d['Close'], 50)
    at = atr(d, 14)
    ml, sg, h = macd(d['Close'], 12, 26, 9)
    
    # Buy: RSI drops below 30 AND current candle is green (buyers stepping in)
    green = d['Close'] > d['Open']
    red = d['Close'] < d['Open']
    
    lc = (r < 30) & green & (h > h.shift(1))
    sc = (r > 70) & red & (h < h.shift(1))
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc; s['short_entry'] = sc
    s['long_exit'] = (r > 55) | ((r < 25) & red)  # Exit if RSI recovers OR if it drops lower
    s['short_exit'] = (r < 45) | ((r > 75) & green)
    s['sl'] = np.where(lc, d['Close'] - 2.0 * at, np.where(sc, d['Close'] + 2.0 * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 3.5 * at, np.where(sc, d['Close'] - 3.5 * at, np.nan))
    return s

def s_extreme_fear_buyer(df):
    """
    HYBRID F: Only buy at extreme fear moments (multiple oversold signals).
    Extremely selective, high conviction.
    """
    d = df.copy()
    r = rsi(d['Close'], 7)  # Fast RSI for quicker signal
    sma, upper, lower = bollinger_bands(d['Close'], n=20, std=2.5)
    k, d_line = stochastic(d, 14, 3)
    at = atr(d, 14)
    e100 = ema(d['Close'], 100)
    
    # Extreme oversold: fast RSI < 20, below lower BB, stochastic < 15
    extreme_sell = (r < 20) & (d['Close'] < lower) & (k < 15)
    # Must not be a total market collapse (price still within range of e100)
    not_collapse = d['Close'] > e100 * 0.88
    # Green candle (buyers coming in)
    green = d['Close'] > d['Open']
    
    lc = extreme_sell & not_collapse & green
    
    # Extreme overbought
    extreme_buy = (r > 80) & (d['Close'] > upper) & (k > 85)
    red = d['Close'] < d['Open']
    sc = extreme_buy & red & (d['Close'] < e100 * 1.12)
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc; s['short_entry'] = sc
    s['long_exit'] = r > 50
    s['short_exit'] = r < 50
    s['sl'] = np.where(lc, d['Close'] - 3.0 * at, np.where(sc, d['Close'] + 3.0 * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 5.0 * at, np.where(sc, d['Close'] - 5.0 * at, np.nan))
    return s

# ─── GRID SEARCH FOR HYBRID A ───────────────────────────────────────────────
def grid_search_hybrid_a(df):
    """Grid search over Hybrid A parameters."""
    print("\n  --- GRID SEARCH: Hybrid A (BB + Triple Confirmation) ---")
    best = None
    for rsi_enter in [25, 30, 35]:
        for rsi_exit in [50, 55, 60]:
            for bb_std in [2.0, 2.5]:
                for sl_mult in [2.0, 2.5, 3.0]:
                    for tp_mult in [3.5, 4.0, 5.0]:
                        sig = s_hybrid_trend_reversion(df, rsi_enter, rsi_exit, bb_std, sl_mult, tp_mult)
                        tr = Engine(df).run(sig)
                        if len(tr) < 3: continue
                        pnl = tr['pnl'].values; w = tr['win'].values
                        n = len(pnl); wr = w.sum()/n
                        gp = pnl[pnl>0].sum() if (pnl>0).any() else 0
                        gl = abs(pnl[pnl<0].sum()) if (pnl<0).any() else 1e-9
                        pf = gp/gl
                        eq = np.cumprod(1+pnl); ret = eq[-1]-1
                        if wr >= 0.50 and pf >= 1.30:
                            score = pf * wr * (1+ret)
                            if best is None or score > best['score']:
                                best = dict(rsi_enter=rsi_enter, rsi_exit=rsi_exit, bb_std=bb_std,
                                           sl_mult=sl_mult, tp_mult=tp_mult, 
                                           wr=wr, pf=pf, ret=ret, n=n, score=score)
    if best:
        print(f"  BEST: RSI_enter={best['rsi_enter']} RSI_exit={best['rsi_exit']} BB_std={best['bb_std']}")
        print(f"        SL={best['sl_mult']}x TP={best['tp_mult']}x")
        print(f"        N={best['n']} WR={best['wr']*100:.0f}% PF={best['pf']:.2f} Ret={best['ret']*100:+.1f}%")
    else:
        print("  No passing configuration found.")
    return best

# ─── RUNNER ──────────────────────────────────────────────────────────────────
def run_all():
    df_raw = build_dataset()
    phase_a = df_raw[~df_raw['is_oos']]
    phase_b = df_raw[df_raw['is_oos']]
    
    strats = [
        ("Hybrid A: BB + Triple Confirmation", lambda df: s_hybrid_trend_reversion(df)),
        ("Hybrid B: Adaptive Momentum (Regime Switch)", s_adaptive_momentum),
        ("Hybrid C: Golden Sniper V2 (Stoch)", s_golden_sniper_v2),
        ("Hybrid D: BB Squeeze V2", s_volatility_regime_v2),
        ("Hybrid E: Smart RSI + PA", s_rsi_divergence_smart),
        ("Hybrid F: Extreme Fear Buyer", s_extreme_fear_buyer),
    ]
    
    print(f"\n{'='*90}")
    print(f"  BTC 4H — ROUND 2 OPTIMIZATION — Phase A (In-Sample 80%)")
    print(f"  {len(phase_a)} bars | {phase_a['Date'].min()} → {phase_a['Date'].max()}")
    print(f"{'='*90}")
    
    for name, fn in strats:
        tr = Engine(phase_a).run(fn(phase_a))
        rpt(tr, f"A | {name}")
    
    # Grid search
    best_params = grid_search_hybrid_a(phase_a)
    
    print(f"\n{'='*90}")
    print(f"  BTC 4H — ROUND 2 OPTIMIZATION — Phase B (Out-of-Sample 20%)")
    print(f"  {len(phase_b)} bars | {phase_b['Date'].min()} → {phase_b['Date'].max()}")
    print(f"{'='*90}")
    
    for name, fn in strats:
        tr = Engine(phase_b).run(fn(phase_b))
        rpt(tr, f"B | {name}")
    
    # Test grid search best on Phase B
    if best_params:
        sig = s_hybrid_trend_reversion(phase_b, best_params['rsi_enter'], best_params['rsi_exit'],
                                        best_params['bb_std'], best_params['sl_mult'], best_params['tp_mult'])
        tr = Engine(phase_b).run(sig)
        rpt(tr, "B | Hybrid A OPTIMIZED (Grid)")

if __name__ == '__main__':
    run_all()
