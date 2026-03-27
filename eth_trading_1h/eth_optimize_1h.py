"""
================================================================================
ETH 1H — STRATEGY RESEARCH LAB (1H Specific)
================================================================================
4H strategies translated poorly to 1H due to fakeouts.
Building new models specifically tailored for the 1H noise profile.
================================================================================
"""
import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(DIR, "ETH_1H_2Y.csv")
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
        df['is_warmup'] = False
        df.loc[0:100, 'is_warmup'] = True
    else: df['is_warmup'] = False
    return df

ema = lambda s,n: s.ewm(span=n,adjust=False).mean()
def atr(df,n=14):
    h,l,c=df['High'],df['Low'],df['Close']; pc=c.shift(1)
    return pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1).ewm(com=n-1,min_periods=n).mean()
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
    ok=wr>=0.50 and pf>=1.30
    print(f"  {label:60s} N={n:3d} WR={wr*100:4.0f}% PF={pf:5.2f} Ret={ret*100:+7.1f}% MDD={mdd*100:5.1f}% {'✅' if ok else '❌'}")
    return dict(n=n,wr=wr,pf=pf,ret=ret)

# ─── 1H FOCUSED STRATEGIES ──────────────────────────────────────────────────

def s_1h_mean_reversion_scalp(df):
    """
    1H Mean Reversion Scalper: Highly selective, fast TP.
    """
    d = df.copy()
    e200 = ema(d['Close'], 200)
    sma, upper, lower = bollinger_bands(d['Close'], n=20, std=2.5) # Wide bands
    r = rsi(d['Close'], 10)
    at = atr(d, 14)
    
    lc = (d['Close'] < lower) & (r < 25) & (d['Close'] > e200)
    sc = (d['Close'] > upper) & (r > 75) & (d['Close'] < e200)
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc
    s['short_entry'] = sc
    s['long_exit'] = d['Close'] > sma
    s['short_exit'] = d['Close'] < sma
    
    # Very tight SL and fast TP
    s['sl'] = np.where(lc, d['Close'] - 1.5 * at, np.where(sc, d['Close'] + 1.5 * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 2.0 * at, np.where(sc, d['Close'] - 2.0 * at, np.nan))
    return s

def s_1h_momentum_breakout(df):
    """
    1H Breakout: High momentum candles breaking ranges.
    """
    d = df.copy()
    d['body'] = (d['Close'] - d['Open']).abs()
    d['avg_body'] = d['body'].rolling(20).mean()
    r = rsi(d['Close'], 14)
    at = atr(d, 14)
    e50 = ema(d['Close'], 50)
    
    big_up = (d['Close'] > d['Open']) & (d['body'] > d['avg_body'] * 2.5) & (d['Close'] > e50)
    big_dn = (d['Close'] < d['Open']) & (d['body'] > d['avg_body'] * 2.5) & (d['Close'] < e50)
    
    lc = big_up & (r < 75)
    sc = big_dn & (r > 25)
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc
    s['short_entry'] = sc
    s['long_exit'] = r > 80
    s['short_exit'] = r < 20
    
    # Trade the continuation
    s['sl'] = np.where(lc, d['Close'] - 2.0 * at, np.where(sc, d['Close'] + 2.0 * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 4.0 * at, np.where(sc, d['Close'] - 4.0 * at, np.nan))
    return s

def s_1h_stoch_rsi_fade(df):
    """
    1H Fade: Uses Stochastic RSI to fade extreme hourly moves.
    """
    d = df.copy()
    r = rsi(d['Close'], 14)
    min_r = r.rolling(14).min()
    max_r = r.rolling(14).max()
    stoch_rsi = (r - min_r) / (max_r - min_r)
    # Smooth it
    k = stoch_rsi.rolling(3).mean()
    d_line = k.rolling(3).mean()
    
    at = atr(d, 14)
    e100 = ema(d['Close'], 100)
    
    lc = (k > d_line) & (k.shift(1) <= d_line.shift(1)) & (k < 0.2) & (d['Close'] > e100)
    sc = (k < d_line) & (k.shift(1) >= d_line.shift(1)) & (k > 0.8) & (d['Close'] < e100)
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc; s['short_entry'] = sc
    s['long_exit'] = k > 0.8; s['short_exit'] = k < 0.2
    
    s['sl'] = np.where(lc, d['Close'] - 2.0 * at, np.where(sc, d['Close'] + 2.0 * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 3.0 * at, np.where(sc, d['Close'] - 3.0 * at, np.nan))
    return s

def s_1h_pure_divergence(df):
    """
    1H Pure Divergence (Price lower, RSI higher).
    """
    d = df.copy()
    r = rsi(d['Close'], 14)
    at = atr(d, 14)
    
    # Find pivots
    lookback = 10
    d['low_pivot'] = (d['Low'] < d['Low'].shift(1)) & (d['Low'] < d['Low'].shift(-1))
    d['high_pivot'] = (d['High'] > d['High'].shift(1)) & (d['High'] > d['High'].shift(-1))
    
    # Simple proxy for bullish divergence: price is lower than 15 bars ago, but RSI is higher
    lc = (d['Close'] < d['Close'].shift(15)) & (r > r.shift(15)) & (r < 35) & (d['Close'] > d['Open'])
    sc = (d['Close'] > d['Close'].shift(15)) & (r < r.shift(15)) & (r > 65) & (d['Close'] < d['Open'])
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc; s['short_entry'] = sc
    s['long_exit'] = r > 60; s['short_exit'] = r < 40
    
    s['sl'] = np.where(lc, d['Close'] - 2.5 * at, np.where(sc, d['Close'] + 2.5 * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 5.0 * at, np.where(sc, d['Close'] - 5.0 * at, np.nan))
    return s

def run_all():
    df_raw = build_dataset()
    phase_a = df_raw[~df_raw['is_oos']]
    phase_b = df_raw[df_raw['is_oos']]
    
    strats = [
        ("S1: 1H Mean Reversion Scalp", s_1h_mean_reversion_scalp),
        ("S2: 1H Momentum Breakout", s_1h_momentum_breakout),
        ("S3: 1H Stoch RSI Fade", s_1h_stoch_rsi_fade),
        ("S4: 1H Simple Divergence", s_1h_pure_divergence),
    ]
    
    print(f"\n{'='*90}")
    print(f"  ETH 1H — BRAND NEW STRATEGIES — Phase A (80%)")
    print(f"  {len(phase_a)} bars | {phase_a['Date'].min()} → {phase_a['Date'].max()}")
    print(f"{'='*90}")
    
    for name, fn in strats:
        tr = Engine(phase_a).run(fn(phase_a))
        rpt(tr, f"A | {name}")
    
    print(f"\n{'='*90}")
    print(f"  ETH 1H — BRAND NEW STRATEGIES — Phase B (20%)")
    print(f"  {len(phase_b)} bars | {phase_b['Date'].min()} → {phase_b['Date'].max()}")
    print(f"{'='*90}")
    
    for name, fn in strats:
        tr = Engine(phase_b).run(fn(phase_b))
        rpt(tr, f"B | {name}")

if __name__ == '__main__':
    run_all()
