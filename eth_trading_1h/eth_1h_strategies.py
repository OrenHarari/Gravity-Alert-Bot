"""
================================================================================
ETH 1H — STRATEGY RESEARCH LAB
================================================================================
Testing the top 4H strategies on 1H timeframe. 
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


# ─── 4H STRATEGIES ADAPTED FOR 1H ─────────────────────────────────────────────

def s_1h_trend_pullback(df):
    """Trend + Deep Pullback (The 4H winner adapted)"""
    d = df.copy()
    
    # Using 80 periods to track the 4H equivalent 20 EMA (4x)
    sma, upper, lower = bollinger_bands(d['Close'], n=80, std=2.0)
    ml, _, macd_h = macd(d['Close'], 24, 52, 18) 
    r = rsi(d['Close'], 14)
    e100 = ema(d['Close'], 200) # Similar to 50 EMA on 4H
    at = atr(d, 14)
    
    bull = (d['Close'] > e100) & (ml > 0)
    bear = (d['Close'] < e100) & (ml < 0)
    
    lc = bull & ((d['Close'] <= lower) | (r < 25))
    sc = bear & ((d['Close'] >= upper) | (r > 65))
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc
    s['short_entry'] = sc
    s['long_exit'] = (d['Close'] > sma) | (r > 60)
    s['short_exit'] = (d['Close'] < sma) | (r < 40)
    
    s['sl'] = np.where(lc, d['Close'] - 2.0 * at, np.where(sc, d['Close'] + 2.0 * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 5.0 * at, np.where(sc, d['Close'] - 5.0 * at, np.nan))
    return s

def s_1h_price_action(df):
    """Pure Price Action Sweeps 1H"""
    d = df.copy()
    d['is_green'] = d['Close'] > d['Open']
    d['is_red'] = d['Close'] < d['Open']
    d['body'] = abs(d['Close'] - d['Open'])
    d['avg_body'] = d['body'].rolling(20).mean()
    
    three_red = d['is_red'].shift(1) & d['is_red'].shift(2) & d['is_red'].shift(3)
    engulfing = d['is_green'] & (d['Close'] > d['Open'].shift(1)) & (d['Open'] < d['Close'].shift(1))
    strong = d['body'] > (d['avg_body'] * 1.5)
    
    lc = three_red & engulfing & strong
    
    three_green = d['is_green'].shift(1) & d['is_green'].shift(2) & d['is_green'].shift(3)
    eng_dn = d['is_red'] & (d['Close'] < d['Open'].shift(1)) & (d['Open'] > d['Close'].shift(1))
    
    sc = three_green & eng_dn & strong
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc
    s['short_entry'] = sc
    s['long_exit'] = d['is_red'] & (d['body'] > d['avg_body'])
    s['short_exit'] = d['is_green'] & (d['body'] > d['avg_body'])
    
    at = atr(d, 14)
    s['sl'] = np.where(lc, d['Low'].shift(1) - 1.0 * at, np.where(sc, d['High'].shift(1) + 1.0 * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 4.0 * at, np.where(sc, d['Close'] - 4.0 * at, np.nan))
    return s

def s_1h_macd_rsi_combo(df):
    """MACD + RSI Combo 1H"""
    d = df.copy()
    ml, sg, h = macd(d['Close'], 24, 52, 18)
    r = rsi(d['Close'], 14)
    at = atr(d, 14)
    
    hup = (h > 0) & (h.shift(1) <= 0)
    hdn = (h < 0) & (h.shift(1) >= 0)
    
    lc = hup & (r > 40) & (r < 65)
    sc = hdn & (r < 60) & (r > 35)
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc
    s['short_entry'] = sc
    s['long_exit'] = h < 0
    s['short_exit'] = h > 0
    s['sl'] = np.where(lc, d['Close'] - 2.5 * at, np.where(sc, d['Close'] + 2.5 * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 4.5 * at, np.where(sc, d['Close'] - 4.5 * at, np.nan))
    return s


# ─── RUNNER & GRID SEARCH ──────────────────────────────────────────────────────

def eval_params(df, bb_window, rsi_high, rsi_low, sl_multi, tp_multi):
    d = df.copy()
    sma, upper, lower = bollinger_bands(d['Close'], n=bb_window, std=2.0)
    ml, _, macd_h = macd(d['Close'], 24, 52, 18) 
    r = rsi(d['Close'], 14)
    e100 = ema(d['Close'], 200) 
    at = atr(d, 14)
    
    bull = (d['Close'] > e100) & (ml > 0)
    bear = (d['Close'] < e100) & (ml < 0)
    
    lc = bull & ((d['Close'] <= lower) | (r < rsi_low))
    sc = bear & ((d['Close'] >= upper) | (r > rsi_high))
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc
    s['short_entry'] = sc
    s['long_exit'] = (d['Close'] > sma) | (r > 60)
    s['short_exit'] = (d['Close'] < sma) | (r < 40)
    
    s['sl'] = np.where(lc, d['Close'] - sl_multi * at, np.where(sc, d['Close'] + sl_multi * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + tp_multi * at, np.where(sc, d['Close'] - tp_multi * at, np.nan))
    return s

def run_grid():
    df_raw = build_dataset()
    phase_a = df_raw[~df_raw['is_oos']]
    phase_b = df_raw[df_raw['is_oos']]
    
    print(f"\n{'='*90}")
    print(f"  ETH 1H — OPTIMIZING 'TREND + DEEP PULLBACK'")
    print(f"{'='*90}")
    
    best_score = -1
    best_p = None
    
    bb_windows = [40, 60, 80]
    rsi_zones = [(65, 30), (70, 25)]
    sl_multis = [1.5, 2.0, 2.5]
    tp_multis = [3.0, 4.0, 5.0, 6.0]
    
    for bb in bb_windows:
        for rh, rl in rsi_zones:
            for sl in sl_multis:
                for tp in tp_multis:
                    sig = eval_params(phase_a, bb, rh, rl, sl, tp)
                    tr = Engine(phase_a).run(sig)
                    if tr is None or len(tr) < 20: continue
                    pnl = tr['pnl'].values; w = tr['win'].values
                    wr = w.sum() / len(pnl)
                    gp = pnl[pnl>0].sum() if (pnl>0).any() else 0
                    gl = abs(pnl[pnl<0].sum()) if (pnl<0).any() else 1e-9
                    pf = gp/gl
                    score = wr * pf * len(pnl)
                    if wr >= 0.50 and pf >= 1.30 and score > best_score:
                        best_score = score
                        best_p = (bb, rh, rl, sl, tp)
                        print(f"  [New Best] BB={bb} RSI=({rh},{rl}) SL={sl} TP={tp} | N={len(pnl)} WR={wr*100:.1f}% PF={pf:.2f}")
    
    if best_p:
        bb, rh, rl, sl, tp = best_p
        print(f"\n  BEST PARAMS FOUND: BB={bb} RSI=({rh},{rl}) SL={sl} TP={tp}")
        
        sig_a = eval_params(phase_a, bb, rh, rl, sl, tp)
        tr_a = Engine(phase_a).run(sig_a)
        rpt(tr_a, "Phase A (Best Params)")
        
        sig_b = eval_params(phase_b, bb, rh, rl, sl, tp)
        tr_b = Engine(phase_b).run(sig_b)
        rpt(tr_b, "Phase B (Best Params)")
    else:
        print("\n  ❌ Could not find a winning combination that satisfies WR>=50% and PF>=1.30")

if __name__ == '__main__':
    run_grid()
