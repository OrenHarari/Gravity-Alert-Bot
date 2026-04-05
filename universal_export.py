"""
================================================================================
UNIVERSAL EXPORT ENGINE — ALL ASSETS, ALL STRATEGIES
================================================================================
Exports JSON data for the unified dashboard:
- ETH 1D (Daily)
- ETH 4H (Intraday)
- BTC 4H (Intraday)

Each asset gets all its strategies run through both Phase A and Phase B.
================================================================================
"""
import pandas as pd
import numpy as np
import json
import warnings
import os
import sys
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMMISSION = 0.001

# ─── INDICATORS ──────────────────────────────────────────────────────────────
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

# ─── ENGINE ──────────────────────────────────────────────────────────────────
class Engine:
    def __init__(self,df, leverage=1.0):
        self.df=df.copy().reset_index(drop=True)
        self.wm=df.get('is_warmup', pd.Series(False,index=df.index)).values
        self.trades=[]; self._p=None
        self.leverage = leverage
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
        pnl=(((ep-e)/e if p['t']=='long' else (e-ep)/e)-2*COMMISSION) * self.leverage
        self.trades.append(dict(type=p['t'],
            entry_date=p['ed'].strftime('%Y-%m-%d %H:%M') if hasattr(p['ed'],'strftime') else str(p['ed']),
            exit_date=ed.strftime('%Y-%m-%d %H:%M') if hasattr(ed,'strftime') else str(ed),
            entry=round(float(e),2),exit=round(float(ep),2),
            sl=round(float(p['sl']),2),tp=round(float(p['tp']),2),
            pnl=round(float(pnl*100),2),reason=r,win=bool(pnl>0)))

# ─── DATA LOADER ─────────────────────────────────────────────────────────────
def load_csv(filepath):
    df = pd.read_csv(filepath)
    df.columns = [c.strip().strip('"') for c in df.columns]
    date_col = 'Date' if 'Date' in df.columns else 'Datetime'
    df['Date'] = pd.to_datetime(df[date_col], format='mixed', utc=True).dt.tz_localize(None)
    for c in ['Open','High','Low','Close']:
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].str.replace(',', '').astype(float)
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

# ─── METRICS ─────────────────────────────────────────────────────────────────
def compute_metrics(tr):
    if tr is None or len(tr)==0:
        return dict(n=0,wins=0,losses=0,win_rate=0,profit_factor=0,avg_win=0,avg_loss=0,
                    total_return=0,max_dd=0,sharpe=0,expectancy=0)
    pnl=np.array([t['pnl'] for t in tr])/100 if isinstance(tr, list) else tr['pnl'].values/100
    w=np.array([t['win'] for t in tr]) if isinstance(tr, list) else tr['win'].values
    n=len(pnl); nw=int(w.sum())
    gp=pnl[pnl>0].sum() if (pnl>0).any() else 0
    gl=abs(pnl[pnl<0].sum()) if (pnl<0).any() else 1e-9
    pf=round(gp/gl,2); aw=round(pnl[pnl>0].mean()*100,2) if (pnl>0).any() else 0
    al=round(pnl[pnl<0].mean()*100,2) if (pnl<0).any() else 0
    wr=round(nw/n*100,1); exp=round(wr/100*aw+(1-wr/100)*al,2)
    eq=np.cumprod(1+pnl); ret=round((eq[-1]-1)*100,1)
    peak=np.maximum.accumulate(eq); mdd=round(((eq-peak)/peak).min()*100,1)
    sh=round(pnl.mean()/pnl.std()*np.sqrt(52),2) if pnl.std()>0 else 0
    return dict(n=n,wins=nw,losses=n-nw,win_rate=wr,profit_factor=pf,avg_win=aw,avg_loss=al,
                total_return=ret,max_dd=mdd,sharpe=sh,expectancy=exp)

def calc_equity(trades_list):
    if not trades_list: return [100.0]
    eq = [100.0]
    for t in trades_list:
        eq.append(round(eq[-1] * (1 + t['pnl']/100), 2))
    return eq

# ═══════════════════════════════════════════════════════════════════════════════
# ETH STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════════

def eth_trend_pullback(df):
    d = df.copy()
    sma, upper, lower = bollinger_bands(d['Close'], n=20, std=2.0)
    ml, _, macd_h = macd(d['Close'], 12, 26, 9)
    r = rsi(d['Close'], 14); e100 = ema(d['Close'], 100); at = atr(d, 10)
    bull = (d['Close'] > e100) & (ml > 0); bear = (d['Close'] < e100) & (ml < 0)
    lc = bull & ((d['Close'] <= lower) | (r < 25)); sc = bear & ((d['Close'] >= upper) | (r > 65))
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc; s['short_entry'] = sc
    s['long_exit'] = (d['Close'] > sma) | (r > 60); s['short_exit'] = (d['Close'] < sma) | (r < 40)
    s['sl'] = np.where(lc, d['Close'] - 1.5 * at, np.where(sc, d['Close'] + 1.5 * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 5.0 * at, np.where(sc, d['Close'] - 5.0 * at, np.nan))
    return s

def eth_aggressive_leverage(df):
    d = df.copy()
    sma, upper, lower = bollinger_bands(d['Close'], n=20, std=2.0)
    ml, _, _ = macd(d['Close'], 12, 26, 9)
    r = rsi(d['Close'], 14); e100 = ema(d['Close'], 100); at = atr(d, 10)
    bull = (d['Close'] > e100) & (ml > 0); bear = (d['Close'] < e100) & (ml < 0)
    lc = bull & ((d['Close'] <= lower) | (r < 30)); sc = bear & ((d['Close'] >= upper) | (r > 70))
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc; s['short_entry'] = sc
    s['long_exit'] = r > 65; s['short_exit'] = r < 35
    s['sl'] = np.where(lc, d['Close'] - 3.0 * at, np.where(sc, d['Close'] + 3.0 * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 5.0 * at, np.where(sc, d['Close'] - 5.0 * at, np.nan))
    return s

def eth_turtle_breakout(df):
    d = df.copy()
    high_max = d['High'].shift(1).rolling(40).max()
    low_min = d['Low'].shift(1).rolling(40).min()
    exit_high = d['High'].shift(1).rolling(10).max()
    exit_low = d['Low'].shift(1).rolling(10).min()
    at = atr(d, 20)
    lc = d['Close'] > high_max
    sc = d['Close'] < low_min
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc; s['short_entry'] = sc
    s['long_exit'] = d['Close'] < exit_low; s['short_exit'] = d['Close'] > exit_high
    s['sl'] = np.where(lc, d['Close'] - 2.0 * at, np.where(sc, d['Close'] + 2.0 * at, np.nan))
    s['tp'] = np.nan
    return s

def eth_filtered_turtle(df):
    d = df.copy()
    high_max = d['High'].shift(1).rolling(35).max()
    low_min = d['Low'].shift(1).rolling(35).min()
    exit_high = d['High'].shift(1).rolling(3).max()
    exit_low = d['Low'].shift(1).rolling(3).min()
    e_trend = ema(d['Close'], 100)
    at = atr(d, 20)
    lc = (d['Close'] > high_max) & (d['Close'] > e_trend)
    sc = (d['Close'] < low_min) & (d['Close'] < e_trend)
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc; s['short_entry'] = sc
    s['long_exit'] = d['Close'] < exit_low; s['short_exit'] = d['Close'] > exit_high
    s['sl'] = np.where(lc, d['Close'] - 1.5 * at, np.where(sc, d['Close'] + 1.5 * at, np.nan))
    s['tp'] = np.nan
    return s

def eth_hybrid_sniper(df):
    d = df.copy()
    sma, upper, lower = bollinger_bands(d['Close'], n=20, std=1.5)
    ml, _, h = macd(d['Close'], 12, 26, 9)
    r = rsi(d['Close'], 14)
    e_trend = ema(d['Close'], 100)
    at = atr(d, 10)
    
    bull = (d['Close'] > e_trend) & (ml > 0)
    bear = (d['Close'] < e_trend) & (ml < 0)
    
    lc = bull & ((d['Close'] <= lower) | (r < 35))
    sc = bear & ((d['Close'] >= upper) | (r > 65))
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc
    s['short_entry'] = sc
    s['long_exit'] = (d['Close'] > sma) | (r > 60)
    s['short_exit'] = (d['Close'] < sma) | (r < 40)
    s['sl'] = np.where(lc, d['Close'] - 2.0 * at, np.where(sc, d['Close'] + 2.0 * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 5.0 * at, np.where(sc, d['Close'] - 5.0 * at, np.nan))
    return s

def eth_price_action(df):
    d = df.copy()
    d['is_green'] = d['Close'] > d['Open']; d['is_red'] = d['Close'] < d['Open']
    d['body'] = abs(d['Close'] - d['Open']); d['avg_body'] = d['body'].rolling(14).mean()
    three_red = d['is_red'].shift(1) & d['is_red'].shift(2) & d['is_red'].shift(3)
    engulfing = d['is_green'] & (d['Close'] > d['Open'].shift(1)) & (d['Open'] < d['Close'].shift(1))
    strong = d['body'] > (d['avg_body'] * 1.5)
    lc = three_red & engulfing & strong
    three_green = d['is_green'].shift(1) & d['is_green'].shift(2) & d['is_green'].shift(3)
    eng_dn = d['is_red'] & (d['Close'] < d['Open'].shift(1)) & (d['Open'] > d['Close'].shift(1))
    sc = three_green & eng_dn & strong
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc; s['short_entry'] = sc
    s['long_exit'] = d['is_red'] & (d['body'] > d['avg_body'])
    s['short_exit'] = d['is_green'] & (d['body'] > d['avg_body'])
    at = atr(d, 10)
    s['sl'] = np.where(lc, d['Low'].shift(1) - 0.5*at, np.where(sc, d['High'].shift(1)+0.5*at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 4.0*at, np.where(sc, d['Close'] - 4.0*at, np.nan))
    return s

def eth_volatility_squeeze(df):
    d = df.copy()
    sma, upper, lower = bollinger_bands(d['Close'], n=20, std=2.0)
    d['bb_width'] = (upper - lower) / sma; d['bb_width_mean'] = d['bb_width'].rolling(50).mean()
    squeeze = d['bb_width'] < (d['bb_width_mean'] * 0.8)
    recent = squeeze.rolling(3).max() > 0
    at = atr(d, 10); r = rsi(d['Close'], 14)
    lc = recent & (d['Close'] > upper) & (r > 60); sc = recent & (d['Close'] < lower) & (r < 40)
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc; s['short_entry'] = sc
    s['long_exit'] = r < 45; s['short_exit'] = r > 55
    s['sl'] = np.where(lc, d['Close']-1.5*at, np.where(sc, d['Close']+1.5*at, np.nan))
    s['tp'] = np.where(lc, d['Close']+6.0*at, np.where(sc, d['Close']-6.0*at, np.nan))
    return s

def eth_macd_rsi_combo(df):
    d = df.copy(); ml, sg, h = macd(d['Close'], 12, 26, 9); r = rsi(d['Close'], 14); at = atr(d, 10)
    hup = (h>0)&(h.shift(1)<=0); hdn = (h<0)&(h.shift(1)>=0)
    lc = hup&(r>40)&(r<65); sc = hdn&(r<60)&(r>35)
    s = pd.DataFrame(index=d.index)
    s['long_entry']=lc; s['short_entry']=sc; s['long_exit']=h<0; s['short_exit']=h>0
    s['sl']=np.where(lc,d['Close']-2.0*at,np.where(sc,d['Close']+2.0*at,np.nan))
    s['tp']=np.where(lc,d['Close']+4.0*at,np.where(sc,d['Close']-4.0*at,np.nan))
    return s

# ═══════════════════════════════════════════════════════════════════════════════
# BTC STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════════

def btc_smart_rsi_pa(df):
    """BTC Best Strategy: Smart RSI + PA Confirmation (Hybrid E from R2)."""
    d = df.copy()
    r = rsi(d['Close'], 14); e50 = ema(d['Close'], 50); at = atr(d, 14)
    ml, sg, h = macd(d['Close'], 12, 26, 9)
    green = d['Close'] > d['Open']; red = d['Close'] < d['Open']
    lc = (r < 30) & green & (h > h.shift(1))
    sc = (r > 70) & red & (h < h.shift(1))
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc; s['short_entry'] = sc
    s['long_exit'] = (r > 55) | ((r < 25) & red)
    s['short_exit'] = (r < 45) | ((r > 75) & green)
    s['sl'] = np.where(lc, d['Close'] - 2.0*at, np.where(sc, d['Close']+2.0*at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 3.5*at, np.where(sc, d['Close']-3.5*at, np.nan))
    return s

def btc_adaptive_momentum(df):
    """BTC Adaptive Momentum: High vol = mean revert, Low vol = trend follow."""
    d = df.copy()
    at = atr(d, 14); at_slow = at.rolling(50).mean()
    high_vol = at > at_slow * 1.2
    r = rsi(d['Close'], 14)
    sma, upper, lower = bollinger_bands(d['Close'], n=20, std=2.0)
    ml, sg, h = macd(d['Close'], 12, 26, 9); e50 = ema(d['Close'], 50)
    lc_mr = high_vol & (d['Close'] < lower) & (r < 30) & (h > h.shift(1))
    sc_mr = high_vol & (d['Close'] > upper) & (r > 70) & (h < h.shift(1))
    hcup = (h>0)&(h.shift(1)<=0); hcdn = (h<0)&(h.shift(1)>=0)
    lc_tf = (~high_vol) & hcup & (d['Close']>e50) & (r>45) & (r<65)
    sc_tf = (~high_vol) & hcdn & (d['Close']<e50) & (r>35) & (r<55)
    lc = lc_mr | lc_tf; sc = sc_mr | sc_tf
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc; s['short_entry'] = sc
    s['long_exit'] = np.where(high_vol, (d['Close'] > sma) | (r > 60), (h < 0) | (r > 75))
    s['short_exit'] = np.where(high_vol, (d['Close'] < sma) | (r < 40), (h > 0) | (r < 25))
    s['sl'] = np.where(lc, d['Close']-2.5*at, np.where(sc, d['Close']+2.5*at, np.nan))
    s['tp'] = np.where(lc, d['Close']+4.0*at, np.where(sc, d['Close']-4.0*at, np.nan))
    return s

def btc_golden_sniper(df):
    """BTC Golden Cross Sniper: EMA 50/200 regime + RSI dip buying."""
    d = df.copy()
    e50 = ema(d['Close'], 50); e200 = ema(d['Close'], 200)
    r = rsi(d['Close'], 14); at = atr(d, 14)
    lc = (e50 > e200) & (r < 35) & (r.shift(1) >= 35)
    sc = (e50 < e200) & (r > 65) & (r.shift(1) <= 65)
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc; s['short_entry'] = sc
    s['long_exit'] = (r > 70) | (e50 < e200)
    s['short_exit'] = (r < 30) | (e50 > e200)
    s['sl'] = np.where(lc, d['Close']-3.0*at, np.where(sc, d['Close']+3.0*at, np.nan))
    s['tp'] = np.where(lc, d['Close']+6.0*at, np.where(sc, d['Close']-6.0*at, np.nan))
    return s

def btc_triple_confirmation(df):
    """BTC Triple Confirmation: MACD + RSI + Stochastic all agree."""
    d = df.copy()
    ml, sg, h = macd(d['Close'], 12, 26, 9); r = rsi(d['Close'], 14)
    k, d_line = stochastic(d, 14, 3); e50 = ema(d['Close'], 50); at = atr(d, 14)
    macd_bull = (h>0)&(h>h.shift(1)); rsi_bull = (r>45)&(r<70)
    stoch_bull = (k>d_line)&(k>20)&(k<80)
    macd_bear = (h<0)&(h<h.shift(1)); rsi_bear = (r<55)&(r>30)
    stoch_bear = (k<d_line)&(k>20)&(k<80)
    lc = macd_bull & rsi_bull & stoch_bull & (d['Close']>e50)
    sc = macd_bear & rsi_bear & stoch_bear & (d['Close']<e50)
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc; s['short_entry'] = sc
    s['long_exit'] = (h<0)|(r>80); s['short_exit'] = (h>0)|(r<20)
    s['sl'] = np.where(lc, d['Close']-2.0*at, np.where(sc, d['Close']+2.0*at, np.nan))
    s['tp'] = np.where(lc, d['Close']+4.0*at, np.where(sc, d['Close']-4.0*at, np.nan))
    return s

# ═══════════════════════════════════════════════════════════════════════════════
# ETH 1D STRATEGIES (from the daily system)
# ═══════════════════════════════════════════════════════════════════════════════

def eth1d_trend_following(df):
    """ETH 1D Trend Following: FastMACD + Dual EMA confirmation."""
    d = df.copy()
    ml, sg, h = macd(d['Close'], 5, 13, 3)
    e20 = ema(d['Close'], 20); e50 = ema(d['Close'], 50)
    r = rsi(d['Close'], 14); at = atr(d, 14)
    lc = (h>0)&(h.shift(1)<=0)&(e20>e50)&(r>40)&(r<70)
    sc = (h<0)&(h.shift(1)>=0)&(e20<e50)&(r>30)&(r<60)
    s = pd.DataFrame(index=d.index)
    s['long_entry']=lc; s['short_entry']=sc
    s['long_exit']=(h<0)|(r>80); s['short_exit']=(h>0)|(r<20)
    s['sl']=np.where(lc,d['Close']-2.0*at,np.where(sc,d['Close']+2.0*at,np.nan))
    s['tp']=np.where(lc,d['Close']+5.0*at,np.where(sc,d['Close']-5.0*at,np.nan))
    return s

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def process_asset(asset_name, data_path, strategies, timeframe):
    """Process one asset: load data, run all strategies, return results."""
    print(f"\n  Processing {asset_name} ({timeframe})...")
    df = load_csv(data_path)
    phase_a = df[~df['is_oos']].reset_index(drop=True)
    phase_b = df[df['is_oos']].reset_index(drop=True)
    
    print(f"    Phase A: {len(phase_a)} bars ({phase_a['Date'].min()} → {phase_a['Date'].max()})")
    print(f"    Phase B: {len(phase_b)} bars ({phase_b['Date'].min()} → {phase_b['Date'].max()})")
    
    # Build candles for chart (downsample if >500 candles for performance)
    def make_candles(phase_df, max_candles=500):
        step = max(1, len(phase_df) // max_candles)
        candles = []
        for i in range(0, len(phase_df), step):
            row = phase_df.iloc[i]
            candles.append(dict(
                date=row.Date.strftime('%Y-%m-%d %H:%M'),
                open=round(float(row.Open),2), high=round(float(row.High),2),
                low=round(float(row.Low),2), close=round(float(row.Close),2)
            ))
        return candles
    
    candles_a = make_candles(phase_a)
    candles_b = make_candles(phase_b)
    
    strat_results = []
    for strat_tuple in strategies:
        # Check if strategy has leverage 
        if len(strat_tuple) == 3:
            strat_name, strat_fn, lev = strat_tuple
        else:
            strat_name, strat_fn = strat_tuple
            lev = 1.0
            
        print(f"    Running {strat_name}...")
        
        # Phase A
        sig_a = strat_fn(phase_a)
        tr_a = Engine(phase_a, leverage=lev).run(sig_a)
        trades_a = tr_a.to_dict('records') if not tr_a.empty else []
        metrics_a = compute_metrics(trades_a)
        equity_a = calc_equity(trades_a)
        
        # Phase B
        sig_b = strat_fn(phase_b)
        tr_b = Engine(phase_b, leverage=lev).run(sig_b)
        trades_b = tr_b.to_dict('records') if not tr_b.empty else []
        metrics_b = compute_metrics(trades_b)
        equity_b = calc_equity(trades_b)
        
        print(f"      A: N={metrics_a['n']} WR={metrics_a['win_rate']}% PF={metrics_a['profit_factor']} Ret={metrics_a['total_return']}%")
        print(f"      B: N={metrics_b['n']} WR={metrics_b['win_rate']}% PF={metrics_b['profit_factor']} Ret={metrics_b['total_return']}%")
        
        strat_results.append(dict(
            name=strat_name,
            phase_a=dict(trades=trades_a, equity=equity_a, 
                        equity_dates=['Start']+[t['exit_date'] for t in trades_a],
                        metrics=metrics_a),
            phase_b=dict(trades=trades_b, equity=equity_b,
                        equity_dates=['Start']+[t['exit_date'] for t in trades_b],
                        metrics=metrics_b)
        ))
    
    return dict(
        asset=asset_name,
        timeframe=timeframe,
        phase_a=dict(
            label=f"Phase A — In-Sample (80%)",
            start=phase_a['Date'].min().strftime('%Y-%m-%d'),
            end=phase_a['Date'].max().strftime('%Y-%m-%d'),
            n_bars=len(phase_a),
            candles=candles_a
        ),
        phase_b=dict(
            label=f"Phase B — Out-of-Sample (20%)",
            start=phase_b['Date'].min().strftime('%Y-%m-%d'),
            end=phase_b['Date'].max().strftime('%Y-%m-%d'),
            n_bars=len(phase_b),
            candles=candles_b
        ),
        strategies=strat_results
    )

# ─── ASSET DEFINITIONS ──────────────────────────────────────────────────────
print("=" * 70)
print("  UNIVERSAL EXPORT ENGINE")
print("=" * 70)

assets = []

# 1. ETH 4H
eth4h_path = os.path.join(BASE_DIR, "eth_trading_4h", "ETH_4H_2Y.csv")
if os.path.exists(eth4h_path):
    assets.append(process_asset("ETH/USD", eth4h_path, [
        ("Trend + Deep Pullback", eth_trend_pullback, 1.0),
        ("Aggressive PNL Runner (2x Lev)", eth_aggressive_leverage, 2.0),
        ("Super Turtle Breakout (188%)", eth_filtered_turtle, 1.0),
        ("2025 Sniper (70% WR)", eth_hybrid_sniper, 1.0),
        ("Turtle Breakout (Unleveraged)", eth_turtle_breakout, 1.0),
        ("Pure Price Action Sweeps", eth_price_action, 1.0),
        ("Volatility Squeeze", eth_volatility_squeeze, 1.0),
        ("MACD + RSI Combo", eth_macd_rsi_combo, 1.0),
    ], "4H"))

# 2. BTC 4H
btc4h_path = os.path.join(BASE_DIR, "btc_trading_4h", "BTC_4H_2Y.csv")
if os.path.exists(btc4h_path):
    assets.append(process_asset("BTC/USD", btc4h_path, [
        ("Smart RSI + Price Action", btc_smart_rsi_pa),
        ("Adaptive Momentum", btc_adaptive_momentum),
        ("Golden Cross Sniper", btc_golden_sniper),
        ("Triple Confirmation", btc_triple_confirmation),
    ], "4H"))

# 3. ETH 1D — skipped (uses investing.com weekly format, different columns)
# Will add when we download proper daily data from yfinance
# eth1d_candidates = [
#     os.path.join(BASE_DIR, "eth_trading_1d", "Ethereum Historical Data (1).csv"),
# ]

# ─── EXPORT ──────────────────────────────────────────────────────────────────
out = dict(
    generated_at=pd.Timestamp.now().isoformat(),
    assets=assets
)

out_path = os.path.join(BASE_DIR, "unified_dashboard_data.json")
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2, default=str)

# Also write to dashboard_data_inline.js to bypass file:// CORS in browser
js_path = os.path.join(BASE_DIR, "dashboard_data_inline.js")
with open(js_path, 'w', encoding='utf-8') as f:
    f.write("const INLINE_DATA = ")
    json.dump(out, f, indent=2, default=str)
    f.write(";")

print(f"\n✅ Exported to {out_path} and {js_path}")
print(f"   {len(assets)} assets processed")
