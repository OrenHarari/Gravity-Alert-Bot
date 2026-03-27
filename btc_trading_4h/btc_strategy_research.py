"""
================================================================================
BTC 4H — STRATEGY RESEARCH LAB
================================================================================
The ETH strategies failed on BTC. BTC trades differently:
- More institutional, cleaner trends
- Less mean-reversion, more breakout/momentum
- Lower % volatility but massive $ moves
- Needs different parameter tuning

This script tests a wide range of BTC-specific strategies.
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

# ─── DATA ────────────────────────────────────────────────────────────────────
def build_dataset():
    df = pd.read_csv(DATA_FILE)
    df.columns = [c.strip().strip('"') for c in df.columns]
    date_col = 'Date' if 'Date' in df.columns else 'Datetime'
    df['Date'] = pd.to_datetime(df[date_col], format='mixed', utc=True).dt.tz_localize(None)
    for c in ['Open','High','Low','Close']:
        if df[c].dtype == object:
            df[c] = df[c].str.replace(',', '').astype(float)
    df = df.sort_values('Date').reset_index(drop=True)
    if 'Volume' not in df.columns:
        df['Volume'] = 1.0
    df = df[['Date','Open','High','Low','Close','Volume']]
    
    split_idx = int(len(df) * 0.8)
    df['is_oos'] = df.index >= split_idx
    df['is_oos_actual'] = df['is_oos'].copy()
    if split_idx > 52:
        df.loc[split_idx-52:split_idx-1, 'is_oos'] = True
        df.loc[split_idx-52:split_idx-1, 'is_warmup'] = True
    else:
        df['is_warmup'] = False
    df['is_warmup'] = df.get('is_warmup', False).fillna(False)
    return df

# ─── INDICATORS ──────────────────────────────────────────────────────────────
ema = lambda s,n: s.ewm(span=n,adjust=False).mean()

def atr(df,n=14):
    h,l,c=df['High'],df['Low'],df['Close']; pc=c.shift(1)
    return pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1).ewm(com=n-1,min_periods=n).mean()

def macd(s,f=12,sl=26,sig=9):
    ml=ema(s,f)-ema(s,sl); return ml,ema(ml,sig),ml-ema(ml,sig)

def rsi(s, n=14):
    d = s.diff()
    u = d.clip(lower=0).ewm(com=n-1, min_periods=n).mean()
    v = (-d.clip(upper=0)).ewm(com=n-1, min_periods=n).mean()
    return 100 - 100 / (1 + u / v.replace(0, np.nan))

def bollinger_bands(s, n=20, std=2):
    sma = s.rolling(window=n).mean()
    roll_std = s.rolling(window=n).std()
    return sma, sma + (roll_std * std), sma - (roll_std * std)

def stochastic(df, k_period=14, d_period=3):
    low_min = df['Low'].rolling(k_period).min()
    high_max = df['High'].rolling(k_period).max()
    k = 100 * (df['Close'] - low_min) / (high_max - low_min)
    d = k.rolling(d_period).mean()
    return k, d

def adx(df, period=14):
    """Average Directional Index - measures trend strength"""
    h, l, c = df['High'], df['Low'], df['Close']
    plus_dm = h.diff()
    minus_dm = -l.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr_val = tr.ewm(span=period, adjust=False).mean()
    
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr_val)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr_val)
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx_val = dx.ewm(span=period, adjust=False).mean()
    return adx_val, plus_di, minus_di

# ─── ENGINE ──────────────────────────────────────────────────────────────────
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
        print(f"  [{label}]: 0 trades fired.")
        return {}
    pnl=tr['pnl'].values; w=tr['win'].values
    n=len(pnl); nw=w.sum(); wr=nw/n
    gp=pnl[pnl>0].sum() if (pnl>0).any() else 0
    gl=abs(pnl[pnl<0].sum()) if (pnl<0).any() else 1e-9
    pf=gp/gl; aw=pnl[pnl>0].mean()*100 if (pnl>0).any() else 0
    al=pnl[pnl<0].mean()*100 if (pnl<0).any() else 0
    eq=np.cumprod(1+pnl); ret=eq[-1]-1
    peak=np.maximum.accumulate(eq); mdd=((eq-peak)/peak).min()
    sh=pnl.mean()/pnl.std()*np.sqrt(52) if pnl.std()>0 else 0
    ok=wr>=0.50 and pf>=1.30
    print(f"  {label:55s} N={n:3d} WR={wr*100:4.0f}% PF={pf:5.2f} Ret={ret*100:+7.1f}% MDD={mdd*100:5.1f}% {'✅' if ok else '❌'}")
    return dict(label=label,n=n,wr=wr,pf=pf,aw=aw,al=al,ret=ret,mdd=mdd,sh=sh,ok=ok)

# ─── BTC-SPECIFIC STRATEGIES ────────────────────────────────────────────────

def s1_trend_rider(df):
    """
    BTC Trend Rider: Only trade with the trend, use ADX to confirm 
    trend strength. Enter on pullbacks to EMA in strong trends.
    """
    d = df.copy()
    e20 = ema(d['Close'], 20)
    e50 = ema(d['Close'], 50)
    adx_val, plus_di, minus_di = adx(d, 14)
    r = rsi(d['Close'], 14)
    at = atr(d, 14)
    
    strong_trend = adx_val > 25
    bull = (e20 > e50) & strong_trend & (plus_di > minus_di)
    bear = (e20 < e50) & strong_trend & (minus_di > plus_di)
    
    # Pullback to EMA20 in a strong uptrend
    near_ema = (d['Low'] <= e20 * 1.005) & (d['Close'] > e20)
    lc = bull & near_ema & (r > 40) & (r < 60)
    
    # Pullback to EMA20 in a strong downtrend  
    near_ema_short = (d['High'] >= e20 * 0.995) & (d['Close'] < e20)
    sc = bear & near_ema_short & (r > 40) & (r < 60)
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc; s['short_entry'] = sc
    s['long_exit'] = (d['Close'] < e50) | (adx_val < 20)
    s['short_exit'] = (d['Close'] > e50) | (adx_val < 20)
    s['sl'] = np.where(lc, d['Close'] - 2.0 * at, np.where(sc, d['Close'] + 2.0 * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 4.0 * at, np.where(sc, d['Close'] - 4.0 * at, np.nan))
    return s

def s2_stoch_oversold_momentum(df):
    """
    Stochastic Oversold + Momentum Burst.
    Buy when stochastic crosses up from oversold, confirmed by MACD.
    """
    d = df.copy()
    k, d_line = stochastic(d, 14, 3)
    ml, sg, h = macd(d['Close'], 12, 26, 9)
    at = atr(d, 14)
    e50 = ema(d['Close'], 50)
    
    stoch_cross_up = (k > d_line) & (k.shift(1) <= d_line.shift(1)) & (k < 30)
    stoch_cross_dn = (k < d_line) & (k.shift(1) >= d_line.shift(1)) & (k > 70)
    
    lc = stoch_cross_up & (h > h.shift(1)) & (d['Close'] > e50)
    sc = stoch_cross_dn & (h < h.shift(1)) & (d['Close'] < e50)
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc; s['short_entry'] = sc
    s['long_exit'] = k > 80
    s['short_exit'] = k < 20
    s['sl'] = np.where(lc, d['Close'] - 2.5 * at, np.where(sc, d['Close'] + 2.5 * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 5.0 * at, np.where(sc, d['Close'] - 5.0 * at, np.nan))
    return s

def s3_range_breakout(df):
    """
    Range Breakout: Detect consolidation using ATR compression,
    then trade the breakout direction.
    """
    d = df.copy()
    at = atr(d, 14)
    at_slow = atr(d, 50)
    r = rsi(d['Close'], 14)
    
    # ATR compression = low volatility = range
    compressed = at < (at_slow * 0.7)
    was_compressed = compressed.rolling(6).max() > 0
    
    # Breakout: big candle (> 1.5x avg body) 
    d['body'] = (d['Close'] - d['Open']).abs()
    d['avg_body'] = d['body'].rolling(20).mean()
    big_move = d['body'] > d['avg_body'] * 2.0
    
    lc = was_compressed & big_move & (d['Close'] > d['Open']) & (r > 55)
    sc = was_compressed & big_move & (d['Close'] < d['Open']) & (r < 45)
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc; s['short_entry'] = sc
    s['long_exit'] = r > 75
    s['short_exit'] = r < 25
    s['sl'] = np.where(lc, d['Close'] - 1.5 * at, np.where(sc, d['Close'] + 1.5 * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 5.0 * at, np.where(sc, d['Close'] - 5.0 * at, np.nan))
    return s

def s4_momentum_continuation(df):
    """
    Momentum Continuation: In strong trends, buy the first pullback
    that doesn't break structure. Uses EMA cascade (9>21>50).
    """
    d = df.copy()
    e9 = ema(d['Close'], 9)
    e21 = ema(d['Close'], 21)
    e50 = ema(d['Close'], 50)
    e200 = ema(d['Close'], 200)
    r = rsi(d['Close'], 14)
    at = atr(d, 14)
    
    # Perfect bullish alignment
    bull_cascade = (e9 > e21) & (e21 > e50) & (d['Close'] > e200)
    bear_cascade = (e9 < e21) & (e21 < e50) & (d['Close'] < e200)
    
    # Pullback: price touched e21 but bounced (closed above it)
    bull_pullback = (d['Low'] <= e21) & (d['Close'] > e21) & bull_cascade
    bear_pullback = (d['High'] >= e21) & (d['Close'] < e21) & bear_cascade
    
    lc = bull_pullback & (r > 35) & (r < 65)
    sc = bear_pullback & (r > 35) & (r < 65)
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc; s['short_entry'] = sc
    s['long_exit'] = (d['Close'] < e50) | (r > 80)
    s['short_exit'] = (d['Close'] > e50) | (r < 20)
    s['sl'] = np.where(lc, d['Low'] - 0.5 * at, np.where(sc, d['High'] + 0.5 * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 3.5 * at, np.where(sc, d['Close'] - 3.5 * at, np.nan))
    return s

def s5_golden_cross_sniper(df):
    """
    Golden Cross Sniper: Wait for EMA 50/200 golden cross,
    then buy the first RSI dip. High conviction, few trades.
    """
    d = df.copy()
    e50 = ema(d['Close'], 50)
    e200 = ema(d['Close'], 200)
    r = rsi(d['Close'], 14)
    at = atr(d, 14)
    
    # Golden cross happened in last 30 bars
    golden = (e50 > e200) & (e50.shift(30) <= e200.shift(30))
    # Death cross in last 30 bars
    death = (e50 < e200) & (e50.shift(30) >= e200.shift(30))
    
    # After golden cross, buy the RSI dips
    lc = (e50 > e200) & (r < 35) & (r.shift(1) >= 35)
    # After death cross, short the RSI pops
    sc = (e50 < e200) & (r > 65) & (r.shift(1) <= 65)
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc; s['short_entry'] = sc
    s['long_exit'] = (r > 70) | (e50 < e200)
    s['short_exit'] = (r < 30) | (e50 > e200)
    s['sl'] = np.where(lc, d['Close'] - 3.0 * at, np.where(sc, d['Close'] + 3.0 * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 6.0 * at, np.where(sc, d['Close'] - 6.0 * at, np.nan))
    return s

def s6_dip_buyer(df):
    """
    Simple Dip Buyer: Buy extreme RSI dips (< 25) in bull regimes, 
    sell extreme pops (> 75) in bear regimes. Wide stops.
    """
    d = df.copy()
    r = rsi(d['Close'], 14)
    e200 = ema(d['Close'], 200)
    at = atr(d, 14)
    
    lc = (r < 25) & (d['Close'] > e200 * 0.92)  # Not in a total crash
    sc = (r > 75) & (d['Close'] < e200 * 1.08)
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc; s['short_entry'] = sc
    s['long_exit'] = r > 55
    s['short_exit'] = r < 45
    s['sl'] = np.where(lc, d['Close'] - 3.0 * at, np.where(sc, d['Close'] + 3.0 * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 5.0 * at, np.where(sc, d['Close'] - 5.0 * at, np.nan))
    return s

def s7_bb_mean_reversion(df):
    """
    Bollinger Band Mean Reversion with trend filter.
    Buy below lower BB when trend is up. Sell above upper BB when trend is down.
    """
    d = df.copy()
    sma, upper, lower = bollinger_bands(d['Close'], n=20, std=2.5)
    r = rsi(d['Close'], 14)
    e100 = ema(d['Close'], 100)
    at = atr(d, 14)
    
    # Buy below lower BB, but only when long-term trend is up
    lc = (d['Close'] < lower) & (d['Close'] > e100 * 0.95) & (r < 30)
    # Sell above upper BB when long-term trend is down
    sc = (d['Close'] > upper) & (d['Close'] < e100 * 1.05) & (r > 70)
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc; s['short_entry'] = sc
    s['long_exit'] = (d['Close'] > sma) | (r > 60)
    s['short_exit'] = (d['Close'] < sma) | (r < 40)
    s['sl'] = np.where(lc, d['Close'] - 2.0 * at, np.where(sc, d['Close'] + 2.0 * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 4.0 * at, np.where(sc, d['Close'] - 4.0 * at, np.nan))
    return s

def s8_triple_confirmation(df):
    """
    Triple Confirmation: MACD + RSI + Stochastic all agree.
    Very selective - waits for perfect alignment.
    """
    d = df.copy()
    ml, sg, h = macd(d['Close'], 12, 26, 9)
    r = rsi(d['Close'], 14)
    k, d_line = stochastic(d, 14, 3)
    e50 = ema(d['Close'], 50)
    at = atr(d, 14)
    
    # All three say buy
    macd_bull = (h > 0) & (h > h.shift(1))
    rsi_bull = (r > 45) & (r < 70)
    stoch_bull = (k > d_line) & (k > 20) & (k < 80)
    
    # All three say sell
    macd_bear = (h < 0) & (h < h.shift(1))
    rsi_bear = (r < 55) & (r > 30)
    stoch_bear = (k < d_line) & (k > 20) & (k < 80)
    
    lc = macd_bull & rsi_bull & stoch_bull & (d['Close'] > e50)
    sc = macd_bear & rsi_bear & stoch_bear & (d['Close'] < e50)
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc; s['short_entry'] = sc
    s['long_exit'] = (h < 0) | (r > 80)
    s['short_exit'] = (h > 0) | (r < 20)
    s['sl'] = np.where(lc, d['Close'] - 2.0 * at, np.where(sc, d['Close'] + 2.0 * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 4.0 * at, np.where(sc, d['Close'] - 4.0 * at, np.nan))
    return s

def s9_weekly_momentum(df):
    """
    Weekly Momentum on 4H: Use 42-bar (1 week) momentum.
    Buy when 1-week return is positive and RSI is recovering.
    """
    d = df.copy()
    d['ret_42'] = d['Close'].pct_change(42)
    r = rsi(d['Close'], 14)
    e100 = ema(d['Close'], 100)
    at = atr(d, 14)
    
    # Strong weekly momentum + RSI not overbought + above trend
    lc = (d['ret_42'] > 0.05) & (r < 60) & (r > 40) & (d['Close'] > e100)
    sc = (d['ret_42'] < -0.05) & (r > 40) & (r < 60) & (d['Close'] < e100)
    
    # Prevent consecutive signals (only first signal in 6-bar window)
    lc = lc & (~lc.shift(1).fillna(False)) & (~lc.shift(2).fillna(False)) & (~lc.shift(3).fillna(False))
    sc = sc & (~sc.shift(1).fillna(False)) & (~sc.shift(2).fillna(False)) & (~sc.shift(3).fillna(False))
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc; s['short_entry'] = sc
    s['long_exit'] = (d['ret_42'] < 0) | (r > 75)
    s['short_exit'] = (d['ret_42'] > 0) | (r < 25)
    s['sl'] = np.where(lc, d['Close'] - 2.5 * at, np.where(sc, d['Close'] + 2.5 * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 5.0 * at, np.where(sc, d['Close'] - 5.0 * at, np.nan))
    return s

def s10_long_only_dip(df):
    """
    BTC Long-Only Dip Buyer (No Shorts).
    BTC has a strong upward bias historically. Only buy extreme dips
    with multiple confirmations: RSI < 30, price below BB, and trend filter.
    """
    d = df.copy()
    r = rsi(d['Close'], 14)
    sma, upper, lower = bollinger_bands(d['Close'], n=20, std=2.0)
    e200 = ema(d['Close'], 200)
    at = atr(d, 14)
    ml, sg, h = macd(d['Close'], 12, 26, 9)
    
    # Extreme dip: RSI < 30 AND below lower BB
    extreme_dip = (r < 30) & (d['Close'] < lower)
    # But NOT in a death spiral (price still above -15% from EMA200)
    not_death = d['Close'] > e200 * 0.85
    # MACD starting to recover (histogram starting to get less negative)
    recovering = (h > h.shift(1)) | (h > h.shift(2))
    
    lc = extreme_dip & not_death & recovering
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc
    s['short_entry'] = False
    s['long_exit'] = (d['Close'] > sma) | (r > 55)
    s['short_exit'] = False
    s['sl'] = np.where(lc, d['Close'] - 3.0 * at, np.nan)
    s['tp'] = np.where(lc, d['Close'] + 5.0 * at, np.nan)
    return s

def s11_momentum_cascade(df):
    """
    Momentum Cascade: Use multiple timeframe momentum (6h, 1d, 3d) all aligned.
    Only enter when all momentum is pointing same direction.
    Uses returns over different lookbacks as a proxy for multi-TF momentum.
    """
    d = df.copy()
    # 6 bars = 24h, 42 = 1 week, 126 = ~3 weeks 
    d['ret_6'] = d['Close'].pct_change(6)
    d['ret_42'] = d['Close'].pct_change(42)
    d['ret_126'] = d['Close'].pct_change(126)
    
    r = rsi(d['Close'], 14)
    at = atr(d, 14)
    
    # All timeframes positive and RSI not overbought
    all_bull = (d['ret_6'] > 0) & (d['ret_42'] > 0) & (d['ret_126'] > 0) & (r < 65) & (r > 50)
    all_bear = (d['ret_6'] < 0) & (d['ret_42'] < 0) & (d['ret_126'] < 0) & (r > 35) & (r < 50)
    
    # Only take first signal in a window
    lc = all_bull & (~all_bull.shift(1).fillna(False))
    sc = all_bear & (~all_bear.shift(1).fillna(False))
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc; s['short_entry'] = sc
    s['long_exit'] = (d['ret_6'] < -0.02) | (r > 80)
    s['short_exit'] = (d['ret_6'] > 0.02) | (r < 20)
    s['sl'] = np.where(lc, d['Close'] - 2.5 * at, np.where(sc, d['Close'] + 2.5 * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 5.0 * at, np.where(sc, d['Close'] - 5.0 * at, np.nan))
    return s

# ─── RUNNER ──────────────────────────────────────────────────────────────────
def run_all():
    df_raw = build_dataset()
    phase_a = df_raw[~df_raw['is_oos']]
    phase_b = df_raw[df_raw['is_oos']]
    
    print(f"\n{'='*90}")
    print(f"  BTC 4H STRATEGY RESEARCH — Phase A (In-Sample 80%)")
    print(f"  {len(phase_a)} bars | {phase_a['Date'].min()} → {phase_a['Date'].max()}")
    print(f"{'='*90}")
    
    strats = [
        ("S1: ADX Trend Rider", s1_trend_rider),
        ("S2: Stoch Oversold Momentum", s2_stoch_oversold_momentum),
        ("S3: Range Breakout", s3_range_breakout),
        ("S4: Momentum Continuation (EMA Cascade)", s4_momentum_continuation),
        ("S5: Golden Cross Sniper", s5_golden_cross_sniper),
        ("S6: Dip Buyer (RSI < 25)", s6_dip_buyer),
        ("S7: BB Mean Reversion", s7_bb_mean_reversion),
        ("S8: Triple Confirmation", s8_triple_confirmation),
        ("S9: Weekly Momentum", s9_weekly_momentum),
        ("S10: Long-Only Extreme Dip", s10_long_only_dip),
        ("S11: Momentum Cascade (Multi-TF)", s11_momentum_cascade),
    ]
    
    results_a = {}
    for name, fn in strats:
        tr = Engine(phase_a).run(fn(phase_a))
        res = rpt(tr, f"A | {name}")
        results_a[name] = res

    print(f"\n{'='*90}")
    print(f"  BTC 4H STRATEGY RESEARCH — Phase B (Out-of-Sample 20%)")
    print(f"  {len(phase_b)} bars | {phase_b['Date'].min()} → {phase_b['Date'].max()}")
    print(f"{'='*90}")
    
    results_b = {}
    for name, fn in strats:
        tr = Engine(phase_b).run(fn(phase_b))
        res = rpt(tr, f"B | {name}")
        results_b[name] = res
    
    print(f"\n{'='*90}")
    print("  STRATEGY SCORECARD")
    print(f"{'='*90}")
    for name in [s[0] for s in strats]:
        ra = results_a.get(name, {})
        rb = results_b.get(name, {})
        wr_a = ra.get('wr', 0) * 100; pf_a = ra.get('pf', 0)
        wr_b = rb.get('wr', 0) * 100; pf_b = rb.get('pf', 0)
        ret_a = ra.get('ret', 0) * 100; ret_b = rb.get('ret', 0) * 100
        print(f"  {name:45s} | A: WR={wr_a:4.0f}% PF={pf_a:5.2f} Ret={ret_a:+6.1f}% | B: WR={wr_b:4.0f}% PF={pf_b:5.2f} Ret={ret_b:+6.1f}%")

if __name__ == '__main__':
    run_all()
