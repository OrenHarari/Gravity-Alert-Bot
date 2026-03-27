"""
================================================================================
BTC 4H BACKTESTING ENGINE
================================================================================
Same architecture as ETH: 80/20 In-Sample/Out-of-Sample split.
Multiple strategies tested and optimized for Bitcoin's unique characteristics.
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

CEO_WR_MIN = 0.50
CEO_PF_MIN = 1.30

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

def macd(s,f=5,sl=13,sig=3):
    ml=ema(s,f)-ema(s,sl); return ml,ema(ml,sig),ml-ema(ml,sig)

def rsi(s, n=14):
    d = s.diff()
    u = d.clip(lower=0).ewm(com=n-1, min_periods=n).mean()
    v = (-d.clip(upper=0)).ewm(com=n-1, min_periods=n).mean()
    return 100 - 100 / (1 + u / v.replace(0, np.nan))

def bollinger_bands(s, n=20, std=2):
    sma = s.rolling(window=n).mean()
    roll_std = s.rolling(window=n).std()
    upper = sma + (roll_std * std)
    lower = sma - (roll_std * std)
    return sma, upper, lower

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

def rpt(tr,label,show_trades=True):
    if tr is None or len(tr)==0:
        print(f"\n  [{label}]: 0 trades fired.")
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
    ok=wr>=CEO_WR_MIN and pf>=CEO_PF_MIN
    print(f"\n  {'='*62}")
    print(f"  QA: {label}")
    print(f"  N={n}({int(nw)}W/{n-int(nw)}L)  WR={wr*100:.1f}%{'✅' if wr>=CEO_WR_MIN else '❌'}  "
          f"PF={pf:.2f}{'✅' if pf>=CEO_PF_MIN else '❌'}")
    print(f"  AvgW={aw:+.1f}%  AvgL={al:+.1f}%  Ret={ret*100:+.1f}%  MDD={mdd*100:.1f}%  Sh={sh:.2f}")
    if show_trades and len(tr) <= 15:
        print(f"\n  Trades:")
        print(tr[['type','entry_date','exit_date','entry','exit','pnl','reason']].to_string(index=False))
    return dict(label=label,n=n,wr=wr,pf=pf,aw=aw,al=al,ret=ret,mdd=mdd,sh=sh,ok=ok)

# ─── STRATEGIES ──────────────────────────────────────────────────────────────

def s_dual_momentum_pullback(df):
    """
    Trend + Deep Pullback (Same as ETH).
    BTC is less volatile than ETH so we may need to adapt thresholds.
    """
    d = df.copy()
    sma, upper, lower = bollinger_bands(d['Close'], n=20, std=2.0)
    ml, _, macd_h = macd(d['Close'], 12, 26, 9)
    r = rsi(d['Close'], 14)
    e100 = ema(d['Close'], 100)
    at = atr(d, 10)
    
    bull_regime = (d['Close'] > e100) & (ml > 0)
    bear_regime = (d['Close'] < e100) & (ml < 0)
    
    lc = bull_regime & ((d['Close'] <= lower) | (r < 25))
    sc = bear_regime & ((d['Close'] >= upper) | (r > 65))
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc; s['short_entry'] = sc
    s['long_exit'] = (d['Close'] > sma) | (r > 60)
    s['short_exit'] = (d['Close'] < sma) | (r < 40)
    s['sl'] = np.where(lc, d['Close'] - 1.5 * at, np.where(sc, d['Close'] + 1.5 * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 5.0 * at, np.where(sc, d['Close'] - 5.0 * at, np.nan))
    return s

def s_creative_price_action(df):
    """Pure Price Action Sweeps (Same as ETH)."""
    d = df.copy()
    d['is_green'] = d['Close'] > d['Open']
    d['is_red'] = d['Close'] < d['Open']
    d['body'] = abs(d['Close'] - d['Open'])
    d['avg_body'] = d['body'].rolling(14).mean()
    
    three_red = d['is_red'].shift(1) & d['is_red'].shift(2) & d['is_red'].shift(3)
    engulfing = d['is_green'] & (d['Close'] > d['Open'].shift(1)) & (d['Open'] < d['Close'].shift(1))
    strong_candle = d['body'] > (d['avg_body'] * 1.5)
    lc = three_red & engulfing & strong_candle
    
    three_green = d['is_green'].shift(1) & d['is_green'].shift(2) & d['is_green'].shift(3)
    engulfing_down = d['is_red'] & (d['Close'] < d['Open'].shift(1)) & (d['Open'] > d['Close'].shift(1))
    sc = three_green & engulfing_down & strong_candle
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc; s['short_entry'] = sc
    s['long_exit'] = d['is_red'] & (d['body'] > d['avg_body'])
    s['short_exit'] = d['is_green'] & (d['body'] > d['avg_body'])
    
    at = atr(d, 10)
    s['sl'] = np.where(lc, d['Low'].shift(1) - 0.5 * at, np.where(sc, d['High'].shift(1) + 0.5 * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 4.0 * at, np.where(sc, d['Close'] - 4.0 * at, np.nan))
    return s

def s_creative_volatility_squeeze(df):
    """The Coiled Spring (Volatility Squeeze)."""
    d = df.copy()
    sma, upper, lower = bollinger_bands(d['Close'], n=20, std=2.0)
    d['bb_width'] = (upper - lower) / sma
    d['bb_width_mean'] = d['bb_width'].rolling(50).mean()
    squeeze = d['bb_width'] < (d['bb_width_mean'] * 0.8)
    recent_squeeze = squeeze.rolling(3).max() > 0
    
    at = atr(d, 10); r = rsi(d['Close'], 14)
    lc = recent_squeeze & (d['Close'] > upper) & (r > 60)
    sc = recent_squeeze & (d['Close'] < lower) & (r < 40)
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc; s['short_entry'] = sc
    s['long_exit'] = r < 45; s['short_exit'] = r > 55
    s['sl'] = np.where(lc, d['Close'] - 1.5 * at, np.where(sc, d['Close'] + 1.5 * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 6.0 * at, np.where(sc, d['Close'] - 6.0 * at, np.nan))
    return s

def s_macd_rsi_combo(df):
    """MACD Histogram Cross + RSI Confirmation."""
    d = df.copy()
    ml, sg, h = macd(d['Close'], 12, 26, 9)
    r = rsi(d['Close'], 14)
    at = atr(d, 10)
    
    h_cross_up = (h > 0) & (h.shift(1) <= 0)
    h_cross_dn = (h < 0) & (h.shift(1) >= 0)
    
    lc = h_cross_up & (r > 40) & (r < 65)
    sc = h_cross_dn & (r < 60) & (r > 35)
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc; s['short_entry'] = sc
    s['long_exit'] = (h < 0)
    s['short_exit'] = (h > 0)
    s['sl'] = np.where(lc, d['Close'] - 2.0 * at, np.where(sc, d['Close'] + 2.0 * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 4.0 * at, np.where(sc, d['Close'] - 4.0 * at, np.nan))
    return s

def s_ema_crossover_momentum(df):
    """
    BTC-SPECIFIC: EMA Crossover w/ Momentum Filter.
    BTC trends harder than ETH, so classic EMA crossovers work better.
    Uses EMA 9/21 cross with RSI momentum confirmation.
    """
    d = df.copy()
    e9 = ema(d['Close'], 9)
    e21 = ema(d['Close'], 21)
    e100 = ema(d['Close'], 100)
    r = rsi(d['Close'], 14)
    at = atr(d, 10)
    
    # Golden cross: fast EMA crosses above slow, price above long-term trend
    cross_up = (e9 > e21) & (e9.shift(1) <= e21.shift(1)) & (d['Close'] > e100)
    cross_dn = (e9 < e21) & (e9.shift(1) >= e21.shift(1)) & (d['Close'] < e100)
    
    lc = cross_up & (r > 50) & (r < 70)
    sc = cross_dn & (r < 50) & (r > 30)
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc; s['short_entry'] = sc
    s['long_exit'] = (e9 < e21) | (r > 80)
    s['short_exit'] = (e9 > e21) | (r < 20)
    s['sl'] = np.where(lc, d['Close'] - 2.0 * at, np.where(sc, d['Close'] + 2.0 * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 5.0 * at, np.where(sc, d['Close'] - 5.0 * at, np.nan))
    return s

def s_structure_breakout(df):
    """
    BTC-SPECIFIC: Market Structure Breakout.
    Detects higher-highs/higher-lows pattern breaks.
    BTC makes clean structural moves that can be exploited.
    """
    d = df.copy()
    n = 10  # lookback for structure
    
    d['high_n'] = d['High'].rolling(n).max()
    d['low_n'] = d['Low'].rolling(n).min()
    d['prev_high'] = d['high_n'].shift(1)
    d['prev_low'] = d['low_n'].shift(1)
    
    # Breakout above recent structure high
    breakout_up = (d['Close'] > d['prev_high']) & (d['Close'].shift(1) <= d['prev_high'].shift(1))
    # Breakdown below recent structure low
    breakout_dn = (d['Close'] < d['prev_low']) & (d['Close'].shift(1) >= d['prev_low'].shift(1))
    
    r = rsi(d['Close'], 14)
    at = atr(d, 10)
    e50 = ema(d['Close'], 50)
    
    # Confirm with trend and momentum
    lc = breakout_up & (d['Close'] > e50) & (r > 55) & (r < 75)
    sc = breakout_dn & (d['Close'] < e50) & (r < 45) & (r > 25)
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc; s['short_entry'] = sc
    s['long_exit'] = (d['Close'] < e50) | (r > 80)
    s['short_exit'] = (d['Close'] > e50) | (r < 20)
    s['sl'] = np.where(lc, d['Close'] - 1.8 * at, np.where(sc, d['Close'] + 1.8 * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 4.5 * at, np.where(sc, d['Close'] - 4.5 * at, np.nan))
    return s

# ─── RUNNER ──────────────────────────────────────────────────────────────────
def run_all():
    df_raw = build_dataset()
    phase_a = df_raw[~df_raw['is_oos']]
    phase_b = df_raw[df_raw['is_oos']]
    
    print(f"\n{'#'*65}")
    print(f"  BTC — PHASE A — IN-SAMPLE VERIFICATION (80%)")
    print(f"  {len(phase_a)} 4H bars | {phase_a['Date'].min()} → {phase_a['Date'].max()}")
    print(f"{'#'*65}")
    
    strats = [
        ("Trend + Deep Pullback", s_dual_momentum_pullback),
        ("Pure Price Action Sweeps", s_creative_price_action),
        ("Volatility Squeeze", s_creative_volatility_squeeze),
        ("MACD + RSI Combo", s_macd_rsi_combo),
        ("EMA Crossover Momentum", s_ema_crossover_momentum),
        ("Structure Breakout", s_structure_breakout),
    ]
    
    results_a = {}
    for name, fn in strats:
        tr = Engine(phase_a).run(fn(phase_a))
        res = rpt(tr, f"Phase A | {name}", show_trades=False)
        results_a[name] = res

    print(f"\n{'#'*65}")
    print(f"  BTC — PHASE B — OUT-OF-SAMPLE TEST (20%)")
    print(f"  {len(phase_b)} 4H bars | {phase_b['Date'].min()} → {phase_b['Date'].max()}")
    print(f"{'#'*65}")
    
    results_b = {}
    for name, fn in strats:
        tr = Engine(phase_b).run(fn(phase_b))
        res = rpt(tr, f"Phase B | {name}", show_trades=False)
        results_b[name] = res
    
    # Summary
    print(f"\n{'='*65}")
    print("SUMMARY — BEST STRATEGIES FOR BTC 4H")
    print(f"{'='*65}")
    for name in results_a:
        ra = results_a.get(name, {})
        rb = results_b.get(name, {})
        ok_a = ra.get('ok', False)
        ok_b = rb.get('ok', False)
        status = "✅ PASS" if ok_a and ok_b else ("⚠️ PARTIAL" if ok_a or ok_b else "❌ FAIL")
        wr_a = ra.get('wr', 0) * 100
        pf_a = ra.get('pf', 0)
        wr_b = rb.get('wr', 0) * 100
        pf_b = rb.get('pf', 0)
        print(f"  {name:35s} | A: WR={wr_a:.0f}% PF={pf_a:.2f} | B: WR={wr_b:.0f}% PF={pf_b:.2f} | {status}")

if __name__ == '__main__':
    run_all()
