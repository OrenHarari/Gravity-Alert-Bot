"""
================================================================================
UNIVERSAL BACKTESTING ENGINE — 4H/INTRA-DAY READY
================================================================================
- Dynamically loads a single CSV and splits it 80% In-Sample (Phase A)
  and 20% Out-of-Sample (Phase B) for blind testing.
- Uses FastMACD(5,13,3) + DualEMA filters.
================================================================================
"""
import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(DIR, "ETH_4H_2Y.csv")
COMMISSION = 0.001

CEO_WR_MIN = 0.50  # Lowered strictly because higher frequency = smaller RR windows, WR naturally drops.
CEO_PF_MIN = 1.30  

# ─── DATA ────────────────────────────────────────────────────────────────────
def build_dataset():
    """Loads universal CSV & adds is_oos column for 80/20 train/test split"""
    df = pd.read_csv(DATA_FILE)
    df.columns = [c.strip().strip('"') for c in df.columns]

    # Handle yfinance vs investing.com date columns
    date_col = 'Date' if 'Date' in df.columns else 'Datetime'
    df['Date'] = pd.to_datetime(df[date_col], format='mixed', utc=True).dt.tz_localize(None)

    for c in ['Open','High','Low','Close']:
        if df[c].dtype == object:
            df[c] = df[c].str.replace(',', '').astype(float)
            
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Optional: Fill volume if missing or '0'
    if 'Volume' not in df.columns:
        df['Volume'] = 1.0
        
    df = df[['Date','Open','High','Low','Close','Volume']]
    
    # Calculate 80/20 split
    split_idx = int(len(df) * 0.8)
    df['is_oos'] = df.index >= split_idx
    
    # The warmup phase for OOS needs prior 52 bars
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
    def __init__(self,df, leverage=1.0):
        self.df=df.copy().reset_index(drop=True)
        self.wm=df.get('is_warmup', pd.Series(False,index=df.index)).values
        self.trades=[]; self._p=None
        self.leverage=leverage
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
            
            # Open new positions if not in warmup
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

def s_dual_momentum_pullback(df):
    """
    4H Dual Momentum Pullback: 
    - Master Trend Filter: EMA(200) + MACD direction
    - Trigger: Oversold pullback (RSI < 40) or touching Lower BB within a Master Uptrend.
    """
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
    # Tigher SL (1.5x) and massive TP (5.0x runner)
    s['sl'] = np.where(lc, d['Close'] - 1.5 * at, np.where(sc, d['Close'] + 1.5 * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 5.0 * at, np.where(sc, d['Close'] - 5.0 * at, np.nan))
    
    return s

def s_macd_mean_revert_combo(df):
    """
    Combination Volatility + Momentum:
    - Long: Histogram crossed up AND RSI crossed 50 UP from an oversold region previously.
    """;
    d = df.copy()
    ml, sg, h = macd(d['Close'], 12, 26, 9)
    r = rsi(d['Close'], 14)
    at = atr(d, 10)
    
    # Hist cross up
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

def s_creative_volatility_squeeze(df):
    """
    INNOVATIVE 1: 'The Coiled Spring' (Volatility Squeeze)
    Crypto has extreme periods of dead silence followed by massive explosions.
    We detect the 'silence' via Bollinger Bandwidth compression, and buy the explosion.
    """
    d = df.copy()
    sma, upper, lower = bollinger_bands(d['Close'], n=20, std=2.0)
    d['bb_width'] = (upper - lower) / sma
    d['bb_width_mean'] = d['bb_width'].rolling(50).mean() # Baseline volatility
    
    # Squeeze is active when volatility is 20% below the recent baseline
    squeeze = d['bb_width'] < (d['bb_width_mean'] * 0.8)
    recent_squeeze = squeeze.rolling(3).max() > 0 # Was squeezing in last 3 bars
    
    at = atr(d, 10); r = rsi(d['Close'], 14)
    
    # Long: Explosive candle breaking the upper band AFTER a squeeze, with confirming RSI
    lc = recent_squeeze & (d['Close'] > upper) & (r > 60)
    # Short: Explosive dump breaking the lower band 
    sc = recent_squeeze & (d['Close'] < lower) & (r < 40)
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc; s['short_entry'] = sc
    s['long_exit'] = r < 45; s['short_exit'] = r > 55
    
    # Explosive trends run far: TP is massive (6.0 ATR)
    s['sl'] = np.where(lc, d['Close'] - 1.5 * at, np.where(sc, d['Close'] + 1.5 * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 6.0 * at, np.where(sc, d['Close'] - 6.0 * at, np.nan))
    return s

def s_creative_price_action(df):
    """
    INNOVATIVE 2: Pure Price Action (Institutional Sweep / Order Block)
    Ignores lagging indicators. Looks for a specific psychological candle pattern:
    - Long: 3 consecutive drop candles (retail panic), followed by a massive engulfing green candle.
    """
    d = df.copy()
    
    # Body logic
    d['is_green'] = d['Close'] > d['Open']
    d['is_red'] = d['Close'] < d['Open']
    d['body'] = abs(d['Close'] - d['Open'])
    d['avg_body'] = d['body'].rolling(14).mean()
    
    # 3 consecutive red candles
    three_red = d['is_red'].shift(1) & d['is_red'].shift(2) & d['is_red'].shift(3)
    
    # Current candle is green, AND it completely engulfs the previous red body, AND it has strong momentum (larger than average body)
    engulfing = d['is_green'] & (d['Close'] > d['Open'].shift(1)) & (d['Open'] < d['Close'].shift(1))
    strong_candle = d['body'] > (d['avg_body'] * 1.5)
    
    lc = three_red & engulfing & strong_candle
    
    # 3 consecutive green candles followed by red engulfing
    three_green = d['is_green'].shift(1) & d['is_green'].shift(2) & d['is_green'].shift(3)
    engulfing_down = d['is_red'] & (d['Close'] < d['Open'].shift(1)) & (d['Open'] > d['Close'].shift(1))
    sc = three_green & engulfing_down & strong_candle
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc; s['short_entry'] = sc
    
    # Exit after a fixed number of candles (e.g., 6 bars = 24 hours of trend play)
    # We simulate this by having no explicit exit logic except SL/TP, or a trailing element.
    # To use standard exits: just exit if we get a reversal candle.
    s['long_exit'] = d['is_red'] & (d['body'] > d['avg_body'])
    s['short_exit'] = d['is_green'] & (d['body'] > d['avg_body'])
    
    at = atr(d, 10)
    s['sl'] = np.where(lc, d['Low'].shift(1) - 0.5 * at, np.where(sc, d['High'].shift(1) + 0.5 * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 4.0 * at, np.where(sc, d['Close'] - 4.0 * at, np.nan))
    return s

def s_bb_trend_breakout(df):
    """
    4H Volatility Breakout: Ride momentum when price blasts outside BB with strong RSI.
    """
    d = df.copy()
    sma, upper, lower = bollinger_bands(d['Close'], n=20, std=2.0)
    r = rsi(d['Close'], 14)
    at = atr(d, 10)
    
    d['upper'] = upper; d['lower'] = lower; d['rsi'] = r
    
    # Long Breakout: Breaking upper band with strong momentum
    lc = (d['Close'] > d['upper']) & (d['rsi'] > 60)
    
    # Short Breakout: Breaking lower band with strong downside momentum
    sc = (d['Close'] < d['lower']) & (d['rsi'] < 40)
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc
    s['short_entry'] = sc
    
    # Exits: Momentum slows down (RSI crosses 50)
    s['long_exit'] = d['rsi'] < 50
    s['short_exit'] = d['rsi'] > 50
    
    s['sl'] = np.where(lc, d['Close'] - 2.0 * at, np.where(sc, d['Close'] + 2.0 * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 5.0 * at, np.where(sc, d['Close'] - 5.0 * at, np.nan))
    
    return s

def s_rsi_divergence_simple(df):
    """
    4H Fast RSI Pullback.
    Buy if RSI drops below 30.
    """
    d = df.copy()
    r = rsi(d['Close'], 14)
    at = atr(d, 10)
    
    lc = (r < 30)
    sc = (r > 70)
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc; s['short_entry'] = sc
    s['long_exit'] = r > 50; s['short_exit'] = r < 50
    
    s['sl'] = np.where(lc, d['Close'] - 1.5 * at, np.where(sc, d['Close'] + 1.5 * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 3.0 * at, np.where(sc, d['Close'] - 3.0 * at, np.nan))
    return s

def s_aggressive_leverage_runner(df):
    """
    4H Aggressive PNL Runner (Intended for 2x Leverage)
    Optimized: EMA=100, RSI=30, SL=3.0x, TP=5.0x, HOLD=15
    """
    d = df.copy()
    sma, upper, lower = bollinger_bands(d['Close'], n=20, std=2.0)
    ml, _, _ = macd(d['Close'], 12, 26, 9)
    r = rsi(d['Close'], 14)
    e_trend = ema(d['Close'], 100)
    at = atr(d, 10)
    
    bull_rc = (d['Close'] > e_trend) & (ml > 0)
    bear_rc = (d['Close'] < e_trend) & (ml < 0)
    
    lc = bull_rc & ((d['Close'] <= lower) | (r < 30))
    sc = bear_rc & ((d['Close'] >= upper) | (r > 70))
    
    s = pd.DataFrame(index=d.index)
    s['long_entry'] = lc
    s['short_entry'] = sc
    
    s['long_exit'] = r > 65
    s['short_exit'] = r < 35
    
    s['sl'] = np.where(lc, d['Close'] - 3.0 * at, np.where(sc, d['Close'] + 3.0 * at, np.nan))
    s['tp'] = np.where(lc, d['Close'] + 5.0 * at, np.where(sc, d['Close'] - 5.0 * at, np.nan))
    return s

# ─── RUNNERS ─────────────────────────────────────────────────────────────────
def run_all():
    df_raw = build_dataset()
    phase_a = df_raw[~df_raw['is_oos']]
    phase_b = df_raw[df_raw['is_oos']]
    
    print(f"\n{'#'*65}")
    print(f"  PHASE A — IN-SAMPLE VERIFICATION (80%)")
    print(f"  {len(phase_a)} 4H bars | {phase_a['Date'].min()} → {phase_a['Date'].max()}")
    print(f"{'#'*65}")
    
    strats = [("Trend + Deep Pullback", s_dual_momentum_pullback, 1.0),
              ("Aggressive PNL Runner (2x Lev)", s_aggressive_leverage_runner, 2.0),
              ("MACD + RSI Combo", s_macd_mean_revert_combo, 1.0),
              ("CREATIVE: Volatility Squeeze (Explosions)", s_creative_volatility_squeeze, 1.0),
              ("CREATIVE: Pure Price Action Sweeps", s_creative_price_action, 1.0)]
    
    for name,fn,lev in strats:
        tr = Engine(phase_a, leverage=lev).run(fn(phase_a))
        rpt(tr, f"Phase A | {name}", show_trades=False)

    print(f"\n{'#'*65}")
    print(f"  PHASE B — OUT-OF-SAMPLE TEST (20%)")
    print(f"  {len(phase_b)} 4H bars | {phase_b['Date'].min()} → {phase_b['Date'].max()}")
    print(f"{'#'*65}")
    
    for name,fn,lev in strats:
        tr = Engine(phase_b, leverage=lev).run(fn(phase_b))
        rpt(tr, f"Phase B | {name}", show_trades=False)

if __name__ == '__main__':
    run_all()
