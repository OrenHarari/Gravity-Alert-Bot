"""
Export 4H Backtest data to JSON for the Intraday UI dashboard.
"""
import pandas as pd
import numpy as np
import json
import warnings
import os
warnings.filterwarnings('ignore')

DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(DIR, "ETH_4H_2Y.csv")
COMMISSION = 0.001

# ─── DATA LOADER & SPLIT ─────────────────────────────────────────────────────
def build_dataset():
    df = pd.read_csv(DATA_FILE)
    df.columns = [c.strip().strip('"') for c in df.columns]
    date_col = 'Date' if 'Date' in df.columns else 'Datetime'
    df['Date'] = pd.to_datetime(df[date_col], format='mixed', utc=True).dt.tz_localize(None)
    for c in ['Open','High','Low','Close']:
        if df[c].dtype == object: df[c] = df[c].str.replace(',', '').astype(float)
    df = df.sort_values('Date').reset_index(drop=True)
    if 'Volume' not in df.columns: df['Volume'] = 1.0
    
    split_idx = int(len(df) * 0.8)
    df['is_oos'] = df.index >= split_idx
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
def rsi(s, n=14):
    d = s.diff()
    u = d.clip(lower=0).ewm(com=n-1, min_periods=n).mean()
    v = (-d.clip(upper=0)).ewm(com=n-1, min_periods=n).mean()
    return 100 - 100 / (1 + u / v.replace(0, np.nan))
def bollinger_bands(s, n=20, std=2):
    sma = s.rolling(window=n).mean()
    roll_std = s.rolling(window=n).std()
    return sma, sma + (roll_std * std), sma - (roll_std * std)
def macd(s,f=5,sl=13,sig=3):
    ml=ema(s,f)-ema(s,sl); return ml,ema(ml,sig),ml-ema(ml,sig)

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
                    elif s.get('long_exit',False): r,ep='sig',row['Close']
                else:
                    if hi>=p['sl']:   r,ep='SL',p['sl']
                    elif lo<=p['tp']: r,ep='TP',p['tp']
                    elif s.get('short_exit',False): r,ep='sig',row['Close']
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
        self.trades.append(dict(type=p['t'],entry_date=p['ed'].strftime('%Y-%m-%d %H:%M'),
                                exit_date=ed.strftime('%Y-%m-%d %H:%M'),
                                entry=round(e,2),exit=round(ep,2),sl=round(p['sl'],2),tp=round(p['tp'],2),
                                pnl=round(pnl*100,2),reason=r,win=bool(pnl>0)))

# ─── STRATEGIES ──────────────────────────────────────────────────────────────
def s_dual_momentum_pullback(df):
    d = df.copy()
    sma, upper, lower = bollinger_bands(d['Close'], n=20, std=2.0)
    ml, _, macd_h = macd(d['Close'], 12, 26, 9)
    r = rsi(d['Close'], 14); e100 = ema(d['Close'], 100); at = atr(d, 10)
    
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
    s['e200'] = e100; s['lower'] = lower; s['upper'] = upper; s['rsi'] = r
    return s

def s_creative_price_action(df):
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

def compute_metrics(tr):
    if tr is None or len(tr)==0:
        return dict(n=0,wins=0,losses=0,win_rate=0,profit_factor=0,avg_win=0,avg_loss=0,total_return=0,max_dd=0,sharpe=0,expectancy=0)
    pnl=tr['pnl'].values/100; w=tr['win'].values
    n=len(pnl); nw=int(w.sum())
    gp=pnl[pnl>0].sum() if (pnl>0).any() else 0
    gl=abs(pnl[pnl<0].sum()) if (pnl<0).any() else 1e-9
    pf=round(gp/gl,2); aw=round(pnl[pnl>0].mean()*100,1) if (pnl>0).any() else 0
    al=round(pnl[pnl<0].mean()*100,1) if (pnl<0).any() else 0
    wr=round(nw/n*100,1); exp=round(wr/100*aw+(1-wr/100)*al,2)
    eq=np.cumprod(1+pnl); ret=round((eq[-1]-1)*100,1)
    peak=np.maximum.accumulate(eq); mdd=round(((eq-peak)/peak).min()*100,1)
    sh=round(pnl.mean()/pnl.std()*np.sqrt(52),2) if pnl.std()>0 else 0
    return dict(n=n,wins=nw,losses=n-nw,win_rate=wr,profit_factor=pf,avg_win=aw,avg_loss=al,total_return=ret,max_dd=mdd,sharpe=sh,expectancy=exp)

def calc_equity(trades_df):
    if trades_df.empty: return [100.0]
    eq = [100.0]
    for pnl in trades_df['pnl'].values:
        eq.append(round(eq[-1] * (1 + pnl/100), 2))
    return eq

# ─── BUILD ───────────────────────────────────────────────────────────────────
print("Building dataset...")
df = build_dataset()
phase_a = df[~df['is_oos']].reset_index(drop=True)
phase_b = df[df['is_oos']].reset_index(drop=True)

print("Running strategies...")
sig_a_trend = s_dual_momentum_pullback(phase_a)
sig_a_pa    = s_creative_price_action(phase_a)
tr_a_trend  = Engine(phase_a).run(sig_a_trend)
tr_a_pa     = Engine(phase_a).run(sig_a_pa)

sig_b_trend = s_dual_momentum_pullback(phase_b)
sig_b_pa    = s_creative_price_action(phase_b)
tr_b_trend  = Engine(phase_b).run(sig_b_trend)
tr_b_pa     = Engine(phase_b).run(sig_b_pa)

candles_a = [dict(
    date=row.Date.strftime('%Y-%m-%d %H:%M'),
    open=round(row.Open,2), high=round(row.High,2), low=round(row.Low,2), close=round(row.Close,2),
    e200=round(sig_a_trend['e200'].iloc[i],2) if not pd.isna(sig_a_trend['e200'].iloc[i]) else None,
    lower=round(sig_a_trend['lower'].iloc[i],2) if not pd.isna(sig_a_trend['lower'].iloc[i]) else None,
) for i,row in phase_a.iterrows()]

candles_b = [dict(
    date=row.Date.strftime('%Y-%m-%d %H:%M'),
    open=round(row.Open,2), high=round(row.High,2), low=round(row.Low,2), close=round(row.Close,2),
) for _,row in phase_b.iterrows()]

out = dict(
    generated_at=pd.Timestamp.now().isoformat(),
    phase_a=dict(
        label="Phase A — In-Sample (80% 4H Data)",
        start=phase_a['Date'].min().strftime('%Y-%m-%d'), end=phase_a['Date'].max().strftime('%Y-%m-%d'),
        n_bars=len(phase_a), candles=candles_a,
        strategy_trend=dict(
            name="Trend + Deep Pullback",
            trades=tr_a_trend.to_dict('records') if not tr_a_trend.empty else [],
            equity=calc_equity(tr_a_trend),
            equity_dates=['Start'] + list(tr_a_trend['exit_date']) if not tr_a_trend.empty else ['Start'],
            metrics=compute_metrics(tr_a_trend)
        ),
        strategy_pa=dict(
            name="Pure Price Action Sweep",
            trades=tr_a_pa.to_dict('records') if not tr_a_pa.empty else [],
            equity=calc_equity(tr_a_pa),
            equity_dates=['Start'] + list(tr_a_pa['exit_date']) if not tr_a_pa.empty else ['Start'],
            metrics=compute_metrics(tr_a_pa)
        )
    ),
    phase_b=dict(
        label="Phase B — Out-of-Sample (20% blind 4H data)",
        start=phase_b['Date'].min().strftime('%Y-%m-%d'), end=phase_b['Date'].max().strftime('%Y-%m-%d'),
        n_bars=len(phase_b), daily_candles=candles_b,
        strategy_trend=dict(
            name="Trend + Deep Pullback OOS",
            trades=tr_b_trend.to_dict('records') if not tr_b_trend.empty else [],
            equity=calc_equity(tr_b_trend),
            equity_dates=['Start'] + list(tr_b_trend['exit_date']) if not tr_b_trend.empty else ['Start'],
            metrics=compute_metrics(tr_b_trend)
        ),
        strategy_pa=dict(
            name="Price Action Sweep OOS",
            trades=tr_b_pa.to_dict('records') if not tr_b_pa.empty else [],
            equity=calc_equity(tr_b_pa),
            equity_dates=['Start'] + list(tr_b_pa['exit_date']) if not tr_b_pa.empty else ['Start'],
            metrics=compute_metrics(tr_b_pa)
        )
    )
)

out_path = os.path.join(DIR, "backtest_4h_data.json")
with open(out_path,'w') as f: json.dump(out, f, indent=2, default=str)
print(f"Data exported to {out_path}")
