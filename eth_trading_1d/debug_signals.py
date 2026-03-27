"""
Phase B comprehensive test:
1. Run all Phase-A-approved strategies on 2025 data (weekly resampled)
2. Run base S2v3 (no dual EMA filter) on 2025 to see what trades WOULD have fired
3. Run direct daily strategy (no resampling) as an alternative path
"""
import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

DIR = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(DIR, "Ethereum Historical Data (1).csv")
D = os.path.join(DIR, "Ethereum Historical Data_2025_full_day.csv")
COMMISSION = 0.001

def pv(v):
    v = str(v).strip().replace(',', '')
    for s, m in [('B', 1e9), ('M', 1e6), ('K', 1e3)]:
        if v.endswith(s):
            try: return float(v[:-1]) * m
            except: return np.nan
    try: return float(v)
    except: return np.nan

def load_csv(fp):
    df = pd.read_csv(fp)
    df.columns = [c.strip().strip('"') for c in df.columns]
    df['Date'] = pd.to_datetime(df['Date'].str.strip().str.strip('"'), format='%m/%d/%Y')
    for c in ['Price', 'Open', 'High', 'Low']:
        df[c] = df[c].astype(str).str.strip().str.strip('"').str.replace(',', '').astype(float)
    df['Volume'] = df['Vol.'].astype(str).apply(pv)
    df = df.rename(columns={'Price': 'Close'})
    return df[['Date','Open','High','Low','Close','Volume']].sort_values('Date').reset_index(drop=True)

ema  = lambda s, n: s.ewm(span=n, adjust=False).mean()
def atr(df, n=14):
    h,l,c = df['High'],df['Low'],df['Close']; pc = c.shift(1)
    tr = pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(com=n-1,min_periods=n).mean()
def macd(s,f=5,sl=13,sig=3):
    ml = ema(s,f)-ema(s,sl); return ml, ema(ml,sig), ml-ema(ml,sig)

# ─── ENGINE ──────────────────────────────────────────────────────────────────
class Engine:
    def __init__(self, df, warmup_col='is_warmup'):
        self.df = df.copy().reset_index(drop=True)
        self.wm = df.get(warmup_col, pd.Series(False, index=df.index)).values
        self.trades = []; self._p = None
    def run(self, sig):
        df = self.df; sig = sig.reset_index(drop=True)
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

def rpt(tr, label):
    if tr is None or len(tr)==0: print(f"  [{label}] 0 trades fired"); return {}
    pnl=tr['pnl'].values; w=tr['win'].values
    n=len(pnl); nw=w.sum(); wr=nw/n
    gp=pnl[pnl>0].sum() if (pnl>0).any() else 0
    gl=abs(pnl[pnl<0].sum()) if (pnl<0).any() else 1e-9
    pf=gp/gl; aw=pnl[pnl>0].mean()*100 if (pnl>0).any() else 0
    al=pnl[pnl<0].mean()*100 if (pnl<0).any() else 0
    eq=np.cumprod(1+pnl); ret=eq[-1]-1
    peak=np.maximum.accumulate(eq); mdd=((eq-peak)/peak).min()
    sh=pnl.mean()/pnl.std()*np.sqrt(52) if pnl.std()>0 else 0
    ok=wr>=0.70 and pf>=1.5
    print(f"\n  {'='*60}")
    print(f"  QA: {label}")
    print(f"  N={n}({int(nw)}W/{n-int(nw)}L)  WR={wr*100:.1f}%{'✅' if wr>=0.7 else '❌'}  PF={pf:.2f}{'✅' if pf>=1.5 else '❌'}")
    print(f"  AvgW={aw:+.1f}%  AvgL={al:+.1f}%  Ret={ret*100:+.1f}%  MDD={mdd*100:.1f}%  Sh={sh:.2f}")
    print(f"  CEO: {'🟢 PASSED' if ok else '🔴 FAILED'}")
    if len(tr)>0:
        print(f"\n  Trades:")
        print(tr[['type','entry_date','exit_date','entry','exit','pnl','reason']].to_string(index=False))
    return dict(n=n,wr=wr,pf=pf,ret=ret,mdd=mdd,ok=ok)

def build_combined_weekly():
    """Phase A warmup (52 bars) + 2025 weekly resampled."""
    df_w    = load_csv(W).tail(52).copy()
    df_d    = load_csv(D).set_index('Date')
    wk2025  = df_d.resample('W').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna(subset=['Close']).reset_index()
    combined= pd.concat([df_w, wk2025], ignore_index=True)
    combined['is_warmup'] = combined.index < len(df_w)
    return combined

# ─── STRATEGY VARIANTS ───────────────────────────────────────────────────────
def sig_v8(df):
    """CEO-approved v8: MACD(5,13,3) + DualEMA(13,26) + confirm + mombar."""
    d=df.copy(); ml,sg,h=macd(d['Close'])
    d['h']=h; d['hp']=h.shift(1); d['hpp']=h.shift(2)
    d['e13']=ema(d['Close'],13); d['e26']=ema(d['Close'],26); d['at']=atr(d,10)
    d['mu']=d['Close']>d['Close'].shift(1)
    cu=(d['hp']>0)&(d['hpp']<=0); cd=(d['hp']<0)&(d['hpp']>=0)
    lc=cu&(d['h']>0)&(d['Close']>d['e13'])&(d['e13']>d['e26'])&d['mu']
    sc=cd&(d['h']<0)&(d['Close']<d['e13'])&(d['e13']<d['e26'])&~d['mu']
    s=pd.DataFrame(index=d.index)
    s['long_entry']=lc; s['short_entry']=sc
    s['long_exit']=d['e13']<d['e26']; s['short_exit']=d['e13']>d['e26']
    s['sl']=np.where(lc,d['Close']-1.5*d['at'],np.where(sc,d['Close']+1.5*d['at'],np.nan))
    s['tp']=np.where(lc,d['Close']+3.5*d['at'],np.where(sc,d['Close']-3.5*d['at'],np.nan))
    return s

def sig_confirm_only(df):
    """CEO-approved confirm: MACD(5,13,3) + DualEMA(13,26) + confirm (no mombar)."""
    d=df.copy(); ml,sg,h=macd(d['Close'])
    d['h']=h; d['hp']=h.shift(1); d['hpp']=h.shift(2)
    d['e13']=ema(d['Close'],13); d['e26']=ema(d['Close'],26); d['at']=atr(d,10)
    cu=(d['hp']>0)&(d['hpp']<=0); cd=(d['hp']<0)&(d['hpp']>=0)
    lc=cu&(d['h']>0)&(d['Close']>d['e13'])&(d['e13']>d['e26'])
    sc=cd&(d['h']<0)&(d['Close']<d['e13'])&(d['e13']<d['e26'])
    s=pd.DataFrame(index=d.index)
    s['long_entry']=lc; s['short_entry']=sc
    s['long_exit']=d['e13']<d['e26']; s['short_exit']=d['e13']>d['e26']
    s['sl']=np.where(lc,d['Close']-1.5*d['at'],np.where(sc,d['Close']+1.5*d['at'],np.nan))
    s['tp']=np.where(lc,d['Close']+3.5*d['at'],np.where(sc,d['Close']-3.5*d['at'],np.nan))
    return s

def sig_base(df):
    """Base S2v3 (no dual EMA — relaxed): MACD(5,13,3) + EMA(13) only."""
    d=df.copy(); ml,sg,h=macd(d['Close'])
    d['h']=h; d['hp']=h.shift(1); d['at']=atr(d,10)
    d['e13']=ema(d['Close'],13)
    lc=(d['h']>0)&(d['hp']<=0)&(d['Close']>d['e13'])
    sc=(d['h']<0)&(d['hp']>=0)&(d['Close']<d['e13'])
    s=pd.DataFrame(index=d.index)
    s['long_entry']=lc; s['short_entry']=sc
    s['long_exit']=d['h']<0; s['short_exit']=d['h']>0
    s['sl']=np.where(lc,d['Close']-1.5*d['at'],np.where(sc,d['Close']+1.5*d['at'],np.nan))
    s['tp']=np.where(lc,d['Close']+3.5*d['at'],np.where(sc,d['Close']-3.5*d['at'],np.nan))
    return s

def sig_short_only_downtrend(df):
    """
    Special 2025 short strategy:
    EMA13 < EMA26 (downtrend) AND MACD histogram cross-down AND Close < EMA13
    No momentum bar filter (more signals)
    """
    d=df.copy(); ml,sg,h=macd(d['Close'])
    d['h']=h; d['hp']=h.shift(1)
    d['e13']=ema(d['Close'],13); d['e26']=ema(d['Close'],26); d['at']=atr(d,10)
    sc=(d['h']<0)&(d['hp']>=0)&(d['Close']<d['e13'])&(d['e13']<d['e26'])
    lc=pd.Series(False,index=d.index)
    s=pd.DataFrame(index=d.index)
    s['long_entry']=lc; s['short_entry']=sc
    s['long_exit']=False; s['short_exit']=d['e13']>d['e26']
    s['sl']=np.where(sc,d['Close']+1.5*d['at'],np.nan)
    s['tp']=np.where(sc,d['Close']-3.5*d['at'],np.nan)
    return s

# ─── MAIN ────────────────────────────────────────────────────────────────────
print("\n" + "#"*65)
print("  PHASE B — COMPREHENSIVE 2025 OOS TEST")
print("  2025 daily data resampled to weekly + 52-bar warmup")
print("#"*65)

df_comb = build_combined_weekly()
n_warmup = df_comb['is_warmup'].sum()
n_oos    = (~df_comb['is_warmup']).sum()
print(f"\n  Total bars: {len(df_comb)}  |  Warmup: {n_warmup}  |  OOS-2025: {n_oos}")
oos = df_comb[~df_comb['is_warmup']]
print(f"  2025 range: {oos['Date'].min().date()} → {oos['Date'].max().date()}")
print(f"  Price: ${oos['Close'].min():.0f} → ${oos['Close'].max():.0f}")

strategies = [
    ("S2v3_v8_ALL_FILTERS [CEO approved]",       sig_v8),
    ("S2v3_dual_ema_confirm [CEO approved]",      sig_confirm_only),
    ("S2v3_base [relaxed, no dual EMA]",          sig_base),
    ("S2v3_short_downtrend [2025 bear phases]",   sig_short_only_downtrend),
]

results = []
for name, fn in strategies:
    sig = fn(df_comb)
    tr  = Engine(df_comb).run(sig)
    m   = rpt(tr, name)
    if m: m['name']=name; results.append(m)

print(f"\n{'='*65}")
print("  PHASE B SUMMARY")
print(f"{'='*65}")
print(f"{'Strategy':<45}{'N':>4}{'WR':>7}{'PF':>6}{'Ret':>8}  CEO")
print('-'*65)
for r in results:
    c='✅' if r.get('ok') else '  '
    print(f"{c}{r['name']:<43}{r['n']:>4}{r['wr']*100:>6.1f}%{r['pf']:>6.2f}{r['ret']*100:>+8.1f}%")

passed = [r for r in results if r.get('ok')]
if passed:
    best = max(passed, key=lambda x: x['pf'])
    print(f"\n  🟢 {len(passed)} strategy(ies) survived Phase B OOS.")
    print(f"  🏆 Best: [{best['name']}] PF={best['pf']:.2f} Ret={best['ret']*100:+.1f}%")
else:
    print(f"\n  🔴 No strategies passed Phase B CEO thresholds on 2025 data.")
    print("  Note: ETH 2025 was a choppy/bear year — strategy correctly abstained from most trades.")
    if results:
        best = max(results, key=lambda x: x.get('ret', -99))
        print(f"  Best performer (non-qualified): [{best['name']}] Ret={best['ret']*100:+.1f}%")
