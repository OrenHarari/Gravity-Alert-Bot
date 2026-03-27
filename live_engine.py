"""
================================================================================
GRAVITY TRADING LAB — LIVE 4H ALERT ENGINE
================================================================================
Monitors ETH-USD every 4 hours for the 'Trend + Deep Pullback' Sniper Strategy.
Sends real-time entry/exit alerts to Telegram.
================================================================================
"""
import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime
import requests
import os
import warnings
warnings.filterwarnings('ignore')

# ─── SECRETS & SETUP ────────────────────────────────────────────────────────
DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(DIR, ".env")

def load_env():
    env = {}
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE, "r") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    env[k] = v
    return env

ENV = load_env()
TG_TOKEN = ENV.get("TG_TOKEN", "")
TG_CHAT_ID = ENV.get("TG_CHAT_ID", "")

def send_telegram(message):
    if not TG_TOKEN or not TG_CHAT_ID:
        print("Telegram not configured. Message:", message)
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {
        "chat_id": TG_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Telegram alert sent!")
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")

# ─── INDICATORS ─────────────────────────────────────────────────────────────
ema = lambda s,n: s.ewm(span=n,adjust=False).mean()

def atr(df, n=14):
    h = df['High']; l = df['Low']; c = df['Close']; pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(com=n-1, min_periods=n).mean()

def rsi(s, n=14):
    d = s.diff()
    u = d.clip(lower=0).ewm(com=n-1, min_periods=n).mean()
    v = (-d.clip(upper=0)).ewm(com=n-1, min_periods=n).mean()
    return 100 - 100 / (1 + u / v.replace(0, np.nan))

def bollinger_bands(s, n=20, std=2.0):
    sma = s.rolling(window=n).mean()
    rstd = s.rolling(window=n).std()
    return sma, sma + (rstd * std), sma - (rstd * std)

def macd(s, f=12, sl=26, sig=9):
    ml = ema(s, f) - ema(s, sl)
    return ml, ema(ml, sig), ml - ema(ml, sig)

# ─── STRATEGY ENGINE ────────────────────────────────────────────────────────
def check_market(symbol="ETH-USD"):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetching latest market data for {symbol}...")
    
    # Download enough data for 200 EMA (need ~250 candles)
    df = yf.download(symbol, period="60d", interval="4h", progress=False)
    if df.empty:
        print("Failed to fetch data.")
        return
        
    df.reset_index(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    
    if 'Datetime' in df.columns:
        df.rename(columns={'Datetime': 'Date'}, inplace=True)
        
    # Calculate Indicators (Trend + Deep Pullback exact specs)
    sma, upper, lower = bollinger_bands(df['Close'], n=20, std=2.0)
    ml, _, _ = macd(df['Close'], 12, 26, 9)
    r = rsi(df['Close'], 14)
    e_trend = ema(df['Close'], 200)
    current_atr = atr(df, 14)
    
    df['lower_bb'] = lower
    df['upper_bb'] = upper
    df['macd_line'] = ml
    df['rsi'] = r
    df['ema_200'] = e_trend
    df['atr'] = current_atr
    
    # We analyze the LAST COMPLETED candle
    # iloc[-1] is the current open candle, iloc[-2] is the last closed candle
    last_closed = df.iloc[-2]
    current_price = df.iloc[-1]['Close'] # Live price
    
    bull_rc = (last_closed['Close'] > last_closed['ema_200']) and (last_closed['macd_line'] > 0)
    bear_rc = (last_closed['Close'] < last_closed['ema_200']) and (last_closed['macd_line'] < 0)
    
    long_signal = bull_rc and ((last_closed['Close'] <= last_closed['lower_bb']) or (last_closed['rsi'] < 25))
    short_signal = bear_rc and ((last_closed['Close'] >= last_closed['upper_bb']) or (last_closed['rsi'] > 65))
    
    print(f"  > Closed Price: ${last_closed['Close']:.2f} | RSI: {last_closed['rsi']:.1f} | EMA200: ${last_closed['ema_200']:.2f}")
    
    if long_signal:
        sl = current_price - (2.0 * last_closed['atr'])
        tp = current_price + (5.0 * last_closed['atr'])
        msg = (
            f"🟢 *GRAVITY LAB: LONG ALERT* 🟢\n"
            f"Asset: `{symbol}` (4H Timeframe)\n"
            f"Strategy: *Trend + Deep Pullback (Sniper)*\n\n"
            f"💵 *Entry Price:* ${current_price:.2f} (Market)\n"
            f"🛡️ *Stop-Loss (2x ATR):* ${sl:.2f}\n"
            f"🎯 *Take-Profit (5x ATR):* ${tp:.2f}\n\n"
            f"📊 *Context:* Trend is UP (Above EMA200), MACD bullish, Price hit Lower BB or RSI < 25."
        )
        send_telegram(msg)
        return "LONG"
        
    elif short_signal:
        sl = current_price + (2.0 * last_closed['atr'])
        tp = current_price - (5.0 * last_closed['atr'])
        msg = (
            f"🔴 *GRAVITY LAB: SHORT ALERT* 🔴\n"
            f"Asset: `{symbol}` (4H Timeframe)\n"
            f"Strategy: *Trend + Deep Pullback (Sniper)*\n\n"
            f"💵 *Entry Price:* ${current_price:.2f} (Market)\n"
            f"🛡️ *Stop-Loss (2x ATR):* ${sl:.2f}\n"
            f"🎯 *Take-Profit (5x ATR):* ${tp:.2f}\n\n"
            f"📊 *Context:* Trend is DOWN (Below EMA200), MACD bearish, Price hit Upper BB or RSI > 65."
        )
        send_telegram(msg)
        return "SHORT"
        
    else:
        print("  > No trade signal found at this time. Monitoring...")
        return "NONE"

# ─── MAIN SCHEDULER ─────────────────────────────────────────────────────────
def run_live():
    print("="*60)
    print("🚀 GRAVITY LIVE ALERT ENGINE STARTED")
    print("Monitoring ETH-USD 4H timeframe...")
    print("Waiting for the next 4H candle close...")
    print("="*60)
    
    # Run once at startup just to make sure everything works
    check_market()
    
    while True:
        now = datetime.utcnow()
        # 4H candles typically close at 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC
        # We wait until exactly 1 minute past the closing hour to ensure data is updated
        if now.hour % 4 == 0 and now.minute == 1:
            check_market()
            # Sleep for 50 minutes to avoid running multiple times in the same minute
            time.sleep(3000)
        else:
            time.sleep(30) # Check time every 30 seconds

if __name__ == "__main__":
    run_live()
