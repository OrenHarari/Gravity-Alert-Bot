import os
import time
import asyncio
import traceback
from datetime import datetime, timedelta
from threading import Thread
import yfinance as yf
import pandas as pd
import numpy as np
import requests

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="GRAVITY Trading Lab - Live System")

# Ensure static folder exists
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Data Store for the UI
SYSTEM_STATE = {
    "last_update": None,
    "next_candle": None,
    "status": "Initializing",
    "assets": {
        "ETH-USD": {
            "price": 0.0,
            "rsi": 0.0,
            "ema200": 0.0,
            "macd": 0.0,
            "lower_bb": 0.0,
            "upper_bb": 0.0,
            "atr": 0.0,
            "trend": "Unknown",
            "distance_to_long": 0.0,
            "distance_to_short": 0.0,
            "signal": "NONE"
        }
    },
    "recent_alerts": []
}

# Load Environment Secrets
from dotenv import load_dotenv
load_dotenv(".env")
TG_TOKEN = os.getenv("TG_TOKEN", "")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "")

# ─── CORE INDICATOR FUNCTIONS ────────────────────────
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

def send_telegram(message):
    if not TG_TOKEN or not TG_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TG_CHAT_ID, "text": message, "parse_mode": "Markdown"})
    # Keep last 5 alerts
    SYSTEM_STATE["recent_alerts"].insert(0, {"time": datetime.now().strftime('%H:%M:%S'), "msg": message})
    if len(SYSTEM_STATE["recent_alerts"]) > 5:
        SYSTEM_STATE["recent_alerts"].pop()

# ─── UPDATE ENGINE ──────────────────────────────────
def update_market_data(symbol="ETH-USD"):
    try:
        df = yf.download(symbol, period="60d", interval="4h", progress=False)
        if df.empty: return False
            
        df.reset_index(inplace=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        if 'Datetime' in df.columns:
            df.rename(columns={'Datetime': 'Date'}, inplace=True)
            
        sma, upper, lower = bollinger_bands(df['Close'], n=20, std=2.0)
        ml, _, _ = macd(df['Close'], 12, 26, 9)
        r = rsi(df['Close'], 14)
        e_trend = ema(df['Close'], 200)
        current_atr = atr(df, 14)
        
        last_closed = df.iloc[-2]
        current_price = df.iloc[-1]['Close']
        
        bull_rc = (last_closed['Close'] > e_trend.iloc[-2]) and (ml.iloc[-2] > 0)
        bear_rc = (last_closed['Close'] < e_trend.iloc[-2]) and (ml.iloc[-2] < 0)
        
        trend_status = "UP" if bull_rc else "DOWN" if bear_rc else "NEUTRAL"
        
        # UI Store Update
        SYSTEM_STATE["assets"][symbol] = {
            "price": float(current_price),
            "rsi": float(r.iloc[-1]),
            "ema200": float(e_trend.iloc[-1]),
            "macd": float(ml.iloc[-1]),
            "lower_bb": float(lower.iloc[-1]),
            "upper_bb": float(upper.iloc[-1]),
            "atr": float(current_atr.iloc[-1]),
            "trend": trend_status,
            "distance_to_long": float((current_price / lower.iloc[-1]) - 1) * 100,
            "distance_to_short": float((upper.iloc[-1] / current_price) - 1) * 100,
            "signal": "NONE"
        }
        
        # Signal Check exactly at candle close event
        long_signal = bull_rc and ((last_closed['Close'] <= lower.iloc[-2]) or (r.iloc[-2] < 25))
        short_signal = bear_rc and ((last_closed['Close'] >= upper.iloc[-2]) or (r.iloc[-2] > 65))
        
        # Calculate Next UTC 4H Close
        now = datetime.utcnow()
        hours_to_add = 4 - (now.hour % 4)
        next_close = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=hours_to_add)
        SYSTEM_STATE["next_candle"] = next_close.isoformat() + "Z"
        SYSTEM_STATE["last_update"] = now.isoformat() + "Z"
        SYSTEM_STATE["status"] = "Active & Monitoring"
        
        # Emit Signal logic handles
        # (Assuming the main loop triggers this precisely at minutes 0-2 of the hour)
        if now.hour % 4 == 0 and now.minute < 5 and not hasattr(update_market_data, "last_signal_hour"):
            update_market_data.last_signal_hour = now.hour
            
        if hasattr(update_market_data, "last_signal_hour") and update_market_data.last_signal_hour != now.hour:
            # We are crossing into a new 4H window
            if long_signal:
                sl = current_price - (2.0 * last_closed['atr'])
                tp = current_price + (5.0 * last_closed['atr'])
                msg = f"🟢 *LONG ENTRY* 🟢\nAsset: `{symbol}`\nPrice: ${current_price:.2f}\nSL: ${sl:.2f}\nTP: ${tp:.2f}"
                send_telegram(msg)
                SYSTEM_STATE["assets"][symbol]["signal"] = "LONG"
            elif short_signal:
                sl = current_price + (2.0 * last_closed['atr'])
                tp = current_price - (5.0 * last_closed['atr'])
                msg = f"🔴 *SHORT ENTRY* 🔴\nAsset: `{symbol}`\nPrice: ${current_price:.2f}\nSL: ${sl:.2f}\nTP: ${tp:.2f}"
                send_telegram(msg)
                SYSTEM_STATE["assets"][symbol]["signal"] = "SHORT"
                
            update_market_data.last_signal_hour = now.hour
            
        return True
    except Exception as e:
        print(f"Error updating market data: {e}")
        traceback.print_exc()
        SYSTEM_STATE["status"] = f"Error: {str(e)}"
        return False

# ─── BACKGROUND LOOP ─────────────────────────────────
def background_worker():
    print("Starting background Live Tracker...")
    while True:
        try:
            update_market_data()
        except:
            pass
        # Refresh Data every 60 seconds
        time.sleep(60)

# Initialize background thread without blocking FastAPI
Thread(target=background_worker, daemon=True).start()

# ─── API ROUTES ──────────────────────────────────────
@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")

@app.get("/api/state")
def get_state():
    return SYSTEM_STATE

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
