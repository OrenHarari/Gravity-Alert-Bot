"""
Download 2 years of BTC/USD 4H data from Yahoo Finance (same source as ETH).
"""
import yfinance as yf
import pandas as pd
import os

DIR = os.path.dirname(os.path.abspath(__file__))
OUT_FILE = os.path.join(DIR, "BTC_4H_2Y.csv")

print("Downloading BTC-USD 4H data from Yahoo Finance...")
ticker = yf.Ticker("BTC-USD")
df = ticker.history(period="2y", interval="4h")

df = df.reset_index()
# Rename Datetime column if needed
if 'Datetime' in df.columns:
    df = df.rename(columns={'Datetime': 'Date'})
elif 'date' in df.columns:
    df = df.rename(columns={'date': 'Date'})

# Keep only OHLCV
df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
df.to_csv(OUT_FILE, index=False)
print(f"Saved {len(df)} rows to {OUT_FILE}")
print(f"Date range: {df.Date.iloc[0]} to {df.Date.iloc[-1]}")
print(f"Price range: ${df.Close.min():.2f} to ${df.Close.max():.2f}")
