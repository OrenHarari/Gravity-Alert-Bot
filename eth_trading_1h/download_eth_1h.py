import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

def download_crypto_1h(symbol="ETH-USD", days=720):
    print(f"Downloading {symbol} 1H data for the last {days} days from Yahoo Finance...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Download 1H data
    # yfinance limits 1h data to max 730 days
    df = yf.download(symbol, start=start_date, end=end_date, interval="1h", progress=False)
    
    if df.empty:
        print("Failed to download data. The ticker might be incorrect or data unavailable.")
        return
        
    df.reset_index(inplace=True)
    
    # Clean column names (remove multi-index if present)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if c[1] == '' or c[1] == symbol else c[0] for c in df.columns]
        
    # Standardize column name for Date/Datetime
    if 'Datetime' in df.columns:
        df.rename(columns={'Datetime': 'Date'}, inplace=True)
        
    # Save to CSV
    # make sure we get exactly what we want: Date, Open, High, Low, Close, Volume
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_file = os.path.join(out_dir, "ETH_1H_2Y.csv")
    
    df.to_csv(out_file, index=False)
    
    print(f"Saved {len(df)} 1H bars to {out_file}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Price range: ${float(df['Low'].min()):.2f} to ${float(df['High'].max()):.2f}")

if __name__ == "__main__":
    download_crypto_1h()
