import yfinance as yf
import os

ticker = '^GSPC'
df = yf.download(ticker, period='5y', auto_adjust=True)
df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
os.makedirs('data', exist_ok=True)
df.to_csv('data/raw_data.csv')
print(f'Downloaded {len(df)} rows. Saved to data/raw_data.csv')
print(df.tail())
