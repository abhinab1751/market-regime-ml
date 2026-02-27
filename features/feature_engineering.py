import pandas as pd
import numpy as np


def compute_sma(df, windows=[20, 50, 200]):
    for w in windows:
        df[f'SMA_{w}'] = df['Close'].rolling(w).mean()
    return df


def compute_ema(df, span=20):
    df[f'EMA_{span}'] = df['Close'].ewm(span=span, adjust=False).mean()
    return df


def compute_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def compute_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df


def compute_roc(df, period=10):
    df[f'ROC_{period}'] = df['Close'].pct_change(period) * 100
    return df


def compute_bollinger_bands(df, window=20, num_std=2):
    rolling_mean = df['Close'].rolling(window).mean()
    rolling_std = df['Close'].rolling(window).std()
    df['BB_Upper'] = rolling_mean + (num_std * rolling_std)
    df['BB_Lower'] = rolling_mean - (num_std * rolling_std)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Width'] + 1e-10)
    return df


def compute_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(period).mean()
    return df


def compute_volatility(df, window=20):
    df['Rolling_Std'] = df['Close'].pct_change().rolling(window).std()
    return df


def compute_statistical_features(df, window=20):
    returns = df['Close'].pct_change()
    df['Rolling_Return'] = returns.rolling(window).mean()
    df['Z_Score'] = (returns - returns.rolling(window).mean()) / (returns.rolling(window).std() + 1e-10)
    
    rolling_max = df['Close'].rolling(window).max()
    df['Drawdown'] = (df['Close'] - rolling_max) / (rolling_max + 1e-10)
    
    df['Volume_Pct_Change'] = df['Volume'].pct_change() * 100
    
    return df


def compute_ma_crossover(df):
    df['SMA_20_50_Cross'] = (df['SMA_20'] - df['SMA_50'])
    df['SMA_50_200_Cross'] = (df['SMA_50'] - df['SMA_200'])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = compute_sma(df)
    df = compute_ema(df)
    df = compute_rsi(df)
    df = compute_macd(df)
    df = compute_roc(df)
    df = compute_bollinger_bands(df)
    df = compute_atr(df)
    df = compute_volatility(df)
    df = compute_statistical_features(df)
    df = compute_ma_crossover(df)
    df.dropna(inplace=True)
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/raw_data.csv", index_col=0, parse_dates=True)
    df_features = engineer_features(df)
    print(df_features.shape)
    print(df_features.head())
    df_features.to_csv("data/features_data.csv")
    print("Saved feature data to data/features_data.csv")