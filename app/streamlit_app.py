import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import yfinance as yf
import plotly.graph_objects as go

from features.feature_engineering import engineer_features

st.set_page_config(page_title="Market Regime Classifier", layout="wide", page_icon="📈")

FEATURE_COLS = [
    'SMA_20', 'SMA_50', 'SMA_200', 'EMA_20',
    'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'ROC_10', 'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_Position',
    'ATR', 'Rolling_Std', 'Rolling_Return', 'Z_Score', 'Drawdown',
    'Volume_Pct_Change', 'SMA_20_50_Cross', 'SMA_50_200_Cross'
]

REGIME_COLORS = {
    'Bullish':  '#00C853',
    'Bearish':  '#D50000',
    'Sideways': '#FFD600'
}

REGIME_EMOJI = {
    'Bullish':  '🟢',
    'Bearish':  '🔴',
    'Sideways': '🟡'
}


@st.cache_resource
def load_model():
    with open("models/saved_model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data(ttl=3600)
def fetch_data(ticker: str, period: str = "3y") -> pd.DataFrame:
    local_path = "data/raw_data.csv"
    if os.path.exists(local_path):
        df = pd.read_csv(local_path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index.date)
        return df

    for attempt in range(3):
        try:
            t = yf.Ticker(ticker)
            df = t.history(period=period)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                df.index = pd.to_datetime(df.index.date)
                return df
            time.sleep(5)
        except Exception as e:
            if attempt < 2:
                time.sleep(5)
                continue
            st.error(f"Error fetching data: {e}")
    return pd.DataFrame()


def predict_regime(df: pd.DataFrame, artifacts: dict):
    df_feat = engineer_features(df.copy())
    df_feat = df_feat.replace([float('inf'), float('-inf')], float('nan'))
    df_feat = df_feat.dropna(subset=FEATURE_COLS)

    X      = df_feat[FEATURE_COLS]
    X_sc   = artifacts['scaler'].transform(X)
    preds  = artifacts['model'].predict(X_sc)
    probas = artifacts['model'].predict_proba(X_sc)
    labels = artifacts['label_encoder'].inverse_transform(preds)
    return df_feat, labels, probas, artifacts['label_encoder'].classes_


st.sidebar.title("⚙️ Settings")
ticker = st.sidebar.text_input("Ticker Symbol", value="SPY")
period = st.sidebar.selectbox("Data Period", ["1y", "2y", "3y", "5y"], index=2)
st.sidebar.markdown("---")
st.sidebar.info("Model: XGBoost | Labels: Bullish / Bearish / Sideways")
st.sidebar.markdown("**Ticker Examples:**")
st.sidebar.markdown("- `SPY` — S&P 500 ETF")
st.sidebar.markdown("- `QQQ` — NASDAQ ETF")
st.sidebar.markdown("- `^NSEI` — NIFTY 50")
st.sidebar.markdown("- `AAPL` — Apple Inc")
st.sidebar.markdown("- `TSLA` — Tesla")

st.title("📊 Intelligent Market Regime Classifier")
st.markdown("Classifies market conditions into **Bullish**, **Bearish**, or **Sideways** using ML + technical indicators.")

try:
    artifacts = load_model()
except FileNotFoundError:
    st.error("Model not found. Please run `python -m models.train` first.")
    st.stop()

with st.spinner("Fetching market data..."):
    raw_df = fetch_data(ticker, period)

if raw_df.empty:
    st.error(f"No data found for ticker '{ticker}'. Try SPY, QQQ, AAPL, or TSLA.")
    st.stop()

try:
    df_feat, labels, probas, classes = predict_regime(raw_df, artifacts)
except Exception as e:
    st.error(f"Prediction error: {e}")
    st.stop()

if len(labels) == 0:
    st.error("Not enough data to generate predictions. Try a longer period.")
    st.stop()

current_regime = labels[-1]
current_probs  = probas[-1]

col1, col2, col3 = st.columns(3)
col1.metric(
    "Current Regime",
    f"{REGIME_EMOJI[current_regime]} {current_regime}",
    delta=f"{current_probs.max()*100:.1f}% confidence"
)
col2.metric(
    "Latest Close",
    f"${df_feat['Close'].iloc[-1]:,.2f}",
    delta=f"{df_feat['Close'].pct_change().iloc[-1]*100:.2f}%"
)
col3.metric(
    "RSI",
    f"{df_feat['RSI'].iloc[-1]:.1f}",
    delta="Overbought" if df_feat['RSI'].iloc[-1] > 70 else
          "Oversold"   if df_feat['RSI'].iloc[-1] < 30 else "Neutral"
)

st.markdown("---")


st.subheader("📊 Regime Probabilities (Latest)")
prob_cols = st.columns(3)
for i, cls in enumerate(classes):
    prob_cols[i].metric(cls, f"{current_probs[i]*100:.1f}%")

st.subheader("📈 Price Chart with Regime Overlay")

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df_feat.index,
    open=df_feat['Open'],
    high=df_feat['High'],
    low=df_feat['Low'],
    close=df_feat['Close'],
    name="Price",
    increasing_line_color='#00C853',
    decreasing_line_color='#D50000'
))

label_series = pd.Series(labels, index=df_feat.index)
for regime, color in REGIME_COLORS.items():
    mask  = label_series == regime
    dates = label_series[mask].index
    if len(dates) == 0:
        continue
    groups, start = [], dates[0]
    for i in range(1, len(dates)):
        if (dates[i] - dates[i-1]).days > 5:
            groups.append((start, dates[i-1]))
            start = dates[i]
    groups.append((start, dates[-1]))
    for s, e in groups:
        fig.add_vrect(
            x0=s, x1=e,
            fillcolor=color,
            opacity=0.10,
            layer="below",
            line_width=0
        )

fig.add_trace(go.Scatter(
    x=df_feat.index, y=df_feat['SMA_50'],
    name="SMA 50", line=dict(color='orange', width=1)
))
fig.add_trace(go.Scatter(
    x=df_feat.index, y=df_feat['SMA_200'],
    name="SMA 200", line=dict(color='purple', width=1)
))

fig.update_layout(
    height=500,
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
    legend=dict(orientation="h")
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("🔁 RSI")
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(
    x=df_feat.index, y=df_feat['RSI'],
    name="RSI", line=dict(color='cyan')
))
fig_rsi.add_hline(y=70, line_dash="dash", line_color="red",   annotation_text="Overbought")
fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
fig_rsi.update_layout(height=250, template="plotly_dark")
st.plotly_chart(fig_rsi, use_container_width=True)

st.subheader("📉 MACD")
fig_macd = go.Figure()
fig_macd.add_trace(go.Scatter(
    x=df_feat.index, y=df_feat['MACD'],
    name="MACD", line=dict(color='#2196F3')
))
fig_macd.add_trace(go.Scatter(
    x=df_feat.index, y=df_feat['MACD_Signal'],
    name="Signal", line=dict(color='orange')
))
fig_macd.add_trace(go.Bar(
    x=df_feat.index, y=df_feat['MACD_Hist'],
    name="Histogram", marker_color='gray'
))
fig_macd.update_layout(height=250, template="plotly_dark")
st.plotly_chart(fig_macd, use_container_width=True)

st.subheader("📊 Bollinger Bands")
fig_bb = go.Figure()
fig_bb.add_trace(go.Scatter(
    x=df_feat.index, y=df_feat['Close'],
    name="Close", line=dict(color='white', width=1)
))
fig_bb.add_trace(go.Scatter(
    x=df_feat.index, y=df_feat['BB_Upper'],
    name="Upper Band", line=dict(color='red', width=1, dash='dash')
))
fig_bb.add_trace(go.Scatter(
    x=df_feat.index, y=df_feat['BB_Lower'],
    name="Lower Band", line=dict(color='green', width=1, dash='dash'),
    fill='tonexty', fillcolor='rgba(0,100,80,0.05)'
))
fig_bb.update_layout(height=300, template="plotly_dark")
st.plotly_chart(fig_bb, use_container_width=True)

st.subheader("📋 Recent Regime Predictions (Last 30 Days)")

classes_list = list(classes)
bullish_idx  = classes_list.index('Bullish')
bearish_idx  = classes_list.index('Bearish')
sideways_idx = classes_list.index('Sideways')

recent = pd.DataFrame({
    'Date':       df_feat.index[-30:],
    'Close':      df_feat['Close'].values[-30:].round(2),
    'Regime':     labels[-30:],
    'Bullish %':  (probas[-30:, bullish_idx]  * 100).round(1),
    'Bearish %':  (probas[-30:, bearish_idx]  * 100).round(1),
    'Sideways %': (probas[-30:, sideways_idx] * 100).round(1),
}).set_index('Date').sort_index(ascending=False)

st.dataframe(recent, use_container_width=True)

st.markdown("---")
st.markdown("Built with XGBoost · scikit-learn · yfinance · Streamlit · Plotly")
