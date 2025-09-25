import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="Intraday Trading Dashboard")

st.sidebar.title("Dashboard Controls")
symbol = st.sidebar.selectbox("Select Stock/Index", ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS"])
interval = st.sidebar.selectbox("Interval", ["1m", "5m", "15m"])
strategy = st.sidebar.selectbox("Strategy", ["EMA Crossover", "Bollinger Breakout"])
capital = st.sidebar.number_input("Capital per Trade", value=10000)
risk_percent = st.sidebar.slider("Risk % per Trade", 1, 10, 2)
start_button = st.sidebar.button("Start Live Feed")

# Fetch intraday data
@st.cache_data
def fetch_data(symbol, interval):
    end = datetime.now()
    start = end - timedelta(days=7)
    data = yf.download(symbol, start=start, end=end, interval=interval)
    data.reset_index(inplace=True)
    return data

df = fetch_data(symbol, interval)

# Technical indicators
df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()

# Plot candlestick chart
fig = go.Figure(data=[go.Candlestick(
    x=df['Datetime'],
    open=df['Open'], high=df['High'],
    low=df['Low'], close=df['Close'],
    name="Candlestick"
)])
fig.add_trace(go.Scatter(x=df['Datetime'], y=df['EMA20'], line=dict(color='blue', width=1), name="EMA20"))
fig.add_trace(go.Scatter(x=df['Datetime'], y=df['EMA50'], line=dict(color='orange', width=1), name="EMA50"))

st.plotly_chart(fig, use_container_width=True)

# Trade signals
st.subheader("Trade Signals")
if strategy == "EMA Crossover":
    df['Signal'] = 0
    df.loc[df['EMA20'] > df['EMA50'], 'Signal'] = 1  # Buy
    df.loc[df['EMA20'] < df['EMA50'], 'Signal'] = -1 # Sell

signals = df[df['Signal'] != 0][['Datetime', 'Close', 'Signal']]
st.dataframe(signals)

# PnL simulation (paper trading)
st.subheader("Paper Trading PnL")
df['PnL'] = df['Signal'].shift(1) * (df['Close'].pct_change()) * capital
st.line_chart(df['PnL'].cumsum())
