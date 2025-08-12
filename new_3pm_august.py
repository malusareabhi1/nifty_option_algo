import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Streamlit page config
st.set_page_config(page_title="NIFTY Chart with 3PM Reference", layout="wide")

st.title("📈 NIFTY Chart – Last Day 3PM Candle Reference")

# Fetch NIFTY intraday data
symbol = "^NSEI"  # Nifty 50 Index
interval = "15m"
period = "5d"  # fetch enough data for last two trading days

df = yf.download(symbol, interval=interval, period=period)
df = df.dropna()

# Get last two trading days
df["DateOnly"] = df.index.date
trading_days = sorted(df["DateOnly"].unique())

if len(trading_days) < 2:
    st.error("Not enough trading days data found.")
    st.stop()

last_day = trading_days[-2]
today_day = trading_days[-1]

# Get last day's 3:00 PM candle
last_day_df = df[df["DateOnly"] == last_day]
last_day_3pm_candle = last_day_df.between_time("15:00", "15:15")

if last_day_3pm_candle.empty:
    st.error("No 3:00 PM candle found for last trading day.")
    st.stop()

open_3pm = last_day_3pm_candle["Open"].iloc[0]
close_3pm = last_day_3pm_candle["Close"].iloc[0]

# Get today's data
today_df = df[df["DateOnly"] == today_day]

# Create Plotly chart
fig = go.Figure()

# Add today's candles
fig.add_trace(go.Candlestick(
    x=today_df.index,
    open=today_df['Open'],
    high=today_df['High'],
    low=today_df['Low'],
    close=today_df['Close'],
    name="Today's Candles"
))

# Add last day's 3PM Open and Close reference lines
fig.add_hline(y=open_3pm, line_dash="dash", line_color="blue", annotation_text="3PM Open", annotation_position="top left")
fig.add_hline(y=close_3pm, line_dash="dash", line_color="orange", annotation_text="3PM Close", annotation_position="bottom left")

fig.update_layout(
    title=f"NIFTY – Today with Last Day's 3PM Candle Reference ({last_day})",
    yaxis_title="Price",
    xaxis_title="Time",
    template="plotly_white",
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True)

# Display values
st.write(f"**Last Day's 3PM Candle:** Open = {open_3pm}, Close = {close_3pm}")
