import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="NIFTY Last & Today with 3PM Mark", layout="wide")
st.title("📈 NIFTY 5-Min Chart – Last Day & Today with 3PM Candle")

# Download last 5 days of 5-min data
ticker = "^NSEI"  # NIFTY 50 index
df = yf.download(ticker, period="5d", interval="5m")

# Ensure timezone is Asia/Kolkata
if df.index.tz is None:
    df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
else:
    df.index = df.index.tz_convert("Asia/Kolkata")

# Filter last 2 trading days
df["Date"] = df.index.date
trading_days = sorted(df["Date"].unique())
if len(trading_days) < 2:
    st.error("Not enough trading days found in data.")
    st.stop()

last_day = trading_days[-2]
today_day = trading_days[-1]

df_last_day = df[df["Date"] == last_day]
df_today = df[df["Date"] == today_day]

# Find last day's 3:00 PM candle
three_pm_time = datetime.combine(last_day, datetime.strptime("15:00", "%H:%M").time())
three_pm_candle = df_last_day[df_last_day.index.time == three_pm_time.time()]

if three_pm_candle.empty:
    st.warning("No 3:00 PM candle found for last trading day.")
    three_pm_open = three_pm_close = None
else:
    three_pm_open = three_pm_candle["Open"].iloc[0]
    three_pm_close = three_pm_candle["Close"].iloc[0]

# Combine last day + today
df_plot = pd.concat([df_last_day, df_today])

# Plot chart
fig = go.Figure(data=[go.Candlestick(
    x=df_plot.index,
    open=df_plot["Open"],
    high=df_plot["High"],
    low=df_plot["Low"],
    close=df_plot["Close"],
    name="Candles"
)])

# Mark last day's 3PM candle
if three_pm_open and three_pm_close:
    fig.add_hline(y=three_pm_open, line_dash="dash", line_color="blue",
                  annotation_text="Last Day 3PM Open", annotation_position="top left")
    fig.add_hline(y=three_pm_close, line_dash="dash", line_color="red",
                  annotation_text="Last Day 3PM Close", annotation_position="bottom left")

fig.update_layout(
    title=f"NIFTY – Last Day ({last_day}) & Today ({today_day})",
    xaxis_rangeslider_visible=False,
    height=700
)

st.plotly_chart(fig, use_container_width=True)
