import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import pytz
from datetime import datetime, timedelta

st.set_page_config(page_title="NIFTY Last Day & Today Chart", layout="wide")
st.title("📊 NIFTY Chart with Last Day 3PM Candle Marked")

# Fetch last 2 days intraday data (15-min interval)
symbol = "^NSEI"  # NIFTY 50 Index
end = datetime.now()
start = end - timedelta(days=5)  # fetch extra to ensure 2 trading days
df = yf.download(symbol, start=start, end=end, interval="15m")

# Convert to IST
df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
df = df.dropna()

# Get yesterday's date (last trading day excluding today)
today_date = pd.Timestamp.now(tz="Asia/Kolkata").date()
yesterday_data = df[df.index.date < today_date]
yesterday_date = yesterday_data.index.date[-1]  # last trading day

# Find 3:00 PM candle for yesterday
three_pm_time = pd.Timestamp(f"{yesterday_date} 15:00:00", tz="Asia/Kolkata")
three_pm_candle = yesterday_data[yesterday_data.index == three_pm_time]

if three_pm_candle.empty:
    st.error("⚠️ No 3:00 PM candle found for last trading day.")
else:
    open_price = three_pm_candle["Open"].iloc[0]
    close_price = three_pm_candle["Close"].iloc[0]

    # Filter yesterday + today data
    display_df = df[df.index.date >= yesterday_date]

    # Plot chart
    fig = go.Figure()

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=display_df.index,
        open=display_df['Open'],
        high=display_df['High'],
        low=display_df['Low'],
        close=display_df['Close'],
        name="NIFTY"
    ))

    # Mark reference lines from 3PM candle
    fig.add_hline(y=open_price, line_dash="dot", line_color="blue", annotation_text="3PM Open", annotation_position="top left")
    fig.add_hline(y=close_price, line_dash="dot", line_color="red", annotation_text="3PM Close", annotation_position="bottom left")

    fig.update_layout(
        title=f"NIFTY Chart – Yesterday ({yesterday_date}) & Today",
        xaxis_rangeslider_visible=False,
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)
