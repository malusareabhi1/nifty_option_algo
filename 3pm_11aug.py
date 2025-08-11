import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta, time

st.set_page_config(page_title="Nifty 3PM Breakout Options Strategy", layout="wide")

st.title("Nifty 3PM Candle Breakout & Gap Options Strategy")

# User Input - Select date range (minimum 2 days)
start_date = st.date_input("Start Date", datetime.today() - timedelta(days=10))
end_date = st.date_input("End Date", datetime.today())

if start_date >= end_date:
    st.error("End Date must be after Start Date")
    st.stop()

# Fetch Nifty 15-min data from yfinance (NIFTY index ticker: ^NSEI)
# yfinance interval '15m' works only for last 60 days approx, so limit accordingly

data = yf.download("^NSEI", start=start_date - timedelta(days=1), end=end_date + timedelta(days=1), interval="15m", progress=False)
if data.empty:
    st.error("No data fetched. Try different date range.")
    st.stop()

data = data.reset_index()
data['Datetime'] = pd.to_datetime(data['Datetime']).dt.tz_localize(None)  # Remove timezone

# Helper function to find 3:00-3:15 PM candle for a trading day
def get_3pm_candle(df, day):
    # yfinance timestamp is candle END time
    # So 3:15 PM candle ends exactly at 3:15 PM
    mask = (df['Datetime'].dt.date == day) & (df['Datetime'].dt.time == time(15, 15))
    candle = df.loc[mask]
    if candle.empty:
        # If exact 3:15 not found, try nearest between 3:10 to 3:20 PM
        mask = (df['Datetime'].dt.date == day) & (df['Datetime'].dt.time >= time(15, 10)) & (df['Datetime'].dt.time <= time(15, 20))
        candle = df.loc[mask]
    if candle.empty:
        return None
    return candle.iloc[0]


# Show 3PM candle lines on Day 0
all_days = sorted(data['Datetime'].dt.date.unique())
if len(all_days) < 2:
    st.error("Need at least two trading days for strategy.")
    st.stop()

day0 = all_days[-2]
day1 = all_days[-1]

candle_3pm = get_3pm_candle(data, day0)
if candle_3pm is None:
    st.error(f"No 3:00-3:15 PM candle found on {day0}")
    st.stop()

open_3pm = candle_3pm['Open']
close_3pm = candle_3pm['Close']

st.markdown(f"### Reference Day 0: {day0} 3:00-3:15 PM candle")
st.write(f"Open: {open_3pm:.2f}, Close: {close_3pm:.2f}")

# Prepare Plotly Candlestick with reference lines
fig = go.Figure(data=[go.Candlestick(
    x=data['Datetime'],
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'],
    name="Nifty 15-min"
)])

# Add 3PM candle open and close lines (horizontal across day1)
fig.add_hline(y=open_3pm, line_dash="dash", line_color="blue", annotation_text="3PM Open (Day 0)", annotation_position="bottom left")
fig.add_hline(y=close_3pm, line_dash="dash", line_color="red", annotation_text="3PM Close (Day 0)", annotation_position="top left")

st.plotly_chart(fig, use_container_width=True)

st.info("This is the 3PM candle of Day 0 with open and close lines marked.\nNext steps: detect gap and first candle of Day 1 for trade signals.")

# You can continue here to add logic for conditions and trading simulation

