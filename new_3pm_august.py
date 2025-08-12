import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="NIFTY Chart with 3PM Marker", layout="wide")
st.title("📈 NIFTY Last Day & Today with 3:00 PM Candle Marker")

# Fetch last 2 days intraday data
symbol = "^NSEI"
df = yf.download(symbol, period="5d", interval="5m")
df = df.reset_index()

# Convert timezone to IST
df['Datetime'] = pd.to_datetime(df['Datetime']).dt.tz_convert('Asia/Kolkata')

# Get last trading day and today
last_day = df['Datetime'].dt.date.unique()[-2]
today = df['Datetime'].dt.date.unique()[-1]

# Last day's 3 PM candle
three_pm_time = pd.Timestamp(f"{last_day} 15:00:00").tz_localize('Asia/Kolkata')
three_pm_candle = df[df['Datetime'] == three_pm_time]

if not three_pm_candle.empty:
    three_pm_open = three_pm_candle['Open'].iloc[0]
    three_pm_close = three_pm_candle['Close'].iloc[0]
else:
    three_pm_open = None
    three_pm_close = None
    st.warning("⚠ No 3:00 PM candle found for last trading day.")

# Filter for last day + today
df_filtered = df[df['Datetime'].dt.date.isin([last_day, today])]

# Plot candlestick chart
fig = go.Figure(data=[go.Candlestick(
    x=df_filtered['Datetime'],
    open=df_filtered['Open'],
    high=df_filtered['High'],
    low=df_filtered['Low'],
    close=df_filtered['Close'],
    name="NIFTY"
)])

# Add 3PM markers if data available
if pd.notna(three_pm_open) and pd.notna(three_pm_close):
    fig.add_hline(y=three_pm_open, line=dict(color="blue", width=2, dash="dot"), name="3PM Open")
    fig.add_hline(y=three_pm_close, line=dict(color="orange", width=2, dash="dot"), name="3PM Close")

fig.update_layout(title=f"NIFTY – {last_day} & {today}",
                  xaxis_rangeslider_visible=False,
                  template="plotly_dark")

st.plotly_chart(fig, use_container_width=True)
