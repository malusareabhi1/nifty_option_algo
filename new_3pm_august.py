import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="NIFTY Chart with 3PM Marker", layout="wide")
st.title("📈 NIFTY Last Day & Today with 3:00 PM Candle Marker")

# Fetch last 5 days intraday data (to ensure we have last 2 trading days)
symbol = "^NSEI"
df = yf.download(symbol, period="5d", interval="5m")
df = df.reset_index()

# Convert timezone to IST
#df['Datetime'] = pd.to_datetime(df['Datetime']).dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
# Convert to datetime first (if not already)
df['Datetime'] = pd.to_datetime(df['Datetime'])

# Check if tz-aware or not, then apply appropriate conversion
if df['Datetime'].dt.tz is None:
    # tz-naive -> localize then convert
    df['Datetime'] = df['Datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
else:
    # tz-aware -> just convert to target tz
    df['Datetime'] = df['Datetime'].dt.tz_convert('Asia/Kolkata')


# Get unique trading dates
unique_dates = sorted(df['Datetime'].dt.date.unique())

if len(unique_dates) < 2:
    st.error("Not enough data for last two trading days.")
    st.stop()

last_day = unique_dates[-2]
today = unique_dates[-1]

# 3:00 PM timestamp on last trading day in Asia/Kolkata timezone
three_pm_time = pd.Timestamp(f"{last_day} 15:00:00").tz_localize('Asia/Kolkata')

three_pm_candle = df[df['Datetime'] == three_pm_time]

if not three_pm_candle.empty:
    # Explicitly get scalar values
    three_pm_open = three_pm_candle['Open'].values[0]
    three_pm_close = three_pm_candle['Close'].values[0]
else:
    three_pm_open = None
    three_pm_close = None
    st.warning("⚠ No 3:00 PM candle found for last trading day.")

# Filter candles for last day and today
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

# Add horizontal lines for 3PM open and close if available
if (three_pm_open is not None) and (three_pm_close is not None):
    fig.add_hline(y=three_pm_open, line=dict(color="blue", width=2, dash="dot"), annotation_text="3PM Open", annotation_position="top left")
    fig.add_hline(y=three_pm_close, line=dict(color="orange", width=2, dash="dot"), annotation_text="3PM Close", annotation_position="bottom left")

fig.update_layout(title=f"NIFTY – {last_day} & {today}",
                  xaxis_rangeslider_visible=False,
                  template="plotly_dark")

st.plotly_chart(fig, use_container_width=True)
