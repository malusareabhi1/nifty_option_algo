import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz

st.set_page_config(page_title="3PM Candle High/Low", layout="wide")

st.title("📊 15-min Chart with 3PM Candle High/Low")

# Sidebar inputs
symbol_input = st.text_input("Enter NSE Stock/Index Symbol", value="NIFTY")
if not symbol_input.endswith(".NS"):
    symbol = symbol_input + ".NS"
else:
    symbol = symbol_input

days = 3  # last 3 days
interval = "15m"

# Fetch last 3 days data
end_date = datetime.utcnow()
start_date = end_date - timedelta(days=days)

df = yf.download(
    symbol,
    start=start_date,
    end=end_date,
    interval=interval,
    prepost=False
)

if df.empty:
    st.error("No data fetched. Check the symbol or try again later.")
else:
    # Reset index and convert UTC to IST
    df = df.reset_index()
    df['Datetime'] = df['Datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')

    # Find 3 PM candle for each day
    markers = []
    for date in df['Datetime'].dt.date.unique():
        day_data = df[df['Datetime'].dt.date == date]
        candle_3pm = day_data[day_data['Datetime'].dt.strftime("%H:%M") == "15:00"]
        if not candle_3pm.empty:
            high_val = candle_3pm['High'].values[0]
            low_val = candle_3pm['Low'].values[0]
            markers.append((candle_3pm['Datetime'].iloc[0], high_val, low_val))

    # Plot candlestick chart
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df['Datetime'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="15-min candles"
    ))

    # Add horizontal lines for 3 PM candle high/low
    for ts, high_val, low_val in markers:
        fig.add_hline(y=high_val, line_dash="dash", line_color="green",
                      annotation_text=f"3PM High {high_val}", annotation_position="top right")
        fig.add_hline(y=low_val, line_dash="dash", line_color="red",
                      annotation_text=f"3PM Low {low_val}", annotation_position="bottom right")

    fig.update_layout(
        title=f"{symbol_input} - Last {days} Days (15-min)",
        xaxis_rangeslider_visible=False,
        height=700
    )

    st.plotly_chart(fig, use_container_width=True)
