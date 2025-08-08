import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="3PM Candle High/Low", layout="wide")

st.title("📊 15-min Chart with 3PM Candle High/Low")

# --- Sidebar Inputs ---
symbol = st.text_input("Enter NSE Stock/Index Symbol", value="NIFTY")
if not symbol.endswith(".NS"):
    symbol = symbol + ".NS"  # Append NSE suffix for yfinance

days = 3  # Last 3 days data
interval = "15m"

# --- Download Data ---
end_date = datetime.now()
start_date = end_date - timedelta(days=days)
df = yf.download(symbol, start=start_date, end=end_date, interval=interval)

if df.empty:
    st.error("No data fetched. Check the symbol or try again later.")
else:
    df.reset_index(inplace=True)

    # Convert to IST
    df['Datetime'] = df['Datetime'].dt.tz_convert('Asia/Kolkata')

    # Identify 3PM candle for each day
    markers = []
    for date in df['Datetime'].dt.date.unique():
        day_data = df[df['Datetime'].dt.date == date]
        candle_3pm = day_data[day_data['Datetime'].dt.strftime("%H:%M") == "15:00"]
        if not candle_3pm.empty:
            high_val = candle_3pm['High'].values[0]
            low_val = candle_3pm['Low'].values[0]
            markers.append((candle_3pm['Datetime'].values[0], high_val, low_val))

    # --- Plot ---
    fig = go.Figure()

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['Datetime'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="15-min candles"
    ))

    # Mark 3PM candle high & low
    for ts, high_val, low_val in markers:
        ts_dt = pd.to_datetime(ts)
        fig.add_hline(y=high_val, line_dash="dash", line_color="green", 
                      annotation_text=f"3PM High {high_val}", annotation_position="top right")
        fig.add_hline(y=low_val, line_dash="dash", line_color="red", 
                      annotation_text=f"3PM Low {low_val}", annotation_position="bottom right")

    fig.update_layout(
        title=f"{symbol.replace('.NS','')} - Last {days} Days (15-min)",
        xaxis_rangeslider_visible=False,
        height=700
    )

    st.plotly_chart(fig, use_container_width=True)
