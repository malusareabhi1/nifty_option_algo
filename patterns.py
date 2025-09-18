import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Chart Pattern Finder", layout="wide")

st.title("📊 Multi-Timeframe Chart Pattern Finder & Trend Predictor")

# Sidebar Inputs
st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input("Enter Stock/Index Symbol (e.g. SBIN.NS, NIFTY50)", value="SBIN.NS")

# FIXED: Ensure default values exist in options
timeframe_options = ['1m','5m','15m','30m','60m','1d','1wk','1mo']
default_timeframes = ['15m','60m','1d']  # changed '1h' -> '60m' to match options
timeframes = st.sidebar.multiselect("Timeframes to analyze (multiple)", options=timeframe_options, default=default_timeframes)

start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=30))
end_date = st.sidebar.date_input("End Date", value=datetime.now())

if st.sidebar.button("Fetch & Analyze"):
    st.info(f"Fetching data for {ticker} from {start_date} to {end_date}...")

    for tf in timeframes:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, interval=tf)
            if df.empty:
                st.warning(f"No data for {tf} timeframe.")
                continue

            df.dropna(inplace=True)

            st.subheader(f"Timeframe: {tf}")

            # Plot candlestick chart
            fig = go.Figure(data=[go.Candlestick(x=df.index,
                                                 open=df['Open'],
                                                 high=df['High'],
                                                 low=df['Low'],
                                                 close=df['Close'])])
            fig.update_layout(title=f"{ticker} - {tf} Chart", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            # Simple pattern detection example: Higher Highs / Lower Lows
            df['HigherHigh'] = df['High'] > df['High'].shift(1)
            df['LowerLow'] = df['Low'] < df['Low'].shift(1)

            hh_count = df['HigherHigh'].sum()
            ll_count = df['LowerLow'].sum()

            trend = "Bullish" if hh_count > ll_count else "Bearish"

            st.markdown(f"**Detected Trend:** `{trend}`  | Higher Highs: {hh_count}, Lower Lows: {ll_count}")

        except Exception as e:
            st.error(f"Error fetching data for {tf}: {e}")
else:
    st.info("Select parameters and click 'Fetch & Analyze' to start.")
