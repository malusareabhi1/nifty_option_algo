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

timeframe_options = ['1m','5m','15m','30m','60m','1d','1wk','1mo']
default_timeframes = ['15m','60m','1d']
timeframes = st.sidebar.multiselect("Timeframes to analyze (multiple)", options=timeframe_options, default=default_timeframes)

# Convert to datetime for safe comparison
start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=30))
end_date = st.sidebar.date_input("End Date", value=datetime.now())

# FIX: convert date -> datetime at 00:00 for start, 23:59 for end
start_date = datetime.combine(start_date, datetime.min.time())
end_date = datetime.combine(end_date, datetime.max.time())

if st.sidebar.button("Fetch & Analyze"):
    st.info(f"Fetching data for {ticker} from {start_date.date()} to {end_date.date()}...")

    for tf in timeframes:
        try:
            # Auto adjust max range for intraday intervals
            max_days = 59 if tf != "1m" else 6
            adjusted_start = max(start_date, datetime.now() - timedelta(days=max_days))

            df = yf.download(ticker, start=adjusted_start, end=end_date, interval=tf)

            if df.empty:
                st.warning(f"No data found for {tf} (try a smaller date range).")
                continue

            # ✅ FIX: Flatten multi-index columns if they exist
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join(col).strip() for col in df.columns.values]

            df.dropna(inplace=True)
            df.reset_index(inplace=True)

            # Ensure datetime column exists
            if 'Datetime' not in df.columns:
                df.rename(columns={"Date": "Datetime"}, inplace=True)

            df['Datetime'] = pd.to_datetime(df['Datetime'])

            st.subheader(f"Timeframe: {tf}")

            # Plot candlestick chart
            fig = go.Figure(data=[go.Candlestick(
                x=df['Datetime'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close']
            )])

            fig.update_layout(
                title=f"{ticker} - {tf} Chart",
                xaxis_rangeslider_visible=False,
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
             # Display downloaded data table
            st.markdown("**Downloaded Data (OHLCV):**")
            st.dataframe(df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(20))  # show last 20 rows


            # Basic pattern detection
            df['HigherHigh'] = df['High'] > df['High'].shift(1)
            df['LowerLow'] = df['Low'] < df['Low'].shift(1)

            hh_count = df['HigherHigh'].sum()
            ll_count = df['LowerLow'].sum()

            trend = "📈 Bullish" if hh_count > ll_count else "📉 Bearish"
            st.markdown(f"**Detected Trend:** {trend}  | HH: {hh_count}, LL: {ll_count}")

        except Exception as e:
            st.error(f"Error fetching or plotting data for {tf}: {e}")
else:
    st.info("Select parameters and click 'Fetch & Analyze' to start.")
