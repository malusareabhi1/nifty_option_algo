import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

st.title("📈 Opening Range Breakout (ORB) Strategy")

# --- Data Input ---
data_source = st.radio("Select Data Source:", ["Online (Yahoo Finance)", "Offline (CSV)"])

if data_source == "Online (Yahoo Finance)":
    ticker = st.text_input("Enter Ticker Symbol (e.g., ^NSEI for NIFTY50, ^NSEBANK for BANKNIFTY):", "^NSEI")
    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=5))
    end_date = st.date_input("End Date", datetime.now())
    interval = st.selectbox("Select Interval", ["5m", "15m", "30m", "1h"])

    if st.button("Fetch Online Data"):
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        # 🔹 Fix MultiIndex (flatten columns)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    
    # 🔹 Standardize column names
    df.columns = [col.capitalize() for col in df.columns]

    # Normalize column names to lowercase
    df.columns = [col.lower() for col in df.columns]
    
    # Make sure we have required OHLC columns
    required_cols = ["open", "high", "low", "close"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"Missing columns in data: {missing}")
    # 🔹 Normalize column names: Open, High, Low, Close, Volume
    df.columns = [col.capitalize() for col in df.columns]


    df.reset_index(inplace=True)
    st.write("Sample Data", df.head())
    

elif data_source == "Offline (CSV)":
    file = st.file_uploader("Upload CSV File", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        # Try to normalize column names
        df.columns = [col.strip().capitalize() for col in df.columns]
        if "Datetime" in df.columns:
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            df.set_index("Datetime", inplace=True)
        st.write("Sample Data", df.head())

# --- ORB Strategy Logic ---
if "df" in locals() and not df.empty:
    st.subheader("🔎 ORB Strategy Analysis")

    # Make sure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Datetime" in df.columns:
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            df.set_index("Datetime", inplace=True)

   # Extract opening range (first 15 minutes)
    opening_range = df.between_time("09:15", "09:30")
    
    OR_high = float(opening_range["High_^nsei"].max())
    OR_low = float(opening_range["Low_^nsei"].min())
    
    # Get latest price (last available close)
    latest_price = float(df["Close_^nsei"].iloc[-1])
    
    # ORB condition
    if latest_price > OR_high:
        signal = "BUY"
    elif latest_price < OR_low:
        signal = "SELL"
    else:
        signal = "NO TRADE"

    # Display results
    st.metric("Opening Range High", f"{OR_high:.2f}")
    st.metric("Opening Range Low", f"{OR_low:.2f}")
    st.metric("Latest Price", f"{latest_price:.2f}")
    st.success(f"📌 ORB Signal: {signal}")

    # Chart
    st.line_chart(df["Close"])
