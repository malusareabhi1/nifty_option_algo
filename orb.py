import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="ORB Strategy Backtester", layout="wide")

st.title("📈 Opening Range Breakout (ORB) Strategy")

# ---------------------------
# 1. Data Input Options
# ---------------------------
data_source = st.radio("Select Data Source:", ["📂 Upload CSV", "🌐 Fetch Online (yFinance)"])

df = None

if data_source == "📂 Upload CSV":
    uploaded_file = st.file_uploader("Upload Intraday OHLC CSV (Datetime, Open, High, Low, Close, Volume)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "Datetime" not in df.columns:
            st.error("CSV must have 'Datetime' column in format YYYY-MM-DD HH:MM:SS")
        else:
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            df.set_index("Datetime", inplace=True)

elif data_source == "🌐 Fetch Online (yFinance)":
    ticker = st.text_input("Enter Stock Symbol (e.g., ^NSEI for NIFTY, RELIANCE.NS, INFY.NS, BANKNIFTY.NS)", "^NSEI")
    interval = st.selectbox("Select Timeframe", ["5m", "15m"])
    days = st.slider("Number of past days", 1, 30, 5)

    if st.button("Fetch Data"):
        df = yf.download(ticker, period=f"{days}d", interval=interval)
        df.reset_index(inplace=True)
        df.rename(columns={"Datetime": "Datetime", "Open": "Open", "High": "High", "Low": "Low", "Close": "Close"}, inplace=True)
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df.set_index("Datetime", inplace=True)

# ---------------------------
# 2. ORB Strategy Logic
# ---------------------------
if df is not None and not df.empty:
    st.subheader("Uploaded / Fetched Data Sample")
    st.write(df.head())

    # Group data by day
    unique_days = sorted(list(set(df.index.date)))
    results = []

    for day in unique_days:
        day_data = df[df.index.date == day]

        # Skip if no trading session
        if day_data.empty:
            continue

        # Opening Range (first 15 minutes → 9:15–9:30)
        opening_range = day_data.between_time("09:15", "09:30")
        if opening_range.empty:
            continue

        OR_high = opening_range["High"].max()
        OR_low = opening_range["Low"].min()

        position = None
        entry_price, stop_loss, target = 0, 0, 0
        risk_reward = 2
        trades = []

        for time, row in day_data.between_time("09:30", "15:15").iterrows():
            price = row["Close"]

            if position is None:
                if price > OR_high:
                    position = "LONG"
                    entry_price = price
                    stop_loss = OR_low
                    target = entry_price + (entry_price - stop_loss) * risk_reward
                    trades.append((time, "BUY", price))

                elif price < OR_low:
                    position = "SHORT"
                    entry_price = price
                    stop_loss = OR_high
                    target = entry_price - (stop_loss - entry_price) * risk_reward
                    trades.append((time, "SELL", price))

            else:
                if position == "LONG":
                    if price <= stop_loss:
                        trades.append((time, "EXIT SL", price))
                        position = None
                    elif price >= target:
                        trades.append((time, "EXIT TARGET", price))
                        position = None

                elif position == "SHORT":
                    if price >= stop_loss:
                        trades.append((time, "EXIT SL", price))
                        position = None
                    elif price <= target:
                        trades.append((time, "EXIT TARGET", price))
                        position = None

        # Force exit at EOD
        if position is not None:
            trades.append((day_data.index[-1], "EXIT EOD", day_data["Close"].iloc[-1]))

        trade_df = pd.DataFrame(trades, columns=["Time", "Action", "Price"])

        # Calculate PnL
        pnl = 0
        for i in range(0, len(trade_df), 2):
            if i + 1 < len(trade_df):
                entry, exit_trade = trade_df.iloc[i], trade_df.iloc[i+1]
                if entry["Action"] == "BUY":
                    pnl += exit_trade["Price"] - entry["Price"]
                elif entry["Action"] == "SELL":
                    pnl += entry["Price"] - exit_trade["Price"]

        results.append({"Date": day, "PnL": pnl, "Trades": trade_df})

    # ---------------------------
    # 3. Show Results
    # ---------------------------
    st.subheader("ORB Strategy Results")
    results_df = pd.DataFrame(results).drop(columns=["Trades"])
    st.write(results_df)

    total_pnl = results_df["PnL"].sum()
    st.success(f"✅ Total PnL over {len(results_df)} days: {total_pnl:.2f} points")

    # Plot last day's chart
    if results:
        last_day = results[-1]
        day_data = df[df.index.date == last_day["Date"]]
        trades = last_day["Trades"]

        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(day_data.index, day_data["Close"], label="Close Price", color="blue")
        ax.axhline(day_data.between_time("09:15","09:30")["High"].max(), color="green", linestyle="--", label="OR High")
        ax.axhline(day_data.between_time("09:15","09:30")["Low"].min(), color="red", linestyle="--", label="OR Low")

        for _, row in trades.iterrows():
            if "BUY" in row["Action"]:
                ax.scatter(row["Time"], row["Price"], color="green", marker="^", s=100, label="BUY")
            elif "SELL" in row["Action"]:
                ax.scatter(row["Time"], row["Price"], color="red", marker="v", s=100, label="SELL")
            else:
                ax.scatter(row["Time"], row["Price"], color="black", marker="o", s=60, label="EXIT")

        ax.set_title(f"ORB Strategy Chart - {last_day['Date']}")
        ax.set_ylabel("Price")
        ax.legend(loc="best")
        st.pyplot(fig)
