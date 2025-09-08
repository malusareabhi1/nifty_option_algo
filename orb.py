import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.title("📈 Opening Range Breakout (ORB) Strategy with Trade Log, P&L & Candlestick Chart")

# --- Data Input ---
data_source = st.radio("Select Data Source:", ["Online (Yahoo Finance)", "Offline (CSV)"])

if data_source == "Online (Yahoo Finance)":
    ticker = st.text_input("Enter Ticker Symbol (e.g., ^NSEI for NIFTY50, ^NSEBANK for BANKNIFTY):", "^NSEI")
    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=5))
    end_date = st.date_input("End Date", datetime.now())
    interval = st.selectbox("Select Interval", ["5m", "15m", "30m", "1h"])

    if st.button("Fetch Online Data"):
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval)

        # 🔹 Fix MultiIndex (flatten columns if needed)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]

        # 🔹 Clean column names
        df.columns = df.columns.str.replace(r'_.*', '', regex=True)  # remove suffix like _^nsei
        df.columns = [col.capitalize() for col in df.columns]        # Open, High, Low, Close, Volume

        df.reset_index(inplace=True)
        st.write("Sample Data", df.head())

elif data_source == "Offline (CSV)":
    file = st.file_uploader("Upload CSV File", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)

        # 🔹 Clean column names
        df.columns = df.columns.str.replace(r'_.*', '', regex=True)
        df.columns = [col.capitalize() for col in df.columns]

        if "Datetime" in df.columns:
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            df.set_index("Datetime", inplace=True)

        st.write("Sample Data", df.head())

# --- ORB Strategy Logic ---
if "df" in locals() and not df.empty:
    st.subheader("🔎 ORB Strategy Analysis")

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Datetime" in df.columns:
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            df.set_index("Datetime", inplace=True)

    # Extract opening range (first 15 minutes)
    opening_range = df.between_time("09:15", "09:30")
    OR_high = float(opening_range["High"].max())
    OR_low = float(opening_range["Low"].min())

    # Trade log storage
    trades = []
    position = None
    entry_price = None

    for i, row in df.iterrows():
        price = row["Close"]

        if position is None:  # No trade yet
            if price > OR_high:
                position = "LONG"
                entry_price = price
                trades.append([i, "BUY", entry_price, None, None])

            elif price < OR_low:
                position = "SHORT"
                entry_price = price
                trades.append([i, "SELL", entry_price, None, None])

        elif position == "LONG" and price < OR_low:
            exit_price = price
            pnl = exit_price - entry_price
            trades[-1][3] = exit_price
            trades[-1][4] = pnl
            position, entry_price = None, None

        elif position == "SHORT" and price > OR_high:
            exit_price = price
            pnl = entry_price - exit_price
            trades[-1][3] = exit_price
            trades[-1][4] = pnl
            position, entry_price = None, None

    # Final exit at last candle
    if position is not None:
        exit_price = df["Close"].iloc[-1]
        pnl = exit_price - entry_price if position == "LONG" else entry_price - exit_price
        trades[-1][3] = exit_price
        trades[-1][4] = pnl

    # Convert trades to DataFrame
    trade_log = pd.DataFrame(trades, columns=["Datetime", "Signal", "EntryPrice", "ExitPrice", "PnL"])

    # Show Metrics
    latest_price = float(df["Close"].iloc[-1])
    st.metric("Opening Range High", f"{OR_high:.2f}")
    st.metric("Opening Range Low", f"{OR_low:.2f}")
    st.metric("Latest Price", f"{latest_price:.2f}")

    # Results
    st.subheader("📒 Trade Log")
    st.dataframe(trade_log)

    st.subheader("💰 Strategy P&L")
    total_pnl = trade_log["PnL"].sum()
    st.metric("Total P&L", f"{total_pnl:.2f}")

    # --- Plotly Candlestick Chart with Buy/Sell markers ---
    st.subheader("📊 ORB Candlestick Chart")

    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Candlesticks"
    )])

    # Add OR High/Low lines
    fig.add_hline(y=OR_high, line_dash="dot", line_color="green", annotation_text="OR High")
    fig.add_hline(y=OR_low, line_dash="dot", line_color="red", annotation_text="OR Low")

    # Add Buy/Sell markers
    for _, trade in trade_log.iterrows():
        if trade["Signal"] == "BUY":
            fig.add_trace(go.Scatter(
                x=[trade["Datetime"]],
                y=[trade["EntryPrice"]],
                mode="markers+text",
                marker=dict(color="green", size=12, symbol="triangle-up"),
                text=["BUY"],
                textposition="top center",
                name="BUY"
            ))
        elif trade["Signal"] == "SELL":
            fig.add_trace(go.Scatter(
                x=[trade["Datetime"]],
                y=[trade["EntryPrice"]],
                mode="markers+text",
                marker=dict(color="red", size=12, symbol="triangle-down"),
                text=["SELL"],
                textposition="bottom center",
                name="SELL"
            ))

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)
