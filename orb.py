import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go

# --- Page Config ---
st.set_page_config(page_title="ORB Strategy Dashboard", layout="wide")

st.title("📈 Opening Range Breakout (ORB) Strategy")

# --- Sidebar (Inputs) ---
st.sidebar.header("⚙️ Parameters")

data_source = st.sidebar.radio("Select Data Source:", ["Online (Yahoo Finance)", "Offline (CSV)"])

df = None

if data_source == "Online (Yahoo Finance)":
    ticker = st.sidebar.text_input("Ticker Symbol", "^NSEI")
    start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=5))
    end_date = st.sidebar.date_input("End Date", datetime.now())
    interval = st.sidebar.selectbox("Interval", ["5m", "15m", "30m", "1h"])
    
    if st.sidebar.button("Fetch Online Data"):
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval)

        # Flatten if MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]

        # Clean column names
        df.columns = df.columns.str.replace(r'_.*', '', regex=True)
        df.columns = [col.capitalize() for col in df.columns]

        # Reset index & convert timezone
        df.reset_index(inplace=True)
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df['Datetime'] = df['Datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')

        # Keep only market hours (09:15 – 15:30 IST)
        df = df.set_index("Datetime").between_time("09:15", "15:30").reset_index()

        st.write("📊 Data (IST, NSE Hours Only)", df.head())

elif data_source == "Offline (CSV)":
    file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)

        df.columns = df.columns.str.replace(r'_.*', '', regex=True)
        df.columns = [col.capitalize() for col in df.columns]

        if "Datetime" in df.columns:
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            df.set_index("Datetime", inplace=True)

# --- Main Panel (Outputs) ---
if df is not None and not df.empty:
    tab1, tab2, tab3 = st.tabs(["🔎 ORB Analysis", "📒 Trade Log", "📊 Chart"])

    with tab1:
        st.subheader("ORB Strategy Analysis")

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

            if position is None:
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

        # Final exit
        if position is not None:
            exit_price = df["Close"].iloc[-1]
            pnl = exit_price - entry_price if position == "LONG" else entry_price - exit_price
            trades[-1][3] = exit_price
            trades[-1][4] = pnl

        # Convert trades to DataFrame
        trade_log = pd.DataFrame(trades, columns=["Datetime", "Signal", "EntryPrice", "ExitPrice", "PnL"])

        latest_price = float(df["Close"].iloc[-1])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Opening Range High", f"{OR_high:.2f}")
        col2.metric("Opening Range Low", f"{OR_low:.2f}")
        col3.metric("Latest Price", f"{latest_price:.2f}")
        col4.metric("Total P&L", f"{trade_log['PnL'].sum():.2f}")

        st.write("Sample Data", df.head())

    with tab2:
        st.subheader("📒 Trade Log")
        st.dataframe(trade_log)

    with tab3:
        st.subheader("📊 ORB Candlestick Chart")

        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Candlesticks"
        )])

        # OR lines
        fig.add_hline(y=OR_high, line_dash="dot", line_color="green", annotation_text="OR High")
        fig.add_hline(y=OR_low, line_dash="dot", line_color="red", annotation_text="OR Low")

        # Markers
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
