import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import ta  # For EMA and RSI

# --- Page Config ---
st.set_page_config(page_title="EMA + RSI Intraday Strategy", layout="wide")
st.title("📈 EMA + RSI Intraday Momentum Strategy")

# --- Sidebar Inputs ---
st.sidebar.header("⚙️ Parameters")
data_source = st.sidebar.radio("Data Source:", ["Online (Yahoo Finance)", "Offline (CSV)"])

df = None

# --- Fetch Data ---
if data_source == "Online (Yahoo Finance)":
    ticker = st.sidebar.text_input("Ticker Symbol", "^NSEI")
    start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=5))
    end_date = st.sidebar.date_input("End Date", datetime.now())
    interval = st.sidebar.selectbox("Interval", ["5m", "15m", "30m", "1h"])
    
    if st.sidebar.button("Fetch Online Data"):
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        # Flatten MultiIndex if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]
        # Clean column names
        df.columns = df.columns.str.replace(r'_.*', '', regex=True)
        df.columns = [col.capitalize() for col in df.columns]
        # Convert index to IST and reset as column
        # Ensure datetime column exists
        if 'Datetime' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True).dt.tz_convert('Asia/Kolkata')
        elif 'Date' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Date'], utc=True).dt.tz_convert('Asia/Kolkata')
        else:
            # If datetime is index (Yahoo Finance)
            df.index = pd.to_datetime(df.index, utc=True).tz_convert('Asia/Kolkata')
            df.reset_index(inplace=True)
            df.rename(columns={'index':'Datetime'}, inplace=True)
            st.write("📊 Data Sample", df.head())
            # Filter only NSE market hours
            # Show column names before processing
            st.write("🔹 Column names before datetime handling:", df.columns.tolist())
            df = df.set_index('Datetime').between_time("09:15", "15:30").reset_index()


elif data_source == "Offline (CSV)":
    file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        df.columns = df.columns.str.replace(r'_.*', '', regex=True)
        df.columns = [col.capitalize() for col in df.columns]
        if "Datetime" in df.columns:
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            df = df.set_index("Datetime").between_time("09:15", "15:30").reset_index()
        st.write("📊 Data Sample", df.head())

# --- Strategy Logic ---
if df is not None and not df.empty:
    tab1, tab2, tab3 = st.tabs(["🔎 Strategy Analysis", "📒 Trade Log", "📊 Candlestick Chart"])

    # Calculate EMA & RSI
    df['EMA20'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
    df['EMA50'] = ta.trend.EMAIndicator(df['Close'], window=50).ema_indicator()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()

    # --- Trade Logic ---
    trades = []
    position = None
    entry_price = None

    for i, row in df.iterrows():
        timestamp = row['Datetime']
        price = row['Close']
        ema50 = row['EMA50']
        rsi = row['RSI']

        if position is None:
            if price > ema50 and rsi > 30:
                position = "LONG"
                entry_price = price
                trades.append([timestamp, "BUY", entry_price, None, None])
            elif price < ema50 and rsi < 70:
                position = "SHORT"
                entry_price = price
                trades.append([timestamp, "SELL", entry_price, None, None])
        elif position == "LONG" and (price < ema50 or rsi > 70):
            exit_price = price
            pnl = exit_price - entry_price
            trades[-1][3] = exit_price
            trades[-1][4] = pnl
            position, entry_price = None, None
        elif position == "SHORT" and (price > ema50 or rsi < 30):
            exit_price = price
            pnl = entry_price - exit_price
            trades[-1][3] = exit_price
            trades[-1][4] = pnl
            position, entry_price = None, None

    # Final exit at last candle
    if position is not None:
        exit_price = df['Close'].iloc[-1]
        pnl = exit_price - entry_price if position=="LONG" else entry_price - exit_price
        trades[-1][3] = exit_price
        trades[-1][4] = pnl

    trade_log = pd.DataFrame(trades, columns=["Datetime","Signal","EntryPrice","ExitPrice","PnL"])

    # --- Tabs Display ---
    with tab1:
        st.subheader("Strategy Metrics")
        col1, col2 = st.columns(2)
        col1.metric("Latest Close", f"{df['Close'].iloc[-1]:.2f}")
        col2.metric("Total P&L", f"{trade_log['PnL'].sum():.2f}")
        st.write(df.tail())

    with tab2:
        st.subheader("Trade Log")
        st.dataframe(trade_log)

    with tab3:
        st.subheader("Candlestick Chart with EMA & Signals")
        fig = go.Figure(data=[go.Candlestick(
            x=df['Datetime'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Candles"
        )])
        # Plot EMAs
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['EMA20'], mode='lines', name='EMA20', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['EMA50'], mode='lines', name='EMA50', line=dict(color='orange')))

        # Add Buy/Sell markers
        for _, trade in trade_log.iterrows():
            color = "green" if trade["Signal"]=="BUY" else "red"
            symbol = "triangle-up" if trade["Signal"]=="BUY" else "triangle-down"
            fig.add_trace(go.Scatter(
                x=[trade["Datetime"]],
                y=[trade["EntryPrice"]],
                mode="markers+text",
                marker=dict(color=color, size=12, symbol=symbol),
                text=[trade["Signal"]],
                textposition="top center" if trade["Signal"]=="BUY" else "bottom center",
                name=trade["Signal"]
            ))

        fig.update_layout(template="plotly_white", xaxis_rangeslider_visible=False, height=600)
        st.plotly_chart(fig, use_container_width=True)
