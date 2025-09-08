import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import ta  # EMA and RSI

# --- Page Config ---
st.set_page_config(page_title="EMA + RSI Intraday Strategy", layout="wide")
st.title("📈 EMA + RSI Intraday Momentum Strategy")

# --- Sidebar Inputs ---
st.sidebar.header("⚙️ Parameters")
data_source = st.sidebar.radio("Data Source:", ["Online (Yahoo Finance)", "Offline (CSV)"])

df = None  # Initialize DataFrame

# --- Helper Function to Preprocess Data ---
def preprocess_dataframe(df):
    # Flatten MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]

    # Remove second level (ticker) if MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]  # Keep only first level
    
    # Reset index to get Datetime as a column
    df = df.reset_index()
    
    # Rename for consistency
    df.rename(columns={
        'Date': 'Datetime',
        'Adj Close': 'Adj Close',
        'Open': 'Open',
        'High': 'High',
        'Low': 'Low',
        'Close': 'Close',
        'Volume': 'Volume'
    }, inplace=True)
    
    # Ensure timezone is India
    #df['Datetime'] = pd.to_datetime(df['Datetime']).dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')

    if df is not None:
            st.write(df.columns.tolist())
    # Clean column names
    df.columns = [col.strip().capitalize() for col in df.columns]

    # Detect datetime column or use index
    datetime_col = None
    for col in df.columns:
        if col.lower() in ['datetime', 'date', 'time', 'timestamp']:
            datetime_col = col
            break

    if datetime_col is None:
        # Use index if datetime
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={'index':'Datetime'})
            datetime_col = 'Datetime'
        else:
            st.error("No datetime information found. This ticker may not support intraday data for the selected range/interval.")
            return None

    # Convert to datetime
    df['Datetime'] = pd.to_datetime(df[datetime_col], errors='coerce', utc=True)
    df['Datetime'] = df['Datetime'].dt.tz_convert('Asia/Kolkata')
    df = df.dropna(subset=['Datetime'])

    # Only filter market hours if intraday
    if df['Datetime'].dt.time.nunique() > 1:
        df = df.set_index('Datetime').between_time("09:15", "15:30").reset_index()
    else:
        st.warning("Data has no intraday timestamps. Market hours filter skipped.")

    # Ensure OHLC columns exist
    ohlc_cols = ['Open','High','Low','Close','Volume']
    for col in ohlc_cols:
        if col not in df.columns:
            # Try common alternatives (like 'Adj Close')
            if col=='Close' and 'Adj Close' in df.columns:
                df['Close'] = df['Adj Close']
            else:
                st.error(f"Column {col} not found in data.")
                return None

    return df

# --- Fetch Online Data ---
if data_source == "Online (Yahoo Finance)":
    ticker = st.sidebar.text_input("Ticker Symbol", "^NSEI")
    start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=5))
    end_date = st.sidebar.date_input("End Date", datetime.now())
    interval = st.sidebar.selectbox("Interval", ["5m", "15m", "30m", "1h"])

    if st.sidebar.button("Fetch Online Data"):
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        st.write("📊 Sample Data", df.head())
        if df is not None:
            st.write(df.columns.tolist())
        try:
            df = preprocess_dataframe(df)
        except Exception as e:
            st.error(f"Error processing data: {e}")
        #if df is not None:
            #st.write("📊 Sample Data", df.head())

# --- Upload Offline CSV Data ---
elif data_source == "Offline (CSV)":
    file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        try:
            df = preprocess_dataframe(df)
        except Exception as e:
            st.error(f"Error processing data: {e}")
        if df is not None:
            st.write("📊 Sample Data", df.head())

# --- Strategy Logic ---
if df is not None and not df.empty:
    # Tabs for organization
    tab1, tab2, tab3 = st.tabs(["🔎 Strategy Analysis", "📒 Trade Log", "📊 Candlestick Chart"])

    # --- Calculate EMA & RSI ---
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
        # EMAs
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['EMA20'], mode='lines', name='EMA20', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['EMA50'], mode='lines', name='EMA50', line=dict(color='orange')))

        # Buy/Sell markers
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
