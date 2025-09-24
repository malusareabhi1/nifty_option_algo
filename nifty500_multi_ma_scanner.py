# Streamlit App: NIFTY500 Multi-MA Strategy Scanner
# Single-file Streamlit application to scan multiple tickers (NIFTY500) for Moving Average strategies.
# Requirements: streamlit, yfinance, pandas, numpy, plotly

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import io
from datetime import date, timedelta
import plotly.graph_objects as go

st.set_page_config(page_title="NIFTY500 Multi-MA Scanner", layout="wide")

# ---------------------------- Helper functions ----------------------------
@st.cache_data
def load_default_ticker_list():
    # A small example list. Replace this with a full NIFTY500 ticker list or upload.
    return [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
        "ICICIBANK.NS", "BHARTIARTL.NS", "LT.NS", "KOTAKBANK.NS", "ITC.NS"
    ]

@st.cache_data
def fetch_ohlcv(tickers, start_date, end_date):
    # Use yfinance to download multiple tickers's daily OHLCV
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, group_by='ticker', threads=True)
    results = {}
    for t in tickers:
        try:
            if len(tickers) == 1:
                df = data.copy()
            else:
                df = data[t].dropna()
            df = df[['Open','High','Low','Close','Volume']].dropna()
            results[t] = df
        except Exception:
            continue
    return results

# Moving averages
def add_moving_averages(df, ma_windows):
    for w in ma_windows:
        df[f'SMA_{w}'] = df['Close'].rolling(window=w).mean()
        df[f'EMA_{w}'] = df['Close'].ewm(span=w, adjust=False).mean()
    return df

# Simple SMA crossover signal generator
def generate_signals(df, fast, slow, ma_type='SMA'):
    fast_col = f'{ma_type}_{fast}'
    slow_col = f'{ma_type}_{slow}'
    df = df.copy().dropna(subset=[fast_col, slow_col])
    df['signal'] = 0
    df.loc[(df[fast_col].shift(1) < df[slow_col].shift(1)) & (df[fast_col] > df[slow_col]), 'signal'] = 1
    df.loc[(df[fast_col].shift(1) > df[slow_col].shift(1)) & (df[fast_col] < df[slow_col]), 'signal'] = -1
    return df

# Backtest a simple position strategy: enter on signal day close, exit on opposite signal
def backtest_signals(df, entry_signal=1):
    df = df.copy()
    df['position'] = 0
    pos = 0
    entry_price = 0.0
    trades = []
    for idx, row in df.iterrows():
        sig = row['signal']
        if pos == 0 and sig == entry_signal:
            pos = 1 if entry_signal == 1 else -1
            entry_price = row['Close']
            entry_date = idx
        elif pos != 0 and sig == -entry_signal:
            exit_price = row['Close']
            exit_date = idx
            pnl = (exit_price - entry_price) * pos
            ret = pnl / entry_price
            trades.append({'entry_date': entry_date, 'exit_date': exit_date, 'entry': entry_price, 'exit': exit_price, 'pnl': pnl, 'return': ret})
            pos = 0
    trades_df = pd.DataFrame(trades)
    return trades_df

# ---------------------------- UI ----------------------------
st.title("NIFTY500 Multi-MA Strategy Scanner")

col1, col2 = st.columns([1,2])
with col1:
    st.header("Inputs")
    ticker_source = st.radio("Tickers source", ["Default sample list", "Upload CSV (one ticker per line)"])
    if ticker_source == "Default sample list":
        tickers = load_default_ticker_list()
        st.write(f"Loaded sample {len(tickers)} tickers. Replace with full NIFTY500 for production.")
        st.text_area("Tickers (editable)", value='\n'.join(tickers), height=120, key='tickers_area')
        tickers = [t.strip().upper() for t in st.session_state['tickers_area'].splitlines() if t.strip()]
    else:
        uploaded = st.file_uploader("Upload CSV", type=['csv','txt'])
        if uploaded is not None:
            try:
                content = pd.read_csv(uploaded, header=None)
                tickers = content.iloc[:,0].astype(str).str.strip().str.upper().tolist()
                st.success(f"Loaded {len(tickers)} tickers from file")
            except Exception as e:
                st.error("Cannot read file. Make sure it has one ticker per line.")
                tickers = []
        else:
            tickers = []

    today = date.today()
    default_start = today - timedelta(days=365)
    start_date = st.date_input("Start date", value=default_start)
    end_date = st.date_input("End date", value=today)

    st.markdown("---")
    st.subheader("Strategy settings")
    ma_type = st.selectbox("MA type for crossover", ['SMA','EMA'])
    fast_ma = st.number_input("Fast MA window", min_value=2, max_value=200, value=20)
    slow_ma = st.number_input("Slow MA window", min_value=2, max_value=400, value=50)
    if slow_ma <= fast_ma:
        st.warning("Slow MA should be larger than Fast MA. Adjusting automatically.")
        slow_ma = fast_ma + 10

    run_button = st.button("Run scan")

with col2:
    st.header("Summary / Output")
    output_container = st.container()

# ---------------------------- Run ----------------------------
if run_button:
    if not tickers:
        st.error("No tickers provided. Upload or provide tickers to scan.")
    else:
        with st.spinner("Downloading data and scanning..."):
            data_dict = fetch_ohlcv(tickers, start_date.isoformat(), end_date.isoformat())

        results = []
        equity_fig = go.Figure()
        for t, df in data_dict.items():
            df = add_moving_averages(df, [fast_ma, slow_ma])
            df_sign = generate_signals(df, fast_ma, slow_ma, ma_type=ma_type)
            if df_sign['signal'].abs().sum() == 0:
                # no signals
                results.append({'Ticker': t, 'Signals': 0, 'Last Signal': 'None', 'Last Price': df['Close'].iloc[-1]})
                continue
            trades = backtest_signals(df_sign, entry_signal=1)
            total_return = trades['return'].sum() if not trades.empty else 0.0
            wins = trades[trades['pnl']>0].shape[0] if not trades.empty else 0
            losses = trades[trades['pnl']<=0].shape[0] if not trades.empty else 0
            last_signal = df_sign['signal'].iloc[-1]
            last_signal_str = 'BUY' if last_signal==1 else ('SELL' if last_signal==-1 else 'NONE')
            results.append({'Ticker': t, 'Signals': int(df_sign['signal'].abs().sum()), 'Last Signal': last_signal_str, 'Last Price': df['Close'].iloc[-1], 'Trades': trades.shape[0], 'Total_Return': total_return, 'Wins': wins, 'Losses': losses})

        results_df = pd.DataFrame(results).sort_values('Signals', ascending=False).reset_index(drop=True)

        with output_container:
            st.subheader("Scan results")
            st.dataframe(results_df)

            # Allow user to select a ticker to view chart/trades
            sel = st.selectbox("Select ticker for detailed view", options=results_df['Ticker'].tolist())
            if sel:
                sel_df = data_dict[sel].copy()
                sel_df = add_moving_averages(sel_df, [fast_ma, slow_ma])
                sel_df = generate_signals(sel_df, fast_ma, slow_ma, ma_type=ma_type)

                # Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=sel_df.index, y=sel_df['Close'], name='Close', mode='lines'))
                fig.add_trace(go.Scatter(x=sel_df.index, y=sel_df[f'{ma_type}_{fast_ma}'], name=f'{ma_type} {fast_ma}', mode='lines'))
                fig.add_trace(go.Scatter(x=sel_df.index, y=sel_df[f'{ma_type}_{slow_ma}'], name=f'{ma_type} {slow_ma}', mode='lines'))
                buys = sel_df[sel_df['signal']==1]
                sells = sel_df[sel_df['signal']==-1]
                fig.add_trace(go.Scatter(x=buys.index, y=buys['Close'], mode='markers', marker_symbol='triangle-up', marker_size=10, name='BUY'))
                fig.add_trace(go.Scatter(x=sells.index, y=sells['Close'], mode='markers', marker_symbol='triangle-down', marker_size=10, name='SELL'))
                fig.update_layout(height=500, showlegend=True, title=f'{sel} Price and MA Signals')
                st.plotly_chart(fig, use_container_width=True)

                trades = backtest_signals(sel_df, entry_signal=1)
                if not trades.empty:
                    st.subheader("Trades (simple entry/exit) for selected ticker")
                    st.dataframe(trades)
                    st.download_button("Download trades CSV", data=trades.to_csv(index=False).encode('utf-8'), file_name=f"{sel}_trades.csv")
                else:
                    st.info("No completed trades found in the backtest for this ticker with current settings.")

        st.success("Scan complete.")

# ---------------------------- Footer ----------------------------
st.markdown("---")
st.markdown(
    "**Notes:** This is a basic scanner and backtester for educational/useful screening purposes.\n"
    "- Replace sample tickers with a full NIFTY500 ticker list for a complete scan.\n"
    "- Strategy/backtest rules are intentionally simple (enter on crossover, exit on opposite crossover).\n"
    "- Consider adding transaction costs, slippage, position sizing, risk management, and more realistic execution logic before trading live.")
