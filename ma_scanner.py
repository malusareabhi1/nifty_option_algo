import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="Nifty 500 Multi-MA Scanner", layout="wide")

st.title("📊 Nifty 500 Multi-MA Strategy Scanner")

st.sidebar.header("Scanner Settings")
ma_short = st.sidebar.number_input("Short MA Period", min_value=5, max_value=50, value=20)
ma_medium = st.sidebar.number_input("Medium MA Period", min_value=20, max_value=100, value=50)
ma_long = st.sidebar.number_input("Long MA Period", min_value=50, max_value=200, value=200)

# ✅ Auto-fetch NIFTY500 tickers
def fetch_nifty500_tickers():
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    df = pd.read_csv(url)
    return df['Symbol'].tolist()

try:
    tickers = fetch_nifty500_tickers()
    st.sidebar.success(f"Fetched {len(tickers)} Nifty 500 tickers ✅")
except Exception as e:
    st.sidebar.error("Could not fetch tickers automatically. Using fallback list.")
    tickers = ["RELIANCE", "TCS", "INFY", "HDFCBANK"]

selected_tickers = st.sidebar.multiselect("Select Stocks", tickers, default=tickers[:5])

results = []

for symbol in selected_tickers:
    try:
        data = yf.download(symbol + ".NS", period="6mo", interval="1d")
        data['SMA_short'] = data['Close'].rolling(ma_short).mean()
        data['SMA_medium'] = data['Close'].rolling(ma_medium).mean()
        data['SMA_long'] = data['Close'].rolling(ma_long).mean()

        latest = data.iloc[-1]

        signal = None
        if latest['SMA_short'] > latest['SMA_medium'] > latest['SMA_long']:
            signal = "Bullish Alignment"
        elif latest['SMA_short'] < latest['SMA_medium'] < latest['SMA_long']:
            signal = "Bearish Alignment"

        results.append({
            "Stock": symbol,
            "Close": round(latest['Close'], 2),
            "SMA_short": round(latest['SMA_short'], 2),
            "SMA_medium": round(latest['SMA_medium'], 2),
            "SMA_long": round(latest['SMA_long'], 2),
            "Signal": signal or "No Clear Trend"
        })

    except Exception as e:
        st.warning(f"⚠️ Could not fetch data for {symbol}")

if results:
    df_results = pd.DataFrame(results)
    st.subheader("📋 Scanned Results")
    st.dataframe(df_results)

    bullish = df_results[df_results['Signal'] == "Bullish Alignment"]
    bearish = df_results[df_results['Signal'] == "Bearish Alignment"]

    st.subheader("📈 Bullish Stocks")
    st.write(bullish[['Stock', 'Close']])

    st.subheader("📉 Bearish Stocks")
    st.write(bearish[['Stock', 'Close']])

    # Option to plot a single stock chart
    selected_stock = st.selectbox("Select Stock to View Chart", df_results['Stock'])
    if selected_stock:
        data = yf.download(selected_stock + ".NS", period="6mo", interval="1d")
        data['SMA_short'] = data['Close'].rolling(ma_short).mean()
        data['SMA_medium'] = data['Close'].rolling(ma_medium).mean()
        data['SMA_long'] = data['Close'].rolling(ma_long).mean()

        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=data.index,
                                     open=data['Open'],
                                     high=data['High'],
                                     low=data['Low'],
                                     close=data['Close'],
                                     name='Candles'))
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_short'], mode='lines', name=f'SMA {ma_short}'))
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_medium'], mode='lines', name=f'SMA {ma_medium}'))
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_long'], mode='lines', name=f'SMA {ma_long}'))

        fig.update_layout(title=f"{selected_stock} Price Chart with Moving Averages", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
