import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta

# --- 1. Streamlit Inputs ---
st.set_page_config(layout="wide")
st.title("3PM Base Zone Strategy - Multi-Day Backtest (Any NSE F&O stock)")

symbol = st.text_input("Enter NSE Symbol (F&O stocks only)", value="NIFTY").upper()
start_date = st.date_input("Start date", value=pd.to_datetime("2023-01-01"))
end_date = st.date_input("End date", value=pd.to_datetime("today"))

# --- 2. Download F&O List (to validate) (Optional) ---
# Typically would use NSE's F&O lot size file, but for this example assume any user input is F&O-eligible

# --- 3. Data Fetching ---
@st.cache_data
def load_data(symbol, start_date, end_date):
    df = yf.download(symbol, start=start_date, end=end_date, interval="15m")
    if not df.empty:
        df.reset_index(inplace=True)
        df['Date'] = df['Datetime'].dt.date
        df['Time'] = df['Datetime'].dt.time
        df.rename(columns={'Open':'OpenNSEI', 'High':'HighNSEI', 'Low':'LowNSEI', 'Close':'CloseNSEI'}, inplace=True)
    return df

df = load_data(symbol, start_date, end_date)
if df.empty:
    st.warning("No data found for this symbol and time period.")
    st.stop()

# --- 4. Option Chain Fetch ---
def get_option_chain(symbol):
    if symbol in ("NIFTY", "BANKNIFTY", "FINNIFTY"):
        endpoint = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    else:
        endpoint = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com"
    }
    with requests.Session() as s:
        s.get("https://www.nseindia.com", headers=headers)
        r = s.get(endpoint, headers=headers)
        if r.ok:
            return r.json()
        return None

# --- 5. Utility Functions ---
def plot_base_zone_candles(df, tradingdays, symbol):
    fig = go.Figure()
    for i in range(1, len(tradingdays)):
        day0 = tradingdays[i-1]
        day1 = tradingdays[i]
        df_day1 = df[df['Date'] == day1]
        fig.add_trace(go.Candlestick(
            x=df_day1['Datetime'],
            open=df_day1['OpenNSEI'],
            high=df_day1['HighNSEI'],
            low=df_day1['LowNSEI'],
            close=df_day1['CloseNSEI'],
            name=f"{day1}"
        ))
        # Previous day's 3PM candle
        df_day0_3pm = df[(df['Date'] == day0) & (df['Datetime'].dt.hour == 15) & (df['Datetime'].dt.minute == 0)]
        if not df_day0_3pm.empty:
            open_3pm = df_day0_3pm.iloc[0]['OpenNSEI']
            close_3pm = df_day0_3pm.iloc[0]['CloseNSEI']
            fig.add_shape(type='line', x0=df_day1['Datetime'].min(), x1=df_day1['Datetime'].max(),
                          y0=open_3pm, y1=open_3pm, line=dict(color='blue', dash='dot', width=1))
            fig.add_shape(type='line', x0=df_day1['Datetime'].min(), x1=df_day1['Datetime'].max(),
                          y0=close_3pm, y1=close_3pm, line=dict(color='red', dash='dot', width=1))
    fig.update_layout(title=f"{symbol} 15-min Candles with Previous Day 3PM Open/Close", 
                      xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# --- 6. Base Zone Trading Signal Logic ---
def base_zone_trade_signal(df):
    output = []
    tradingdays = sorted(df['Date'].unique())
    for i in range(1, len(tradingdays)):
        day0 = tradingdays[i-1]
        day1 = tradingdays[i]
        # Previous day 3PM candle
        df3pm = df[(df['Date'] == day0) & (df['Datetime'].dt.hour == 15) & (df['Datetime'].dt.minute == 0)]
        if df3pm.empty:
            continue
        open_3pm = df3pm.iloc[0]['OpenNSEI']
        close_3pm = df3pm.iloc[0]['CloseNSEI']
        # First candle next day (9:30)
        df930 = df[(df['Date'] == day1) & (df['Datetime'].dt.hour == 9) & (df['Datetime'].dt.minute == 30)]
        if df930.empty:
            continue
        o930 = df930.iloc[0]['OpenNSEI']
        c930 = df930.iloc[0]['CloseNSEI']
        l930 = df930.iloc[0]['LowNSEI']
        crossedopen = (l930 < open_3pm) and (c930 > open_3pm)
        crossedclose = (l930 < close_3pm) and (c930 > close_3pm)
        closesaboveboth = (c930 > open_3pm) and (c930 > close_3pm)
        if crossedopen and crossedclose and closesaboveboth:
            output.append({
                'EntryDatetime': df930.iloc[0]['Datetime'],
                'BuyPrice': c930,
                'StopLoss': c930 * 0.9,
                'TakeProfit': c930 * 1.10,
                'EntryMessage': f'On {day1} (close={c930}) crossed above 3PM open ({open_3pm}) and close ({close_3pm}) -- Buy Signal'
            })
    return pd.DataFrame(output)

# --- 7. Show Plot and Results ---
tradingdays = sorted(df['Date'].unique())
if len(tradingdays) < 2:
    st.warning("Not enough trading days for analysis.")
else:
    plot_base_zone_candles(df, tradingdays, symbol)
    signaldf = base_zone_trade_signal(df)
    if not signaldf.empty:
        st.write("### Trade Signal Log")
        st.write(signaldf)
    else:
        st.info("No BUY signals found for the period.")

# --- 8. Option Chain Section (on demand) ---
if st.button("Show Option Chain Data (raw, for info)"):
    oc = get_option_chain(symbol)
    if oc:
        st.json(oc)
    else:
        st.warning("Option chain not available for this symbol or bad gateway. Make sure it's a current F&O stock.")

# --- 9. Further: Add functions to select correct expiry, lot size, option price, and simulate option trades by integrating this option data. ---
# (Omitted for brevity; use symbol-variable everywhere and connect the dots per your original business logic.)
