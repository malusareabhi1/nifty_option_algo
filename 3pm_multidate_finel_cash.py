import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("Nifty 3PM Trailing SL and Take Profit  Strategy - Multi-Day Backtest")




##################################START To Execute ################################################

data_source = st.radio("Select Data Source", ["Yahoo Finance", "Upload CSV"])
if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        start_date = df['Datetime'].min().date()
        end_date = df['Datetime'].max().date()
        
    else:
        st.stop()
else:
       # Text input for any stock symbol
       selected_stock = st.text_input("Enter Stock Symbol (e.g., RELIANCE.NS)", value="RELIANCE.NS")  
       start_date = st.date_input("Select Start Date", value=datetime.today() - timedelta(days=15))
       end_date = st.date_input("Select End Date", value=datetime.today())

if start_date >= end_date:
    st.warning("End date must be after start date")
    st.stop()

# ✅ Download full data for range (start-1 day to end)
download_start = start_date - timedelta(days=1)  # To include previous day for first day
df = yf.download(selected_stock, start=download_start, end=end_date + timedelta(days=1), interval="15m")
if df.empty:
    st.warning("No data for selected range")
    st.stop()
df.columns = ['_'.join(col).strip() for col in df.columns.values]

df.reset_index(inplace=True)
df.rename(columns={'index': 'Datetime'}, inplace=True)  # Ensure proper name
#st.write(df.columns)
#st.write(df.columns.tolist())

# ✅ Normalize columns
if isinstance(df.columns, pd.MultiIndex):
    df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]

df['Datetime'] = pd.to_datetime(df['Datetime'])
if df['Datetime'].dt.tz is None:
    df['Datetime'] = df['Datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
else:
    df['Datetime'] = df['Datetime'].dt.tz_convert('Asia/Kolkata')

# ✅ Filter only NSE trading hours
df = df[(df['Datetime'].dt.time >= datetime.strptime("09:15", "%H:%M").time()) &
        (df['Datetime'].dt.time <= datetime.strptime("15:30", "%H:%M").time())]

# ✅ Get all unique trading days
unique_days = sorted(df['Datetime'].dt.date.unique())

# ✅ Filter for user-selected range
unique_days = [d for d in unique_days if start_date <= d <= end_date]

if len(unique_days) < 2:
    st.warning("Not enough trading days in the selected range")
    st.stop()

# ✅ Initialize combined trade log
combined_trade_log = []
# trading_days = list of unique trading days in selected range

trading_days = sorted([d for d in df['Datetime'].dt.date.unique() if start_date <= d <= end_date])

fig = plot_nifty_multiday(df, trading_days)
st.plotly_chart(fig, use_container_width=True)



#st.write(signal)
# 2. Fetch option premium data for that strike
#option_df = get_option_data_realtime(signal['spot_price'], signal['expiry'], signal['strike'])  # 15m candles

# 3. Track exit
#trade_result = track_trade_exit(signal, option_df)
#st.write(trade_result)
