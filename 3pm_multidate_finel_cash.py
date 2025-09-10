import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("CASH  3PM Trailing SL and Take Profit  Strategy - Multi-Day Backtest")



def plot_nifty_multiday(df, trading_days):
    fig = go.Figure()

    for i in range(1, len(trading_days)):
        day0 = trading_days[i - 1]   # Previous day
        day1 = trading_days[i]       # Current day

        # Filter data for current day
        df_day1 = df[df['Datetime'].dt.date == day1]
        if df_day1.empty:
            continue

        # Plot current day candles
        fig.add_trace(go.Candlestick(
            x=df_day1['Datetime'],
            open=df_day1['Open'],
            high=df_day1['High'],
            low=df_day1['Low'],
            close=df_day1['Close'],
            name=str(day1)
        ))

        # Get 3PM candle of previous day
        candle_3pm = df[
            (df['Datetime'].dt.date == day0) &
            (df['Datetime'].dt.hour == 15) &
            (df['Datetime'].dt.minute == 0)
        ]
        if not candle_3pm.empty:
            open_3pm = candle_3pm.iloc[0]['Open']
            close_3pm = candle_3pm.iloc[0]['Close']

            # Day1 session range
            x_start = df_day1['Datetime'].min()
            day1_3pm = df_day1[
                (df_day1['Datetime'].dt.hour == 15) &
                (df_day1['Datetime'].dt.minute == 0)
            ]
            if not day1_3pm.empty:
                x_end = day1_3pm['Datetime'].iloc[0]
            else:
                x_end = df_day1['Datetime'].max()

            # 🔵 Line for 3PM Open (mark as "High")
            fig.add_shape(
                type="line",
                x0=x_start, x1=x_end,
                y0=open_3pm, y1=open_3pm,
                line=dict(color="blue", width=1, dash="dot"),
            )
            fig.add_annotation(
                x=x_end, y=open_3pm, text="3PM Open", 
                showarrow=False, font=dict(color="blue", size=10)
            )

            # 🔴 Line for 3PM Close (mark as "Low")
            fig.add_shape(
                type="line",
                x0=x_start, x1=x_end,
                y0=close_3pm, y1=close_3pm,
                line=dict(color="red", width=1, dash="dot"),
            )
            fig.add_annotation(
                x=x_end, y=close_3pm, text="3PM Close", 
                showarrow=False, font=dict(color="red", size=10)
            )

    # Layout adjustments
    fig.update_layout(
        title="Multi-Day 3PM Strategy (Prev Day 3PM Open = High, Close = Low)",
        xaxis_rangeslider_visible=False,
        xaxis=dict(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),   # skip weekends
                dict(bounds=[15.5, 9.25], pattern="hour")  # skip off-hours
            ]
        )
    )
    return fig

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
#
# ✅ Normalize columns for any stock (Yahoo / CSV)
df.columns = [str(c) for c in df.columns]

rename_map = {}
for col in df.columns:
    col_low = col.lower()
    if 'date' in col_low or 'time' in col_low: rename_map[col] = 'Datetime'
    elif 'open' in col_low: rename_map[col] = 'Open'
    elif 'high' in col_low: rename_map[col] = 'High'
    elif 'low' in col_low: rename_map[col] = 'Low'
    elif 'close' in col_low: rename_map[col] = 'Close'
    elif 'volume' in col_low: rename_map[col] = 'Volume'
df.rename(columns=rename_map, inplace=True)

# ✅ Keep only needed columns
keep_cols = ['Datetime', 'Open', 'High', 'Low', 'Close']
df = df[[c for c in keep_cols if c in df.columns]]

# ✅ Ensure datetime type
df['Datetime'] = pd.to_datetime(df['Datetime'])
#st.write(df.columns)
#
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

# Initialize empty list to store signals
signal_log_list = []
# ✅ Loop through each day (starting from 2nd day in range)
for i in range(1, len(unique_days)):
    day0 = unique_days[i-1]
    day1 = unique_days[i]

    day_df = df[df['Datetime'].dt.date.isin([day0, day1])]

    # Call your trading signal function
    signal = trading_signal_all_conditions1(day_df)

#st.write(signal)
# 2. Fetch option premium data for that strike
#option_df = get_option_data_realtime(signal['spot_price'], signal['expiry'], signal['strike'])  # 15m candles

# 3. Track exit
#trade_result = track_trade_exit(signal, option_df)
#st.write(trade_result)
