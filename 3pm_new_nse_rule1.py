import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("Nifty 3PM Trailing SL and Take Profit  Strategy - Multi-Day Backtest")



def plot_nifty_multiday(df, trading_days):
    """
    Plots Nifty 15-min candles for multiple trading days with each previous day's 3PM Open/Close
    marked only on the next trading day and extending only until 3PM candle.
    
    Parameters:
    - df : DataFrame with columns ['Datetime', 'Open_^NSEI', 'High_^NSEI', 'Low_^NSEI', 'Close_^NSEI']
    - trading_days : list of sorted trading dates (datetime.date)
    """
    
    fig = go.Figure()
    
    for i in range(1, len(trading_days)):
        day0 = trading_days[i-1]  # Previous day (for Base Zone)
        day1 = trading_days[i]    # Current day
        
        # Filter data for current day only
        df_day1 = df[df['Datetime'].dt.date == day1]
        
        # Add candlestick trace for current day
        fig.add_trace(go.Candlestick(
            x=df_day1['Datetime'],
            open=df_day1['Open_^NSEI'],
            high=df_day1['High_^NSEI'],
            low=df_day1['Low_^NSEI'],
            close=df_day1['Close_^NSEI'],
            name=f"{day1}"
        ))
        
        # Get 3 PM candle of previous day (Base Zone)
        candle_3pm = df[df['Datetime'].dt.date == day0]
        candle_3pm = candle_3pm[(candle_3pm['Datetime'].dt.hour == 15) &
                                (candle_3pm['Datetime'].dt.minute == 0)]
        
        if not candle_3pm.empty:
            open_3pm = candle_3pm.iloc[0]['Open_^NSEI']
            close_3pm = candle_3pm.iloc[0]['Close_^NSEI']
            
            # Get day1 3PM candle time for line end
            day1_3pm_candle = df_day1[(df_day1['Datetime'].dt.hour == 15) &
                                       (df_day1['Datetime'].dt.minute == 0)]
            if not day1_3pm_candle.empty:
                x_end = day1_3pm_candle['Datetime'].iloc[0]
                x_start = df_day1['Datetime'].min()
                
                # Horizontal line for Open
                fig.add_shape(
                    type="line",
                    x0=x_start,
                    x1=x_end,
                    y0=open_3pm,
                    y1=open_3pm,
                    line=dict(color="blue", width=1, dash="dot"),
                )
                
                # Horizontal line for Close
                fig.add_shape(
                    type="line",
                    x0=x_start,
                    x1=x_end,
                    y0=close_3pm,
                    y1=close_3pm,
                    line=dict(color="red", width=1, dash="dot"),
                )
    
    # Layout adjustments
    fig.update_layout(
        title="Nifty 15-min Candles with Previous Day 3PM Open/Close Lines (to next day 3PM)",
        xaxis_rangeslider_visible=False,
        xaxis=dict(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),          # Hide weekends
                dict(bounds=[15.5, 9.25], pattern="hour")  # Hide off-hours
            ]
        )
    )
    
    return fig


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
       start_date = st.date_input("Select Start Date", value=datetime.today() - timedelta(days=15))
       end_date = st.date_input("Select End Date", value=datetime.today())

if start_date >= end_date:
    st.warning("End date must be after start date")
    st.stop()

# ✅ Download full data for range (start-1 day to end)
download_start = start_date - timedelta(days=1)  # To include previous day for first day
df = yf.download("^NSEI", start=download_start, end=end_date + timedelta(days=1), interval="15m")
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


# Initialize empty list to store signals
signal_log_list = []
# ✅ Loop through each day (starting from 2nd day in range)
for i in range(1, len(unique_days)):
    day0 = unique_days[i-1]
    day1 = unique_days[i]

    day_df = df[df['Datetime'].dt.date.isin([day0, day1])]

    # Call your trading signal function
    signal = trading_signal_all_conditions1(day_df)
    if isinstance(signal, dict):
        try:
            # Wrap dict into a list to create a single-row DataFrame
            signal_df = pd.DataFrame([signal])
            #st.write("signal converted to DataFrame:", signal_df)
        except Exception as e:
            st.error(f"Cannot convert signal to DataFrame: {e}")
            st.write("Raw signal dictionary:", signal)
    else:
        signal_df = signal

    if signal is None:
        continue
     # Convert dict -> list
    signal_list = [signal] if isinstance(signal, dict) else signal

    for sig in signal_list:
        # ✅ Get ITM strike based on signal's option type and spot price
        if "spot_price" not in sig:
            sig["spot_price"] = day_df['Close_^NSEI'].iloc[-1]

        if "option_type" in sig:
            sig["itm_strike"] = get_nearest_itm_strike(sig["spot_price"], sig["option_type"])

        # Calculate nearest expiry (next Thursday)
        today = day1
        expiry = today + timedelta((3 - today.weekday()) % 7)
        sig["expiry"] = expiry

        signal_log_list.append(sig)
    # If function returns a dict (single signal)
    if isinstance(signal, dict):
        signal_log_list.append(signal)

    # If function returns a list of signals
    elif isinstance(signal, list):
        signal_log_list.extend(signal)

# ✅ Convert all collected signals into DataFrame
if signal_log_list:
    signal_log_df = pd.DataFrame(signal_log_list)
    #st.write("Signal Log")
    #st.dataframe(signal_log_df, use_container_width=True)
    # Optional: reorder columns for cleaner display
    cols_order = ['entry_time','condition','option_type','buy_price','stoploss',
                  'exit_price','status','message','spot_price','quantity','expiry']
    signal_log_df = signal_log_df[[c for c in cols_order if c in signal_log_df.columns]]

    # ✅ Display in table instead of row-by-row
    st.write("Signal Log")
    st.dataframe(signal_log_df, use_container_width=True)

    # ✅ Also allow CSV download
    csv = signal_log_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Signal Log (CSV)", data=csv, file_name="signal_log.csv", mime="text/csv")

else:
    st.info("No signals generated for the selected period.")

################################################################################################
