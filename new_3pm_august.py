import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# Place at the very top of your script (or just before plotting)
st_autorefresh(interval=240000, limit=None, key="refresh")

# Now your whole app will auto-refresh every 4 minutes (240000 ms)


# Load data (replace this with your data loading logic)
df = yf.download("^NSEI", period="7d", interval="15m")
if isinstance(df.index, pd.DatetimeIndex):
    df.reset_index(inplace=True)

if 'Datetime_' in df.columns:
    df.rename(columns={'Datetime_': 'Datetime'}, inplace=True)
elif 'Date' in df.columns:
    df.rename(columns={'Date': 'Datetime'}, inplace=True)
# Add any other detected name if needed


#st.write(df.columns)
#st.write(df.head(10))
# Flatten columns if MultiIndex
if isinstance(df.columns, pd.MultiIndex):
    df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]

# Rename datetime column if needed
if 'Datetime' not in df.columns and 'datetime' in df.columns:
    df.rename(columns={'datetime': 'Datetime'}, inplace=True)

#st.write(df.columns)
# Convert to datetime & timezone aware
#df['Datetime'] = pd.to_datetime(df['Datetime'])
if df['Datetime_'].dt.tz is None:
    df['Datetime'] = df['Datetime_'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
else:
    df['Datetime'] = df['Datetime_'].dt.tz_convert('Asia/Kolkata')

#st.write(df.columns)
#st.write(df.head(10))

# Filter for last two trading days to plot
unique_days = df['Datetime'].dt.date.unique()
if len(unique_days) < 2:
    st.warning("Not enough data for two trading days")
else:
    last_day = unique_days[-2]
    today = unique_days[-1]

    df_plot = df[df['Datetime'].dt.date.isin([last_day, today])]

    # Get last day 3PM candle open and close
    candle_3pm = df_plot[(df_plot['Datetime'].dt.date == last_day) &
                         (df_plot['Datetime'].dt.hour == 15) &
                         (df_plot['Datetime'].dt.minute == 0)]

    if not candle_3pm.empty:
        open_3pm = candle_3pm.iloc[0]['Open_^NSEI']
        close_3pm = candle_3pm.iloc[0]['Close_^NSEI']
    else:
        open_3pm = None
        close_3pm = None
        st.warning("No 3:00 PM candle found for last trading day.")

    # Plot candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=df_plot['Datetime'],
        open=df_plot['Open_^NSEI'],
        high=df_plot['High_^NSEI'],
        low=df_plot['Low_^NSEI'],
        close=df_plot['Close_^NSEI']
    )])

    if open_3pm and close_3pm:
        fig.add_hline(y=open_3pm, line_dash="dot", line_color="blue", annotation_text="3PM Open")
        fig.add_hline(y=close_3pm, line_dash="dot", line_color="red", annotation_text="3PM Close")




    # Draw horizontal lines as line segments only between 3PM last day and 3PM next day

    
    fig.update_layout(title="Nifty 15-min candles - Last Day & Today", xaxis_rangeslider_visible=False)
    fig.update_layout(
    xaxis=dict(
        rangebreaks=[
            # Hide weekends (Saturday and Sunday)
            dict(bounds=["sat", "mon"]),
            # Hide hours outside of trading hours (NSE trading hours 9:15 to 15:30)
            dict(bounds=[15.5, 9.25], pattern="hour"),
        ]
    )
)


    st.plotly_chart(fig, use_container_width=True)

def display_3pm_candle_info(df, day):
    """
    Display the 3PM candle Open and Close prices for a given day (datetime.date).
    
    Parameters:
    - df: DataFrame with 'Datetime' column (timezone-aware datetime)
    - day: datetime.date object representing the trading day
    
    Returns:
    - (open_price, close_price) tuple or (None, None) if candle not found
    """
    candle = df[(df['Datetime'].dt.date == day) &
                (df['Datetime'].dt.hour == 15) &
                (df['Datetime'].dt.minute == 0)]
    
    if candle.empty:
        st.warning(f"No 3:00 PM candle found for {day}")
        return None, None
    
    open_price = candle.iloc[0]['Open_^NSEI']
    close_price = candle.iloc[0]['Close_^NSEI']
    
    st.info(f"3:00 PM Candle for {day}: Open = {open_price}, Close = {close_price}")
    st.write(f"🔵 3:00 PM Open for {day}: {open_price}")
    st.write(f"🔴 3:00 PM Close for {day}: {close_price}")
    
    return open_price, close_price


#import streamlit as st

def display_current_candle(df):
    """
    Display the latest candle OHLC prices from a DataFrame with a 'Datetime' column.
    Assumes df is sorted by datetime ascending.
    
    Parameters:
    - df: DataFrame with columns ['Open_^NSEI', 'High_^NSEI', 'Low_^NSEI', 'Close_^NSEI', 'Datetime']
    """
    st.write(df.columns)
    # Ensure datetime is timezone-aware and converted to Asia/Kolkata
    local_dt = current_candle['Datetime_']
    if local_dt.tzinfo is None:
        local_dt = local_dt.tz_localize('UTC').tz_convert('Asia/Kolkata')
    else:
        local_dt = local_dt.tz_convert('Asia/Kolkata')
    
    st.info(f"Current Candle @ {local_dt.strftime('%Y-%m-%d %H:%M')} (Asia/Kolkata)")

    if df.empty:
        st.warning("No candle data available.")
        return
    
    current_candle = df.iloc[-1]
    
    #st.info(f"Current Candle @ {current_candle['Datetime']}")
    st.write(f"Open: {current_candle['Open_^NSEI']}")
    st.write(f"High: {current_candle['High_^NSEI']}")
    st.write(f"Low: {current_candle['Low_^NSEI']}")
    st.write(f"Close: {current_candle['Close_^NSEI']}")

# After you get last_day and df_plot

open_3pm, close_3pm = display_3pm_candle_info(df_plot, last_day)

# Now you have values to use in plotting or other logic

display_current_candle(df)


