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
#import pandas as pd
#import streamlit as st

def display_current_trend(df):
    """
    Display only the trend (Bullish/Bearish/Doji) of the latest candle.

    Parameters:
    - df: DataFrame with columns ['Open_^NSEI', 'Close_^NSEI']
    """
    if df.empty:
        st.warning("No candle data available.")
        return
    
    current_candle = df.iloc[-1]
    open_price = current_candle['Open_^NSEI']
    close_price = current_candle['Close_^NSEI']
    
    if close_price > open_price:
        trend_text = "Bullish 🔥"
        trend_color = "green"
    elif close_price < open_price:
        trend_text = "Bearish ❄️"
        trend_color = "red"
    else:
        trend_text = "Doji / Neutral ⚪"
        trend_color = "gray"
    
    st.markdown(f"<span style='color:{trend_color}; font-weight:bold; font-size:20px;'>Trend: {trend_text}</span>", unsafe_allow_html=True)



#import pandas as pd

def condition_1_trade_signal(nifty_df):
    """
    Parameters:
    - nifty_df: pd.DataFrame with columns ['Datetime', 'Open', 'High', 'Low', 'Close']
      'Datetime' is timezone-aware pandas Timestamp.
      Data must cover trading day 0 3PM candle and trading day 1 9:30AM candle.
      
    Returns:
    - dict with keys:
      - 'buy_signal': bool
      - 'buy_price': float (close of first candle next day if buy_signal True)
      - 'stoploss': float (10% below buy_price)
      - 'take_profit': float (10% above buy_price)
      - 'entry_time': pd.Timestamp of buy candle close time
      - 'message': str for info/logging
    """
    
    # Step 1: Identify trading days
    unique_days = sorted(nifty_df['Datetime'].dt.date.unique())
    if len(unique_days) < 2:
        return {'buy_signal': False, 'message': 'Not enough trading days in data.'}
    
    day0 = unique_days[-2]
    day1 = unique_days[-1]
    
    # Step 2: Get 3PM candle of day0 closing at 3:15 PM
    candle_3pm = nifty_df[
        (nifty_df['Datetime'].dt.date == day0) &
        (nifty_df['Datetime'].dt.hour == 15) &
        (nifty_df['Datetime'].dt.minute == 0)
    ]
    if candle_3pm.empty:
        return {'buy_signal': False, 'message': '3 PM candle on day0 not found.'}
    open_3pm = candle_3pm.iloc[0]['Open_^NSEI']
    close_3pm = candle_3pm.iloc[0]['Close_^NSEI']
    
    # Step 3: Get first 15-min candle of day1 closing at 9:30 AM (9:15-9:30)
    candle_930 = nifty_df[
        (nifty_df['Datetime'].dt.date == day1) &
        (nifty_df['Datetime'].dt.hour == 9) &
        (nifty_df['Datetime'].dt.minute == 15)
    ]
    if candle_930.empty:
        return {'buy_signal': False, 'message': '9:30 AM candle on day1 not found.'}
    open_930 = candle_930.iloc[0]['Open_^NSEI']
    close_930 = candle_930.iloc[0]['Close_^NSEI']
    high_930 = candle_930.iloc[0]['High_^NSEI']
    low_930 = candle_930.iloc[0]['Low_^NSEI']
    close_time_930 = candle_930.iloc[0]['Datetime']
    
    # Step 4: Check if candle cuts both lines from below to above and closes above both lines
    # "cuts lines from below to upwards" means low < open_3pm and close > open_3pm, same for close_3pm
    
    crossed_open_line = (low_930 < open_3pm) and (close_930 > open_3pm)
    crossed_close_line = (low_930 < close_3pm) and (close_930 > close_3pm)
    closes_above_both = (close_930 > open_3pm) and (close_930 > close_3pm)
    
    if crossed_open_line and crossed_close_line and closes_above_both:
        buy_price = close_930  # use close of 9:30 candle as buy price
        stoploss = buy_price * 0.9  # 10% trailing stoploss below buy price
        take_profit = buy_price * 1.10  # 10% profit booking target
        return {
            'buy_signal': True,
            'buy_price': buy_price,
            'stoploss': stoploss,
            'take_profit': take_profit,
            'entry_time': close_time_930,
            'message': 'Buy signal triggered: Crossed above 3PM open and close lines.'
        }
    
    return {'buy_signal': False, 'message': 'No buy signal: conditions not met.'}
 #######################################################################################################################################   
def condition_1_trade_signal_for_candle(nifty_df, candle_time):
    """
    Check buy condition for a specific candle_time on day1.
    candle_time: pd.Timestamp of candle close time to check (e.g. 9:30 AM, 9:45 AM, ...)
    
    Returns buy signal dict if condition met, else no signal.
    """
    unique_days = sorted(nifty_df['Datetime'].dt.date.unique())
    if len(unique_days) < 2:
        return {'buy_signal': False, 'message': 'Not enough trading days.'}
    
    day0 = unique_days[-2]
    day1 = unique_days[-1]
    
    # 3PM candle day0
    candle_3pm = nifty_df[
        (nifty_df['Datetime'].dt.date == day0) &
        (nifty_df['Datetime'].dt.hour == 15) &
        (nifty_df['Datetime'].dt.minute == 0)
    ]
    if candle_3pm.empty:
        return {'buy_signal': False, 'message': '3 PM candle day0 not found.'}
    open_3pm = candle_3pm.iloc[0]['Open_^NSEI']
    close_3pm = candle_3pm.iloc[0]['Close_^NSEI']
    
    # The candle on day1 at candle_time
    candle = nifty_df[nifty_df['Datetime'] == candle_time]
    if candle.empty:
        return {'buy_signal': False, 'message': f'No candle found at {candle_time}.'}
    
    open_c = candle.iloc[0]['Open_^NSEI']
    close_c = candle.iloc[0]['Close_^NSEI']
    low_c = candle.iloc[0]['Low_^NSEI']
    close_time = candle.iloc[0]['Datetime']
    
    crossed_open_line = (low_c < open_3pm) and (close_c > open_3pm)
    crossed_close_line = (low_c < close_3pm) and (close_c > close_3pm)
    closes_above_both = (close_c > open_3pm) and (close_c > close_3pm)
    
    if crossed_open_line and crossed_close_line and closes_above_both:
        buy_price = close_c
        stoploss = buy_price * 0.9
        take_profit = buy_price * 1.10
        return {
            'buy_signal': True,
            'buy_price': buy_price,
            'stoploss': stoploss,
            'take_profit': take_profit,
            'entry_time': close_time,
            'message': 'Buy signal triggered.'
        }
    
    return {'buy_signal': False, 'message': 'Condition not met.'}

#####################################################################################################################################
open_3pm, close_3pm = display_3pm_candle_info(df_plot, last_day)


##########################################################################################################

import pandas as pd
import streamlit as st

def display_todays_candles_with_trend(df):
    """
    Display all today's candles with OHLC + Trend column in Streamlit.

    Args:
    - df: DataFrame with columns ['Datetime', 'Open_^NSEI', 'High_^NSEI', 'Low_^NSEI', 'Close_^NSEI']
          'Datetime' must be timezone-aware or convertible to datetime.

    Output:
    - Shows table in Streamlit with added Trend column.
    """
    if df.empty:
        st.warning("No candle data available.")
        return
    
    # Get today date from last datetime in df (assumes df sorted)
    today_date = df['Datetime'].dt.date.max()
    
    # Filter today's data
    todays_df = df[df['Datetime'].dt.date == today_date].copy()
    if todays_df.empty:
        st.warning(f"No data for today: {today_date}")
        return
    
    # Calculate Trend column
    def calc_trend(row):
        if row['Close_^NSEI'] > row['Open_^NSEI']:
            return "Bullish 🔥"
        elif row['Close_^NSEI'] < row['Open_^NSEI']:
            return "Bearish ❄️"
        else:
            return "Doji ⚪"
    
    todays_df['Trend'] = todays_df.apply(calc_trend, axis=1)
    
    # Format datetime for display
    todays_df['Time'] = todays_df['Datetime'].dt.strftime('%H:%M')
    
    # Select and reorder columns to display
    display_df = todays_df[['Time', 'Open_^NSEI', 'High_^NSEI', 'Low_^NSEI', 'Close_^NSEI', 'Trend']].copy()
    display_df.rename(columns={
        'Open_^NSEI': 'Open',
        'High_^NSEI': 'High',
        'Low_^NSEI': 'Low',
        'Close_^NSEI': 'Close'
    }, inplace=True)
    
    st.write(f"All 15-min candles for today ({today_date}):")
    st.table(display_df)

###########################################################################################
#run_check_for_all_candles(df)  # df = your full OHLC DataFrame

display_todays_candles_with_trend(df)
