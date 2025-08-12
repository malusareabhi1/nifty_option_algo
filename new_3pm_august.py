import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import requests

st.set_page_config(layout="wide")
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

#import pandas as pd
#import streamlit as st

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
import pandas as pd
import streamlit as st

def display_todays_candles_with_trend_and_signal(df):
    """
    Display all today's candles with OHLC + Trend + Signal columns in Streamlit.

    Args:
    - df: DataFrame with columns ['Datetime', 'Open_^NSEI', 'High_^NSEI', 'Low_^NSEI', 'Close_^NSEI']
          'Datetime' must be timezone-aware or convertible to datetime.

    Output:
    - Shows table in Streamlit with added Trend and Signal columns.
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
    
    # Calculate Signal column
    signals = []
    for i in range(len(todays_df)):
        if i == 0:
            # No previous candle, so no signal
            signals.append("-")
        else:
            prev_high = todays_df.iloc[i-1]['High_^NSEI']
            prev_low = todays_df.iloc[i-1]['Low_^NSEI']
            curr_close = todays_df.iloc[i]['Close_^NSEI']
            curr_trend = todays_df.iloc[i]['Trend']
            
            if curr_trend == "Bullish 🔥" and curr_close > prev_high:
                signals.append("Buy")
            elif curr_trend == "Bearish ❄️" and curr_close < prev_low:
                signals.append("Sell")
            else:
                signals.append("-")
    
    todays_df['Signal'] = signals
    
    # Format datetime for display
    todays_df['Time'] = todays_df['Datetime'].dt.strftime('%H:%M')
    
    # Select and reorder columns to display
    display_df = todays_df[['Time', 'Open_^NSEI', 'High_^NSEI', 'Low_^NSEI', 'Close_^NSEI', 'Trend', 'Signal']].copy()
    display_df.rename(columns={
        'Open_^NSEI': 'Open',
        'High_^NSEI': 'High',
        'Low_^NSEI': 'Low',
        'Close_^NSEI': 'Close'
    }, inplace=True)
    
    st.write(f"All 15-min candles for today ({today_date}):")
    st.table(display_df)


###################################################################################################

import pandas as pd

def get_nearest_weekly_expiry(today):
    """
    Placeholder: implement your own logic to find nearest weekly expiry date
    For demo, returns today + 7 days (Saturday)
    """
    return today + pd.Timedelta(days=7)

def trading_signal_all_conditions(df, quantity=10*750):
    """
    Evaluate all 4 conditions and return trade signals if any triggered.

    Returns:
    dict with keys:
    - 'condition': 1/2/3/4
    - 'option_type': 'CALL' or 'PUT'
    - 'buy_price': float (price from 9:30 candle close for now)
    - 'stoploss': float (10% trailing SL below buy price)
    - 'take_profit': float (10% above buy price)
    - 'quantity': int (10 lots = 7500 assumed)
    - 'expiry': datetime.date
    - 'entry_time': pd.Timestamp
    - 'message': explanation string
    Or None if no signal.
    """

    df = df.copy()
    df['Date'] = df['Datetime'].dt.date
    unique_days = sorted(df['Date'].unique())
    if len(unique_days) < 2:
        return None  # not enough data
    
    day0 = unique_days[-2]
    day1 = unique_days[-1]
    
    # 1) Get day0 3PM candle (15:00 - 15:15)
    candle_3pm = df[(df['Date'] == day0) & 
                    (df['Datetime'].dt.hour == 15) & 
                    (df['Datetime'].dt.minute == 0)]
    if candle_3pm.empty:
        return None
    
    open_3pm = candle_3pm.iloc[0]['Open_^NSEI']
    close_3pm = candle_3pm.iloc[0]['Close_^NSEI']

    # 2) Day1 first 15-min candle (9:15 - 9:30)
    candle_930 = df[(df['Date'] == day1) & 
                    (df['Datetime'].dt.hour == 9) & 
                    (df['Datetime'].dt.minute == 15)]
    if candle_930.empty:
        return None
    
    open_930 = candle_930.iloc[0]['Open_^NSEI']
    close_930 = candle_930.iloc[0]['Close_^NSEI']
    high_930 = candle_930.iloc[0]['High_^NSEI']
    low_930 = candle_930.iloc[0]['Low_^NSEI']
    entry_time = candle_930.iloc[0]['Datetime']
    
    # 3) Day1 open price = first tick open or 9:15 candle open assumed here
    day1_open_price = open_930

    # Calculate nearest weekly expiry (example, user replace with correct method)
    expiry = get_nearest_weekly_expiry(pd.to_datetime(day1))

    # Helper values
    buy_price = close_930
    stoploss = buy_price * 0.9
    take_profit = buy_price * 1.10

    # Detect gap up / gap down vs day0 close
    gap_up = day1_open_price > close_3pm
    gap_down = day1_open_price < close_3pm
    within_range = (day1_open_price >= open_3pm) and (day1_open_price <= close_3pm)

    # Condition 1
    # Candle cuts the 3PM open & close lines from below upwards and closes above both
    cond1_cuts_open = (low_930 < open_3pm) and (close_930 > open_3pm)
    cond1_cuts_close = (low_930 < close_3pm) and (close_930 > close_3pm)
    cond1_closes_above_both = (close_930 > open_3pm) and (close_930 > close_3pm)

    if cond1_cuts_open and cond1_cuts_close and cond1_closes_above_both:
        return {
            'condition': 1,
            'option_type': 'CALL',
            'buy_price': buy_price,
            'stoploss': stoploss,
            'take_profit': take_profit,
            'quantity': quantity,
            'expiry': expiry,
            'entry_time': entry_time,
            'message': 'Condition 1 met: Buy nearest ITM CALL option.'
        }

    # Condition 2: Major gap down + 9:30 candle closes below 3PM open & close lines
    if gap_down and (close_930 < open_3pm) and (close_930 < close_3pm):
        # Reference candle 2 = first 9:30 candle
        ref_high = high_930
        ref_low = low_930
        
        # Look for next candle after 9:30 to cross below low of ref candle 2
        # We'll scan candles after 9:30 for this signal:
        day1_after_930 = df[(df['Date'] == day1) & (df['Datetime'] > entry_time)].sort_values('Datetime')
        for _, next_candle in day1_after_930.iterrows():
            if next_candle['Low_^NSEI'] < ref_low:
                # Signal to buy PUT option
                buy_price_put = next_candle['Close_^NSEI']
                stoploss_put = buy_price_put * 0.9
                take_profit_put = buy_price_put * 1.10
                return {
                    'condition': 2,
                    'option_type': 'PUT',
                    'buy_price': buy_price_put,
                    'stoploss': stoploss_put,
                    'take_profit': take_profit_put,
                    'quantity': quantity,
                    'expiry': expiry,
                    'entry_time': next_candle['Datetime'],
                    'message': 'Condition 2 met: Gap down + next candle crosses low of 9:30 candle, buy nearest ITM PUT.'
                }
        # If no such candle crossed, no trade yet
    
    # Condition 3: Major gap up + 9:30 candle closes above 3PM open & close lines
    if gap_up and (close_930 > open_3pm) and (close_930 > close_3pm):
        # Reference candle 2 = first 9:30 candle
        ref_high = high_930
        ref_low = low_930
        
        # Look for next candle after 9:30 to cross above high of ref candle 2
        day1_after_930 = df[(df['Date'] == day1) & (df['Datetime'] > entry_time)].sort_values('Datetime')
        for _, next_candle in day1_after_930.iterrows():
            if next_candle['High_^NSEI'] > ref_high:
                buy_price_call = next_candle['Close_^NSEI']
                stoploss_call = buy_price_call * 0.9
                take_profit_call = buy_price_call * 1.10
                return {
                    'condition': 3,
                    'option_type': 'CALL',
                    'buy_price': buy_price_call,
                    'stoploss': stoploss_call,
                    'take_profit': take_profit_call,
                    'quantity': quantity,
                    'expiry': expiry,
                    'entry_time': next_candle['Datetime'],
                    'message': 'Condition 3 met: Gap up + next candle crosses high of 9:30 candle, buy nearest ITM CALL.'
                }
        # No trade if not crossed yet
    
    # Condition 4:
    # Candle cuts the 3PM open & close lines from above downwards and closes below both
    cond4_cuts_open = (high_930 > open_3pm) and (close_930 < open_3pm)
    cond4_cuts_close = (high_930 > close_3pm) and (close_930 < close_3pm)
    cond4_closes_below_both = (close_930 < open_3pm) and (close_930 < close_3pm)

    if cond4_cuts_open and cond4_cuts_close and cond4_closes_below_both:
        return {
            'condition': 4,
            'option_type': 'PUT',
            'buy_price': buy_price,
            'stoploss': stoploss,
            'take_profit': take_profit,
            'quantity': quantity,
            'expiry': expiry,
            'entry_time': entry_time,
            'message': 'Condition 4 met: Buy nearest ITM PUT option.'
        }
    
    # No condition met
    return None



#########################################################################################################

#import pandas as pd
#import requests
#import pandas as pd

def get_live_nifty_option_chain():
    url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.nseindia.com/option-chain",
        "Connection": "keep-alive",
    }

    session = requests.Session()
    # First request to get cookies
    session.get("https://www.nseindia.com", headers=headers)

    # Now get option chain JSON
    response = session.get(url, headers=headers)
    data = response.json()

    records = []
    for record in data['records']['data']:
        strike_price = record['strikePrice']
        expiry_dates = data['records']['expiryDates']
        # Calls data
        if 'CE' in record:
            ce = record['CE']
            ce_row = {
                'strikePrice': strike_price,
                'expiryDate': ce['expiryDate'],
                'optionType': 'CE',
                'lastPrice': ce.get('lastPrice', None),
                'bidQty': ce.get('bidQty', None),
                'askQty': ce.get('askQty', None),
                'openInterest': ce.get('openInterest', None),
                'changeinOpenInterest': ce.get('changeinOpenInterest', None),
                'impliedVolatility': ce.get('impliedVolatility', None),
                'underlying': ce.get('underlying', None),
            }
            records.append(ce_row)
        # Puts data
        if 'PE' in record:
            pe = record['PE']
            pe_row = {
                'strikePrice': strike_price,
                'expiryDate': pe['expiryDate'],
                'optionType': 'PE',
                'lastPrice': pe.get('lastPrice', None),
                'bidQty': pe.get('bidQty', None),
                'askQty': pe.get('askQty', None),
                'openInterest': pe.get('openInterest', None),
                'changeinOpenInterest': pe.get('changeinOpenInterest', None),
                'impliedVolatility': pe.get('impliedVolatility', None),
                'underlying': pe.get('underlying', None),
            }
            records.append(pe_row)

    option_chain_df = pd.DataFrame(records)
    # Convert expiryDate column to datetime
    option_chain_df['expiryDate'] = pd.to_datetime(option_chain_df['expiryDate'])

    return option_chain_df

# Usage:
#option_chain_df = get_live_nifty_option_chain()
#st.write(option_chain_df.head())

################################################################################################
def find_nearest_itm_option():
    import nsepython
    from nsepython import nse_optionchain_scrapper


    option_chain = nse_optionchain_scrapper('NIFTY')
    df = []
    
    for item in option_chain['records']['data']:
        strike = item['strikePrice']
        expiry = item['expiryDate']
        if 'CE' in item:
            ce = item['CE']
            ce['strikePrice'] = strike
            ce['expiryDate'] = expiry
            ce['optionType'] = 'CE'
            df.append(ce)
        if 'PE' in item:
            pe = item['PE']
            pe['strikePrice'] = strike
            pe['expiryDate'] = expiry
            pe['optionType'] = 'PE'
            df.append(pe)
    
    import pandas as pd
    option_chain_df = pd.DataFrame(df)
    option_chain_df['expiryDate'] = pd.to_datetime(option_chain_df['expiryDate'])
    #st.write(option_chain_df.head())
    return  option_chain_df



##################################################################################
import pandas as pd

def option_chain_finder(option_chain_df, spot_price, option_type, lots=10, lot_size=75):
    """
    Find nearest ITM option in option chain DataFrame.

    Parameters:
    - option_chain_df: pd.DataFrame with columns including ['strikePrice', 'expiryDate', 'optionType', ...]
    - spot_price: float, current underlying price
    - option_type: str, 'CE' for Call or 'PE' for Put
    - lots: int, number of lots to trade (default 10)
    - lot_size: int, lot size per option contract (default 75)

    Returns:
    - dict with keys:
        'strikePrice', 'expiryDate', 'optionType', 'total_quantity', 'option_data' (pd.Series row)
    """

    # Ensure expiryDate is datetime
    if not pd.api.types.is_datetime64_any_dtype(option_chain_df['expiryDate']):
        option_chain_df['expiryDate'] = pd.to_datetime(option_chain_df['expiryDate'])

    today = pd.Timestamp.today().normalize()

    # Find nearest expiry on or after today
    expiries = option_chain_df.loc[option_chain_df['expiryDate'] >= today, 'expiryDate'].unique()
    if len(expiries) == 0:
        raise ValueError("No expiry dates found on or after today.")
    nearest_expiry = min(expiries)

    # Filter for nearest expiry and option type
    df_expiry = option_chain_df[
        (option_chain_df['expiryDate'] == nearest_expiry) &
        (option_chain_df['optionType'] == option_type)
    ]

    if df_expiry.empty:
        raise ValueError(f"No options found for expiry {nearest_expiry.date()} and type {option_type}")

    # Find nearest ITM strike
    if option_type == 'CE':
        itm_strikes = df_expiry[df_expiry['strikePrice'] <= spot_price]
        if itm_strikes.empty:
            # fallback to minimum strike (OTM)
            nearest_strike = df_expiry['strikePrice'].min()
        else:
            nearest_strike = itm_strikes['strikePrice'].max()
    else:  # 'PE'
        itm_strikes = df_expiry[df_expiry['strikePrice'] >= spot_price]
        if itm_strikes.empty:
            # fallback to maximum strike (OTM)
            nearest_strike = df_expiry['strikePrice'].max()
        else:
            nearest_strike = itm_strikes['strikePrice'].min()

    # Get option row
    option_row = df_expiry[df_expiry['strikePrice'] == nearest_strike].iloc[0]

    total_qty = lots * lot_size

    return {
        'strikePrice': nearest_strike,
        'expiryDate': nearest_expiry,
        'optionType': option_type,
        'total_quantity': total_qty,
        'option_data': option_row
    }

############################################################################################################
#run_check_for_all_candles(df)  # df = your full OHLC DataFrame

#display_todays_candles_with_trend(df)
display_todays_candles_with_trend_and_signal(df)
########################
result_chain=find_nearest_itm_option()
#calling all condition in one function
signal = trading_signal_all_conditions(df)
if signal:
    st.write(f"Trade signal detected:\n{signal['message']}")
    st.table(pd.DataFrame([signal]))
    # Find nearest ITM option to buy
    result = option_chain_finder(result_chain, spot_price, option_type='CE', lots=10, lot_size=75)

    st.write("Nearest ITM Call option to BUY:")
    st.table(pd.DataFrame([result['option_data']]))

    st.write(f"Total Quantity: {result['total_quantity']}")
else:
    st.write("No trade signal for today based on conditions.")


#st.write(result_chain.tail())
#################################################
