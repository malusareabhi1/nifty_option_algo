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
    

##############################################################################
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
    
    #st.info(f"3:00 PM Candle for {day}: Open = {open_price}, Close = {close_price}")
    #st.write(f"🔵 3:00 PM Open for {day}: {open_price}")
    #st.write(f"🔴 3:00 PM Close for {day}: {close_price}")
    
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
    
    #st.write(f"All 15-min candles for today ({today_date}):")
    #st.table(display_df.tail(5))

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
    
    #st.write(f"All 15-min candles for today ({today_date}):")
    #st.table(display_df)


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
    spot_price = df['Close_^NSEI'].iloc[-1]  # last close price
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
            'message': 'Condition 1 met: Buy nearest ITM CALL option.',
            'spot_price': spot_price
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
                    'message': 'Condition 2 met: Gap down + next candle crosses low of 9:30 candle, buy nearest ITM PUT.',
                    'spot_price': spot_price
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
                    'message': 'Condition 3 met: Gap up + next candle crosses high of 9:30 candle, buy nearest ITM CALL.',
                    'spot_price': spot_price
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
            'message': 'Condition 4 met: Buy nearest ITM PUT option.',
            'spot_price': spot_price
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
def find_nearest_itm_option_old():
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
    
    #import pandas as pd
    option_chain_df = pd.DataFrame(df)
    option_chain_df['expiryDate'] = pd.to_datetime(option_chain_df['expiryDate'])
    #st.write(option_chain_df.head())
    return  option_chain_df
##############################################################.



def find_nearest_itm_option():
    import pandas as pd
    from nsepython import nse_optionchain_scrapper

    try:
        option_chain = nse_optionchain_scrapper('NIFTY')
    except Exception as e:
        print(f"❌ Error fetching option chain: {e}")
        return pd.DataFrame()  # return empty DataFrame

    # Check if 'records' and 'data' exist
    if not isinstance(option_chain, dict) or 'records' not in option_chain or 'data' not in option_chain['records']:
        print("⚠️ Option chain response does not have expected structure.")
        print(f"Received: {option_chain}")
        return pd.DataFrame()

    df = []
    for item in option_chain['records']['data']:
        strike = item.get('strikePrice')
        expiry = item.get('expiryDate')
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

    option_chain_df = pd.DataFrame(df)
    if not option_chain_df.empty:
        option_chain_df['expiryDate'] = pd.to_datetime(option_chain_df['expiryDate'])
    return option_chain_df



##################################################################################
import pandas as pd

def option_chain_finder_old(option_chain_df, spot_price, option_type, lots=10, lot_size=75):
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
    if option_type == "CALL":
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

##############################################################


def option_chain_finder(option_chain_df, spot_price, option_type="CE", lots=1, lot_size=50):
    import pandas as pd
    
    # ✅ Guard clause to avoid KeyError
    if option_chain_df.empty:
        print("⚠️ option_chain_df is empty — no option chain data available.")
        return None
    
    if "expiryDate" not in option_chain_df.columns:
        print("⚠️ expiryDate column missing in option_chain_df.")
        print(option_chain_df.head())  # for debugging
        return None

    # ✅ Ensure expiryDate is datetime
    if not pd.api.types.is_datetime64_any_dtype(option_chain_df['expiryDate']):
        option_chain_df['expiryDate'] = pd.to_datetime(option_chain_df['expiryDate'], errors='coerce')

    # ✅ Filter for nearest expiry
    nearest_expiry = option_chain_df['expiryDate'].min()
    df_nearest = option_chain_df[option_chain_df['expiryDate'] == nearest_expiry]

    # ✅ Find nearest strike (ITM)
    df_nearest['strike_diff'] = abs(df_nearest['strikePrice'] - spot_price)
    df_nearest = df_nearest.sort_values(by='strike_diff')
    
    selected_option = df_nearest[df_nearest['optionType'] == option_type].head(1)
    
    if selected_option.empty:
        print("⚠️ No matching option found for given type.")
        return None
    
    # ✅ Add lot calculation
    selected_option['lots'] = lots
    selected_option['lot_size'] = lot_size
    selected_option['total_qty'] = lots * lot_size

    return selected_option




###########################################################

import pandas as pd
from datetime import timedelta

def generate_trade_log_from_option(result, trade_signal):
    if result is None or trade_signal is None:
        return None
    # Determine exit reason and price
    stoploss_hit = False
    target_hit = False
    
    #exit_time = pd.to_datetime(exit_time)
    #result.index = pd.to_datetime(result.index)

    
    
    

    option = result['option_data']
    qty = result['total_quantity']

    condition = trade_signal['condition']
    entry_time = trade_signal['entry_time']
    message = trade_signal['message']

    buy_price = option.get('lastPrice', trade_signal.get('buy_price'))
    expiry = option.get('expiryDate', trade_signal.get('expiry'))
    option_type = option.get('optionType', trade_signal.get('option_type'))

    stoploss = buy_price * 0.9
    take_profit = buy_price * 1.10
    partial_qty = qty // 2
    time_exit = entry_time + timedelta(minutes=16)

    

    

    trade_log = {
        "Condition": condition,
        "Option Type": option_type,
        "Strike Price": option.get('strikePrice'),
        #"Exit Price": exit_price,  # ✅ new column
        "Buy Premium": buy_price,
        "Stoploss (Trailing 10%)": stoploss,
        "Take Profit (10% rise)": take_profit,
        "Quantity": qty,
        "Partial Profit Booking Qty (50%)": partial_qty,
        "Expiry Date": expiry.strftime('%Y-%m-%d') if hasattr(expiry, 'strftime') else expiry,
        "Entry Time": entry_time.strftime('%Y-%m-%d %H:%M:%S') if hasattr(entry_time, 'strftime') else entry_time,
        "Time Exit (16 mins after entry)": time_exit.strftime('%Y-%m-%d %H:%M:%S'),
        "Trade Message": message
        #"Trade Details": row["Trade Details"],
        #"Exit Reason": reason
    }

    # Add condition-specific details
    if condition == 1:
        trade_log["Trade Details"] = (
            "Buy nearest ITM CALL option. Stoploss trailing 10% below buy premium. "
            "Book 50% qty profit when premium rises 10%. "
            "Time exit after 16 minutes if no target hit."
        )
    elif condition == 2:
        trade_log["Trade Details"] = (
            "Major gap down. Buy nearest ITM PUT option when next candle crosses low of 9:30 candle. "
            "Stoploss trailing 10% below buy premium."
        )
    elif condition == 3:
        trade_log["Trade Details"] = (
            "Major gap up. Buy nearest ITM CALL option. Stoploss trailing 10% below buy premium. "
            "Book 50% qty profit when premium rises 10%. "
            "Time exit after 16 minutes if no target hit."
        )
    elif condition == 4:
        trade_log["Trade Details"] = (
            "Buy nearest ITM PUT option. Stoploss trailing 10% below buy premium. "
            "Book 50% qty profit when premium rises 10%. "
            "Time exit after 16 minutes if no target hit."
        )
    else:
        trade_log["Trade Details"] = "No specific trade details available."

    trade_log_df = pd.DataFrame([trade_log])
    return trade_log_df

################################################################################################################

#import pandas as pd
#import matplotlib.pyplot as plt

#import matplotlib.pyplot as plt

def plot_option_trade(option_symbol, entry_time, exit_time, entry_price, exit_price, reason_exit, pnl):
    """
    Plot option price movement with entry/exit markers and P&L.
    """
    #import yfinance as yf
    #import pandas as pd

    # Fetch historical intraday data for the option
    data = yf.download(option_symbol, start=entry_time.date(), end=exit_time.date() + pd.Timedelta(days=1), interval="5m")

    if data.empty:
        st.warning(f"No historical data found for {option_symbol}")
        return

    # Plot price movement
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Price chart
    ax[0].plot(data.index, data['Close'], label='Option Price', color='blue')
    ax[0].axvline(entry_time, color='green', linestyle='--', label='Entry')
    ax[0].axvline(exit_time, color='red', linestyle='--', label='Exit')
    ax[0].scatter(entry_time, entry_price, color='green', s=80, zorder=5)
    ax[0].scatter(exit_time, exit_price, color='red', s=80, zorder=5)
    ax[0].set_ylabel("Price")
    ax[0].legend()
    ax[0].set_title(f"Trade on {option_symbol} | Exit Reason: {reason_exit}")

    # P&L curve
    data['P&L'] = (data['Close'] - entry_price) * (1 if exit_price > entry_price else -1)
    ax[1].plot(data.index, data['P&L'], label="P&L", color='orange')
    ax[1].axhline(0, color='black', linestyle='--')
    ax[1].set_ylabel("P&L")
    ax[1].legend()

    plt.tight_layout()
    st.pyplot(fig)
###################################################################################

#import streamlit as st
#from playsound import playsound
import time

def price_alert(current_price, high, low, sound_file="alert.mp3"):
    """
    Play alert if price crosses high or low.

    Parameters:
    - current_price: float, current market price
    - high: float, high threshold
    - low: float, low threshold
    - sound_file: str, path to alert sound
    """
    if current_price > high:
        st.warning(f"⚠️ Price above HIGH! Current: {current_price}, High: {high}")
        playsound(sound_file)
    elif current_price < low:
        st.warning(f"⚠️ Price below LOW! Current: {current_price}, Low: {low}")
        playsound(sound_file)
    else:
        st.write(f"Current Price: {current_price}")

# -----------------------------
# Example usage in Streamlit
# -----------------------------
#st.title("Price Alert Function Example")

#high = st.number_input("Set High Price", value=200)
#low = st.number_input("Set Low Price", value=180)

# Simulate live price (replace with real API later)
#current_price = st.number_input("Current Price", value=190)

#if st.button("Check Alert"):
    #price_alert(current_price, high, low)
########################################################################################################

import pandas as pd
import numpy as np

def trading_signal_all_condition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to generate trading signals based on conditions:
    - 3PM candle (day 0) open/close marked
    - Next day first 15-min breakout check
    - EMA20 crossover + Volume breakout (if volume available)

    Parameters:
        df (pd.DataFrame): NIFTY 15-min OHLC data with columns
                           ['Datetime','Open_^NSEI','High_^NSEI','Low_^NSEI','Close_^NSEI']
                           (optional 'Volume')

    Returns:
        pd.DataFrame: df with extra columns ['EMA20','Signal']
    """
    
    df = df.copy()
    
    # Ensure datetime is in pandas datetime
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values("Datetime").reset_index(drop=True)
    
    # Rename columns to generic OHLC
    df = df.rename(columns={
        'Open_^NSEI': 'Open',
        'High_^NSEI': 'High',
        'Low_^NSEI': 'Low',
        'Close_^NSEI': 'Close'
    })
    
    # Add EMA20
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # Initialize signal column
    df['Signal'] = np.nan
    
    # Step 1: Find 3:15 PM candle (previous day reference)
    df['Time'] = df['Datetime'].dt.time
    df['Date'] = df['Datetime'].dt.date
    
    reference_levels = {}  # Store {date: (open, close)}
    
    for d in df['Date'].unique():
        day_data = df[df['Date'] == d]
        # Find 3:15 PM candle (close at 15:15)
        ref_candle = day_data[day_data['Time'] == pd.to_datetime("15:15").time()]
        if not ref_candle.empty:
            o = ref_candle['Open'].values[0]
            c = ref_candle['Close'].values[0]
            reference_levels[d] = (o, c)
    
    # Step 2: Next day 9:30 AM breakout
    for d in reference_levels.keys():
        next_days = [x for x in df['Date'].unique() if x > d]
        if not next_days:
            continue
        next_day = next_days[0]
        
        first_candle = df[(df['Date'] == next_day) & (df['Time'] == pd.to_datetime("09:30").time())]
        if not first_candle.empty:
            open_val = reference_levels[d][0]
            close_val = reference_levels[d][1]
            fc = first_candle.iloc[0]
            
            if fc['High'] > max(open_val, close_val):  # Breakout above
                df.loc[first_candle.index, 'Signal'] = "Breakout_Up"
            elif fc['Low'] < min(open_val, close_val):  # Breakdown below
                df.loc[first_candle.index, 'Signal'] = "Breakout_Down"
    
    # Step 3: EMA20 + Volume condition (if volume exists)
    if 'Volume' in df.columns:
        df['Prev_Close'] = df['Close'].shift(1)
        df['Prev_EMA20'] = df['EMA20'].shift(1)
        
        vol_ma = df['Volume'].rolling(20).mean()
        
        for i in range(1, len(df)):
            if (
                df.loc[i, 'Close'] > df.loc[i, 'EMA20'] 
                and df.loc[i-1, 'Prev_Close'] <= df.loc[i-1, 'Prev_EMA20']
            ):
                if df.loc[i, 'Volume'] > vol_ma.iloc[i]:
                    df.loc[i, 'Signal'] = "EMA20_Buy"
            
            elif (
                df.loc[i, 'Close'] < df.loc[i, 'EMA20'] 
                and df.loc[i-1, 'Prev_Close'] >= df.loc[i-1, 'Prev_EMA20']
            ):
                if df.loc[i, 'Volume'] > vol_ma.iloc[i]:
                    df.loc[i, 'Signal'] = "EMA20_Sell"
        
        df = df.drop(columns=['Prev_Close','Prev_EMA20'])
    
    # Clean up temp cols
    df = df.drop(columns=['Time','Date'])
    
    return df




##################################################New function Change #######################################################

def trading_signal_all_conditions1_old_working(df, quantity=10*75, return_all_signals=False):
    """
    Evaluate trading conditions based on Base Zone strategy with modified stop loss logic:
    - CALL stop loss = recent swing low (before entry)
    - PUT stop loss = recent swing high (before entry)
    """

    signals = []
    spot_price = df['Close_^NSEI'].iloc[-1]
    df = df.copy()
    df['Date'] = df['Datetime'].dt.date
    unique_days = sorted(df['Date'].unique())
    if len(unique_days) < 2:
        return None

    day0 = unique_days[-2]
    day1 = unique_days[-1]

    candle_3pm = df[(df['Date'] == day0) &
                    (df['Datetime'].dt.hour == 15) &
                    (df['Datetime'].dt.minute == 0)]
    if candle_3pm.empty:
        return None

    base_open = candle_3pm.iloc[0]['Open_^NSEI']
    base_close = candle_3pm.iloc[0]['Close_^NSEI']
    base_low = min(base_open, base_close)
    base_high = max(base_open, base_close)

    candle_915 = df[(df['Date'] == day1) &
                    (df['Datetime'].dt.hour == 9) &
                    (df['Datetime'].dt.minute == 15)]
    if candle_915.empty:
        return None

    H1 = candle_915.iloc[0]['High_^NSEI']
    L1 = candle_915.iloc[0]['Low_^NSEI']
    C1 = candle_915.iloc[0]['Close_^NSEI']
    entry_time = candle_915.iloc[0]['Datetime']

    expiry = get_nearest_weekly_expiry(pd.to_datetime(day1))
    day1_after_915 = df[(df['Date'] == day1) & (df['Datetime'] > entry_time)].sort_values('Datetime')

    # Helper to find recent swing high and low before a given time
    def get_recent_swing(df, current_time):
        recent_data = df[(df['Datetime'] < current_time)].tail(10)  # last 10 candles before entry
        swing_high = recent_data['High_^NSEI'].max()
        swing_low = recent_data['Low_^NSEI'].min()
        return swing_high, swing_low

    # Get recent swing values before 09:15
    swing_high, swing_low = get_recent_swing(df[df['Date'] == day1], entry_time)

    # Condition 1: CALL breakout
    if (L1 < base_high and H1 > base_low) and (C1 > base_high):
        sig = {
            'condition': 1,
            'option_type': 'CALL',
            'buy_price': H1,
            'stoploss': swing_low,             # Updated stoploss
            'take_profit': H1 * 1.10,
            'quantity': quantity,
            'expiry': expiry,
            'entry_time': entry_time,
            'message': 'Condition 1: Bullish breakout above Base Zone → Buy CALL above H1',
            'spot_price': spot_price
        }
        signals.append(sig)
        if not return_all_signals:
            return sig

    if C1 < base_low:
        for _, next_candle in day1_after_915.iterrows():
            swing_high, swing_low = get_recent_swing(df[df['Date'] == day1], next_candle['Datetime'])

            # Condition 2: PUT continuation
            if next_candle['Low_^NSEI'] < L1:
                sig = {
                    'condition': 2,
                    'option_type': 'PUT',
                    'buy_price': L1,
                    'stoploss': swing_high,        # Updated stoploss
                    'take_profit': L1 * 0.90,
                    'quantity': quantity,
                    'expiry': expiry,
                    'entry_time': next_candle['Datetime'],
                    'message': 'Condition 2: Gap down confirmed → Buy PUT below L1',
                    'spot_price': spot_price
                }
                signals.append(sig)
                if not return_all_signals:
                    return sig

            # Flip rule 2.7: CALL
            if next_candle['Close_^NSEI'] > base_high:
                ref_high = next_candle['High_^NSEI']
                sig_flip = {
                    'condition': 2.7,
                    'option_type': 'CALL',
                    'buy_price': ref_high,
                    'stoploss': swing_low,        # Updated stoploss
                    'take_profit': ref_high * 1.10,
                    'quantity': quantity,
                    'expiry': expiry,
                    'entry_time': next_candle['Datetime'],
                    'message': 'Condition 2 Flip: Later candle closed above Base Zone → Buy CALL above Candle 2 high',
                    'spot_price': spot_price
                }
                signals.append(sig_flip)
                if not return_all_signals:
                    return sig_flip

    if C1 > base_high:
        for _, next_candle in day1_after_915.iterrows():
            swing_high, swing_low = get_recent_swing(df[df['Date'] == day1], next_candle['Datetime'])

            # Condition 3: CALL continuation
            if next_candle['High_^NSEI'] > H1:
                sig = {
                    'condition': 3,
                    'option_type': 'CALL',
                    'buy_price': H1,
                    'stoploss': swing_low,        # Updated stoploss
                    'take_profit': H1 * 1.10,
                    'quantity': quantity,
                    'expiry': expiry,
                    'entry_time': next_candle['Datetime'],
                    'message': 'Condition 3: Gap up confirmed → Buy CALL above H1',
                    'spot_price': spot_price
                }
                signals.append(sig)
                if not return_all_signals:
                    return sig

            # Flip rule 3.7: PUT
            if next_candle['Close_^NSEI'] < base_low:
                ref_low = next_candle['Low_^NSEI']
                sig_flip = {
                    'condition': 3.7,
                    'option_type': 'PUT',
                    'buy_price': ref_low,
                    'stoploss': swing_high,       # Updated stoploss
                    'take_profit': ref_low * 0.90,
                    'quantity': quantity,
                    'expiry': expiry,
                    'entry_time': next_candle['Datetime'],
                    'message': 'Condition 3 Flip: Later candle closed below Base Zone → Buy PUT below Candle 3 low',
                    'spot_price': spot_price
                }
                signals.append(sig_flip)
                if not return_all_signals:
                    return sig_flip

    # Condition 4: PUT breakdown
    if (L1 < base_high and H1 > base_low) and (C1 < base_low):
        sig = {
            'condition': 4,
            'option_type': 'PUT',
            'buy_price': L1,
            'stoploss': swing_high,             # Updated stoploss
            'take_profit': L1 * 0.90,
            'quantity': quantity,
            'expiry': expiry,
            'entry_time': entry_time,
            'message': 'Condition 4: Bearish breakdown below Base Zone → Buy PUT below L1',
            'spot_price': spot_price
        }
        signals.append(sig)
        if not return_all_signals:
            return sig

    return signals if signals else None
#######################################################################################################################################

def trading_signal_all_conditions1_trail_sl(df, quantity=10*75, return_all_signals=False):
    """
    Evaluate trading conditions based on Base Zone strategy with Trailing Stop Loss:
    - CALL stop loss = recent swing low (before and after entry, updated dynamically)
    - PUT stop loss = recent swing high (before and after entry, updated dynamically)
    """

    signals = []
    spot_price = df['Close_^NSEI'].iloc[-1]
    df = df.copy()
    df['Date'] = df['Datetime'].dt.date
    unique_days = sorted(df['Date'].unique())
    if len(unique_days) < 2:
        return None

    day0 = unique_days[-2]
    day1 = unique_days[-1]

    candle_3pm = df[(df['Date'] == day0) &
                    (df['Datetime'].dt.hour == 15) &
                    (df['Datetime'].dt.minute == 0)]
    if candle_3pm.empty:
        return None

    base_open = candle_3pm.iloc[0]['Open_^NSEI']
    base_close = candle_3pm.iloc[0]['Close_^NSEI']
    base_low = min(base_open, base_close)
    base_high = max(base_open, base_close)

    candle_915 = df[(df['Date'] == day1) &
                    (df['Datetime'].dt.hour == 9) &
                    (df['Datetime'].dt.minute == 15)]
    if candle_915.empty:
        return None

    H1 = candle_915.iloc[0]['High_^NSEI']
    L1 = candle_915.iloc[0]['Low_^NSEI']
    C1 = candle_915.iloc[0]['Close_^NSEI']
    entry_time = candle_915.iloc[0]['Datetime']

    expiry = get_nearest_weekly_expiry(pd.to_datetime(day1))
    day1_after_915 = df[(df['Date'] == day1) & (df['Datetime'] > entry_time)].sort_values('Datetime')

    # Helper to find recent swing high and low before a given time
    def get_recent_swing(df, current_time):
        recent_data = df[(df['Datetime'] < current_time)].tail(10)  # last 10 candles
        swing_high = recent_data['High_^NSEI'].max()
        swing_low = recent_data['Low_^NSEI'].min()
        return swing_high, swing_low

    # Initial recent swing values before 09:15
    swing_high, swing_low = get_recent_swing(df[df['Date'] == day1], entry_time)

    # Condition 1: CALL breakout
    if (L1 < base_high and H1 > base_low) and (C1 > base_high):
        sig = {
            'condition': 1,
            'option_type': 'CALL',
            'buy_price': H1,
            'stoploss': swing_low,  # initial SL
            'take_profit': H1 * 1.10,
            'quantity': quantity,
            'expiry': expiry,
            'entry_time': entry_time,
            'message': 'Condition 1: Bullish breakout above Base Zone → Buy CALL above H1',
            'spot_price': spot_price
        }

        # Trailing Stop Loss for CALL
        for _, candle in day1_after_915.iterrows():
            new_swing_high, new_swing_low = get_recent_swing(df[df['Date'] == day1], candle['Datetime'])
            if new_swing_low > sig['stoploss']:
                sig['stoploss'] = new_swing_low  # trail SL upward
        signals.append(sig)
        if not return_all_signals:
            return sig

    if C1 < base_low:
        for _, next_candle in day1_after_915.iterrows():
            swing_high, swing_low = get_recent_swing(df[df['Date'] == day1], next_candle['Datetime'])

            # Condition 2: PUT continuation
            if next_candle['Low_^NSEI'] < L1:
                sig = {
                    'condition': 2,
                    'option_type': 'PUT',
                    'buy_price': L1,
                    'stoploss': swing_high,  # initial SL
                    'take_profit': L1 * 0.90,
                    'quantity': quantity,
                    'expiry': expiry,
                    'entry_time': next_candle['Datetime'],
                    'message': 'Condition 2: Gap down confirmed → Buy PUT below L1',
                    'spot_price': spot_price
                }

                # Trailing Stop Loss for PUT
                for _, candle in day1_after_915.iterrows():
                    new_swing_high, new_swing_low = get_recent_swing(df[df['Date'] == day1], candle['Datetime'])
                    if new_swing_high < sig['stoploss']:
                        sig['stoploss'] = new_swing_high  # trail SL downward
                signals.append(sig)
                if not return_all_signals:
                    return sig

            # Flip rule 2.7: CALL
            if next_candle['Close_^NSEI'] > base_high:
                ref_high = next_candle['High_^NSEI']
                sig_flip = {
                    'condition': 2.7,
                    'option_type': 'CALL',
                    'buy_price': ref_high,
                    'stoploss': swing_low,
                    'take_profit': ref_high * 1.10,
                    'quantity': quantity,
                    'expiry': expiry,
                    'entry_time': next_candle['Datetime'],
                    'message': 'Condition 2 Flip: Later candle closed above Base Zone → Buy CALL above Candle 2 high',
                    'spot_price': spot_price
                }

                # Trailing Stop Loss for CALL flip
                for _, candle in day1_after_915.iterrows():
                    new_swing_high, new_swing_low = get_recent_swing(df[df['Date'] == day1], candle['Datetime'])
                    if new_swing_low > sig_flip['stoploss']:
                        sig_flip['stoploss'] = new_swing_low
                signals.append(sig_flip)
                if not return_all_signals:
                    return sig_flip

    if C1 > base_high:
        for _, next_candle in day1_after_915.iterrows():
            swing_high, swing_low = get_recent_swing(df[df['Date'] == day1], next_candle['Datetime'])

            # Condition 3: CALL continuation
            if next_candle['High_^NSEI'] > H1:
                sig = {
                    'condition': 3,
                    'option_type': 'CALL',
                    'buy_price': H1,
                    'stoploss': swing_low,
                    'take_profit': H1 * 1.10,
                    'quantity': quantity,
                    'expiry': expiry,
                    'entry_time': next_candle['Datetime'],
                    'message': 'Condition 3: Gap up confirmed → Buy CALL above H1',
                    'spot_price': spot_price
                }

                # Trailing Stop Loss for CALL
                for _, candle in day1_after_915.iterrows():
                    new_swing_high, new_swing_low = get_recent_swing(df[df['Date'] == day1], candle['Datetime'])
                    if new_swing_low > sig['stoploss']:
                        sig['stoploss'] = new_swing_low
                signals.append(sig)
                if not return_all_signals:
                    return sig

            # Flip rule 3.7: PUT
            if next_candle['Close_^NSEI'] < base_low:
                ref_low = next_candle['Low_^NSEI']
                sig_flip = {
                    'condition': 3.7,
                    'option_type': 'PUT',
                    'buy_price': ref_low,
                    'stoploss': swing_high,
                    'take_profit': ref_low * 0.90,
                    'quantity': quantity,
                    'expiry': expiry,
                    'entry_time': next_candle['Datetime'],
                    'message': 'Condition 3 Flip: Later candle closed below Base Zone → Buy PUT below Candle 3 low',
                    'spot_price': spot_price
                }

                # Trailing Stop Loss for PUT flip
                for _, candle in day1_after_915.iterrows():
                    new_swing_high, new_swing_low = get_recent_swing(df[df['Date'] == day1], candle['Datetime'])
                    if new_swing_high < sig_flip['stoploss']:
                        sig_flip['stoploss'] = new_swing_high
                signals.append(sig_flip)
                if not return_all_signals:
                    return sig_flip

    # Condition 4: PUT breakdown
    if (L1 < base_high and H1 > base_low) and (C1 < base_low):
        sig = {
            'condition': 4,
            'option_type': 'PUT',
            'buy_price': L1,
            'stoploss': swing_high,
            'take_profit': L1 * 0.90,
            'quantity': quantity,
            'expiry': expiry,
            'entry_time': entry_time,
            'message': 'Condition 4: Bearish breakdown below Base Zone → Buy PUT below L1',
            'spot_price': spot_price
        }

        # Trailing Stop Loss for PUT
        for _, candle in day1_after_915.iterrows():
            new_swing_high, new_swing_low = get_recent_swing(df[df['Date'] == day1], candle['Datetime'])
            if new_swing_high < sig['stoploss']:
                sig['stoploss'] = new_swing_high
        signals.append(sig)
        if not return_all_signals:
            return sig

    return signals if signals else None

######################################################take profit with trailing SL #######################################################################################




def trading_signal_all_conditions1_changes_working(df, quantity=10*75, return_all_signals=False):
    """
    Evaluate trading conditions based on Base Zone strategy with:
    - CALL stop loss = recent swing low (last 10 candles)
    - PUT stop loss = recent swing high (last 10 candles)
    - Dynamic trailing stop loss based on swing points
    - Take Profit triggered by trailing SL (exit when price hits SL)
    """

    signals = []
    spot_price = df['Close_^NSEI'].iloc[-1]
    df = df.copy()
    df['Date'] = df['Datetime'].dt.date
    unique_days = sorted(df['Date'].unique())
    if len(unique_days) < 2:
        return None

    day0 = unique_days[-2]  # Previous day
    day1 = unique_days[-1]  # Current day

    # Get Base Zone from 3 PM candle of previous day
    candle_3pm = df[(df['Date'] == day0) &
                    (df['Datetime'].dt.hour == 15) &
                    (df['Datetime'].dt.minute == 0)]
    if candle_3pm.empty:
        return None

    base_open = candle_3pm.iloc[0]['Open_^NSEI']
    base_close = candle_3pm.iloc[0]['Close_^NSEI']
    base_low = min(base_open, base_close)
    base_high = max(base_open, base_close)

    # Get 09:15 candle of current day
    candle_915 = df[(df['Date'] == day1) &
                    (df['Datetime'].dt.hour == 9) &
                    (df['Datetime'].dt.minute == 15)]
    if candle_915.empty:
        return None

    H1 = candle_915.iloc[0]['High_^NSEI']
    L1 = candle_915.iloc[0]['Low_^NSEI']
    C1 = candle_915.iloc[0]['Close_^NSEI']
    entry_time = candle_915.iloc[0]['Datetime']

    expiry = get_nearest_weekly_expiry(pd.to_datetime(day1))
    day1_after_915 = df[(df['Date'] == day1) & (df['Datetime'] > entry_time)].sort_values('Datetime')

    # Helper: Get recent swing points from last 10 candles
    def get_recent_swing(current_time):
        recent_data = df[(df['Date'] == day1) & (df['Datetime'] < current_time)].tail(10)
        swing_high = recent_data['High_^NSEI'].max()
        swing_low = recent_data['Low_^NSEI'].min()
        return swing_high, swing_low

    # Helper: Update trailing stop loss
    def update_trailing_sl(option_type, current_sl, current_time):
        new_high, new_low = get_recent_swing(current_time)
        if option_type == 'CALL':  # SL below recent swing low
            if new_low > current_sl:
                return new_low
        elif option_type == 'PUT':  # SL above recent swing high
            if new_high < current_sl:
                return new_high
        return current_sl

    # Common function to monitor trade after entry
    def monitor_trade(sig):
        current_sl = sig['stoploss']
        for _, candle in day1_after_915.iterrows():
            # Update trailing stop loss dynamically
            current_sl = update_trailing_sl(sig['option_type'], current_sl, candle['Datetime'])
            sig['stoploss'] = current_sl

            # Exit if price hits SL
            if sig['option_type'] == 'CALL' and candle['Low_^NSEI'] <= current_sl:
                sig['exit_price'] = current_sl
                sig['status'] = 'Exited at Trailing SL'
                break
            elif sig['option_type'] == 'PUT' and candle['High_^NSEI'] >= current_sl:
                sig['exit_price'] = current_sl
                sig['status'] = 'Exited at Trailing SL'
                break
        return sig

    # Condition 1: CALL breakout
    if (L1 < base_high and H1 > base_low) and (C1 > base_high):
        sig = {
            'condition': 1,
            'option_type': 'CALL',
            'buy_price': H1,
            'stoploss': get_recent_swing(entry_time)[1],  # swing low
            'quantity': quantity,
            'expiry': expiry,
            'entry_time': entry_time,
            'message': 'Condition 1: Bullish breakout above Base Zone → Buy CALL above H1',
            'spot_price': spot_price
        }
        sig = monitor_trade(sig)
        signals.append(sig)
        if not return_all_signals:
            return sig

    # Condition 2: PUT continuation after gap down
    if C1 < base_low:
        for _, next_candle in day1_after_915.iterrows():
            swing_high, swing_low = get_recent_swing(next_candle['Datetime'])
            if next_candle['Low_^NSEI'] < L1:
                sig = {
                    'condition': 2,
                    'option_type': 'PUT',
                    'buy_price': L1,
                    'stoploss': swing_high,  # swing high
                    'quantity': quantity,
                    'expiry': expiry,
                    'entry_time': next_candle['Datetime'],
                    'message': 'Condition 2: Gap down confirmed → Buy PUT below L1',
                    'spot_price': spot_price
                }
                sig = monitor_trade(sig)
                signals.append(sig)
                if not return_all_signals:
                    return sig

    # Condition 3: CALL continuation after gap up
    if C1 > base_high:
        for _, next_candle in day1_after_915.iterrows():
            swing_high, swing_low = get_recent_swing(next_candle['Datetime'])
            if next_candle['High_^NSEI'] > H1:
                sig = {
                    'condition': 3,
                    'option_type': 'CALL',
                    'buy_price': H1,
                    'stoploss': swing_low,
                    'quantity': quantity,
                    'expiry': expiry,
                    'entry_time': next_candle['Datetime'],
                    'message': 'Condition 3: Gap up confirmed → Buy CALL above H1',
                    'spot_price': spot_price
                }
                sig = monitor_trade(sig)
                signals.append(sig)
                if not return_all_signals:
                    return sig

    # Condition 4: PUT breakdown
    if (L1 < base_high and H1 > base_low) and (C1 < base_low):
        sig = {
            'condition': 4,
            'option_type': 'PUT',
            'buy_price': L1,
            'stoploss': get_recent_swing(entry_time)[0],  # swing high
            'quantity': quantity,
            'expiry': expiry,
            'entry_time': entry_time,
            'message': 'Condition 4: Bearish breakdown below Base Zone → Buy PUT below L1',
            'spot_price': spot_price
        }
        sig = monitor_trade(sig)
        signals.append(sig)
        if not return_all_signals:
            return sig

    return signals if signals else None

###############################################Here’s the complete implementation of your described logic with all four conditions, proper entry, stop loss rules,
############################trailing stop loss, and time-based exit (after 16 minutes). The function will return either a single trade
####################################signal or all signals based on return_all_signals.#######################################################################################

import numpy as np
import pandas as pd
from datetime import timedelta

def trading_signal_all_conditions1(df, quantity=10*75, return_all_signals=False):
    """
    Evaluate trading conditions based on Base Zone strategy with:
    - CALL stop loss = recent swing low (last 10 candles)
    - PUT stop loss = recent swing high (last 10 candles)
    - Dynamic trailing stop loss based on swing points
    - Time exit after 16 minutes if neither SL nor trailing SL hit
    - Single active trade per day
    """

    signals = []
    spot_price = df['Close_^NSEI'].iloc[-1]

    # Preprocess
    df = df.copy()
    df['Date'] = df['Datetime'].dt.date
    unique_days = sorted(df['Date'].unique())
    if len(unique_days) < 2:
        return None

    # Day 0 and Day 1
    day0 = unique_days[-2]  # Previous trading day
    day1 = unique_days[-1]  # Current trading day

    # Get Base Zone from 3 PM candle of previous day
    candle_3pm = df[(df['Date'] == day0) &
                    (df['Datetime'].dt.hour == 15) &
                    (df['Datetime'].dt.minute == 0)]
    if candle_3pm.empty:
        return None

    base_open = candle_3pm.iloc[0]['Open_^NSEI']
    base_close = candle_3pm.iloc[0]['Close_^NSEI']
    base_low = min(base_open, base_close)
    base_high = max(base_open, base_close)

    # Get 09:15–09:30 candle of current day
    candle_915 = df[(df['Date'] == day1) &
                    (df['Datetime'].dt.hour == 9) &
                    (df['Datetime'].dt.minute == 15)]
    if candle_915.empty:
        return None

    H1 = candle_915.iloc[0]['High_^NSEI']
    L1 = candle_915.iloc[0]['Low_^NSEI']
    C1 = candle_915.iloc[0]['Close_^NSEI']
    entry_time = candle_915.iloc[0]['Datetime']

    expiry = get_nearest_weekly_expiry(pd.to_datetime(day1))

    # Data after 09:30
    day1_after_915 = df[(df['Date'] == day1) & (df['Datetime'] > entry_time)].sort_values('Datetime')

    # Helper functions
    def get_recent_swing(current_time):
        """
        Return scalar swing_high, swing_low from last 10 candles before current_time.
        If insufficient data return (np.nan, np.nan).
        """
        recent_data = df[(df['Date'] == day1) & (df['Datetime'] < current_time)].tail(10)
        if recent_data.empty:
            return np.nan, np.nan
        # ensure scalar float values (not Series)
        swing_high = recent_data['High_^NSEI'].max()
        swing_low = recent_data['Low_^NSEI'].min()
        # convert numpy scalars to python floats when possible
        swing_high = float(swing_high) if not pd.isna(swing_high) else np.nan
        swing_low = float(swing_low) if not pd.isna(swing_low) else np.nan
        return swing_high, swing_low

    def update_trailing_sl(option_type, current_sl, current_time):
        """
        Safely update trailing SL using last-10-candle swing points.
        - For CALL: SL tracks the most recent swing_low (move up only)
        - For PUT: SL tracks the most recent swing_high (move down only)
        """
        new_high, new_low = get_recent_swing(current_time)

        # CALL: set/raise SL to new_low if valid
        if option_type == 'CALL':
            if pd.isna(new_low):
                # nothing to update
                return current_sl
            # if current_sl is missing, initialize it
            if current_sl is None or pd.isna(current_sl):
                return new_low
            # update only if new_low is higher than current_sl (trail upward)
            if new_low > current_sl:
                return new_low
            return current_sl

        # PUT: set/lower SL to new_high if valid
        if option_type == 'PUT':
            if pd.isna(new_high):
                return current_sl
            if current_sl is None or pd.isna(current_sl):
                return new_high
            # update only if new_high is lower than current_sl (trail downward)
            if new_high < current_sl:
                return new_high
            return current_sl

        return current_sl

    def monitor_trade(sig):
        """
        Monitor trade after entry:
        - update trailing SL every new 15-min candle
        - exit when SL is hit or when 16 minutes passed since entry (time exit)
        - safe handling when there are no monitoring candles
        """
        current_sl = sig.get('stoploss', None)
        entry_dt = sig['entry_time']
        exit_deadline = entry_dt + timedelta(minutes=16)

        # if there are no candles to monitor, exit immediately at entry (safe fallback)
        if day1_after_915.empty:
            sig['exit_price'] = sig.get('buy_price', spot_price)
            sig['status'] = 'No candles to monitor - exited'
            return sig

        exited = False
        for _, candle in day1_after_915.iterrows():
            # Time exit check (exit at or after deadline)
            if candle['Datetime'] >= exit_deadline:
                # Exit at market (use candle close as approximation of market exit)
                sig['exit_price'] = candle['Close_^NSEI']
                sig['status'] = 'Exited due to time limit'
                exited = True
                break

            # Update trailing SL safely
            current_sl = update_trailing_sl(sig['option_type'], current_sl, candle['Datetime'])
            sig['stoploss'] = current_sl

            # Only check SL-hit if SL is a valid numeric value
            if sig['option_type'] == 'CALL' and pd.notna(current_sl):
                if candle['Low_^NSEI'] <= current_sl:
                    sig['exit_price'] = current_sl
                    sig['status'] = 'Exited at Trailing SL'
                    exited = True
                    break
            elif sig['option_type'] == 'PUT' and pd.notna(current_sl):
                if candle['High_^NSEI'] >= current_sl:
                    sig['exit_price'] = current_sl
                    sig['status'] = 'Exited at Trailing SL'
                    exited = True
                    break

        # If not exited in loop, set EOD exit (or last candle close)
        if not exited:
            last_close = day1_after_915.iloc[-1]['Close_^NSEI']
            sig['exit_price'] = last_close
            sig['status'] = 'Exited at EOD/no SL hit'

        return sig

    # Condition 1 — Break above Base Zone (CALL)
    if (L1 < base_high and H1 > base_low) and (C1 > base_high):
        swing_high, swing_low = get_recent_swing(entry_time)
        sig = {
            'condition': 1,
            'option_type': 'CALL',
            'buy_price': H1,
            'stoploss': swing_low,  # may be np.nan if insufficient history
            'quantity': quantity,
            'expiry': expiry,
            'entry_time': entry_time,
            'message': 'Condition 1: Bullish breakout above Base Zone → Buy CALL above H1',
            'spot_price': spot_price
        }
        sig = monitor_trade(sig)
        signals.append(sig)
        if not return_all_signals:
            return sig

    # Condition 2 — Major Gap Down (PUT) and flip 2.7
    if C1 < base_low:
        for _, next_candle in day1_after_915.iterrows():
            swing_high, swing_low = get_recent_swing(next_candle['Datetime'])
            # Primary PUT entry on break below L1
            if next_candle['Low_^NSEI'] <= L1:
                sig = {
                    'condition': 2,
                    'option_type': 'PUT',
                    'buy_price': L1,
                    'stoploss': swing_high,
                    'quantity': quantity,
                    'expiry': expiry,
                    'entry_time': next_candle['Datetime'],
                    'message': 'Condition 2: Gap down confirmed → Buy PUT below L1',
                    'spot_price': spot_price
                }
                sig = monitor_trade(sig)
                signals.append(sig)
                if not return_all_signals:
                    return sig

            # Flip rule 2.7: bullish recovery -> CALL
            if next_candle['Close_^NSEI'] > base_high:
                ref_high = next_candle['High_^NSEI']
                sig_flip = {
                    'condition': 2.7,
                    'option_type': 'CALL',
                    'buy_price': ref_high,
                    'stoploss': swing_low,
                    'quantity': quantity,
                    'expiry': expiry,
                    'entry_time': next_candle['Datetime'],
                    'message': 'Condition 2 Flip: Later candle closed above Base Zone → Buy CALL',
                    'spot_price': spot_price
                }
                sig_flip = monitor_trade(sig_flip)
                signals.append(sig_flip)
                if not return_all_signals:
                    return sig_flip

    # Condition 3 — Major Gap Up (CALL) and flip 3.7
    if C1 > base_high:
        for _, next_candle in day1_after_915.iterrows():
            swing_high, swing_low = get_recent_swing(next_candle['Datetime'])
            if next_candle['High_^NSEI'] >= H1:
                sig = {
                    'condition': 3,
                    'option_type': 'CALL',
                    'buy_price': H1,
                    'stoploss': swing_low,
                    'quantity': quantity,
                    'expiry': expiry,
                    'entry_time': next_candle['Datetime'],
                    'message': 'Condition 3: Gap up confirmed → Buy CALL above H1',
                    'spot_price': spot_price
                }
                sig = monitor_trade(sig)
                signals.append(sig)
                if not return_all_signals:
                    return sig

            # Flip rule 3.7: bearish recovery -> PUT
            if next_candle['Close_^NSEI'] < base_low:
                ref_low = next_candle['Low_^NSEI']
                sig_flip = {
                    'condition': 3.7,
                    'option_type': 'PUT',
                    'buy_price': ref_low,
                    'stoploss': swing_high,
                    'quantity': quantity,
                    'expiry': expiry,
                    'entry_time': next_candle['Datetime'],
                    'message': 'Condition 3 Flip: Later candle closed below Base Zone → Buy PUT',
                    'spot_price': spot_price
                }
                sig_flip = monitor_trade(sig_flip)
                signals.append(sig_flip)
                if not return_all_signals:
                    return sig_flip

    # Condition 4 — Break below Base Zone on Day 1 open (PUT)
    if (L1 < base_high and H1 > base_low) and (C1 < base_low):
        swing_high, swing_low = get_recent_swing(entry_time)
        sig = {
            'condition': 4,
            'option_type': 'PUT',
            'buy_price': L1,
            'stoploss': swing_high,
            'quantity': quantity,
            'expiry': expiry,
            'entry_time': entry_time,
            'message': 'Condition 4: Bearish breakdown below Base Zone → Buy PUT below L1',
            'spot_price': spot_price
        }
        sig = monitor_trade(sig)
        signals.append(sig)
        if not return_all_signals:
            return sig

    return signals if signals else None


###########################################################################################################################################


def trading_signal_all_conditions1_old(df, quantity=10*75, return_all_signals=False):
    """
    Evaluate trading conditions based on Base Zone strategy.

    Parameters:
    - df: pd.DataFrame with columns ['Datetime','Open_^NSEI','High_^NSEI','Low_^NSEI','Close_^NSEI']
    - quantity: total quantity to trade (default 10 lots = 750)
    - return_all_signals: if True, returns list of all possible signals; else returns first found

    Returns:
    - None if no signals
    - dict for first signal (if return_all_signals=False)
    - list of dicts for all signals (if return_all_signals=True)
    """

    signals = []
    spot_price = df['Close_^NSEI'].iloc[-1]  # latest price
    df = df.copy()
    df['Date'] = df['Datetime'].dt.date
    unique_days = sorted(df['Date'].unique())
    if len(unique_days) < 2:
        return None  # not enough data

    day0 = unique_days[-2]
    day1 = unique_days[-1]

    # Step 1: Base Zone
    candle_3pm = df[(df['Date'] == day0) &
                    (df['Datetime'].dt.hour == 15) &
                    (df['Datetime'].dt.minute == 0)]
    if candle_3pm.empty:
        return None

    base_open = candle_3pm.iloc[0]['Open_^NSEI']
    base_close = candle_3pm.iloc[0]['Close_^NSEI']
    base_low = min(base_open, base_close)
    base_high = max(base_open, base_close)

    # Step 2: First candle Day1
    candle_915 = df[(df['Date'] == day1) &
                    (df['Datetime'].dt.hour == 9) &
                    (df['Datetime'].dt.minute == 15)]
    if candle_915.empty:
        return None

    H1 = candle_915.iloc[0]['High_^NSEI']
    L1 = candle_915.iloc[0]['Low_^NSEI']
    C1 = candle_915.iloc[0]['Close_^NSEI']
    entry_time = candle_915.iloc[0]['Datetime']

    expiry = get_nearest_weekly_expiry(pd.to_datetime(day1))

    day1_after_915 = df[(df['Date'] == day1) & (df['Datetime'] > entry_time)].sort_values('Datetime')

    # --- Condition 1 ---
    if (L1 < base_high and H1 > base_low) and (C1 > base_high):
        sig = {
            'condition': 1,
            'option_type': 'CALL',
            'buy_price': H1,
            'stoploss': H1 * 0.9,
            'take_profit': H1 * 1.10,
            'quantity': quantity,
            'expiry': expiry,
            'entry_time': entry_time,
            'message': 'Condition 1: Bullish breakout above Base Zone → Buy CALL above H1',
            'spot_price': spot_price
        }
        signals.append(sig)
        if not return_all_signals:
            return sig

    # --- Condition 2 ---
    if C1 < base_low:
        for _, next_candle in day1_after_915.iterrows():
            if next_candle['Low_^NSEI'] < L1:
                sig = {
                    'condition': 2,
                    'option_type': 'PUT',
                    'buy_price': L1,
                    'stoploss': L1 * 1.10,
                    'take_profit': L1 * 0.90,
                    'quantity': quantity,
                    'expiry': expiry,
                    'entry_time': next_candle['Datetime'],
                    'message': 'Condition 2: Gap down confirmed → Buy PUT below L1',
                    'spot_price': spot_price
                }
                signals.append(sig)
                if not return_all_signals:
                    return sig
            # Flip rule 2.7
            if next_candle['Close_^NSEI'] > base_high:
                ref_high = next_candle['High_^NSEI']
                sig_flip = {
                    'condition': 2.7,
                    'option_type': 'CALL',
                    'buy_price': ref_high,
                    'stoploss': ref_high * 0.9,
                    'take_profit': ref_high * 1.10,
                    'quantity': quantity,
                    'expiry': expiry,
                    'entry_time': next_candle['Datetime'],
                    'message': 'Condition 2 Flip: Later candle closed above Base Zone → Buy CALL above Candle 2 high',
                    'spot_price': spot_price
                }
                signals.append(sig_flip)
                if not return_all_signals:
                    return sig_flip

    # --- Condition 3 ---
    if C1 > base_high:
        for _, next_candle in day1_after_915.iterrows():
            if next_candle['High_^NSEI'] > H1:
                sig = {
                    'condition': 3,
                    'option_type': 'CALL',
                    'buy_price': H1,
                    'stoploss': H1 * 0.9,
                    'take_profit': H1 * 1.10,
                    'quantity': quantity,
                    'expiry': expiry,
                    'entry_time': next_candle['Datetime'],
                    'message': 'Condition 3: Gap up confirmed → Buy CALL above H1',
                    'spot_price': spot_price
                }
                signals.append(sig)
                if not return_all_signals:
                    return sig
            # Flip rule 3.7
            if next_candle['Close_^NSEI'] < base_low:
                ref_low = next_candle['Low_^NSEI']
                sig_flip = {
                    'condition': 3.7,
                    'option_type': 'PUT',
                    'buy_price': ref_low,
                    'stoploss': ref_low * 1.10,
                    'take_profit': ref_low * 0.90,
                    'quantity': quantity,
                    'expiry': expiry,
                    'entry_time': next_candle['Datetime'],
                    'message': 'Condition 3 Flip: Later candle closed below Base Zone → Buy PUT below Candle 3 low',
                    'spot_price': spot_price
                }
                signals.append(sig_flip)
                if not return_all_signals:
                    return sig_flip
                    sig
                    {
            'take_profit': L1 * 0.90,
            'quantity': quantity,
            'expiry': expiry,
            'entry_time': entry_time,
            'message': 'Condition 4: Bearish breakdown below Base Zone → Buy PUT below L1',
            'spot_price': spot_price
                    }
        signals.append(sig)
        if not return_all_signals:
            return sig

    return signals if signals else None

###################################################################################################################################################


def compute_trade_pnl_old(signal_log_df, df):
    """
    Compute PnL and exit reason for each signal in signal_log_df based on price data in df.
    Returns updated DataFrame with Sell Price, PnL, and Exit Reason.
    """
    trade_results = []

    for _, row in signal_log_df.iterrows():
        day = row['Date']
        entry_time = row['Entry Time']
        exit_time = row['Time Exit (16 mins after entry)']
        buy_premium = row['Buy Premium']
        qty = row['Quantity']
        stoploss = row['Stoploss (Trailing 10%)']
        take_profit = row['Take Profit (10% rise)']
        option_type = row['Option Selected']

        # Filter df for the trading day and after entry

    # --- Condition 4 ---
    if (L1 < base_high and H1 > base_low) and (C1 < base_low):
        sig = {
            'condition': 4,
            'option_type': 'PUT',
            'buy_price': L1,
            'stoploss': L1 * 1.10
        }
        day_df = df[df['Datetime'].dt.date == day]
        day_after_entry = day_df[day_df['Datetime'] >= entry_time].sort_values('Datetime')

        sell_price = None
        exit_reason = "Time Exit"

        for _, candle in day_after_entry.iterrows():
            price = candle['Close_^NSEI']  # You can use option price if available
            # Check Take Profit
            if take_profit and price >= take_profit:
                sell_price = take_profit
                exit_reason = "Take Profit"
                exit_time = candle['Datetime']
                break
            # Check Stoploss
            elif stoploss and price <= stoploss:
                sell_price = stoploss
                exit_reason = "Stoploss"
                exit_time = candle['Datetime']
                break

        # If neither TP nor SL hit, sell at last available price (time exit)
        if sell_price is None:
            sell_price = day_after_entry['Close_^NSEI'].iloc[0]  # fallback

        pnl = (sell_price - buy_premium) * qty if option_type.upper() == "CE" else (buy_premium - sell_price) * qty

        trade_results.append({
            **row.to_dict(),
            "Sell Price": sell_price,
            "Exit Reason": exit_reason,
            "Actual Exit Time": exit_time,
            "PnL": pnl
        })

    return pd.DataFrame(trade_results)

###############################################################################################

def compute_trade_pnl(signal_log_df, df):
    """
    Compute PnL and exit reason for each signal in signal_log_df based on price data in df.
    Returns updated DataFrame with Sell Price, PnL, and Exit Reason.
    """
    trade_results = []

    for _, row in signal_log_df.iterrows():
        day = row['Date']
        entry_time = row['Entry Time']
        exit_time = row['Time Exit (16 mins after entry)']
        buy_premium = row['Buy Premium']
        qty = row['Quantity']
        stoploss = row['Stoploss (Trailing 10%)']
        take_profit = row['Take Profit (10% rise)']
        option_type = row['Option Selected']

        # Filter df for the trading day and after entry
        day_df = df[df['Datetime'].dt.date == day]
        day_after_entry = day_df[(day_df['Datetime'] >= entry_time) & (day_df['Datetime'] <= exit_time)].sort_values('Datetime')

        sell_price = None
        actual_exit_time = exit_time
        exit_reason = "Time Exit"

        for _, candle in day_after_entry.iterrows():
            price = candle['Close_^NSEI']  # Spot price used for simulation; replace with option price if available
            
            # Check Take Profit for CALL or PUT
            if take_profit and (
                (option_type.upper() == "CE" and price >= take_profit) or
                (option_type.upper() == "PE" and price <= take_profit)
            ):
                sell_price = take_profit
                exit_reason = "Take Profit"
                actual_exit_time = candle['Datetime']
                break

            # Check Stoploss
            elif stoploss and (
                (option_type.upper() == "CE" and price <= stoploss) or
                (option_type.upper() == "PE" and price >= stoploss)
            ):
                sell_price = stoploss
                exit_reason = "Stoploss"
                actual_exit_time = candle['Datetime']
                break

        # If neither TP nor SL hit, exit at last available price (time exit)
        if sell_price is None:
            sell_price = day_after_entry['Close_^NSEI'].iloc[-1]

        # Compute PnL
        pnl = (sell_price - buy_premium) * qty if option_type.upper() == "CE" else (buy_premium - sell_price) * qty

        trade_results.append({
            **row.to_dict(),
            "Sell Price": sell_price,
            "Exit Reason": exit_reason,
            "Actual Exit Time": actual_exit_time,
            "PnL": pnl
        })

    return pd.DataFrame(trade_results)




#####################################################################################


def compute_performance1(signal_df, brokerage_per_trade=20, gst_rate=0.18, stamp_duty_rate=0.00015):
    """
    Compute performance summary from signal log with PnL and include daily costs.
    
    Returns:
    - summary_df: Overall performance summary (including Total Expense)
    - pnl_per_day: Daily PnL with Total PnL, Net Expense, and Net PnL
    """
    day_capital_needed=0
    total_trades = len(signal_df)
    winning_trades = signal_df[signal_df['PnL'] > 0]
    losing_trades = signal_df[signal_df['PnL'] <= 0]
    
    total_pnl = signal_df['PnL'].sum()
    avg_pnl = signal_df['PnL'].mean() if total_trades > 0 else 0
    max_pnl = signal_df['PnL'].max() if total_trades > 0 else 0
    min_pnl = signal_df['PnL'].min() if total_trades > 0 else 0
    
    win_pct = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
    loss_pct = len(losing_trades) / total_trades * 100 if total_trades > 0 else 0
    
    # ✅ Group by date and calculate daily metrics
    pnl_per_day = signal_df.groupby('Date').agg({
        'PnL': 'sum',
        'Quantity': 'sum'
    }).reset_index()
    
    # ✅ Add Net Expense and Net PnL
    cost_per_trade_list = []
    net_pnl_list = []
    capital_needed_list = []
    capital_after_list = []
    
    for idx, row in pnl_per_day.iterrows():
        day_trades = signal_df[signal_df['Date'] == row['Date']]
        day_expense = 0
        
        for _, trade in day_trades.iterrows():
            turnover = trade['Buy Premium'] * trade['Quantity'] * 2  # Buy + Sell
            brokerage = brokerage_per_trade
            gst = brokerage * gst_rate
            stamp_duty = turnover * stamp_duty_rate
            total_cost = brokerage + gst + stamp_duty
            day_expense += total_cost

             # Calculate capital needed for each trade
            trade_capital = trade['Buy Premium'] * trade['Quantity']
            day_capital_needed += trade_capital
        
        cost_per_trade_list.append(round(day_expense, 2))
        net_pnl_list.append(round(row['PnL'] - day_expense, 2))
        capital_needed_list.append(round(day_capital_needed, 2))
    
    pnl_per_day['Total PnL'] = pnl_per_day['PnL'].round(2)
    pnl_per_day['Net Expense'] = cost_per_trade_list
    pnl_per_day['Net PnL'] = net_pnl_list
    pnl_per_day['Capital Needed'] = capital_needed_list  # ✅ Added
    pnl_per_day['Capital After'] = capital_after_list  # ✅ Added
    
    # ✅ Drop old raw PnL column for clarity (optional)
    pnl_per_day = pnl_per_day[['Date', 'Total PnL', 'Net Expense', 'Net PnL']]
    
    # ✅ Compute total expense
    total_expense = sum(cost_per_trade_list)
    
    # ✅ Summary for overall performance
    summary = {
        "Total Trades": total_trades,
        "Winning Trades": len(winning_trades),
        "Losing Trades": len(losing_trades),
        "Win %": round(win_pct, 2),
        "Loss %": round(loss_pct, 2),
        "Total PnL": round(total_pnl, 2),
        "Average PnL": round(avg_pnl, 2),
        "Max PnL": round(max_pnl, 2),
        "Min PnL": round(min_pnl, 2),
        "Total Expense": round(total_expense, 2),  # ✅ Added this line
        "Net PnL (After Expenses)": round(sum(net_pnl_list), 2),
        "Final Capital": round(current_capital, 2)
    }
    
    summary_df = pd.DataFrame([summary])
    return summary_df, pnl_per_day

###################################################################################


def compute_performance(signal_df, brokerage_per_trade=20, gst_rate=0.18, stamp_duty_rate=0.00015, starting_capital=0):
    """
    Compute performance summary from signal log with PnL and include daily costs.
    
    Returns:
    - summary_df: Overall performance summary (including Total Expense)
    - pnl_per_day: Daily PnL with Total PnL, Net Expense, and Net PnL
    """
    import pandas as pd
    
    total_trades = len(signal_df)
    winning_trades = signal_df[signal_df['PnL'] > 0]
    losing_trades = signal_df[signal_df['PnL'] <= 0]
    
    total_pnl = signal_df['PnL'].sum()
    avg_pnl = signal_df['PnL'].mean() if total_trades > 0 else 0
    max_pnl = signal_df['PnL'].max() if total_trades > 0 else 0
    min_pnl = signal_df['PnL'].min() if total_trades > 0 else 0
    
    win_pct = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
    loss_pct = len(losing_trades) / total_trades * 100 if total_trades > 0 else 0
    
    # Group by date and calculate daily PnL
    pnl_per_day = signal_df.groupby('Date').agg({'PnL': 'sum', 'Quantity': 'sum'}).reset_index()
    
    cost_per_trade_list = []
    net_pnl_list = []
    capital_needed_list = []
    capital_after_list = []
    
    current_capital = starting_capital  # Initialize overall capital tracker
    
    for idx, row in pnl_per_day.iterrows():
        day_trades = signal_df[signal_df['Date'] == row['Date']]
        day_expense = 0
        day_capital_needed = 0  # Initialize daily capital before summing
        
        for _, trade in day_trades.iterrows():
            turnover = trade['Buy Premium'] * trade['Quantity'] * 2  # Buy + Sell
            brokerage = brokerage_per_trade
            gst = brokerage * gst_rate
            stamp_duty = turnover * stamp_duty_rate
            total_cost = brokerage + gst + stamp_duty
            day_expense += total_cost

            # Calculate capital needed for each trade
            trade_capital = trade['Buy Premium'] * trade['Quantity']
            day_capital_needed += trade_capital
        
        # Update capital after daily PnL
        current_capital += row['PnL'] - day_expense
        
        cost_per_trade_list.append(round(day_expense, 2))
        net_pnl_list.append(round(row['PnL'] - day_expense, 2))
        capital_needed_list.append(round(day_capital_needed, 2))
        capital_after_list.append(round(current_capital, 2))
    
    pnl_per_day['Total PnL'] = pnl_per_day['PnL'].round(2)
    pnl_per_day['Net Expense'] = cost_per_trade_list
    pnl_per_day['Net PnL'] = net_pnl_list
    pnl_per_day['Capital Needed'] = capital_needed_list
    pnl_per_day['Capital After'] = capital_after_list
    
    pnl_per_day = pnl_per_day[['Date', 'Total PnL', 'Net Expense', 'Net PnL', 'Capital Needed', 'Capital After']]
    
    total_expense = sum(cost_per_trade_list)
    
    summary = {
        "Total Trades": total_trades,
        "Winning Trades": len(winning_trades),
        "Losing Trades": len(losing_trades),
        "Win %": round(win_pct, 2),
        "Loss %": round(loss_pct, 2),
        "Total PnL": round(total_pnl, 2),
        "Average PnL": round(avg_pnl, 2),
        "Max PnL": round(max_pnl, 2),
        "Min PnL": round(min_pnl, 2),
        "Total Expense": round(total_expense, 2),
        "Net PnL (After Expenses)": round(sum(net_pnl_list), 2),
        "Final Capital": round(current_capital, 2)
    }
    
    summary_df = pd.DataFrame([summary])
    return summary_df, pnl_per_day


##########################################################################################################################

def calculate_trade_cost(buy_price, sell_price, quantity, option_type="CE", brokerage_type="fixed"):
    """
    Calculate total cost/charges per trade.
    
    Params:
    - buy_price: Entry price per unit
    - sell_price: Exit price per unit
    - quantity: Number of units
    - option_type: "CE" or "PE"
    - brokerage_type: "fixed" or "percentage"
    
    Returns total charges
    """
    turnover = (buy_price + sell_price) * quantity

    # Brokerage
    if brokerage_type == "fixed":
        brokerage = 20  # assume ₹20 per trade
    else:  # percentage
        brokerage = turnover * 0.0003  # 0.03%

    # Exchange Transaction Charges
    exchange_charges = turnover * 0.0000325  # 0.00325%

    # GST on brokerage (18%)
    gst = 0.18 * brokerage

    # SEBI Charges (approx)
    sebi_charges = turnover * 0.000001

    # Stamp Duty (approx)
    stamp_duty = turnover * 0.00003

    total_charges = brokerage + exchange_charges + gst + sebi_charges + stamp_duty
    return total_charges

##################################################################################

def compute_trade_pnl_with_costs(signal_log_df, df):
    """
    Compute PnL, exit reason, and capital impact per trade.
    """
    trade_results = []

    capital = 0  # running capital (cumulative PnL)
    

    for _, row in signal_log_df.iterrows():
        day = row['Date']
        entry_time = row['Entry Time']
        exit_time = row['Time Exit (16 mins after entry)']
        buy_premium = row['Buy Premium']
        qty = row['Quantity']
        stoploss = row['Stoploss (Trailing 10%)']
        take_profit = row['Take Profit (10% rise)']
        option_type = row['Option Selected']
        # Capital needed for this trade (premium × quantity)
        capital_needed = buy_premium * qty

        day_df = df[df['Datetime'].dt.date == day]
        day_after_entry = day_df[day_df['Datetime'] >= entry_time].sort_values('Datetime')

        sell_price = None
        exit_reason = "Time Exit"

        for _, candle in day_after_entry.iterrows():
            price = candle['Close_^NSEI']  # Option price if available
            if take_profit and price >= take_profit:
                sell_price = take_profit
                exit_reason = "Take Profit"
                exit_time = candle['Datetime']
                break
            elif stoploss and price <= stoploss:
                sell_price = stoploss
                exit_reason = "Stoploss"
                exit_time = candle['Datetime']
                break

        if sell_price is None:
            sell_price = day_after_entry['Close_^NSEI'].iloc[0]  # fallback

        raw_pnl = (sell_price - buy_premium) * qty if option_type.upper() == "CE" else (buy_premium - sell_price) * qty

        # Calculate brokerage & charges
        total_charges = calculate_trade_cost(buy_premium, sell_price, qty, option_type)
        
        net_pnl = raw_pnl - total_charges
        capital += net_pnl

        trade_results.append({
            **row.to_dict(),
            "Sell Price": sell_price,
            "Exit Reason": exit_reason,
            "Actual Exit Time": exit_time,
            "Raw PnL": raw_pnl,
            "Total Charges": total_charges,
            "Net PnL": net_pnl,
            "Capital Needed": capital_needed,  # ✅ Added column
            "Capital After Trade": capital
        })

    return pd.DataFrame(trade_results)
####################################### New Log ########################################################


def get_nearest_itm_option(spot_price, option_type="CALL", strike_step=50):
    """
    Returns the nearest ITM strike for given spot price and option type.
    Example: If spot = 19765 and option_type = CALL → 19750 (nearest ITM call).
             If spot = 19765 and option_type = PUT  → 19800 (nearest ITM put).
    """
    spot_price = round(spot_price / strike_step) * strike_step  # round to nearest strike

    if option_type.upper() == "CALL":
        return spot_price - strike_step  # ITM Call = one step below spot
    elif option_type.upper() == "PUT":
        return spot_price + strike_step  # ITM Put = one step above spot
    else:
        raise ValueError("option_type must be either 'CALL' or 'PUT'")

#################################################################################################################################



import pandas as pd

def trading_signal_all_conditions2_newlogic(df, quantity=10*75, return_all_signals=False):
    """
    Revised Base Zone Strategy (returns signal dict(s) that include 'message' key)
    """
    signals = []
    df = df.copy()
    df['Date'] = df['Datetime'].dt.date

    unique_days = sorted(df['Date'].unique())
    if len(unique_days) < 2:
        return None

    day0 = unique_days[-2]
    day1 = unique_days[-1]

    # Base Zone: Day0 15:00 (15:00-15:15 candle)
    candle_base = df[(df['Date'] == day0) &
                     (df['Datetime'].dt.hour == 15) &
                     (df['Datetime'].dt.minute == 0)]
    if candle_base.empty:
        return None

    base_open = candle_base.iloc[0]['Open_^NSEI']
    base_close = candle_base.iloc[0]['Close_^NSEI']
    base_low = min(base_open, base_close)
    base_high = max(base_open, base_close)

    # Day1 first 15-min candle (09:15-09:30)
    candle1 = df[(df['Date'] == day1) &
                 (df['Datetime'].dt.hour == 9) &
                 (df['Datetime'].dt.minute == 15)]
    if candle1.empty:
        return None

    H1 = candle1.iloc[0]['High_^NSEI']
    L1 = candle1.iloc[0]['Low_^NSEI']
    C1 = candle1.iloc[0]['Close_^NSEI']
    entry_time = candle1.iloc[0]['Datetime']
    spot_price = df['Close_^NSEI'].iloc[-1]

    # expiry & strike helpers must exist in your module:
    expiry = get_nearest_weekly_expiry(pd.to_datetime(day1))

    def make_signal(condition, option_type, trigger_level, trigger_time, ref_candle_label, message):
        strike = get_nearest_itm_option(spot_price, option_type)
        entry_price = trigger_level  # placeholder: in live you would fetch option premium at fill
        return {
            'condition': condition,
            'message': message,
            'ref_candle': ref_candle_label,
            'option_type': option_type,
            'strike': strike,
            'trigger_level': trigger_level,
            'entry_time': pd.to_datetime(trigger_time),
            'expiry': expiry,
            'quantity': quantity,
            'spot_price': spot_price,
            # Order management / risk fields (as per your rules)
            'sl_init': round(entry_price * 0.90, 4),          # 10% trailing SL from entry
            'partial_profit': round(entry_price * 1.10, 4),   # 10% target for 50% book
            'trail_sl_rule': '10% trailing SL on option premium (ratchet up only)',
            'time_exit_mins': 16,
        }

    # ---------- Condition 1: Break above Base Zone on Day1 open ----------
    # Cuts Base Zone (intersects) and closes above base_high
    if (H1 >= base_low and L1 <= base_high) and (C1 > base_high):
        msg = f"Cond 1: Candle cuts Base Zone and closes above base_high ({base_high}). Entry CALL above H1={H1}."
        sig = make_signal(1, 'CALL', H1, entry_time, 'Candle 1', msg)
        signals.append(sig)
        if not return_all_signals:
            return sig

    # ---------- Condition 2: Major Gap Down ----------
    # First candle closes entirely below Base Zone
    if C1 < base_low:
        # Primary entry: next candles break below L1
        day1_after = df[(df['Date'] == day1) & (df['Datetime'] > entry_time)].sort_values('Datetime')
        triggered = False
        for _, nxt in day1_after.iterrows():
            if nxt['Low_^NSEI'] <= L1:
                msg = f"Cond 2: Gap down confirmed. Enter PUT when price <= L1 ({L1})."
                sig = make_signal(2, 'PUT', L1, nxt['Datetime'], 'Reference Candle 2', msg)
                signals.append(sig)
                triggered = True
                if not return_all_signals:
                    return sig
                break
            # Flip / recovery: later candle closes above Base Zone -> flip to CALL (2.7)
            if nxt['Close_^NSEI'] > base_high:
                msg = f"Cond 2.7 Flip: Price closed above Base Zone. Enter CALL above high {nxt['High_^NSEI']}."
                sig = make_signal(2.7, 'CALL', nxt['High_^NSEI'], nxt['Datetime'], 'Candle 2 (flip)', msg)
                signals.append(sig)
                triggered = True
                if not return_all_signals:
                    return sig
                break
        # continue to next condition if not triggered

    # ---------- Condition 3: Major Gap Up ----------
    if C1 > base_high:
        day1_after = df[(df['Date'] == day1) & (df['Datetime'] > entry_time)].sort_values('Datetime')
        triggered = False
        for _, nxt in day1_after.iterrows():
            if nxt['High_^NSEI'] >= H1:
                msg = f"Cond 3: Gap up confirmed. Enter CALL when price >= H1 ({H1})."
                sig = make_signal(3, 'CALL', H1, nxt['Datetime'], 'Reference Candle 3', msg)
                signals.append(sig)
                triggered = True
                if not return_all_signals:
                    return sig
                break
            # Flip to PUT (3.7) if later candle closes below Base Zone
            if nxt['Close_^NSEI'] < base_low:
                msg = f"Cond 3.7 Flip: Price closed below Base Zone. Enter PUT below low {nxt['Low_^NSEI']}."
                sig = make_signal(3.7, 'PUT', nxt['Low_^NSEI'], nxt['Datetime'], 'Candle 3 (flip)', msg)
                signals.append(sig)
                triggered = True
                if not return_all_signals:
                    return sig
                break

    # ---------- Condition 4: Break below Base Zone on Day1 open ----------
    if (H1 >= base_low and L1 <= base_high) and (C1 < base_low):
        msg = f"Cond 4: Candle cuts Base Zone and closes below base_low ({base_low}). Entry PUT below L1={L1}."
        sig = make_signal(4, 'PUT', L1, entry_time, 'Candle 4', msg)
        signals.append(sig)
        if not return_all_signals:
            return sig

    return signals if signals else None


####################################################################################################################################
import ta   # technical analysis library for EMA, ATR etc.###########

def trading_signal_all_conditions2_improved(df, quantity=10*75, return_all_signals=False):
    """
    Improved Base Zone Strategy - NIFTY Index Options (15-min candles)

    Enhancements:
    - Volume filter
    - ATR-based SL & Target
    - Trend filter using 200 EMA
    - Adaptive time-based exit
    - Skip abnormal Base Zone days

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ['Datetime','Open_^NSEI','High_^NSEI','Low_^NSEI','Close_^NSEI','Volume_^NSEI']
    quantity : int
        Total units (default 10 lots = 750, since 1 lot = 75)
    return_all_signals : bool
        If True -> return list of all signals
        If False -> return first valid signal

    Returns
    -------
    dict or list(dict) or None
    """

    signals = []
    df = df.copy()
    df['Date'] = df['Datetime'].dt.date

    # Add indicators
    df['EMA200'] = ta.trend.ema_indicator(df['Close_^NSEI'], window=200)
    df['ATR'] = ta.volatility.average_true_range(df['High_^NSEI'], df['Low_^NSEI'], df['Close_^NSEI'], window=14)

    unique_days = sorted(df['Date'].unique())
    if len(unique_days) < 2:
        return None  # Not enough history

    day0 = unique_days[-2]
    day1 = unique_days[-1]

    # --- Step 1: Base Zone ---
    candle_base = df[(df['Date'] == day0) &
                     (df['Datetime'].dt.hour == 15) &
                     (df['Datetime'].dt.minute == 0)]
    if candle_base.empty:
        return None

    base_open = candle_base.iloc[0]['Open_^NSEI']
    base_close = candle_base.iloc[0]['Close_^NSEI']
    base_low = min(base_open, base_close)
    base_high = max(base_open, base_close)
    base_range = base_high - base_low

    # Skip abnormal zones (too narrow or too wide)
    if base_range < 0.001 * base_close or base_range > 0.015 * base_close:
        return None

    # --- Step 2: Day 1 first candle ---
    candle1 = df[(df['Date'] == day1) &
                 (df['Datetime'].dt.hour == 9) &
                 (df['Datetime'].dt.minute == 15)]
    if candle1.empty:
        return None

    H1 = candle1.iloc[0]['High_^NSEI']
    L1 = candle1.iloc[0]['Low_^NSEI']
    C1 = candle1.iloc[0]['Close_^NSEI']
    entry_time = candle1.iloc[0]['Datetime']
    spot_price = df['Close_^NSEI'].iloc[-1]

    expiry = get_nearest_weekly_expiry(pd.to_datetime(day1))

    # --- Volume Filter ---
    avg_vol = df[df['Date'] == day0]['Volume_^NSEI'].mean()
    if candle1.iloc[0]['Volume_^NSEI'] < 1.0 * avg_vol:  # Require at least 20% higher volume
        return None

    # --- Helper for creating a signal ---
    def make_signal(condition, option_type, trigger_level, entry_time, message):
        strike = get_nearest_itm_option(spot_price, option_type)  # placeholder
        entry_price = trigger_level  # placeholder, in practice fetch option premium

        atr_val = df['ATR'].iloc[-1] or 50  # fallback ATR ~ 50 points

        return {
            'condition': condition,
            'option_type': option_type,
            'strike': strike,
            'trigger_level': trigger_level,
            'entry_time': entry_time,
            'expiry': expiry,
            'quantity': quantity,
            'spot_price': spot_price,
            'sl_init': round(entry_price - 0.5 * atr_val, 2),   # SL = 0.5 * ATR
            'target1': round(entry_price + 1.0 * atr_val, 2),   # Target = 1 ATR
            'trail_sl_rule': 'Trail SL by 0.3*ATR every new high/low',
            'time_exit': 'Exit by 11:00 AM (morning trade) or 3:00 PM (afternoon trade)',
            'message': message
        }

    # --- Step 3: Apply Conditions ---

    # Condition 1: Bullish breakout
    if (L1 <= base_high and H1 >= base_low) and (C1 > base_high) and (C1 > candle1.iloc[0]['EMA200']):
        sig = make_signal(1, 'CALL', H1, entry_time,
                          'Condition 1: Bullish breakout → Buy CALL above H1')
        signals.append(sig)
        if not return_all_signals: return sig

    # Condition 2: Major Gap Down
    if C1 < base_low and C1 < candle1.iloc[0]['EMA200']:
        day1_after = df[(df['Date'] == day1) & (df['Datetime'] > entry_time)].sort_values('Datetime')
        for _, nxt in day1_after.iterrows():
            if nxt['Low_^NSEI'] <= L1:
                sig = make_signal(2, 'PUT', L1, nxt['Datetime'],
                                  'Condition 2: Gap down confirmed → Buy PUT below L1')
                signals.append(sig)
                if not return_all_signals: return sig
                break
            if nxt['Close_^NSEI'] > base_high:  # Flip 2.7
                ref_high = nxt['High_^NSEI']
                sig = make_signal(2.7, 'CALL', ref_high, nxt['Datetime'],
                                  'Condition 2 Flip: Close above Base Zone → Buy CALL above ref high')
                signals.append(sig)
                if not return_all_signals: return sig
                break

    # Condition 3: Major Gap Up
    if C1 > base_high and C1 > candle1.iloc[0]['EMA200']:
        day1_after = df[(df['Date'] == day1) & (df['Datetime'] > entry_time)].sort_values('Datetime')
        for _, nxt in day1_after.iterrows():
            if nxt['High_^NSEI'] >= H1:
                sig = make_signal(3, 'CALL', H1, nxt['Datetime'],
                                  'Condition 3: Gap up confirmed → Buy CALL above H1')
                signals.append(sig)
                if not return_all_signals: return sig
                break
            if nxt['Close_^NSEI'] < base_low:  # Flip 3.7
                ref_low = nxt['Low_^NSEI']
                sig = make_signal(3.7, 'PUT', ref_low, nxt['Datetime'],
                                  'Condition 3 Flip: Close below Base Zone → Buy PUT below ref low')
                signals.append(sig)
                if not return_all_signals: return sig
                break

    # Condition 4: Bearish breakdown
    if (L1 <= base_high and H1 >= base_low) and (C1 < base_low) and (C1 < candle1.iloc[0]['EMA200']):
        sig = make_signal(4, 'PUT', L1, entry_time,
                          'Condition 4: Bearish breakdown → Buy PUT below L1')
        signals.append(sig)
        if not return_all_signals: return sig

    return signals if signals else None

################################################################################################


def trading_signal_all_conditions_2_improved(df, quantity=10*750, return_all_signals=False):
    """
    Evaluate trading conditions based on Base Zone strategy.
    
    ✅ Includes:
        - Support Flip trades (2.7 & 3.7)
        - Time filter (avoid trades after 14:30)
        - Skip abnormal Base Zones (too tight or too wide)
    
    Parameters:
    - df: pd.DataFrame with columns ['Datetime','Open_^NSEI','High_^NSEI','Low_^NSEI','Close_^NSEI']
    - quantity: trade size
    - return_all_signals: if True, returns all signals list instead of last signal

    Returns:
    - dict or list of signals
    """
    signals = []
    spot_price = df['Close_^NSEI'].iloc[-1]
    
    # Ensure datetime is pandas datetime
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df['Date'] = df['Datetime'].dt.date
    unique_days = sorted(df['Date'].unique())
    
    if len(unique_days) < 2:
        return None  # Need at least 2 days of data

    # Day 0 (previous day) Base Zone calculation
    day0 = unique_days[-2]
    day1 = unique_days[-1]
    
    prev_day_data = df[df['Date'] == day0]
    today_data = df[df['Date'] == day1]

    # Base Zone from 15:00 - 15:15 candle
    base_candle = prev_day_data[(prev_day_data['Datetime'].dt.time >= pd.to_datetime('15:00').time()) & 
                                (prev_day_data['Datetime'].dt.time <= pd.to_datetime('15:15').time())]

    if base_candle.empty:
        return None
    
    base_open = base_candle['Open_^NSEI'].iloc[0]
    base_close = base_candle['Close_^NSEI'].iloc[0]
    base_low = min(base_open, base_close)
    base_high = max(base_open, base_close)
    base_range = base_high - base_low
    
    # ✅ Skip abnormal Base Zones
    if base_range < 5 or base_range > 100:  # too tight or too wide
        return None

    # Day 1 first candle (09:15 - 09:30)
    first_candle = today_data[(today_data['Datetime'].dt.time >= pd.to_datetime('09:15').time()) & 
                              (today_data['Datetime'].dt.time <= pd.to_datetime('09:30').time())]

    if first_candle.empty:
        return None

    H1 = first_candle['High_^NSEI'].iloc[0]
    L1 = first_candle['Low_^NSEI'].iloc[0]

    # Loop through today's data to check breakout
    for i, row in today_data.iterrows():
        current_time = row['Datetime'].time()
        
        # ✅ Time filter: Avoid trades after 14:30
        if current_time >= pd.to_datetime('14:30').time():
            break

        price_high = row['High_^NSEI']
        price_low = row['Low_^NSEI']

        # Buy Condition: Breaks above base_high
        if price_high > base_high:
            signals.append({
                'signal': 'BUY',
                'price': price_high,
                'time': row['Datetime'],
                'reason': 'Breakout Above Base Zone',
                'quantity': quantity
            })
            base_high = price_high  # ✅ Flip support (Base High updates)
        
        # Sell Condition: Breaks below base_low
        elif price_low < base_low:
            signals.append({
                'signal': 'SELL',
                'price': price_low,
                'time': row['Datetime'],
                'reason': 'Breakout Below Base Zone',
                'quantity': quantity
            })
            base_low = price_low  # ✅ Flip support (Base Low updates)

    return signals if return_all_signals else (signals[-1] if signals else None)
    
###################################################################################################    

def get_nearest_itm_strike(spot_price, option_type):
    """Return nearest ITM strike for NIFTY (50-point strikes)."""
    nearest_strike = round(spot_price / 50) * 50

    if option_type == "CALL":
        # ITM CALL means strike < spot
        if nearest_strike >= spot_price:
            nearest_strike -= 50
    elif option_type == "PUT":
        # ITM PUT means strike > spot
        if nearest_strike <= spot_price:
            nearest_strike += 50

    return nearest_strike


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
##########################################################################################
from functools import lru_cache
import pandas as pd
import requests

from functools import lru_cache
import requests
#####################################################################################################
@lru_cache(maxsize=10)
def fetch_option_chain_cached(symbol="NIFTY", date_key=None):
    """
    Fetch NSE Option Chain data with caching.
    Handles 403/timeout gracefully (returns None).
    """
    url_home = "https://www.nseindia.com"
    url_oc = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "accept-encoding": "gzip, deflate, br",
        "accept-language": "en-US,en;q=0.9",
        "referer": "https://www.nseindia.com/option-chain"
    }

    s = requests.Session()

    try:
        # 1️⃣ Get cookies
        s.get(url_home, headers=headers, timeout=5)

        # 2️⃣ Get option chain data
        r = s.get(url_oc, headers=headers, timeout=5)
        if r.status_code != 200:
            print(f"⚠️ NSE returned {r.status_code} for {symbol} on {date_key}")
            return None

        return r.json()

    except requests.exceptions.RequestException as e:
        print(f"⚠️ Could not fetch option chain for {symbol} on {date_key}: {e}")
        return None
####################################################################################################
def option_chain_to_df(option_chain, expiry=None):
    df_list = []
    records = option_chain.get("records", {}).get("data", [])
    for rec in records:
        if expiry and rec.get("expiryDate") != expiry:
            continue
        strike = rec.get("strikePrice")
        if 'CE' in rec:
            ce = rec['CE']
            ce['strikePrice'] = strike
            ce['optionType'] = "CE"
            ce['expiryDate'] = rec.get("expiryDate")
            df_list.append(ce)
        if 'PE' in rec:
            pe = rec['PE']
            pe['strikePrice'] = strike
            pe['optionType'] = "PE"
            pe['expiryDate'] = rec.get("expiryDate")
            df_list.append(pe)
    return pd.DataFrame(df_list)

def get_itm_contract(df_oc, spot_price, option_type="CE"):
    df2 = df_oc[df_oc['optionType'] == option_type]
    if option_type == "CE":
        df_itm = df2[df2['strikePrice'] <= spot_price]
        if df_itm.empty:
            return None
        selected = df_itm.iloc[df_itm['strikePrice'].sub(spot_price).abs().idxmin()]
    else:
        df_itm = df2[df2['strikePrice'] >= spot_price]
        if df_itm.empty:
            return None
        selected = df_itm.iloc[df_itm['strikePrice'].sub(spot_price).abs().idxmin()]
    return {
        "strike": selected['strikePrice'],
        "optionType": selected['optionType'],
        "ltp": selected.get('lastPrice'),
        "expiryDate": selected.get('expiryDate')
    }
#######################################################################################################


from functools import lru_cache

@lru_cache(maxsize=10)
def fetch_option_chain_zerodha(symbol="NIFTY", date_key=None):
    try:
        instruments = kite.instruments("NFO")
        df = pd.DataFrame(instruments)
        df = df[(df['name'] == symbol) & (df['segment'] == "NFO-OPT")]
        if df.empty:
            print(f"[⚠️] No option contracts found for {symbol} on {date_key}")
            return None

        latest_expiry = sorted(df['expiry'].unique())[0]
        df = df[df['expiry'] == latest_expiry]

        # Get LTPs
        tokens = df.set_index("tradingsymbol")["instrument_token"].to_dict()
        ltps = kite.ltp(tokens.values())
        df["ltp"] = df["tradingsymbol"].map(lambda x: ltps[tokens[x]]["last_price"])

        return df  # return DataFrame directly
    except Exception as e:
        print(f"[⚠️] Skipping {date_key} — Zerodha OC fetch failed ({e})")
        return None

###################################################################################################




def get_itm_contract(df_oc, spot_price, option_type="CE"):
    """
    Select the closest ITM option (CE/PE) from Zerodha option chain DataFrame.
    df_oc: DataFrame returned by fetch_option_chain_zerodha()
    spot_price: current NIFTY spot price
    option_type: "CE" or "PE"
    """
    # Filter only CE or PE
    df_filtered = df_oc[df_oc['instrument_type'] == option_type].copy()
    if df_filtered.empty:
        return None

    # ITM selection logic
    if option_type == "CE":
        df_itm = df_filtered[df_filtered['strike'] <= spot_price]
        if df_itm.empty:
            return None
        selected = df_itm.iloc[df_itm['strike'].sub(spot_price).abs().idxmin()]
    else:  # PE
        df_itm = df_filtered[df_filtered['strike'] >= spot_price]
        if df_itm.empty:
            return None
        selected = df_itm.iloc[df_itm['strike'].sub(spot_price).abs().idxmin()]

    return {
        "strike": selected['strike'],
        "optionType": selected['instrument_type'],
        "ltp": selected['ltp'],
        "expiryDate": selected['expiry'],
        "tradingsymbol": selected['tradingsymbol'],
        "instrument_token": selected['instrument_token']
    }



###################################################################################################
# ✅ Inside your signal loop
for i in range(1, len(unique_days)):
    day0 = unique_days[i-1]
    day1 = unique_days[i]

    day_df = df[df['Datetime'].dt.date.isin([day0, day1])]

    # ✅ Fetch option chain once per day (cached)
    #option_chain_data = fetch_option_chain_cached(symbol="NIFTY", date_key=str(day1))
    # ✅ Fetch option chain once per day
    # ✅ Fetch Zerodha option chain once per day
option_chain_data = fetch_option_chain_zerodha(symbol="NIFTY", date_key=str(day1))
if option_chain_data is None:
    st.warning(f"Skipping {day1} — Zerodha option chain not available.")
    continue

# ✅ Call trading signal function
signal = trading_signal_all_conditions1(day_df)

if signal is None:
    continue

# ✅ Attach ITM data
if isinstance(signal, dict):
    spot_price = signal.get("spot_price")
    option_type = signal.get("option_type")

    itm = get_itm_contract(option_chain_data, spot_price, option_type)
    if itm:
        signal.update({
            "itm_strike": itm["strike"],
            "itm_ltp": itm["ltp"],
            "expiry": itm["expiryDate"],
            "tradingsymbol": itm["tradingsymbol"],
            "token": itm["instrument_token"]
        })
    else:
        signal.update({"itm_strike": None, "itm_ltp": None, "expiry": None})

    signal_log_list.append(signal)

elif isinstance(signal, list):
    for sig in signal:
        spot_price = sig.get("spot_price")
        option_type = sig.get("option_type")
        itm = get_itm_contract(option_chain_data, spot_price, option_type)
        if itm:
            sig.update({
                "itm_strike": itm["strike"],
                "itm_ltp": itm["ltp"],
                "expiry": itm["expiryDate"],
                "tradingsymbol": itm["tradingsymbol"],
                "token": itm["instrument_token"]
            })
        else:
            sig.update({"itm_strike": None, "itm_ltp": None, "expiry": None})

        signal_log_list.append(sig)

   
