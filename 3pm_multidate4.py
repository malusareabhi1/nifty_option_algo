import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests

st.set_page_config(layout="wide")
st.title("Nifty 3PM Base Zone Strategy - Multi-Day Backtest")



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



 #######################################################################################################################################   



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
    
    #import pandas as pd
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




import pandas as pd
import numpy as np



#########################################################################################################


#####################################################################################

###################################################################################


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






########################################### With Sell/Exit ########################################################




def trading_signal_all_conditions4(df, quantity=10*75, previous_trade=None, return_all_signals=False):
    """
    Generate BUY and SELL signals based on Base Zone Strategy with reversal conditions.
    """

    signals = []
    df = df.copy()
    df['Date'] = df['Datetime'].dt.date
    unique_days = sorted(df['Date'].unique())
    if len(unique_days) < 2:
        return None

    # --- Base Zone ---
    day0 = unique_days[-2]
    day1 = unique_days[-1]
    candle_base = df[(df['Date'] == day0) &
                     (df['Datetime'].dt.hour == 15) &
                     (df['Datetime'].dt.minute == 0)]
    if candle_base.empty:
        return None

    base_open = candle_base.iloc[0]['Open_^NSEI']
    base_close = candle_base.iloc[0]['Close_^NSEI']
    base_low = min(base_open, base_close)
    base_high = max(base_open, base_close)

    # --- Day 1 first candle ---
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

    def make_signal(condition, option_type, action, trigger_level, entry_time, message):
        strike = get_nearest_itm_option(spot_price, option_type)
        return {
            'condition': condition,
            'option_type': option_type,
            'action': action,  # BUY or SELL
            'strike': strike,
            'trigger_level': trigger_level,
            'entry_time': entry_time,
            'expiry': expiry,
            'quantity': quantity,
            'spot_price': spot_price,
            'message': message
        }

    # ✅ SELL logic if previous trade exists
    if previous_trade:
        if previous_trade['option_type'] == 'CALL' and spot_price < previous_trade['trigger_level']:
            sig = make_signal('Exit', 'CALL', 'SELL', spot_price, df.iloc[-1]['Datetime'],
                              'Reversal: Exit CALL as price dropped below trigger')
            signals.append(sig)
            if not return_all_signals: return sig

        if previous_trade['option_type'] == 'PUT' and spot_price > previous_trade['trigger_level']:
            sig = make_signal('Exit', 'PUT', 'SELL', spot_price, df.iloc[-1]['Datetime'],
                              'Reversal: Exit PUT as price rose above trigger')
            signals.append(sig)
            if not return_all_signals: return sig

    # ✅ Original BUY logic
    if (L1 <= base_high and H1 >= base_low) and (C1 > base_high):
        sig = make_signal(1, 'CALL', 'BUY', H1, entry_time,
                          'Condition 1: Bullish breakout → Buy CALL')
        signals.append(sig)
        if not return_all_signals: return sig

    if C1 < base_low:
        day1_after = df[(df['Date'] == day1) & (df['Datetime'] > entry_time)].sort_values('Datetime')
        for _, nxt in day1_after.iterrows():
            if nxt['Low_^NSEI'] <= L1:
                sig = make_signal(2, 'PUT', 'BUY', L1, nxt['Datetime'],
                                  'Condition 2: Gap down confirmed → Buy PUT')
                signals.append(sig)
                if not return_all_signals: return sig
                break
            if nxt['Close_^NSEI'] > base_high:
                ref_high = nxt['High_^NSEI']
                sig = make_signal(2.7, 'CALL', 'BUY', ref_high, nxt['Datetime'],
                                  'Condition 2 Flip → Buy CALL')
                signals.append(sig)
                if not return_all_signals: return sig
                break

    if C1 > base_high:
        day1_after = df[(df['Date'] == day1) & (df['Datetime'] > entry_time)].sort_values('Datetime')
        for _, nxt in day1_after.iterrows():
            if nxt['High_^NSEI'] >= H1:
                sig = make_signal(3, 'CALL', 'BUY', H1, nxt['Datetime'],
                                  'Condition 3: Gap up confirmed → Buy CALL')
                signals.append(sig)
                if not return_all_signals: return sig
                break
            if nxt['Close_^NSEI'] < base_low:
                ref_low = nxt['Low_^NSEI']
                sig = make_signal(3.7, 'PUT', 'BUY', ref_low, nxt['Datetime'],
                                  'Condition 3 Flip → Buy PUT')
                signals.append(sig)
                if not return_all_signals: return sig
                break

    if (L1 <= base_high and H1 >= base_low) and (C1 < base_low):
        sig = make_signal(4, 'PUT', 'BUY', L1, entry_time,
                          'Condition 4: Bearish breakdown → Buy PUT')
        signals.append(sig)
        if not return_all_signals: return sig

    return signals if signals else None



################################## START To Execute ################################################

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


################################################## find Signals  ########################################################



##################################################################################################################

# ✅ Initialize combined trade log for signals
all_signals = []

# ✅ Loop through each day in the selected range (starting from 2nd day)
for i in range(1, len(unique_days)):
    day0 = unique_days[i-1]  # previous day (for base candle)
    day1 = unique_days[i]    # current day

    # Filter data for these 2 days
    df_two_days = df[(df['Datetime'].dt.date == day0) | (df['Datetime'].dt.date == day1)]

    # Generate signals for these 2 days
    signals = trading_signal_all_conditions4(df_two_days, quantity=10*75, return_all_signals=True)

    if signals:
        for sig in signals:
            sig['Day'] = str(day1)  # add date column for clarity
            all_signals.append(sig)

# ✅ Display all signals in Streamlit
if all_signals:
    signals_df = pd.DataFrame(all_signals)
    st.subheader(f"Trade Signals Between {start_date} and {end_date}")
    st.dataframe(signals_df)

    # ✅ CSV download option
    csv = signals_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download All Signals CSV",
        data=csv,
        file_name="all_signals_between_dates.csv",
        mime="text/csv"
    )
else:
    st.info("No signals generated for the selected date range.")
    
####################################################### ITM #############################################################
if not signals_df.empty:
    # Create new DataFrame for ITM Options
    option_rows = []
    for index, row in signals_df.iterrows():
        spot_price = row['spot_price']
        option_type = row['option_type']
        nearest_itm_strike = get_nearest_itm_option(spot_price, option_type)
        
        option_rows.append({
            'Entry Time': row['entry_time'],
            'Option Type': option_type,
            'Spot Price': spot_price,
            'Nearest ITM Strike': nearest_itm_strike,
            'Expiry': row['expiry']
        })
    
    option_df = pd.DataFrame(option_rows)
    
    st.subheader("Nearest ITM Options for Signals")
    st.table(option_df)

    # Download button
    csv_options = option_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download ITM Options CSV",
        data=csv_options,
        file_name="itm_options.csv",
        mime="text/csv"
    )
else:
    st.write("No signals found to display ITM options.")
    
##############################################################  TRADE LOG  ##################################################################




#st.write("Option DataFrame Columns:", option_df.columns.tolist())
#st.write(option_df.index.tolist())
st.write(signals_df.index.tolist())

######################################################  ✅ Build Trade Log Table ###############################################################################

########################################################################################################################################
