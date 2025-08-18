import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import requests

from datetime import datetime, timedelta


st.set_page_config(layout="wide")

st.title("Nifty 15-min Chart for Date Range")

# Select date range
from_date = st.date_input("From Date", value=datetime.today() - timedelta(days=7))
to_date = st.date_input("To Date", value=datetime.today())

# Ensure to_date is after from_date
if from_date > to_date:
    st.error("❌ 'From Date' must be earlier than 'To Date'")
else:
    # Download data for ^NSEI between from_date and to_date
    df = yf.download(
        "^NSEI", 
        start=from_date.strftime("%Y-%m-%d"), 
        end=(to_date + timedelta(days=1)).strftime("%Y-%m-%d"),  # include 'to_date'
        interval="15m"
    )

    if not df.empty:
        st.success(f"Data loaded for NIFTY between {from_date} and {to_date}")
        #st.dataframe(df.tail(20))  # show last 20 rows
    else:
        st.warning("No data found for the selected range")
        
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
#st.write(df.columns)
# Convert to datetime & timezone aware
#df['Datetime'] = pd.to_datetime(df['Datetime'])
if df['Datetime_'].dt.tz is None:
    df['Datetime'] = df['Datetime_'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
else:
    df['Datetime'] = df['Datetime_'].dt.tz_convert('Asia/Kolkata')

#st.write(df.columns)
#st.write(df.head(10))


#####################################################################################



def plot_with_3pm_levels(df):
    # Ensure datetime index
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # Extract trading days
    df['date'] = df.index.date
    unique_days = sorted(df['date'].unique())

    charts = []
    for i, day in enumerate(unique_days[:-1]):  # skip last day (no next day to extend)
        day_data = df[df['date'] == day]
        next_day = unique_days[i+1]
        next_day_data = df[df['date'] == next_day]

        # Get 3PM candle of current day
        three_pm = day_data.between_time("15:00", "15:15")
        if three_pm.empty:
            continue
        open_3pm = three_pm['Open'].iloc[0]
        close_3pm = three_pm['Close'].iloc[0]

        # Combine current + next day for plotting
        plot_data = pd.concat([day_data, next_day_data])

        # Plot candles
        fig = go.Figure(data=[go.Candlestick(
            x=plot_data.index,
            open=plot_data['Open'],
            high=plot_data['High'],
            low=plot_data['Low'],
            close=plot_data['Close'],
            name="NIFTY 15m"
        )])

        # Add horizontal lines for 3PM open and close
        fig.add_hline(y=open_3pm, line=dict(color="blue", dash="dot"), 
                      annotation_text=f"3PM Open {open_3pm:.2f}", annotation_position="top left")
        fig.add_hline(y=close_3pm, line=dict(color="red", dash="dot"), 
                      annotation_text=f"3PM Close {close_3pm:.2f}", annotation_position="bottom left")

        fig.update_layout(
            title=f"NIFTY 15-min | {day} + Next Day",
            xaxis_rangeslider_visible=False,
            height=600
        )
        charts.append(fig)
    
    return charts

charts = plot_with_3pm_levels(df)
for fig in charts:
    st.plotly_chart(fig, use_container_width=True)


#####################################################################################
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


display_current_trend(df)
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


#####################################################################################################################################
#open_3pm, close_3pm = display_3pm_candle_info(df_plot, selected_date)


##########################################################################################################


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
    
  
###########################################################################################

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

##############
###########################################################




def trading_signal_all_conditions1(df, quantity=10*750, return_all_signals=False):
    """
    Evaluate trading conditions based on Base Zone strategy.

    Parameters:
    - df: pd.DataFrame with columns ['Datetime','Open_^NSEI','High_^NSEI','Low_^NSEI','Close_^NSEI']
    - quantity: total quantity to trade (default 10 lots = 7500)
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

    # --- Condition 4 ---
    if (L1 < base_high and H1 > base_low) and (C1 < base_low):
        sig = {
            'condition': 4,
            'option_type': 'PUT',
            'buy_price': L1,
            'stoploss': L1 * 1.10,
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


#################################################

# Initialize session state for trade logs
if 'trade_log_df' not in st.session_state:
    st.session_state.trade_log_df = pd.DataFrame()

# Display today's candles with trend and signals
display_todays_candles_with_trend_and_signal(df)

# Get current time of latest candle
latest_candle_time = df['Datetime'].max()
candle_time_only = latest_candle_time.time()

# Define trading window
start_time = pd.to_datetime("09:30").time()
end_time = pd.to_datetime("14:00").time()

# Get option chain and signals
result_chain = find_nearest_itm_option()
signal = trading_signal_all_conditions1(df)

# Only take trade if within the window
# if start_time <= candle_time_only <= end_time:    
if signal:
    st.write(f"Trade signal detected:\n{signal['message']}")
    st.table(pd.DataFrame([signal]))
    
    spot_price = signal['spot_price']
    ot = "CE" if signal["option_type"].upper() == "CALL" else "PE"
    
    # Find nearest ITM option to buy
    result = option_chain_finder(result_chain, spot_price, option_type=ot, lots=10, lot_size=75)
    
    st.write("Nearest ITM Call option to BUY:")
    st.table(pd.DataFrame([result['option_data']]))
    st.write(f"Total Quantity: {result['total_quantity']}")
    
    # Generate trade log for current signal
    trade_log_entry = generate_trade_log_from_option(result, signal)
        
    # Only convert 'expiry' if the column exists
    if 'expiry' in trade_log_entry.columns:
        trade_log_entry['expiry'] = pd.to_datetime(trade_log_entry['expiry'])
        
    # Append to session state log
    st.session_state.trade_log_df = pd.concat(
        [st.session_state.trade_log_df, trade_log_entry], ignore_index=True
    )
else:
    st.write("No trade signal for today based on conditions.")   
# else:
#st.write(f"No trades: Outside trading window (9:30 AM – 3:00 PM). Latest candle: {latest_candle_time}")



st.subheader("Cumulative Trade Log")
if not st.session_state.trade_log_df.empty:
    st.table(st.session_state.trade_log_df)
    csv = st.session_state.trade_log_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Trade Log as CSV",
        data=csv,
        file_name="trade_log.csv",
        mime="text/csv",
    )
else:
    st.write("No trade log data available yet.")
