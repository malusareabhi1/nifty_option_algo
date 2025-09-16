import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from datetime import timedelta
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

####################################################################################################################
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

#####################################################################################################################
def get_nearest_weekly_expiry(today):
    """
    Placeholder: implement your own logic to find nearest weekly expiry date
    For demo, returns today + 7 days (Saturday)
    """
    return today + pd.Timedelta(days=7)
######################################################################################################################


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




#####################################################################################################################



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
