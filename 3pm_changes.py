import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta

# ----------------- LOAD DATA FUNCTION -----------------
def load_nifty_data_15min(days_back=3):
    end_date = datetime.today().date() + timedelta(days=1)
    start_date = end_date - timedelta(days=days_back)
    df = yf.download("^NSEI", start=start_date, end=end_date, interval="15m")
    if df.empty:
        return None

    df.reset_index(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]

    if 'Datetime' not in df.columns:
        if 'Datetime_' in df.columns:
            df.rename(columns={'Datetime_': 'Datetime'}, inplace=True)
        elif 'Date' in df.columns:
            df.rename(columns={'Date': 'Datetime'}, inplace=True)

    df['Datetime'] = pd.to_datetime(df['Datetime'])
    if df['Datetime'].dt.tz is None:
        df['Datetime'] = df['Datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
    else:
        df['Datetime'] = df['Datetime'].dt.tz_convert('Asia/Kolkata')

    # Filter NSE trading hours: 9:15–15:30
    df = df[(df['Datetime'].dt.time >= datetime.strptime("09:15", "%H:%M").time()) &
            (df['Datetime'].dt.time <= datetime.strptime("15:30", "%H:%M").time())]

    return df

# ----------------- PLOT FUNCTION -----------------
def plot_all_candles(df):
    # Add previous day 3PM open/close lines
    unique_days = sorted(df['Datetime'].dt.date.unique())
    if len(unique_days) < 2:
        last_day = unique_days[-1]
        prev_day = last_day - timedelta(days=1)
    else:
        prev_day = unique_days[-2]
        last_day = unique_days[-1]

    candle_3pm = df[(df['Datetime'].dt.date == prev_day) &
                     (df['Datetime'].dt.hour == 15) &
                     (df['Datetime'].dt.minute == 0)]
    if not candle_3pm.empty:
        open_3pm = candle_3pm.iloc[0]['Open_^NSEI']
        close_3pm = candle_3pm.iloc[0]['Close_^NSEI']
    else:
        open_3pm = None
        close_3pm = None

    fig = go.Figure(data=[go.Candlestick(
        x=df['Datetime'],
        open=df['Open_^NSEI'],
        high=df['High_^NSEI'],
        low=df['Low_^NSEI'],
        close=df['Close_^NSEI'],
        name="Nifty"
    )])

    if open_3pm is not None:
        fig.add_hline(y=open_3pm, line_dash="dot", line_color="blue", annotation_text="Prev 3PM Open", annotation_position="top left")
    if close_3pm is not None:
        fig.add_hline(y=close_3pm, line_dash="dot", line_color="red", annotation_text="Prev 3PM Close", annotation_position="bottom left")

    fig.update_layout(
        title="Nifty 15-min Candles - Last Day & Today",
        xaxis_rangeslider_visible=False,
        xaxis=dict(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # Hide weekends
                dict(bounds=[15.5, 9.25], pattern="hour"),  # Hide hours outside NSE trading
            ]
        )
    )
    return fig

# ----------------- PLOT FUNCTION -----------------
def plot_last_two_days(df):
    # Get last two trading days
    unique_days = sorted(df['Datetime'].dt.date.unique())
    if len(unique_days) < 2:
        return None
    last_day, today = unique_days[-2], unique_days[-1]

    df_plot = df[df['Datetime'].dt.date.isin([last_day, today])]

    # Previous day 3PM candle
    candle_3pm = df_plot[(df_plot['Datetime'].dt.date == last_day) &
                         (df_plot['Datetime'].dt.hour == 15) &
                         (df_plot['Datetime'].dt.minute == 0)]
    if not candle_3pm.empty:
        open_3pm = candle_3pm.iloc[0]['Open_^NSEI']
        close_3pm = candle_3pm.iloc[0]['Close_^NSEI']
    else:
        open_3pm = None
        close_3pm = None

    fig = go.Figure(data=[go.Candlestick(
        x=df_plot['Datetime'],
        open=df_plot['Open_^NSEI'],
        high=df_plot['High_^NSEI'],
        low=df_plot['Low_^NSEI'],
        close=df_plot['Close_^NSEI'],
        name="Nifty"
    )])

    if open_3pm is not None:
        fig.add_hline(y=open_3pm, line_dash="dot", line_color="blue", annotation_text="Prev 3PM Open", annotation_position="top left")
    if close_3pm is not None:
        fig.add_hline(y=close_3pm, line_dash="dot", line_color="red", annotation_text="Prev 3PM Close", annotation_position="bottom left")

    fig.update_layout(
        title="Nifty 15-min Candles - Last Day & Today",
        xaxis_rangeslider_visible=False,
        xaxis=dict(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # Hide weekends
                dict(bounds=[15.5, 9.25], pattern="hour"),  # Hide hours outside NSE trading
            ]
        )
    )
    return fig
############################################################################################

# ----------------- NEAREST WEEKLY EXPIRY -----------------
def get_nearest_weekly_expiry(today):
    # Placeholder: implement your own logic
    return today + pd.Timedelta(days=7)

# ----------------- SIGNAL FUNCTION -----------------
def trading_signal_all_conditions1(df, quantity=10*750, return_all_signals=False):
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

    # Condition 1
    if (L1 < base_high and H1 > base_low) and (C1 > base_high):
        sig = {
            'condition': 1, 'option_type': 'CALL', 'buy_price': H1,
            'stoploss': H1*0.9, 'take_profit': H1*1.10,
            'quantity': quantity, 'expiry': expiry,
            'entry_time': entry_time,
            'message': 'Condition 1: Bullish breakout above Base Zone → Buy CALL above H1',
            'spot_price': spot_price
        }
        signals.append(sig)
        if not return_all_signals:
            return sig

    # Condition 2
    if C1 < base_low:
        for _, next_candle in day1_after_915.iterrows():
            if next_candle['Low_^NSEI'] < L1:
                sig = {
                    'condition': 2, 'option_type': 'PUT', 'buy_price': L1,
                    'stoploss': L1*1.10, 'take_profit': L1*0.90,
                    'quantity': quantity, 'expiry': expiry,
                    'entry_time': next_candle['Datetime'],
                    'message': 'Condition 2: Gap down confirmed → Buy PUT below L1',
                    'spot_price': spot_price
                }
                signals.append(sig)
                if not return_all_signals:
                    return sig
            if next_candle['Close_^NSEI'] > base_high:
                ref_high = next_candle['High_^NSEI']
                sig_flip = {
                    'condition': 2.7, 'option_type': 'CALL', 'buy_price': ref_high,
                    'stoploss': ref_high*0.9, 'take_profit': ref_high*1.10,
                    'quantity': quantity, 'expiry': expiry,
                    'entry_time': next_candle['Datetime'],
                    'message': 'Condition 2 Flip: Later candle closed above Base Zone → Buy CALL above Candle 2 high',
                    'spot_price': spot_price
                }
                signals.append(sig_flip)
                if not return_all_signals:
                    return sig_flip

    # Condition 3
    if C1 > base_high:
        for _, next_candle in day1_after_915.iterrows():
            if next_candle['High_^NSEI'] > H1:
                sig = {
                    'condition': 3, 'option_type': 'CALL', 'buy_price': H1,
                    'stoploss': H1*0.9, 'take_profit': H1*1.10,
                    'quantity': quantity, 'expiry': expiry,
                    'entry_time': next_candle['Datetime'],
                    'message': 'Condition 3: Gap up confirmed → Buy CALL above H1',
                    'spot_price': spot_price
                }
                signals.append(sig)
                if not return_all_signals:
                    return sig
            if next_candle['Close_^NSEI'] < base_low:
                ref_low = next_candle['Low_^NSEI']
                sig_flip = {
                    'condition': 3.7, 'option_type': 'PUT', 'buy_price': ref_low,
                    'stoploss': ref_low*1.10, 'take_profit': ref_low*0.90,
                    'quantity': quantity, 'expiry': expiry,
                    'entry_time': next_candle['Datetime'],
                    'message': 'Condition 3 Flip: Later candle closed below Base Zone → Buy PUT below Candle 3 low',
                    'spot_price': spot_price
                }
                signals.append(sig_flip)
                if not return_all_signals:
                    return sig_flip

    # Condition 4
    if (L1 < base_high and H1 > base_low) and (C1 < base_low):
        sig = {
            'condition': 4, 'option_type': 'PUT', 'buy_price': L1,
            'stoploss': L1*1.10, 'take_profit': L1*0.90,
            'quantity': quantity, 'expiry': expiry,
            'entry_time': entry_time,
            'message': 'Condition 4: Bearish breakdown below Base Zone → Buy PUT below L1',
            'spot_price': spot_price
        }
        signals.append(sig)
        if not return_all_signals:
            return sig

    return signals if signals else None




# ----------------- STREAMLIT APP -----------------

###############################################################################


# ----------------- STREAMLIT APP -----------------
st.title("Nifty 50 15-min Live Trade Logger")

# Initialize session state
if 'signal_log' not in st.session_state:
    st.session_state.signal_log = pd.DataFrame()
if 'last_candle' not in st.session_state:
    st.session_state.last_candle = None

# Load latest Nifty data
df_nifty = load_nifty_data_15min(days_back=7)
if df_nifty is None or df_nifty.empty:
    st.warning("No data available")
    st.stop()

# Get the last candle
latest_candle_time = df_nifty['Datetime'].max()
if st.session_state.last_candle != latest_candle_time:
    # Only run for new candle
    new_candles_df = df_nifty[df_nifty['Datetime'] > st.session_state.last_candle] if st.session_state.last_candle else df_nifty
    new_signals = trading_signal_all_conditions1(new_candles_df, return_all_signals=True)
    if new_signals:
        st.session_state.signal_log = pd.concat([st.session_state.signal_log, pd.DataFrame(new_signals)], ignore_index=True)
    st.session_state.last_candle = latest_candle_time

# Display trade signals
st.subheader("Trade Signals / Logs")
st.dataframe(st.session_state.signal_log)

# Plot last two days of candles
def plot_last_two_days(df):
    unique_days = sorted(df['Datetime'].dt.date.unique())
    if len(unique_days) < 2:
        return None
    last_day, today = unique_days[-2], unique_days[-1]
    df_plot = df[df['Datetime'].dt.date.isin([last_day, today])]
    fig = go.Figure(data=[go.Candlestick(
        x=df_plot['Datetime'],
        open=df_plot['Open'],
        high=df_plot['High'],
        low=df_plot['Low'],
        close=df_plot['Close']
    )])
    fig.update_layout(xaxis_rangeslider_visible=False)
    return fig

st.plotly_chart(plot_last_two_days(df_nifty), use_container_width=True)

# Auto-refresh every 15 min
st_autorefresh(interval=900000, key="nifty_refresh")
