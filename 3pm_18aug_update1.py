import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import requests

from datetime import datetime, timedelta

# ----------------- LOAD DATA FUNCTION -----------------
def load_nifty_data_15min(days_back=7):
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
def plot_nifty_15min_chart(df, last_day=None, today=None):
    import plotly.graph_objects as go

    if last_day is None or today is None:
        unique_days = df['Datetime'].dt.date.unique()
        if len(unique_days) < 2:
            return None
        last_day, today = unique_days[-2], unique_days[-1]

    # Filter last two trading days
    df_plot = df[df['Datetime'].dt.date.isin([last_day, today])]

    # Get last day 3PM candle
    candle_3pm = df_plot[(df_plot['Datetime'].dt.date == last_day) &
                         (df_plot['Datetime'].dt.hour == 15) &
                         (df_plot['Datetime'].dt.minute == 0)]
    if not candle_3pm.empty:
        open_3pm = candle_3pm.iloc[0]['Open_^NSEI']
        close_3pm = candle_3pm.iloc[0]['Close_^NSEI']
    else:
        open_3pm = None
        close_3pm = None

    # Plot candlestick
    fig = go.Figure(data=[go.Candlestick(
        x=df_plot['Datetime'],
        open=df_plot['Open_^NSEI'],
        high=df_plot['High_^NSEI'],
        low=df_plot['Low_^NSEI'],
        close=df_plot['Close_^NSEI'],
        name="Nifty"
    )])

    # Add horizontal lines for 3PM candle
    if open_3pm is not None:
        fig.add_hline(
            y=open_3pm,
            line_dash="dot",
            line_color="blue",
            annotation_text="3PM Open",
            annotation_position="top left"
        )
    if close_3pm is not None:
        fig.add_hline(
            y=close_3pm,
            line_dash="dot",
            line_color="red",
            annotation_text="3PM Close",
            annotation_position="bottom left"
        )

    # Hide weekends and outside trading hours using rangebreaks
    fig.update_layout(
        title="Nifty 15-min candles - Last Day & Today",
        xaxis_rangeslider_visible=False,
        xaxis=dict(
            rangebreaks=[
                # Hide weekends
                dict(bounds=["sat", "mon"]),
                # Hide hours outside NSE trading hours (15:30–09:15)
                dict(bounds=[15.5, 9.25], pattern="hour"),
            ]
        ),
    )

    return fig

# ----------------- SIGNAL LOGGING FUNCTION -----------------
def run_trading_signals(df, signal_log=None):
    if signal_log is None:
        signal_log = pd.DataFrame()

    # Call the signal function for current df
    signals = trading_signal_all_conditions1(df, return_all_signals=True)
    if signals:
        df_signals = pd.DataFrame(signals)
        signal_log = pd.concat([signal_log, df_signals], ignore_index=True)

    return signal_log


def get_nearest_weekly_expiry(today):
    """
    Placeholder: implement your own logic to find nearest weekly expiry date
    For demo, returns today + 7 days (Saturday)
    """
    return today + pd.Timedelta(days=7)


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
##########################################################################################################
# ----------------- STREAMLIT APP -----------------
st.title("Nifty 50 15-min Live Trade Logger")

# Initialize or load existing signal log
if 'signal_log' not in st.session_state:
    st.session_state.signal_log = pd.DataFrame()

# Load latest Nifty data
df_nifty = load_nifty_data_15min(days_back=7)
if df_nifty is None or df_nifty.empty:
    st.warning("No data available")
    st.stop()

# Run signals and store in session state
st.session_state.signal_log = run_trading_signals(df_nifty, st.session_state.signal_log)
fig = plot_nifty_15min_chart(df_nifty)
if fig:
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Trade Signals / Logs")
st.dataframe(st.session_state.signal_log)

# Auto-refresh every 15 minutes to capture new candle
st_autorefresh(interval=900000, key="nifty_refresh")  # 900000ms = 15 minutes
