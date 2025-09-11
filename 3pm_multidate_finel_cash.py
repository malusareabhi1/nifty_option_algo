import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("CASH  3PM Trailing SL and Take Profit  Strategy - Multi-Day Backtest")

def trading_signal_equity(df, quantity=100, return_all_signals=False):
    """
    Base Zone Strategy for Equity Trading

    Parameters:
    - df: DataFrame with ['Datetime','Open','High','Low','Close']
    - quantity: number of shares to trade (default 100)
    - return_all_signals: if True, returns all signals; else first found

    Logic:
    - Stoploss = recent swing points (last 10 candles)
    - Trailing SL updates with swings
    - Time exit after 16 minutes if no SL hit
    - Single active trade per day
    """

    signals = []
    spot_price = df['Close'].iloc[-1]

    # Preprocess
    df = df.copy()
    df['Date'] = df['Datetime'].dt.date
    unique_days = sorted(df['Date'].unique())
    if len(unique_days) < 2:
        return None

    day0, day1 = unique_days[-2], unique_days[-1]

    # Base Zone (3 PM candle of previous day)
    candle_3pm = df[(df['Date'] == day0) &
                    (df['Datetime'].dt.hour == 15) &
                    (df['Datetime'].dt.minute == 0)]
    if candle_3pm.empty:
        return None

    base_open = candle_3pm.iloc[0]['Open']
    base_close = candle_3pm.iloc[0]['Close']
    base_low = min(base_open, base_close)
    base_high = max(base_open, base_close)

    # First 9:15 candle of current day
    candle_915 = df[(df['Date'] == day1) &
                    (df['Datetime'].dt.hour == 9) &
                    (df['Datetime'].dt.minute == 15)]
    if candle_915.empty:
        return None

    H1, L1, C1 = (candle_915.iloc[0]['High'],
                  candle_915.iloc[0]['Low'],
                  candle_915.iloc[0]['Close'])
    entry_time = candle_915.iloc[0]['Datetime']

    # Day1 after 9:15
    day1_after_915 = df[(df['Date'] == day1) &
                        (df['Datetime'] > entry_time)].sort_values('Datetime')

    # Helper functions
    def get_recent_swing(current_time):
        recent_data = df[(df['Date'] == day1) & (df['Datetime'] < current_time)].tail(10)
        if recent_data.empty:
            return None, None
        return recent_data['High'].max(), recent_data['Low'].min()

    def update_trailing_sl(position, current_sl, current_time):
        new_high, new_low = get_recent_swing(current_time)
        if position == 'BUY':
            if new_low is None:
                return current_sl
            if current_sl is None or new_low > current_sl:
                return new_low
        elif position == 'SELL':
            if new_high is None:
                return current_sl
            if current_sl is None or new_high < current_sl:
                return new_high
        return current_sl

    def monitor_trade(sig):
        current_sl = sig.get('stoploss')
        entry_dt = sig['entry_time']
        exit_deadline = entry_dt + timedelta(minutes=20)

        if day1_after_915.empty:
            sig['exit_price'] = sig['entry_price']
            sig['status'] = 'Exited immediately (no data)'
            return sig

        for _, candle in day1_after_915.iterrows():
            if candle['Datetime'] >= exit_deadline:
                sig['exit_price'] = candle['Close']
                sig['status'] = 'Exited at time limit'
                return sig

            current_sl = update_trailing_sl(sig['position'], current_sl, candle['Datetime'])
            sig['stoploss'] = current_sl

            if sig['position'] == 'BUY' and current_sl is not None:
                if candle['Low'] <= current_sl:
                    sig['exit_price'] = current_sl
                    sig['status'] = 'Exited at Trailing SL'
                    return sig

            if sig['position'] == 'SELL' and current_sl is not None:
                if candle['High'] >= current_sl:
                    sig['exit_price'] = current_sl
                    sig['status'] = 'Exited at Trailing SL'
                    return sig

        sig['exit_price'] = day1_after_915.iloc[-1]['Close']
        sig['status'] = 'Exited at EOD/no SL hit'
        return sig

    # --- Condition 1: Bullish breakout above Base Zone ---
    if (L1 < base_high and H1 > base_low) and (C1 > base_high):
        swing_high, swing_low = get_recent_swing(entry_time)
        sig = {
            'condition': 1,
            'position': 'BUY',
            'entry_price': H1,
            'stoploss': swing_low,
            'quantity': quantity,
            'entry_time': entry_time,
            'message': 'Condition 1: Bullish breakout above Base Zone → BUY',
            'spot_price': spot_price
        }
        sig = monitor_trade(sig)
        signals.append(sig)
        if not return_all_signals:
            return sig

    # --- Condition 2: Gap Down (SELL) + Flip ---
    if C1 < base_low:
        for _, next_candle in day1_after_915.iterrows():
            swing_high, swing_low = get_recent_swing(next_candle['Datetime'])
            if next_candle['Low'] <= L1:
                sig = {
                    'condition': 2,
                    'position': 'SELL',
                    'entry_price': L1,
                    'stoploss': swing_high,
                    'quantity': quantity,
                    'entry_time': next_candle['Datetime'],
                    'message': 'Condition 2: Gap down confirmed → SELL',
                    'spot_price': spot_price
                }
                sig = monitor_trade(sig)
                signals.append(sig)
                if not return_all_signals:
                    return sig

            if next_candle['Close'] > base_high:
                sig_flip = {
                    'condition': 2.7,
                    'position': 'BUY',
                    'entry_price': next_candle['High'],
                    'stoploss': swing_low,
                    'quantity': quantity,
                    'entry_time': next_candle['Datetime'],
                    'message': 'Condition 2 Flip: Closed above Base Zone → BUY',
                    'spot_price': spot_price
                }
                sig_flip = monitor_trade(sig_flip)
                signals.append(sig_flip)
                if not return_all_signals:
                    return sig_flip

    # --- Condition 3: Gap Up (BUY) + Flip ---
    if C1 > base_high:
        for _, next_candle in day1_after_915.iterrows():
            swing_high, swing_low = get_recent_swing(next_candle['Datetime'])
            if next_candle['High'] >= H1:
                sig = {
                    'condition': 3,
                    'position': 'BUY',
                    'entry_price': H1,
                    'stoploss': swing_low,
                    'quantity': quantity,
                    'entry_time': next_candle['Datetime'],
                    'message': 'Condition 3: Gap up confirmed → BUY',
                    'spot_price': spot_price
                }
                sig = monitor_trade(sig)
                signals.append(sig)
                if not return_all_signals:
                    return sig

            if next_candle['Close'] < base_low:
                sig_flip = {
                    'condition': 3.7,
                    'position': 'SELL',
                    'entry_price': next_candle['Low'],
                    'stoploss': swing_high,
                    'quantity': quantity,
                    'entry_time': next_candle['Datetime'],
                    'message': 'Condition 3 Flip: Closed below Base Zone → SELL',
                    'spot_price': spot_price
                }
                sig_flip = monitor_trade(sig_flip)
                signals.append(sig_flip)
                if not return_all_signals:
                    return sig_flip

    # --- Condition 4: Bearish breakdown inside Base Zone ---
    if (L1 < base_high and H1 > base_low) and (C1 < base_low):
        swing_high, swing_low = get_recent_swing(entry_time)
        sig = {
            'condition': 4,
            'position': 'SELL',
            'entry_price': L1,
            'stoploss': swing_high,
            'quantity': quantity,
            'entry_time': entry_time,
            'message': 'Condition 4: Bearish breakdown below Base Zone → SELL',
            'spot_price': spot_price
        }
        sig = monitor_trade(sig)
        signals.append(sig)
        if not return_all_signals:
            return sig

    return signals if signals else None
###########################################################################################

def plot_nifty_multiday(df, trading_days):
    fig = go.Figure()

    for i in range(1, len(trading_days)):
        day0 = trading_days[i - 1]   # Previous day
        day1 = trading_days[i]       # Current day

        # Filter data for current day
        df_day1 = df[df['Datetime'].dt.date == day1]
        if df_day1.empty:
            continue

        # Plot current day candles
        fig.add_trace(go.Candlestick(
            x=df_day1['Datetime'],
            open=df_day1['Open'],
            high=df_day1['High'],
            low=df_day1['Low'],
            close=df_day1['Close'],
            name=str(day1)
        ))

        # Get 3PM candle of previous day
        candle_3pm = df[
            (df['Datetime'].dt.date == day0) &
            (df['Datetime'].dt.hour == 15) &
            (df['Datetime'].dt.minute == 0)
        ]
        if not candle_3pm.empty:
            open_3pm = candle_3pm.iloc[0]['Open']
            close_3pm = candle_3pm.iloc[0]['Close']

            # Day1 session range
            x_start = df_day1['Datetime'].min()
            day1_3pm = df_day1[
                (df_day1['Datetime'].dt.hour == 15) &
                (df_day1['Datetime'].dt.minute == 0)
            ]
            if not day1_3pm.empty:
                x_end = day1_3pm['Datetime'].iloc[0]
            else:
                x_end = df_day1['Datetime'].max()

            # 🔵 Line for 3PM Open (mark as "High")
            fig.add_shape(
                type="line",
                x0=x_start, x1=x_end,
                y0=open_3pm, y1=open_3pm,
                line=dict(color="blue", width=1, dash="dot"),
            )
            fig.add_annotation(
                x=x_end, y=open_3pm, text="3PM Open", 
                showarrow=False, font=dict(color="blue", size=10)
            )

            # 🔴 Line for 3PM Close (mark as "Low")
            fig.add_shape(
                type="line",
                x0=x_start, x1=x_end,
                y0=close_3pm, y1=close_3pm,
                line=dict(color="red", width=1, dash="dot"),
            )
            fig.add_annotation(
                x=x_end, y=close_3pm, text="3PM Close", 
                showarrow=False, font=dict(color="red", size=10)
            )

    # Layout adjustments
    fig.update_layout(
        title="Multi-Day 3PM Strategy (Prev Day 3PM Open = High, Close = Low)",
        xaxis_rangeslider_visible=False,
        xaxis=dict(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),   # skip weekends
                dict(bounds=[15.5, 9.25], pattern="hour")  # skip off-hours
            ]
        )
    )
    return fig

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
       # Text input for any stock symbol
       selected_stock = st.text_input("Enter Stock Symbol (e.g., RELIANCE.NS)", value="RELIANCE.NS")  
       start_date = st.date_input("Select Start Date", value=datetime.today() - timedelta(days=15))
       end_date = st.date_input("Select End Date", value=datetime.today())

if start_date >= end_date:
    st.warning("End date must be after start date")
    st.stop()

# ✅ Download full data for range (start-1 day to end)
download_start = start_date - timedelta(days=1)  # To include previous day for first day
df = yf.download(selected_stock, start=download_start, end=end_date + timedelta(days=1), interval="15m")
if df.empty:
    st.warning("No data for selected range")
    st.stop()
df.columns = ['_'.join(col).strip() for col in df.columns.values]

df.reset_index(inplace=True)
df.rename(columns={'index': 'Datetime'}, inplace=True)  # Ensure proper name
#
# ✅ Normalize columns for any stock (Yahoo / CSV)
df.columns = [str(c) for c in df.columns]

rename_map = {}
for col in df.columns:
    col_low = col.lower()
    if 'date' in col_low or 'time' in col_low: rename_map[col] = 'Datetime'
    elif 'open' in col_low: rename_map[col] = 'Open'
    elif 'high' in col_low: rename_map[col] = 'High'
    elif 'low' in col_low: rename_map[col] = 'Low'
    elif 'close' in col_low: rename_map[col] = 'Close'
    elif 'volume' in col_low: rename_map[col] = 'Volume'
df.rename(columns=rename_map, inplace=True)

# ✅ Keep only needed columns
keep_cols = ['Datetime', 'Open', 'High', 'Low', 'Close']
df = df[[c for c in keep_cols if c in df.columns]]

# ✅ Ensure datetime type
df['Datetime'] = pd.to_datetime(df['Datetime'])
#st.write(df.columns)
#
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
# ✅ Initialize trade log list
# ✅ Initialize trade log list
trade_log = []

for i in range(1, len(unique_days)):
    day0 = unique_days[i-1]
    day1 = unique_days[i]

    day_df = df[df['Datetime'].dt.date.isin([day0, day1])]

    # Call your trading signal function
    signal = trading_signal_equity(day_df)

    if signal:   # append only if signal generated
        # ✅ Calculate PnL
        entry_price = signal.get("entry_price", None)
        exit_price = signal.get("exit_price", None)
        qty = signal.get("quantity", 1)

        if entry_price and exit_price:
            signal["PnL"] = round((exit_price - entry_price) * qty, 2)
        else:
            signal["PnL"] = None

        trade_log.append(signal)

# ✅ Convert to DataFrame once and display full log
if trade_log:
    trade_log_df = pd.DataFrame(trade_log)
    st.subheader("Trade Log with PnL")
    st.dataframe(trade_log_df, use_container_width=True)

    # ✅ Summary
    total_pnl = trade_log_df["PnL"].sum()
    st.metric("Total PnL", f"{total_pnl:,.2f}")
else:
    st.info("No trades found in the selected period.")
