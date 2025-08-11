import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, time
import plotly.graph_objects as go
import pytz

st.title("Nifty 15-Min Options Strategy Backtesting")

# Sidebar: Select number of past days (weekdays) to analyze
days_to_analyze = st.sidebar.slider(
    "Select number of past days to analyze", 
    min_value=3, max_value=20, value=5, step=1
)

@st.cache_data(ttl=600)
def load_data(days):
    # Download more days to ensure enough data for filtering
    df = yf.download("^NSEI", period="40d", interval="15m")
    df.reset_index(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [col.lower() for col in df.columns]

    df['datetime'] = pd.to_datetime(df['datetime'])

    local_tz = pytz.timezone('Asia/Kolkata')

    if df['datetime'].dt.tz is None:
        df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert(local_tz)
    else:
        df['datetime'] = df['datetime'].dt.tz_convert(local_tz)

    # Remove weekends from entire dataset
    df = df[df['datetime'].dt.weekday < 5].copy()

    # Now filter only last N calendar days with data available
    last_date = df['datetime'].dt.date.max()
    # Collect only data from last N weekdays (trading days)
    unique_days = sorted(df['datetime'].dt.date.unique())
    last_n_days = unique_days[-days:]  # last N weekdays with data

    df = df[df['datetime'].dt.date.isin(last_n_days)].copy()

    return df

def detect_all_conditions(df):
    df_3pm = df[(df['datetime'].dt.hour == 15) & (df['datetime'].dt.minute == 0)].copy()
    df_3pm['date'] = df_3pm['datetime'].dt.date

    df_915 = df[(df['datetime'].dt.hour == 9) & (df['datetime'].dt.minute == 15)].copy()
    df_915['date'] = df_915['datetime'].dt.date

    df_930 = df[(df['datetime'].dt.hour == 9) & (df['datetime'].dt.minute == 30)].copy()
    df_930['date'] = df_930['datetime'].dt.date

    records = []

    # Use only dates present in df_3pm (which is filtered already)
    for i in range(len(df_3pm) - 1):
        day0 = df_3pm.iloc[i]
        next_day = df_3pm.iloc[i + 1]['date']

        candle_915 = df_915[df_915['date'] == next_day]
        candle_930 = df_930[df_930['date'] == next_day]

        if candle_915.empty:
            continue
        candle_915 = candle_915.iloc[0]

        candle_930_available = not candle_930.empty
        candle_930 = candle_930.iloc[0] if candle_930_available else None

        ref_open = day0['open']
        ref_close = day0['close']
        upper_ref = max(ref_open, ref_close)
        lower_ref = min(ref_open, ref_close)

        # Condition 1: Next Day Breakout Upwards (No Major Gap)
        open_cond_1 = (candle_915['open'] < ref_open) and (candle_915['open'] < ref_close)
        close_cond_1 = (candle_915['close'] > ref_open) and (candle_915['close'] > ref_close)
        cond1_signal = open_cond_1 and close_cond_1

        # Condition 2: Major Gap Down
        cond2_signal = False
        if candle_930_available:
            gap_down_open = candle_915['open'] < lower_ref
            close_below_ref = (candle_915['close'] < ref_open) and (candle_915['close'] < ref_close)
            if gap_down_open and close_below_ref:
                ref_candle2_low = candle_915['low']
                breaks_below_ref_low = candle_930['low'] < ref_candle2_low
                cond2_signal = breaks_below_ref_low

        # Condition 3: Major Gap Up
        cond3_signal = False
        if candle_930_available:
            gap_up_open = candle_915['open'] > upper_ref
            close_above_ref = (candle_915['close'] > ref_open) and (candle_915['close'] > ref_close)
            if gap_up_open and close_above_ref:
                ref_candle2_high = candle_915['high']
                breaks_above_ref_high = candle_930['high'] > ref_candle2_high
                cond3_signal = breaks_above_ref_high

        # Condition 4: Next Day Breakout Downwards (No Major Gap)
        open_above_both = candle_915['open'] > upper_ref
        low_below_both = candle_915['low'] < lower_ref
        close_below_both = candle_915['close'] < lower_ref
        cond4_signal = open_above_both and low_below_both and close_below_both

        records.append({
            'Day0_3PM_Date': day0['date'],
            'Day1_Date': next_day,
            'Day0_3PM_Open': ref_open,
            'Day0_3PM_Close': ref_close,
            'Day1_9_15_Open': candle_915['open'],
            'Day1_9_15_High': candle_915['high'],
            'Day1_9_15_Low': candle_915['low'],
            'Day1_9_15_Close': candle_915['close'],
            'Day1_9_30_High': candle_930['high'] if candle_930_available else None,
            'Day1_9_30_Low': candle_930['low'] if candle_930_available else None,
            'Condition1_BuyCall': cond1_signal,
            'Condition2_BuyPut': cond2_signal,
            'Condition3_BuyCall': cond3_signal,
            'Condition4_BuyPut': cond4_signal
        })

    return pd.DataFrame(records)

def simulate_option_premium(underlying_price, strike_price, option_type):
    base_premium = 20
    if option_type == 'call':
        intrinsic = max(0, underlying_price - strike_price)
    else:
        intrinsic = max(0, strike_price - underlying_price)
    return intrinsic + base_premium

def simulate_trades(df, signals_df):
    from datetime import time

    trades = []

    lot_size = 750
    position_size = 10  # lots

    for idx, row in signals_df.iterrows():
        day1 = row['Day1_Date']
        day1_candles = df[df['datetime'].dt.date == day1].reset_index(drop=True)
        if day1_candles.empty:
            continue

        entry_candle_idx = day1_candles.index[day1_candles['datetime'].dt.time == time(9, 30)]
        if len(entry_candle_idx) == 0:
            continue
        entry_idx = entry_candle_idx[0]
        entry_candle = day1_candles.loc[entry_idx]

        underlying_entry_price = entry_candle['close']

        strike_price = round(underlying_entry_price / 100) * 100

        for cond_num, (flag_col, opt_type) in enumerate([
            ('Condition1_BuyCall', 'call'),
            ('Condition2_BuyPut', 'put'),
            ('Condition3_BuyCall', 'call'),
            ('Condition4_BuyPut', 'put'),
        ], 1):
            if row[flag_col]:
                entry_premium = simulate_option_premium(underlying_entry_price, strike_price, opt_type)
                stoploss_price = entry_premium * 0.90
                target_price = entry_premium * 1.10

                position_qty = lot_size * position_size
                qty_half = position_qty / 2

                trade_open_time = entry_candle['datetime']
                trade_close_time = None
                exit_reason = None
                exit_price = None
                qty_remaining = position_qty
                profit_realized = 0

                next_idx = entry_idx + 1
                if next_idx < len(day1_candles):
                    next_candle = day1_candles.loc[next_idx]
                    premium_next = simulate_option_premium(next_candle['close'], strike_price, opt_type)

                    if premium_next >= target_price:
                        profit_realized += qty_half * (target_price - entry_premium)
                        qty_remaining = qty_half
                        stoploss_price = entry_premium

                        next_next_idx = next_idx + 1
                        if next_next_idx < len(day1_candles):
                            next_next_candle = day1_candles.loc[next_next_idx]
                            premium_next_next = simulate_option_premium(next_next_candle['close'], strike_price, opt_type)
                            if premium_next_next <= stoploss_price:
                                exit_reason = "Stoploss hit after partial profit"
                                trade_close_time = next_next_candle['datetime']
                                exit_price = premium_next_next
                                profit_realized += qty_remaining * (exit_price - entry_premium)
                                qty_remaining = 0
                            else:
                                exit_reason = "Time exit after partial profit"
                                trade_close_time = next_next_candle['datetime']
                                exit_price = premium_next_next
                                profit_realized += qty_remaining * (exit_price - entry_premium)
                                qty_remaining = 0
                        else:
                            exit_reason = "Time exit after partial profit (no more candles)"
                            trade_close_time = next_candle['datetime']
                            exit_price = premium_next
                            profit_realized += qty_remaining * (exit_price - entry_premium)
                            qty_remaining = 0

                    elif premium_next <= stoploss_price:
                        exit_reason = "Stoploss hit"
                        trade_close_time = next_candle['datetime']
                        exit_price = premium_next
                        profit_realized += qty_remaining * (exit_price - entry_premium)
                        qty_remaining = 0

                    else:
                        exit_reason = "Time exit"
                        trade_close_time = next_candle['datetime']
                        exit_price = premium_next
                        profit_realized += qty_remaining * (exit_price - entry_premium)
                        qty_remaining = 0
                else:
                    exit_reason = "Time exit (no next candle)"
                    trade_close_time = entry_candle['datetime']
                    exit_price = entry_premium
                    profit_realized = 0

                trades.append({
                    'Trade_Condition': f'Condition {cond_num}',
                    'Trade_Date': day1,
                    'Entry_Time': trade_open_time,
                    'Exit_Time': trade_close_time,
                    'Option_Type': opt_type,
                    'Strike_Price': strike_price,
                    'Entry_Premium': entry_premium,
                    'Exit_Premium': exit_price,
                    'Position_Qty': position_qty,
                    'Profit': profit_realized,
                    'Exit_Reason': exit_reason
                })

    return pd.DataFrame(trades)

def plot_candles_with_signals(df, signals_df):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df['datetime'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Nifty 15min'
    ))

    for _, row in signals_df.iterrows():
        day0_date = row['Day0_3PM_Date']
        day1_date = row['Day1_Date']
        candle_3pm = df[(df['datetime'].dt.date == day0_date) &
                        (df['datetime'].dt.hour == 15) &
                        (df['datetime'].dt.minute == 0)]
        if candle_3pm.empty:
            continue
        candle_3pm = candle_3pm.iloc[0]
        open_price = candle_3pm['open']
        close_price = candle_3pm['close']

        fig.add_hline(y=open_price, line_dash="dash", line_color="blue",
                      annotation_text=f"{day0_date} 3PM Open",
                      annotation_position="top left")
        fig.add_hline(y=close_price, line_dash="dash", line_color="red",
                      annotation_text=f"{day0_date} 3PM Close",
                      annotation_position="bottom left")

        candle_915 = df[(df['datetime'].dt.date == day1_date) &
                        (df['datetime'].dt.hour == 9) &
                        (df['datetime'].dt.minute == 15)]
        if candle_915.empty:
            continue
        candle_915 = candle_915.iloc[0]
        x = candle_915['datetime']
        y = candle_915['close']

        if row['Condition1_BuyCall']:
            fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers',
                                     marker=dict(color='green', size=12, symbol='triangle-up'),
                                     name='Cond1 Buy Call'))
        if row['Condition2_BuyPut']:
            fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers',
                                     marker=dict(color='red', size=12, symbol='triangle-down'),
                                     name='Cond2 Buy Put'))
        if row['Condition3_BuyCall']:
            fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers',
                                     marker=dict(color='lime', size=12, symbol='triangle-up'),
                                     name='Cond3 Buy Call'))
        if row['Condition4_BuyPut']:
            fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers',
                                     marker=dict(color='darkred', size=12, symbol='triangle-down'),
                                     name='Cond4 Buy Put'))

    fig.update_layout(title="Nifty 15-min Candles with Strategy Signals",
                      xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# --- Main ---

df = load_data(days_to_analyze)
st.write(f"Loaded data for last {days_to_analyze} trading days (weekdays only).")

st.subheader("Raw Data Sample")
st.dataframe(df.tail(10))

st.subheader("Detecting Strategy Signals...")
signals_df = detect_all_conditions(df)
if not signals_df.empty:
    st.dataframe(signals_df)
else:
    st.write("No signals detected.")

st.subheader("Simulating Trades...")
trade_results = simulate_trades(df, signals_df)
if not trade_results.empty:
    st.dataframe(trade_results)
else:
    st.write("No trades executed in simulation.")

st.subheader("Visualizing Candles and Signals")
plot_candles_with_signals(df, signals_df)
