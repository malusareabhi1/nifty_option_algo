import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
#import plotly.express as px
#import plotly.express as px
st.set_page_config(page_title="NIFTY 15-Min Chart with 3PM Breakout Strategy", layout="wide")

st.title("📈 NIFTY 15-Min Chart – 3PM Breakout/Breakdown Strategy")

st.sidebar.header("Settings")
offset_points = st.sidebar.number_input("Offset Points for Breakout/Breakdown", value=100, step=10)
analysis_days = st.sidebar.slider("Number of Days to Analyze", min_value=5, max_value=20, value=5, step=1)

st.sidebar.subheader("💼 Paper Trading Settings")
initial_capital = st.sidebar.number_input("Starting Capital (₹)", value=100000, step=10000)
risk_per_trade_pct = st.sidebar.slider("Risk per Trade (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)


st.markdown("""
## 📘 Strategy Explanation

This intraday breakout/backtest strategy is based on the NIFTY 15-minute chart.

- 🔼 **Breakout Logic**: At 3:00 PM, capture the high of the 15-minute candle. On the next trading day, if price crosses 3PM High + offset points, mark as a breakout.
- 🔽 **Breakdown Logic**: Track 3PM Close. On the next day, if price crosses below previous close and then drops offset points lower, mark as breakdown.

*Useful for swing and intraday traders planning trades based on end-of-day momentum.*

---
""")

#import plotly.express as px

def plot_cumulative_pnl(df, title="Cumulative P&L"):
    df['cumulative_pnl'] = df['P&L'].cumsum()
    fig = px.line(df, x='3PM Date', y='cumulative_pnl', title=title)
    fig.update_layout(height=400)
    return fig
    
@st.cache_data(ttl=3600)
def load_nifty_data(ticker="^NSEI", interval="15m", period="60d"):
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False)
        if df.empty:
            st.error("❌ No data returned from yfinance.")
            st.stop()

        df.reset_index(inplace=True)

        # ✅ Flatten MultiIndex columns if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]

        # ✅ Find datetime column automatically
        datetime_col = next((col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()), None)

        if not datetime_col:
            st.error("❌ No datetime column found after reset_index().")
            st.write("📋 Available columns:", df.columns.tolist())
            st.stop()

        df.rename(columns={datetime_col: 'datetime'}, inplace=True)

        # ✅ Convert to datetime and localize
        df['datetime'] = pd.to_datetime(df['datetime'])
        if df['datetime'].dt.tz is None:
            df['datetime'] = df['datetime'].dt.tz_localize('UTC')
        df['datetime'] = df['datetime'].dt.tz_convert('Asia/Kolkata')

        # ✅ Now lowercase column names
        df.columns = [col.lower() for col in df.columns]

        # ✅ Filter NSE market hours (9:15 to 15:30)
        df = df[(df['datetime'].dt.time >= pd.to_datetime("09:15").time()) &
                (df['datetime'].dt.time <= pd.to_datetime("15:30").time())]

        return df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()
def filter_last_n_days(df, n_days):
    df['date'] = df['datetime'].dt.date
    unique_days = sorted(df['date'].unique())
    last_days = unique_days[-n_days:]
    filtered_df = df[df['date'].isin(last_days)].copy()
    filtered_df.drop(columns='date', inplace=True)
    return filtered_df

def generate_trade_logs(df, offset):
    df_3pm = df[(df['datetime'].dt.hour == 15) & (df['datetime'].dt.minute == 0)].reset_index(drop=True)
    breakout_logs = []
    breakdown_logs = []

    for i in range(len(df_3pm) - 1):
        current = df_3pm.iloc[i]
        next_day_date = df_3pm.iloc[i + 1]['datetime'].date()

        threepm_high = current['high']
        threepm_close = current['close']
        threepm_low = current['low']

        # Breakout parameters
        entry_breakout = threepm_high + offset
        sl_breakout = threepm_low
        target_breakout = entry_breakout + (entry_breakout - sl_breakout) * 1.5

        # Breakdown parameters
        entry_breakdown = threepm_close
        sl_breakdown = threepm_high
        target_breakdown = entry_breakdown - (sl_breakdown - entry_breakdown) * 1.5

        next_day_data = df[(df['datetime'].dt.date == next_day_date) &
                           (df['datetime'].dt.time > pd.to_datetime("09:30").time())].copy()
        next_day_data.sort_values('datetime', inplace=True)

        # --- Breakout Logic ---
        entry_row = next_day_data[next_day_data['high'] >= entry_breakout]
        if not entry_row.empty:
            entry_time = entry_row.iloc[0]['datetime']
            after_entry = next_day_data[next_day_data['datetime'] >= entry_time]

            target_hit = after_entry[after_entry['high'] >= target_breakout]
            sl_hit = after_entry[after_entry['low'] <= sl_breakout]

            if not target_hit.empty:
                breakout_result = '🎯 Target Hit'
                exit_price = target_hit.iloc[0]['high']
                exit_time = target_hit.iloc[0]['datetime']
            elif not sl_hit.empty:
                breakout_result = '🛑 Stop Loss Hit'
                exit_price = sl_hit.iloc[0]['low']
                exit_time = sl_hit.iloc[0]['datetime']
            else:
                breakout_result = '⏰ Time Exit'
                exit_price = after_entry.iloc[-1]['close']
                exit_time = after_entry.iloc[-1]['datetime']

            pnl = round(exit_price - entry_breakout, 2)

            breakout_logs.append({
                '3PM Date': current['datetime'].date(),
                'Next Day': next_day_date,
                '3PM High': round(threepm_high, 2),
                'Entry': round(entry_breakout, 2),
                'SL': round(sl_breakout, 2),
                'Target': round(target_breakout, 2),
                'Entry Time': entry_time.time(),
                'Exit Time': exit_time.time(),
                'Result': breakout_result,
                'P&L': pnl
            })

        # --- Breakdown Logic ---
        crossed_down = False
        entry_time = None
        exit_time = None
        pnl = 0.0

        for j in range(1, len(next_day_data)):
            prev = next_day_data.iloc[j - 1]
            curr = next_day_data.iloc[j]

            if not crossed_down and prev['high'] > entry_breakdown and curr['low'] < entry_breakdown:
                crossed_down = True
                entry_time = curr['datetime']
                after_entry = next_day_data[next_day_data['datetime'] >= entry_time]

                target_hit = after_entry[after_entry['low'] <= target_breakdown]
                sl_hit = after_entry[after_entry['high'] >= sl_breakdown]

                if not target_hit.empty:
                    breakdown_result = '🎯 Target Hit'
                    exit_price = target_hit.iloc[0]['low']
                    exit_time = target_hit.iloc[0]['datetime']
                elif not sl_hit.empty:
                    breakdown_result = '🛑 Stop Loss Hit'
                    exit_price = sl_hit.iloc[0]['high']
                    exit_time = sl_hit.iloc[0]['datetime']
                else:
                    breakdown_result = '⏰ Time Exit'
                    exit_price = after_entry.iloc[-1]['close']
                    exit_time = after_entry.iloc[-1]['datetime']

                pnl = round(entry_breakdown - exit_price, 2)

                breakdown_logs.append({
                    '3PM Date': current['datetime'].date(),
                    'Next Day': next_day_date,
                    '3PM Close': round(threepm_close, 2),
                    'Entry': round(entry_breakdown, 2),
                    'SL': round(sl_breakdown, 2),
                    'Target': round(target_breakdown, 2),
                    'Entry Time': entry_time.time(),
                    'Exit Time': exit_time.time(),
                    'Result': breakdown_result,
                    'P&L': pnl
                })
                break  # Stop after first valid breakdown entry

    breakout_df = pd.DataFrame(breakout_logs)
    breakdown_df = pd.DataFrame(breakdown_logs)
    return breakout_df, breakdown_df


#df['date'] = df['datetime'].dt.date




def plot_candlestick_chart(df, df_3pm):
    fig = go.Figure(data=[go.Candlestick(
        x=df['datetime'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="NIFTY"
    )])

    fig.update_traces(increasing_line_color='green', decreasing_line_color='red')

    # 🚀 Add horizontal lines from 3PM to next day 3PM
    for i in range(len(df_3pm) - 1):
        start_time = df_3pm.iloc[i]['datetime']
        end_time = df_3pm.iloc[i + 1]['datetime']
        high_val = df_3pm.iloc[i]['high']
        low_val = df_3pm.iloc[i]['low']

        fig.add_trace(go.Scatter(
            x=[start_time, end_time],
            y=[high_val, high_val],
            mode='lines',
            name='3PM High',
            line=dict(color='orange', width=1.5, dash='dot'),
            showlegend=(i == 0)  # Show legend only once
        ))

        fig.add_trace(go.Scatter(
            x=[start_time, end_time],
            y=[low_val, low_val],
            mode='lines',
            name='3PM Low',
            line=dict(color='cyan', width=1.5, dash='dot'),
            showlegend=(i == 0)
        ))

    fig.update_layout(
        title="NIFTY 15-Min Chart (Last {} Trading Days)".format(analysis_days),
        xaxis_title="DateTime (IST)",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        xaxis=dict(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),
                dict(bounds=[16, 9.15], pattern="hour")
            ],
            showgrid=False
        ),
        yaxis=dict(showgrid=True),
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        height=600
    )
    if len(df_3pm) > 0:
    last_candle = df_3pm.iloc[-1]
    start_time = last_candle['datetime']
    # Extend line 1 hour after last candle, or till last data timestamp
    end_time = df['datetime'].max() + pd.Timedelta(minutes=15)  # 15 min beyond last data

    high_val = last_candle['high']
    low_val = last_candle['low']

    fig.add_trace(go.Scatter(
        x=[start_time, end_time],
        y=[high_val, high_val],
        mode='lines',
        name='3PM High',
        line=dict(color='orange', width=1.5, dash='dot'),
        showlegend=False  # No need to show legend again
    ))

    fig.add_trace(go.Scatter(
        x=[start_time, end_time],
        y=[low_val, low_val],
        mode='lines',
        name='3PM Low',
        line=dict(color='cyan', width=1.5, dash='dot'),
        showlegend=False
    ))
    return fig

def simulate_paper_trades(trade_df, initial_capital, risk_pct):
    trade_df = trade_df.copy()
    capital = initial_capital
    capital_log = []

    for i, row in trade_df.iterrows():
        entry_price = row['Entry']
        sl = row['SL']
        risk_per_unit = abs(entry_price - sl)
        risk_amount = capital * (risk_pct / 100)

        qty = int(risk_amount / risk_per_unit) if risk_per_unit != 0 else 0
        capital_used = qty * entry_price
        pnl = row['P&L'] * qty
        capital += pnl  # Update capital with P&L

        trade_df.at[i, 'Qty'] = qty
        trade_df.at[i, 'Capital Used'] = round(capital_used, 2)
        trade_df.at[i, 'Realized P&L'] = round(pnl, 2)
        trade_df.at[i, 'Capital After Trade'] = round(capital, 2)

    return trade_df


def show_trade_metrics(df, label):
    total_trades = len(df)
    wins = df[df['Result'] == '🎯 Target Hit'].shape[0]
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    avg_pnl = df['P&L'].mean() if total_trades > 0 else 0
    total_pnl = df['P&L'].sum() if total_trades > 0 else 0

    st.success(f"✅ {label} – Total Trades: {total_trades}, Wins: {wins} ({win_rate:.2f}%), Avg P&L: ₹{avg_pnl:.2f}, Total P&L: ₹{total_pnl:,.2f}")

def color_pnl(val):
    color = 'green' if val > 0 else 'red' if val < 0 else 'white'
    return f'color: {color}; font-weight: bold;'

# ----------------------- MAIN ------------------------

df = load_nifty_data(period=f"{analysis_days}d")

if df.empty:
    st.stop()

df = filter_last_n_days(df, analysis_days)
df_3pm = df[(df['datetime'].dt.hour == 15) & (df['datetime'].dt.minute == 0)].reset_index(drop=True)
#st.write("Available columns:", df.columns.tolist())
# ✅ Manually set the required columns (works for most tickers)
df = df.rename(columns={
    'datetime': 'datetime',
    'open_^nsei': 'open',
    'high_^nsei': 'high',
    'low_^nsei': 'low',
    'close_^nsei': 'close',
    'volume_^nsei': 'volume'
})
#st.write("Available columns:", df.columns.tolist())
required_cols = ['datetime', 'open', 'high', 'low', 'close']

missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    st.error(f"Missing columns: {missing_cols}")
    st.stop()



trade_log_df, breakdown_df = generate_trade_logs(df, offset_points)


trade_log_df = simulate_paper_trades(trade_log_df, initial_capital, risk_per_trade_pct)
breakdown_df = simulate_paper_trades(breakdown_df, initial_capital, risk_per_trade_pct)

# 🔁 Simulate paper trading
paper_trade_log_df = simulate_paper_trades(trade_log_df, initial_capital, risk_per_trade_pct)
paper_breakdown_df = simulate_paper_trades(breakdown_df, initial_capital, risk_per_trade_pct)


#st.write("📋 df_3pm Columns:", df_3pm.columns.tolist())
df_3pm = df_3pm.rename(columns={
    'datetime': 'datetime',
    'open_^nsei': 'open',
    'high_^nsei': 'high',
    'low_^nsei': 'low',
    'close_^nsei': 'close',
    'volume_^nsei': 'volume'
})
#st.write("📋 df_3pm Columns:", df_3pm.columns.tolist())
# Plot chart
# After the for loop for all but last 3PM candle, add this:



fig = plot_candlestick_chart(df, df_3pm)
st.subheader("🕯️ NIFTY Candlestick Chart (15m)")
st.plotly_chart(fig, use_container_width=True)
