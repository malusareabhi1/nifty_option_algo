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

# ----------------- STREAMLIT APP -----------------
st.title("Nifty 50 15-min Live Candlestick Chart")

# Initialize session_state for all candles
if 'df_plot' not in st.session_state:
    st.session_state.df_plot = pd.DataFrame()

# Load latest Nifty data
df_nifty = load_nifty_data_15min(days_back=2)
if df_nifty is None or df_nifty.empty:
    st.warning("No data available")
    st.stop()

# Append only new candles
if st.session_state.df_plot.empty:
    st.session_state.df_plot = df_nifty
else:
    last_time = st.session_state.df_plot['Datetime'].max()
    new_candles = df_nifty[df_nifty['Datetime'] > last_time]
    if not new_candles.empty:
        st.session_state.df_plot = pd.concat([st.session_state.df_plot, new_candles], ignore_index=True)

# Plot all candles
fig = plot_all_candles(st.session_state.df_plot)
st.plotly_chart(fig, use_container_width=True)

# Auto-refresh every 15 minutes
st_autorefresh(interval=900000, key="nifty_refresh")  # 15 min
