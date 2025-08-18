import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
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
def plot_new_candle(df):
    if 'last_candle_time' not in st.session_state:
        st.session_state.last_candle_time = df['Datetime'].max() - pd.Timedelta(minutes=15)
        df_to_plot = df[df['Datetime'] == st.session_state.last_candle_time + pd.Timedelta(minutes=15)]
    else:
        last_time = st.session_state.last_candle_time
        df_to_plot = df[df['Datetime'] > last_time]

    if df_to_plot.empty:
        st.info("No new candle yet")
        return None

    # Update last plotted candle time
    st.session_state.last_candle_time = df_to_plot['Datetime'].max()

    fig = go.Figure(data=[go.Candlestick(
        x=df_to_plot['Datetime'],
        open=df_to_plot['Open_^NSEI'],
        high=df_to_plot['High_^NSEI'],
        low=df_to_plot['Low_^NSEI'],
        close=df_to_plot['Close_^NSEI'],
        name="Nifty"
    )])

    # Add 3PM Base Zone lines from previous day
    prev_day = df_to_plot['Datetime'].dt.date.min() - pd.Timedelta(days=1)
    candle_3pm = df[(df['Datetime'].dt.date == prev_day) &
                     (df['Datetime'].dt.hour == 15) &
                     (df['Datetime'].dt.minute == 0)]
    if not candle_3pm.empty:
        open_3pm = candle_3pm.iloc[0]['Open_^NSEI']
        close_3pm = candle_3pm.iloc[0]['Close_^NSEI']
        fig.add_hline(y=open_3pm, line_dash="dot", line_color="blue", annotation_text="3PM Open", annotation_position="top left")
        fig.add_hline(y=close_3pm, line_dash="dot", line_color="red", annotation_text="3PM Close", annotation_position="bottom left")

    fig.update_layout(
        title=f"Nifty 15-min new candle - {df_to_plot['Datetime'].max()}",
        xaxis_rangeslider_visible=False
    )
    return fig

# ----------------- STREAMLIT APP -----------------
st.title("Nifty 50 15-min Live Candle Plot")

# Load latest Nifty data
df_nifty = load_nifty_data_15min(days_back=7)
if df_nifty is None or df_nifty.empty:
    st.warning("No data available")
    st.stop()

# Plot only the new candle
fig = plot_new_candle(df_nifty)
if fig:
    st.plotly_chart(fig, use_container_width=True)

# Auto-refresh every 15 minutes
st_autorefresh(interval=900000, key="nifty_refresh")  # 15 min
