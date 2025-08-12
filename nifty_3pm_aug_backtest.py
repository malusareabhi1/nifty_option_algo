import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import requests

from datetime import datetime, timedelta


st.set_page_config(layout="wide")
# Place at the very top of your script (or just before plotting)
st_autorefresh(interval=240000, limit=None, key="refresh")

st.title("Nifty 15-min Chart for Selected Date & Previous Day")

# Select date input (default today)
selected_date = st.date_input("Select date", value=datetime.today())

# Calculate date range to download (7 days before selected_date to day after selected_date)
start_date = selected_date - timedelta(days=7)
end_date = selected_date + timedelta(days=1)

# Download data for ^NSEI from start_date to end_date
df = yf.download("^NSEI", start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), interval="15m")

if df.empty:
    st.warning("No data downloaded for the selected range.")
    st.stop()
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
st.write(df.columns)
#st.write(df.columns)
# Convert to datetime & timezone aware
#df['Datetime'] = pd.to_datetime(df['Datetime'])
if df['Datetime_'].dt.tz is None:
    df['Datetime'] = df['Datetime_'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
else:
    df['Datetime'] = df['Datetime_'].dt.tz_convert('Asia/Kolkata')

#st.write(df.columns)
#st.write(df.head(10))

# Filter for last two trading days to plot
unique_days = df['Datetime'].dt.date.unique()
if len(unique_days) < 2:
    st.warning("Not enough data for two trading days")
else:
    last_day = unique_days[-2]
    today = unique_days[-1]

    df_plot = df[df['Datetime'].dt.date.isin([last_day, today])]

    # Get last day 3PM candle open and close
    candle_3pm = df_plot[(df_plot['Datetime'].dt.date == last_day) &
                         (df_plot['Datetime'].dt.hour == 15) &
                         (df_plot['Datetime'].dt.minute == 0)]

    if not candle_3pm.empty:
        open_3pm = candle_3pm.iloc[0]['Open_^NSEI']
        close_3pm = candle_3pm.iloc[0]['Close_^NSEI']
    else:
        open_3pm = None
        close_3pm = None
        st.warning("No 3:00 PM candle found for last trading day.")

    # Plot candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=df_plot['Datetime'],
        open=df_plot['Open_^NSEI'],
        high=df_plot['High_^NSEI'],
        low=df_plot['Low_^NSEI'],
        close=df_plot['Close_^NSEI']
    )])

    if open_3pm and close_3pm:
        fig.add_hline(y=open_3pm, line_dash="dot", line_color="blue", annotation_text="3PM Open")
        fig.add_hline(y=close_3pm, line_dash="dot", line_color="red", annotation_text="3PM Close")




    # Draw horizontal lines as line segments only between 3PM last day and 3PM next day

    
    fig.update_layout(title="Nifty 15-min candles - Last Day & Today", xaxis_rangeslider_visible=False)
    fig.update_layout(
    xaxis=dict(
        rangebreaks=[
            # Hide weekends (Saturday and Sunday)
            dict(bounds=["sat", "mon"]),
            # Hide hours outside of trading hours (NSE trading hours 9:15 to 15:30)
            dict(bounds=[15.5, 9.25], pattern="hour"),
        ]
    )
)


    st.plotly_chart(fig, use_container_width=True)
