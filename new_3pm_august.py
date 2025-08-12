import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# Load Nifty data - example symbol
symbol = "^NSEI"

df = yf.download(symbol, period="7d", interval="15m")
df.reset_index(inplace=True)

# Convert to datetime
df['Datetime'] = pd.to_datetime(df['Datetime'])

# Timezone handling
if df['Datetime'].dt.tz is None:
    df['Datetime'] = df['Datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
else:
    df['Datetime'] = df['Datetime'].dt.tz_convert('Asia/Kolkata')

# Filter to trading days and times if needed (optional)
df = df[df['Datetime'].dt.weekday < 5]

st.write("Data sample:", df.head())

# Plot candles
fig = go.Figure(data=[go.Candlestick(
    x=df['Datetime'],
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name=symbol
)])

fig.update_layout(title=f"{symbol} 15-min Candlestick Chart", xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)
