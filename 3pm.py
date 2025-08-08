import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.title("📊 15-Min Chart with 3PM Candle High/Low")

# Select stock
symbol = st.text_input("Enter Stock/Nifty Symbol (e.g. RELIANCE.NS or ^NSEI)", "RELIANCE.NS")

# Fetch last 3 days of 15min data
end_date = datetime.now()
start_date = end_date - timedelta(days=3)

try:
    df = yf.download(symbol, start=start_date, end=end_date, interval="15m")
    
    # Ensure column names are standard
   def col_to_str(col):
        if isinstance(col, tuple):
            # Join tuple elements with underscore or space
            col_str = "_".join([str(c) for c in col if c])
        else:
            col_str = str(col)
        return col_str.capitalize()
    
        df.columns = [col_to_str(col) for col in df.columns]

    
    if not set(['Open', 'High', 'Low', 'Close']).issubset(df.columns):
        st.error(f"Missing OHLC columns in downloaded data: {df.columns.tolist()}")
    else:
        df = df.dropna()

        # Identify 3PM candle each day
        df['Date'] = df.index.date
        df['Time'] = df.index.time

        three_pm_candles = df[df.index.time == datetime.strptime("15:00", "%H:%M").time()]

        # Plot candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Candlesticks"
        )])

        # Mark 3PM candle high & low
        for idx, row in three_pm_candles.iterrows():
            fig.add_hline(y=row['High'], line=dict(color="green", dash="dot"), 
                          annotation_text=f"3PM High {row['High']:.2f}", annotation_position="top right")
            fig.add_hline(y=row['Low'], line=dict(color="red", dash="dot"), 
                          annotation_text=f"3PM Low {row['Low']:.2f}", annotation_position="bottom right")

        fig.update_layout(
            title=f"{symbol} - 15min Chart (Last 3 Days)",
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=700
        )

        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Error fetching data: {e}")
