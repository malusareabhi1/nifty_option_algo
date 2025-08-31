import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Streamlit page settings
st.set_page_config(layout="wide")
st.title("NIFTY Index - 15 Min Candle Data")

# Date range selection
st.sidebar.header("Select Date Range")
end_date = st.sidebar.date_input("End Date", datetime.today())
start_date = st.sidebar.date_input("Start Date", datetime.today() - timedelta(days=30))

# Validate date range
if start_date > end_date:
    st.error("Start date cannot be after end date.")
else:
    st.write(f"Fetching NIFTY data from {start_date} to {end_date}")

    # Fetch NIFTY index data (Symbol for NIFTY 50 index is ^NSEI on Yahoo Finance)
    ticker = "^NSEI"
    interval = "15m"

    try:
        nifty_data = yf.download(ticker, start=start_date, end=end_date + timedelta(days=1), interval=interval)

        if nifty_data.empty:
            st.warning("No data found for the selected range. Try different dates.")
        else:
            # Reset index and rename columns
            nifty_data.reset_index(inplace=True)
            nifty_data.rename(columns={"Datetime": "DateTime"}, inplace=True)

            # Display data
            st.subheader("NIFTY 15-Minute Candle Data")
            st.dataframe(nifty_data)

            # Download option
            csv = nifty_data.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", data=csv, file_name="nifty_15min_data.csv", mime="text/csv")

            # Show chart
            st.subheader("Candlestick Chart")
            import plotly.graph_objects as go
            fig = go.Figure(data=[go.Candlestick(x=nifty_data['DateTime'],
                                                 open=nifty_data['Open'],
                                                 high=nifty_data['High'],
                                                 low=nifty_data['Low'],
                                                 close=nifty_data['Close'])])
            fig.update_layout(title="NIFTY 15-Min Candle Chart", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error fetching data: {e}")
