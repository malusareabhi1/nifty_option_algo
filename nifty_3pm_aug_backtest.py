import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

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
if 'Datetime_' in df.columns:
    df.rename(columns={'Datetime_': 'Datetime'}, inplace=True)
elif 'Date' in df.columns:
    df.rename(columns={'Date': 'Datetime'}, inplace=True)
# Reset index to get Datetime as column
df = df.reset_index()
st.write(df.columns)
#df['Datetime'] = pd.to_datetime(df['Datetime'])
if df['Datetime'].dt.tz is None:
    df['Datetime'] = df['Datetime_'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
else:
    df['Datetime'] = df['Datetime'].dt.tz_convert('Asia/Kolkata')


# Get unique sorted trading dates in data
unique_dates = sorted(df['Datetime'].dt.date.unique())

if selected_date not in unique_dates:
    st.warning(f"No trading data for {selected_date}. Showing last available trading date instead.")
    # Pick closest previous trading day
    valid_dates = [d for d in unique_dates if d <= selected_date]
    if not valid_dates:
        st.error("No trading data available before selected date.")
        st.stop()
    plot_date = valid_dates[-1]
else:
    plot_date = selected_date

# Get previous trading day for plot_date
plot_date_idx = unique_dates.index(plot_date)
prev_date_idx = max(0, plot_date_idx - 1)
days_to_plot = unique_dates[prev_date_idx:plot_date_idx+1]

# Filter df for those two days
df_plot = df[df['Datetime'].dt.date.isin(days_to_plot)]

st.write(f"Showing data for {days_to_plot[0]} and {days_to_plot[1]}")

# Plot candlestick chart
fig = go.Figure(data=[go.Candlestick(
    x=df_plot['Datetime'],
    open=df_plot['Open_^NSEI'],
    high=df_plot['High_^NSEI'],
    low=df_plot['Low_^NSEI'],
    close=df_plot['Close_^NSEI']
)])

fig.update_layout(
    title="Nifty 15-min Candlestick Chart",
    xaxis_rangeslider_visible=False,
    xaxis=dict(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),  # hide weekends
            dict(bounds=[15.5, 9.25], pattern="hour")  # hide non-trading hours
        ]
    )
)

st.plotly_chart(fig, use_container_width=True)

