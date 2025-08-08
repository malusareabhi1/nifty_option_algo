import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="15m chart + 3PM high/low", layout="wide")
st.title("📊 15-min Chart — Last 3 Days with 3:00 PM High / Low")

# ---- Inputs ----
symbol_input = st.text_input("Enter NSE symbol (example: NIFTY or INFY)", "NIFTY")
# map common NIFTY keyword to Yahoo symbol
if symbol_input.strip().upper() in ("NIFTY", "NIFTY50", "NIFTY 50", "^NSEI"):
    yf_symbol = "^NSEI"
else:
    yf_symbol = symbol_input.strip().upper()
    if not (yf_symbol.endswith(".NS") or yf_symbol.startswith("^")):
        yf_symbol = yf_symbol + ".NS"

interval = "15m"
days = 3

# ---- Download data ----
end_utc = datetime.utcnow()
start_utc = end_utc - timedelta(days=days + 1)  # +1 to ensure full coverage
with st.spinner(f"Downloading {yf_symbol} {interval} data ..."):
    df = yf.download(yf_symbol, start=start_utc, end=end_utc, interval=interval, progress=False)

if df is None or df.empty:
    st.error("No data returned. Try another symbol or try again later.")
    st.stop()

# Flatten MultiIndex column names to strings
df.columns = [str(c) for c in df.columns]

# reset index and rename datetime column
df = df.reset_index()
dt_col = df.columns[0]
df = df.rename(columns={dt_col: "Datetime"})

# ---- Robust UTC -> IST conversion ----
# ensure datetime dtype
df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")

# If tz-naive: localize to UTC then convert. If already tz-aware: just convert.
try:
    # try localize first (works if tz-naive)
    df["Datetime"] = df["Datetime"].dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata")
except TypeError:
    # already tz-aware -> just convert
    df["Datetime"] = df["Datetime"].dt.tz_convert("Asia/Kolkata")

# For consistent column names (uppercase from yfinance): Open/High/Low/Close -> use first-letter capitalized
open_col = next((c for c in df.columns if c.lower() == "open"), None)
high_col = next((c for c in df.columns if c.lower() == "high"), None)
low_col  = next((c for c in df.columns if c.lower() == "low"), None)
close_col= next((c for c in df.columns if c.lower() == "close"), None)

# rename to standard names used below
df = df.rename(columns={open_col: "Open", high_col: "High", low_col: "Low", close_col: "Close"})

# filter to last N calendar days (by IST date)
df["date_only"] = df["Datetime"].dt.date
unique_dates = sorted(df["date_only"].unique())[-days:]
df = df[df["date_only"].isin(unique_dates)].copy()
df.sort_values("Datetime", inplace=True)
df.reset_index(drop=True, inplace=True)

# ---- find 3:00 PM candle per day ----
markers = []
for d in sorted(df["date_only"].unique()):
    day_df = df[df["date_only"] == d]
    # match time "15:00" in IST
    candle = day_df[day_df["Datetime"].dt.strftime("%H:%M") == "15:00"]
    if not candle.empty:
        ts = candle["Datetime"].iloc[0]
        high_val = float(candle["High"].iloc[0])
        low_val  = float(candle["Low"].iloc[0])
        markers.append({"date": d, "ts": ts, "high": high_val, "low": low_val})

# ---- Plot candlestick ----
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df["Datetime"],
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    name="15-min"
))

# Mark and highlight each day's 3PM candle
for m in markers:
    ts = m["ts"]
    high_val = m["high"]
    low_val = m["low"]
    # horizontal lines
    fig.add_hline(y=high_val, line_dash="dash", line_color="green",
                  annotation_text=f"{m['date']} 3PM High {high_val}", annotation_position="top right")
    fig.add_hline(y=low_val, line_dash="dash", line_color="red",
                  annotation_text=f"{m['date']} 3PM Low {low_val}", annotation_position="bottom right")
    # shaded rectangle ~ highlight 15-min candle (±7 minutes)
    fig.add_vrect(x0=ts - pd.Timedelta(minutes=7), x1=ts + pd.Timedelta(minutes=8),
                  fillcolor="yellow", opacity=0.12, layer="below", line_width=0)

fig.update_layout(
    title=f"{yf_symbol} — Last {days} trading days (15-min)",
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
    height=700
)

st.plotly_chart(fig, use_container_width=True)

# show found markers
st.subheader("3PM candles found")
if markers:
    st.table(pd.DataFrame(markers)[["date", "ts", "high", "low"]].assign(ts=lambda x: x["ts"].dt.strftime("%Y-%m-%d %H:%M:%S %Z")))
else:
    st.info("No 15:00 IST candles found in the fetched range.")
