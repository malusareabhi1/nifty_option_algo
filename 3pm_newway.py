# app.py
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

# =============== PAGE & REFRESH ===============
st.set_page_config(page_title="3PM Strategy Dashboard", layout="wide")
st_autorefresh(interval=240_000, key="auto_refresh")  # 4 min

# =============== SIDEBAR ===============
st.sidebar.title("⚙️ Controls")
mode = st.sidebar.selectbox("Mode", ["Paper Trade", "Backtest (basic)", "Live (stub)"])
symbol = st.sidebar.selectbox("Symbol", ["^NSEI"], index=0)
interval = "15m"

risk_capital = st.sidebar.number_input("Total Capital (₹)", 50_000, 1_00_00_000, 2_00_000, step=10_000)
risk_per_trade_pct = st.sidebar.slider("Risk % per trade", 0.25, 5.0, 1.0, 0.25)
lots = st.sidebar.number_input("Lots", 1, 50, 10)
lot_size = st.sidebar.number_input("Lot Size", 25, 150, 75)
take_profit_pct = st.sidebar.slider("Target %", 1.0, 25.0, 10.0, 0.5)
stop_loss_pct = st.sidebar.slider("Stoploss %", 1.0, 25.0, 10.0, 0.5)
time_exit_minutes = st.sidebar.number_input("Time Exit (mins)", 5, 60, 16)
days_back = st.sidebar.slider("Days back (download)", 3, 20, 7)
trading_start, trading_end = datetime.strptime("09:30","%H:%M").time(), datetime.strptime("14:00","%H:%M").time()

st.sidebar.markdown("---")
selected_date = st.sidebar.date_input("Focus date (today if live/paper)", datetime.today())

# =============== SESSION STATE ===============
if "trade_log" not in st.session_state:
    st.session_state.trade_log = pd.DataFrame()
if "last_signal_key" not in st.session_state:
    st.session_state.last_signal_key = set()     # to avoid repeat orders per candle/condition
if "last_candle_time" not in st.session_state:
    st.session_state.last_candle_time = None

# =============== DATA LOADER ===============
def load_ohlc_15m(sym="^NSEI", days=7):
    end_date = datetime.today().date() + timedelta(days=1)
    start_date = end_date - timedelta(days=days)
    df = yf.download(sym, start=start_date, end=end_date, interval="15m", progress=False)
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    # Standardize columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(col) if isinstance(col, tuple) else col for col in df.columns]
    # Normalize datetime
    if "Datetime" not in df.columns:
        if "Date" in df.columns:
            df.rename(columns={"Date": "Datetime"}, inplace=True)
        elif "Datetime_" in df.columns:
            df.rename(columns={"Datetime_": "Datetime"}, inplace=True)
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True).dt.tz_convert("Asia/Kolkata")
    # Trading hours filter (NSE 09:15–15:30 for display; signals will use 09:30–14:00)
    df = df[(df["Datetime"].dt.time >= datetime.strptime("09:15","%H:%M").time()) &
            (df["Datetime"].dt.time <= datetime.strptime("15:30","%H:%M").time())]
    # Rename OHLC to stable names for this app
    cols_map = {}
    for c in ["Open","High","Low","Close"]:
        if c in df.columns: cols_map[c] = f"{c}_^NSEI"
        elif f"{c}_{sym}" in df.columns: cols_map[f"{c}_{sym}"] = f"{c}_^NSEI"
    df.rename(columns=cols_map, inplace=True)
    return df[["Datetime","Open_^NSEI","High_^NSEI","Low_^NSEI","Close_^NSEI"]].sort_values("Datetime").reset_index(drop=True)

df = load_ohlc_15m(symbol, days_back)
if df.empty:
    st.error("No data available.")
    st.stop()

# =============== STRATEGY CORE ===============
def get_nearest_weekly_expiry(today_date: pd.Timestamp) -> pd.Timestamp:
    # Simple nearest weekly (Thu) expiry; adjust for holidays in live build
    d = pd.Timestamp(today_date).normalize()
    # Next Thursday logic
    while d.weekday() != 3:
        d += pd.Timedelta(days=1)
    return d

def trading_signal_all_conditions1(df: pd.DataFrame, quantity=10*75, return_all_signals=False):
    signals = []
    spot_price = df["Close_^NSEI"].iloc[-1]
    work = df.copy()
    work["Date"] = work["Datetime"].dt.date
    days = sorted(work["Date"].unique())
    if len(days) < 2:
        return None
    day0, day1 = days[-2], days[-1]

    c3 = work[(work["Date"]==day0)&(work["Datetime"].dt.hour==15)&(work["Datetime"].dt.minute==0)]
    if c3.empty: return None
    base_open = c3.iloc[0]["Open_^NSEI"]; base_close = c3.iloc[0]["Close_^NSEI"]
    base_low, base_high = min(base_open, base_close), max(base_open, base_close)

    c915 = work[(work["Date"]==day1)&(work["Datetime"].dt.hour==9)&(work["Datetime"].dt.minute==15)]
    if c915.empty: return None
    H1 = c915.iloc[0]["High_^NSEI"]; L1 = c915.iloc[0]["Low_^NSEI"]; C1 = c915.iloc[0]["Close_^NSEI"]
    entry_time = c915.iloc[0]["Datetime"]
    expiry = get_nearest_weekly_expiry(pd.to_datetime(day1))
    after = work[(work["Date"]==day1)&(work["Datetime"]>entry_time)].sort_values("Datetime")

    # Condition 1
    if (L1 < base_high and H1 > base_low) and (C1 > base_high):
        sig = dict(condition=1, option_type="CALL", buy_price=H1,
                   stoploss=H1*(1-stop_loss_pct/100), take_profit=H1*(1+take_profit_pct/100),
                   quantity=quantity, expiry=expiry, entry_time=entry_time,
                   message="Cond 1: Bullish breakout above Base Zone → Buy CALL above H1",
                   spot_price=spot_price)
        signals.append(sig)
        if not return_all_signals: return sig

    # Condition 2 (+ 2.7 flip)
    if C1 < base_low:
        for _, row in after.iterrows():
            if row["Low_^NSEI"] < L1:
                sig = dict(condition=2, option_type="PUT", buy_price=L1,
                           stoploss=L1*(1+stop_loss_pct/100), take_profit=L1*(1-take_profit_pct/100),
                           quantity=quantity, expiry=expiry, entry_time=row["Datetime"],
                           message="Cond 2: Gap down confirmed → Buy PUT below L1",
                           spot_price=spot_price)
                signals.append(sig)
                if not return_all_signals: return sig
            if row["Close_^NSEI"] > base_high:
                ref_high = row["High_^NSEI"]
                sig = dict(condition=2.7, option_type="CALL", buy_price=ref_high,
                           stoploss=ref_high*(1-stop_loss_pct/100), take_profit=ref_high*(1+take_profit_pct/100),
                           quantity=quantity, expiry=expiry, entry_time=row["Datetime"],
                           message="Cond 2 Flip: Close above Base → Buy CALL above that high",
                           spot_price=spot_price)
                signals.append(sig)
                if not return_all_signals: return sig

    # Condition 3 (+ 3.7 flip)
    if C1 > base_high:
        for _, row in after.iterrows():
            if row["High_^NSEI"] > H1:
                sig = dict(condition=3, option_type="CALL", buy_price=H1,
                           stoploss=H1*(1-stop_loss_pct/100), take_profit=H1*(1+take_profit_pct/100),
                           quantity=quantity, expiry=expiry, entry_time=row["Datetime"],
                           message="Cond 3: Gap up confirmed → Buy CALL above H1",
                           spot_price=spot_price)
                signals.append(sig)
                if not return_all_signals: return sig
            if row["Close_^NSEI"] < base_low:
                ref_low = row["Low_^NSEI"]
                sig = dict(condition=3.7, option_type="PUT", buy_price=ref_low,
                           stoploss=ref_low*(1+stop_loss_pct/100), take_profit=ref_low*(1-take_profit_pct/100),
                           quantity=quantity, expiry=expiry, entry_time=row["Datetime"],
                           message="Cond 3 Flip: Close below Base → Buy PUT below that low",
                           spot_price=spot_price)
                signals.append(sig)
                if not return_all_signals: return sig

    # Condition 4
    if (L1 < base_high and H1 > base_low) and (C1 < base_low):
        sig = dict(condition=4, option_type="PUT", buy_price=L1,
                   stoploss=L1*(1+stop_loss_pct/100), take_profit=L1*(1-take_profit_pct/100),
                   quantity=quantity, expiry=expiry, entry_time=entry_time,
                   message="Cond 4: Bearish breakdown below Base Zone → Buy PUT below L1",
                   spot_price=spot_price)
        signals.append(sig)
        if not return_all_signals: return sig

    return signals if signals else None

# =============== PAPER "BROKER" ===============
def paper_execute(signal: dict, lots: int, lot_size: int):
    """Creates a trade row with SL/TP/time-exit plan (no price feed on options here)."""
    if not signal: return None
    qty = lots * lot_size
    entry_price = signal["buy_price"]       # using underlying level as proxy; swap to option LTP when integrating
    stoploss = entry_price * (1 - stop_loss_pct/100) if signal["option_type"]=="CALL" else entry_price * (1 + stop_loss_pct/100)
    target = entry_price * (1 + take_profit_pct/100) if signal["option_type"]=="CALL" else entry_price * (1 - take_profit_pct/100)
    exit_time = pd.to_datetime(signal["entry_time"]) + pd.Timedelta(minutes=time_exit_minutes)

    details = {
        1: "Buy nearest ITM CALL above H1; SL trailing; book 50% @ +10%; time-exit 16m",
        2: "Gap down; Buy PUT below L1; SL trailing; time-exit",
        2.7:"Flip up; Buy CALL above flip-candle high; SL trailing; time-exit",
        3: "Gap up; Buy CALL above H1; SL trailing; book 50% @ +10%; time-exit",
        3.7:"Flip down; Buy PUT below flip-candle low; SL trailing; time-exit",
        4: "Breakdown; Buy PUT below L1; SL trailing; book 50% @ +10%; time-exit",
    }

    row = {
        "Trade Date": pd.to_datetime(signal["entry_time"]).date(),
        "Entry Time": pd.to_datetime(signal["entry_time"]),
        "Condition": signal["condition"],
        "Side/Type": "CALL" if signal["option_type"].upper()=="CALL" else "PUT",
        "Entry Ref Price": round(entry_price,2),
        "Stoploss Ref": round(stoploss,2),
        "Target Ref": round(target,2),
        "Planned Exit (time)": exit_time,
        "Planned Qty": lots * lot_size,
        "Message": signal["message"],
        "Notes": details.get(signal["condition"], ""),
        "Expiry": pd.to_datetime(signal["expiry"]).date(),
        "Spot At Signal": round(signal["spot_price"],2),
    }
    return pd.DataFrame([row])

# =============== DEDUPE & WINDOW GUARD ===============
def in_trading_window(ts: pd.Timestamp) -> bool:
    t = pd.to_datetime(ts).tz_convert("Asia/Kolkata").time()
    return (t >= trading_start) and (t <= trading_end)

def make_signal_key(sig: dict) -> str:
    d = pd.to_datetime(sig["entry_time"]).strftime("%Y-%m-%d %H:%M")
    return f"{d}|{sig['condition']}|{sig['option_type']}"

# =============== CHART ===============
def plot_last_two_days(df: pd.DataFrame):
    days = df["Datetime"].dt.date.unique()
    if len(days) < 2: return go.Figure()
    d0, d1 = days[-2], days[-1]
    sub = df[df["Datetime"].dt.date.isin([d0,d1])]
    c3 = df[(df["Datetime"].dt.date==d0)&(df["Datetime"].dt.hour==15)&(df["Datetime"].dt.minute==0)]
    o3 = c3.iloc[0]["Open_^NSEI"] if not c3.empty else None
    c3v = c3.iloc[0]["Close_^NSEI"] if not c3.empty else None

    fig = go.Figure(data=[go.Candlestick(
        x=sub["Datetime"], open=sub["Open_^NSEI"], high=sub["High_^NSEI"],
        low=sub["Low_^NSEI"], close=sub["Close_^NSEI"], name="NIFTY 15m"
    )])
    if o3 is not None:
        fig.add_hline(y=o3, line_dash="dot", line_color="blue", annotation_text="Prev 3PM Open")
    if c3v is not None:
        fig.add_hline(y=c3v, line_dash="dot", line_color="red", annotation_text="Prev 3PM Close")

    fig.update_layout(
        title="Nifty 15m — Previous Day & Today",
        xaxis_rangeslider_visible=False,
        xaxis=dict(rangebreaks=[dict(bounds=["sat","mon"]), dict(bounds=[15.5,9.25], pattern="hour")])
    )
    return fig

# =============== MAIN VIEW ===============
st.title("📊 3PM Strategy — Dashboard (MVP)")

colA, colB = st.columns([2,1], gap="large")

with colA:
    st.plotly_chart(plot_last_two_days(df), use_container_width=True)

with colB:
    # Show latest candle trend
    last = df.iloc[-1]
    trend = "Bullish 🔥" if last["Close_^NSEI"]>last["Open_^NSEI"] else ("Bearish ❄️" if last["Close_^NSEI"]<last["Open_^NSEI"] else "Doji ⚪")
    st.metric("Latest Candle Trend", trend)
    st.metric("Latest Close", f"{last['Close_^NSEI']:.2f}")
    st.metric("Selected Mode", mode)

st.markdown("---")

# ===== SIGNALS + EXECUTION (Paper/Live modes behave same here; Backtest is stub) =====
if mode in ["Paper Trade", "Live (stub)"]:
    # Only evaluate once per new candle
    latest_candle_time = df["Datetime"].max()
    if st.session_state.last_candle_time != latest_candle_time:
        # Get signal on full df (needs prev day)
        signal = trading_signal_all_conditions1(df, quantity=lots*lot_size, return_all_signals=False)

        if signal and in_trading_window(signal["entry_time"]):
            sig_key = make_signal_key(signal)
            if sig_key not in st.session_state.last_signal_key:
                # Execute paper order
                trade_row = paper_execute(signal, lots=lots, lot_size=lot_size)
                if trade_row is not None:
                    st.session_state.trade_log = pd.concat([st.session_state.trade_log, trade_row], ignore_index=True)
                    st.session_state.last_signal_key.add(sig_key)
                    st.success(f"New trade logged — {signal['message']}")
            else:
                st.info("Signal already taken for this candle/condition. Skipping duplicate.")
        elif signal:
            st.info("Signal generated but outside 09:30–14:00 window; not trading.")
        else:
            st.write("No trade signal triggered this session.")

        st.session_state.last_candle_time = latest_candle_time

elif mode == "Backtest (basic)":
    st.info("Basic backtest stub: evaluate signals per day; extend with full fill logic later.")
    # Minimal demonstration: show which days would have produced a signal
    # (You can expand this to iterate over history day-by-day with fills)
    sig = trading_signal_all_conditions1(df, return_all_signals=True)
    st.write("Signals found:" if sig else "No signals found.")
    if isinstance(sig, list) and sig:
        st.dataframe(pd.DataFrame(sig))

# =============== TRADE LOG & DOWNLOAD ===============
st.markdown("### 🧾 Cumulative Trade Log")
if not st.session_state.trade_log.empty:
    show = st.session_state.trade_log.copy()
    show["Entry Time"] = pd.to_datetime(show["Entry Time"]).dt.tz_localize(None)
    show["Planned Exit (time)"] = pd.to_datetime(show["Planned Exit (time)"]).dt.tz_localize(None)
    st.dataframe(show, use_container_width=True)
    csv = show.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv, file_name="3pm_trades.csv", mime="text/csv")
else:
    st.write("No trades yet.")

# =============== LIVE TRADING HOOKS (STUB) ===============
if mode == "Live (stub)":
    st.markdown("---")
    st.subheader("🔌 Broker Integration (Next Step)")
    st.write("""
    • Plug Zerodha/Fyers here: authenticate on app start, fetch option LTP to replace `Entry Ref Price`.  
    • On order fill, monitor option price for SL/TP; close by time exit.  
    • Emit Telegram alerts on entry/exit.  
    """)
