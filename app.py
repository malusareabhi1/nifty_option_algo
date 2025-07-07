# streamlit_nifty_option_bot.py
import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import math
import time
import plotly.graph_objects as go

st.set_page_config(page_title="NIFTY Options Algo Trading Bot", layout="wide")
st.title("📈 NIFTY Options Paper Trading Bot – 3PM Breakout Strategy")

# Sidebar settings
st.sidebar.header("⚙️ Settings")
capital = st.sidebar.number_input("Initial Capital (₹)", value=100000, step=10000)
risk_pct = st.sidebar.slider("Risk per Trade (%)", 0.5, 5.0, 2.0, step=0.5)
offset = st.sidebar.number_input("Breakout/Breakdown Offset", value=100, step=10)
refresh_rate = st.sidebar.slider("Refresh Interval (sec)", 30, 300, 60)

# Session state
if "capital" not in st.session_state:
    st.session_state.capital = capital
if "position" not in st.session_state:
    st.session_state.position = None
if "logs" not in st.session_state:
    st.session_state.logs = []

# --- Utility Functions ---
def get_nifty_spot_price():
    df = yf.download("^NSEI", interval="1m", period="1d", progress=False)
    if not df.empty:
        return round(df['Close'].iloc[-1], 2)
    return None

def get_atm_strike(spot):
    return int(round(spot / 50) * 50)

def get_current_week_expiry():
    today = datetime.now()
    weekday = today.weekday()
    expiry = today + timedelta((3 - weekday) % 7)
    return expiry.strftime('%y%b%d').upper()

def get_option_symbol(strike, opt_type='CE'):
    expiry = get_current_week_expiry()
    return f"NIFTY{expiry}{strike}{opt_type}"

def fetch_option_price(symbol):
    try:
        df = yf.download(symbol + ".NS", period="1d", interval="1m", progress=False)
        if not df.empty:
            return round(df['Close'].iloc[-1], 2)
    except:
        return None
    return None

def calculate_quantity(option_price, capital, risk_pct):
    risk_amt = capital * (risk_pct / 100)
    max_loss = option_price * 0.3
    qty = int(risk_amt / max_loss)
    return qty

def simulate_trade(entry_price, qty):
    sl_price = round(entry_price * 0.7, 2)
    target_price = round(entry_price + (entry_price - sl_price) * 1.5, 2)
    return {
        "entry": entry_price,
        "sl": sl_price,
        "target": target_price,
        "qty": qty,
        "status": "OPEN"
    }

def monitor_exit(current_price, trade):
    if current_price <= trade['sl']:
        pnl = (current_price - trade['entry']) * trade['qty']
        return 'SL', pnl
    elif current_price >= trade['target']:
        pnl = (current_price - trade['entry']) * trade['qty']
        return 'TARGET', pnl
    elif datetime.now().time() >= datetime.strptime("15:25", "%H:%M").time():
        pnl = (current_price - trade['entry']) * trade['qty']
        return 'TIME EXIT', pnl
    else:
        return None, None

@st.cache_data(ttl=300)
def get_nifty_15min_chart():
    df = yf.download("^NSEI", interval="15m", period="5d", progress=False)
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    return df


# --- Main Logic ---


spot = get_nifty_spot_price()
#st.write("Spot (raw):", spot, "Type:", type(spot))
spot = float(spot)

# ATM Strike Calculation
strike_gap = 50  # or 100 depending on NIFTY/BANKNIFTY
atm_strike = round(spot / strike_gap) * strike_gap

# Now safe to use
st.write(f"Spot: ₹{spot:.2f}, ATM Strike: {atm_strike}")

st.subheader("📉 NIFTY 15-Min Candlestick Chart (Last 5 Days)")
df_chart = get_nifty_15min_chart()

fig = go.Figure(data=[go.Candlestick(
    x=df_chart['Datetime'],
    open=df_chart['Open'],
    high=df_chart['High'],
    low=df_chart['Low'],
    close=df_chart['Close'],
    increasing_line_color='green',
    decreasing_line_color='red'
)])
fig.update_layout(
    xaxis_title='Time',
    yaxis_title='Price',
    xaxis_rangeslider_visible=False,
    height=500
)
st.plotly_chart(fig, use_container_width=True)

if spot is None:
    st.error("❌ Could not fetch NIFTY spot price")
    st.stop()

atm_strike = get_atm_strike(spot)
st.write(f"Spot: ₹{spot:.2f}, ATM Strike: {atm_strike}")

# Simulate 3PM levels for demo purposes
threepm_high = spot + 50  # Simulated
threepm_close = spot      # Simulated

st.info(f"🕒 3PM High: ₹{threepm_high}, 3PM Close: ₹{threepm_close}")

if not st.session_state.position:
    if spot > threepm_high + offset:
        symbol = get_option_symbol(atm_strike, "CE")
        ltp = fetch_option_price(symbol)
        if ltp:
            qty = calculate_quantity(ltp, st.session_state.capital, risk_pct)
            trade = simulate_trade(ltp, qty)
            trade.update({"symbol": symbol, "type": "Breakout"})
            st.session_state.position = trade
            st.toast(f"🚀 Entered Breakout: {symbol} @ ₹{ltp}", icon="🟢")
    elif spot < threepm_close - offset:
        symbol = get_option_symbol(atm_strike, "PE")
        ltp = fetch_option_price(symbol)
        if ltp:
            qty = calculate_quantity(ltp, st.session_state.capital, risk_pct)
            trade = simulate_trade(ltp, qty)
            trade.update({"symbol": symbol, "type": "Breakdown"})
            st.session_state.position = trade
            st.toast(f"🚨 Entered Breakdown: {symbol} @ ₹{ltp}", icon="🔴")
else:
    pos = st.session_state.position
    ltp = fetch_option_price(pos['symbol'])
    if ltp:
        reason, pnl = monitor_exit(ltp, pos)
        if reason:
            st.toast(f"{reason} @ ₹{ltp} | P&L: ₹{pnl:.2f}", icon="💰" if pnl > 0 else "📉")
            log = {
                'Time': datetime.now(),
                'Symbol': pos['symbol'],
                'Type': pos['type'],
                'Entry': pos['entry'],
                'Exit': ltp,
                'Qty': pos['qty'],
                'P&L': round(pnl, 2),
                'Reason': reason
            }
            st.session_state.logs.append(log)
            st.session_state.capital += pnl
            st.session_state.position = None

# Show current position
if st.session_state.position:
    st.subheader("📌 Open Position")
    st.write(st.session_state.position)

# Trade Logs
st.subheader("📋 Trade Log")
log_df = pd.DataFrame(st.session_state.logs)
if not log_df.empty:
    st.dataframe(log_df)
    st.success(f"Net P&L: ₹{log_df['P&L'].sum():,.2f} | Capital: ₹{st.session_state.capital:,.2f}")
else:
    st.info("No trades yet.")


# Auto-refresh
st.markdown(f"⏳ Refreshing every {refresh_rate} seconds...")
time.sleep(refresh_rate)
#st.experimental_rerun()
