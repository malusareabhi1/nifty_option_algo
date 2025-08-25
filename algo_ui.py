import time
from datetime import datetime, date, timedelta
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------
# Page Config & Global Style
# ------------------------------------------------------------
st.set_page_config(
    page_title="Algo Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject minimal CSS for nice cards & dark/light adaptation
BASE_CSS = """
<style>
:root {
  --card-bg: rgba(255,255,255,0.65);
  --card-border: rgba(0,0,0,0.08);
  --success: #22c55e; /* green-500 */
  --warn: #f59e0b;    /* amber-500 */
  --danger: #ef4444;  /* red-500 */
}
/* Dark theme support */
[data-theme="dark"] :root, .stApp[theme="dark"] :root { 
  --card-bg: rgba(30,41,59,0.65);
  --card-border: rgba(255,255,255,0.08);
}

.card {
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: 16px; 
  padding: 16px; 
  box-shadow: 0 4px 18px rgba(0,0,0,0.06);
}
.card h3 { margin-top: 0 !important; }
.badge { display:inline-block; padding:2px 8px; border-radius:9999px; font-size:12px; }
.badge.success{ background:rgba(34,197,94,0.15); color:#16a34a; }
.badge.warn{ background:rgba(245,158,11,0.15); color:#d97706; }
.badge.danger{ background:rgba(239,68,68,0.15); color:#dc2626; }
.kpi { font-size: 28px; font-weight: 700; }
.subtle { color: rgba(0,0,0,0.6); }
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)

# ------------------------------------------------------------
# Session State Defaults
# ------------------------------------------------------------
def _init_state():
    defaults = dict(
        theme_dark=False,
        api_status={"Zerodha": False, "Fyers": False, "AliceBlue": False},
        connected_broker=None,
        live_running=False,
        live_strategy=None,
        trade_logs=pd.DataFrame(columns=["Time","Symbol","Action","Qty","Price","PnL"]),
        capital=100000.0,
        risk_per_trade_pct=1.0,
        max_trades=3,
        strategies=[
            {
                "name": "Doctor Strategy 1.0",
                "short": "BB 20 SMA breakout with IV filter",
                "metrics": {"CAGR": 18.3, "Win%": 64.8, "MaxDD%": 12.6},
            },
            {
                "name": "ORB (Opening Range Breakout)",
                "short": "Range breakout after first 15m",
                "metrics": {"CAGR": 14.1, "Win%": 57.4, "MaxDD%": 10.9},
            },
            {
                "name": "EMA20 + Volume",
                "short": "Momentum confirmation with volume push",
                "metrics": {"CAGR": 11.7, "Win%": 55.0, "MaxDD%": 9.8},
            },
        ],
        selected_strategy="Doctor Strategy 1.0",
        pricing=[
            {"name": "Basic", "price": 699, "features": [
                "1 live strategy",
                "Backtests & charts",
                "Telegram alerts",
                "Email support"
            ]},
            {"name": "Pro", "price": 1499, "features": [
                "3 live strategies",
                "Automation (paper/live)",
                "Custom risk settings",
                "Priority support"
            ]},
            {"name": "Enterprise", "price": 3999, "features": [
                "Unlimited strategies",
                "Broker API integration",
                "SLA & onboarding",
                "Dedicated manager"
            ]},
        ],
        faq=[
            ("Is algo trading risky?", "Yes. Markets involve risk. Backtests are not guarantees. Manage risk per trade and overall exposure."),
            ("Which brokers are supported?", "Zerodha, Fyers, AliceBlue out-of-the-box. Others upon request."),
            ("Do you store my API keys?", "Keys are stored encrypted on device/server per your deployment. You control revocation."),
        ],
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ------------------------------------------------------------
# Utility Components
# ------------------------------------------------------------
def metric_card(title: str, value: str, delta: str | None = None, help_text: str | None = None):
    with st.container(border=True):
        c1, c2 = st.columns([1,2])
        with c1:
            st.caption(title)
        with c2:
            st.write(f"<div class='kpi'>{value}</div>", unsafe_allow_html=True)
            if delta:
                st.write(delta)
        if help_text:
            st.caption(help_text)


def strategy_card(name: str, desc: str, metrics: Dict[str, float]):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader(name)
    st.write(desc)
    c1, c2, c3 = st.columns(3)
    c1.metric("CAGR", f"{metrics.get('CAGR', 0):.1f}%")
    c2.metric("Win%", f"{metrics.get('Win%', 0):.1f}%")
    c3.metric("Max DD", f"{metrics.get('MaxDD%', 0):.1f}%")
    st.markdown("</div>", unsafe_allow_html=True)


def plot_equity_curve(pnl_series: pd.Series, title: str = "Equity Curve"):
    if pnl_series is None or len(pnl_series) == 0:
        st.info("Upload backtest CSV to see equity curve.")
        return
    cum = pnl_series.cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=cum, mode='lines', name='Equity'))
    fig.update_layout(height=360, title=title, margin=dict(l=20,r=20,t=40,b=20))
    st.plotly_chart(fig, use_container_width=True)


def plot_candles(df: pd.DataFrame, title: str = "Candlestick Chart"):
    req = {"Datetime","Open","High","Low","Close"}
    if not req.issubset(df.columns):
        st.warning("Candlestick requires columns: Datetime, Open, High, Low, Close")
        return
    fig = go.Figure(data=[go.Candlestick(
        x=df['Datetime'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']
    )])
    fig.update_layout(height=420, title=title, xaxis_rangeslider_visible=False, margin=dict(l=20,r=20,t=40,b=20))
    st.plotly_chart(fig, use_container_width=True)


def add_trade_log(symbol: str, side: str, qty: int, price: float, pnl: float):
    row = {
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Symbol": symbol,
        "Action": side,
        "Qty": qty,
        "Price": price,
        "PnL": pnl,
    }
    st.session_state.trade_logs = pd.concat([st.session_state.trade_logs, pd.DataFrame([row])], ignore_index=True)

# ------------------------------------------------------------
# Sidebar Navigation
# ------------------------------------------------------------
with st.sidebar:
    st.title("⚡ Algo Trading")
    # Theme toggle (visual only)
    st.session_state.theme_dark = st.toggle("Dark Theme", value=st.session_state.theme_dark)
    # Mark attribute for downstream CSS targeting
    st.markdown(f"<div style='display:none' data-theme={'dark' if st.session_state.theme_dark else 'light'}></div>", unsafe_allow_html=True)

    st.image("https://assets-global.website-files.com/5e0a1f0d3a9f1b6f7f1b6f34/5e0a1f63a4f62a5534b5f5f9_finance-illustration.png")

    MENU = st.radio(
        "Navigate",
        ["Home", "Strategies", "Broker API", "Dashboard", "Products", "Support"],
        index=0,
    )

# ------------------------------------------------------------
# Home
# ------------------------------------------------------------
if MENU == "Home":
    st.title("Welcome to Your Algo Trading Platform")
    st.write("Automate your trades with smart, auditable strategies. Connect your broker, choose a strategy, and manage risk— all from one dashboard.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### What is Algo Trading?")
        st.write("Algorithmic trading executes orders using pre-defined rules for entries, exits, and risk management.")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Why Choose Us?")
        st.write("Clean UI, safer defaults, backtests, paper trading, and live automation with popular brokers.")
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Risk Disclaimer")
        st.write("Trading involves risk. Past performance does not guarantee future returns. Trade responsibly.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    st.subheader("Quick KPIs")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        metric_card("Capital", f"₹{st.session_state.capital:,.0f}")
    with k2:
        metric_card("Risk / Trade", f"{st.session_state.risk_per_trade_pct:.1f}%")
    with k3:
        metric_card("Max Trades", str(st.session_state.max_trades))
    with k4:
        connected = st.session_state.connected_broker or "—"
        status = "Connected" if connected != "—" else "Not Connected"
        metric_card("Broker", connected, status)

    st.divider()
    st.info("Use the sidebar to explore Strategies, connect Broker APIs, and run the live Dashboard.")

# ------------------------------------------------------------
# Strategies
# ------------------------------------------------------------
elif MENU == "Strategies":
    st.title("Strategies Library")

    # Strategy selector + brief
    colA, colB = st.columns([2, 1])
    with colA:
        names = [s["name"] for s in st.session_state.strategies]
        st.session_state.selected_strategy = st.selectbox("Select strategy", names, index=0)
    with colB:
        st.caption("Filter")
        _min_win = st.slider("Min Win%", 0, 100, 50)

    # Render cards
    for s in st.session_state.strategies:
        if s["metrics"]["Win%"] >= _min_win:
            strategy_card(s["name"], s["short"], s["metrics"])

    st.divider()

    st.subheader("Backtest Viewer")
    st.caption("Upload a CSV with columns: Datetime, Open, High, Low, Close, PnL (optional)")
    up = st.file_uploader("Upload backtest CSV", type=["csv"]) 
    if up:
        df = pd.read_csv(up)
        # Try parsing datetime
        for col in ["Datetime", "Date", "timestamp", "time"]:
            if col in df.columns:
                try:
                    df["Datetime"] = pd.to_datetime(df[col])
                    break
                except Exception:
                    pass
        # Basic displays
        st.dataframe(df.head(200), use_container_width=True)
        if {"Open","High","Low","Close","Datetime"}.issubset(df.columns):
            plot_candles(df, title=f"{st.session_state.selected_strategy} – Price")
        if "PnL" in df.columns:
            plot_equity_curve(df["PnL"], title=f"{st.session_state.selected_strategy} – Equity")

# ------------------------------------------------------------
# Broker API
# ------------------------------------------------------------
elif MENU == "Broker API":
    st.title("Broker Integrations")
    st.write("Connect your broker to enable paper/live trading. This demo stores states locally. Replace with secure key vault in production.")

    brokers = ["Zerodha", "Fyers", "AliceBlue"]
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        sel = st.selectbox("Broker", brokers, index=0)
        if sel == "Zerodha":
            st.text_input("API Key", key="z_key")
            st.text_input("API Secret", type="password", key="z_secret")
            if st.button("Connect Zerodha"):
                st.session_state.api_status["Zerodha"] = True
                st.session_state.connected_broker = "Zerodha"
                st.success("Zerodha connected (demo)")
        elif sel == "Fyers":
            st.text_input("Client ID", key="f_id")
            st.text_input("Secret Key", type="password", key="f_secret")
            if st.button("Connect Fyers"):
                st.session_state.api_status["Fyers"] = True
                st.session_state.connected_broker = "Fyers"
                st.success("Fyers connected (demo)")
        elif sel == "AliceBlue":
            st.text_input("User ID", key="a_id")
            st.text_input("API Key", type="password", key="a_key")
            if st.button("Connect AliceBlue"):
                st.session_state.api_status["AliceBlue"] = True
                st.session_state.connected_broker = "AliceBlue"
                st.success("AliceBlue connected (demo)")

    with bcol2:
        st.subheader("Connection Status")
        for name, ok in st.session_state.api_status.items():
            badge = f"<span class='badge {'success' if ok else 'danger'}'>{'Connected' if ok else 'Not Connected'}</span>"
            st.markdown(f"**{name}**: {badge}", unsafe_allow_html=True)
        st.caption("Replace demo handlers with actual OAuth/API calls. Store tokens securely.")

# ------------------------------------------------------------
# Dashboard (Main Trading Panel)
# ------------------------------------------------------------
elif MENU == "Dashboard":
    st.title("Trading Dashboard")

    # Top KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        metric_card("Capital", f"₹{st.session_state.capital:,.0f}")
    with k2:
        metric_card("Risk / Trade", f"{st.session_state.risk_per_trade_pct:.1f}%")
    with k3:
        metric_card("Max Trades", str(st.session_state.max_trades))
    with k4:
        broker = st.session_state.connected_broker or "—"
        metric_card("Broker", broker)

    st.divider()

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Strategy Control")
        st.session_state.live_strategy = st.selectbox(
            "Select Strategy", [s["name"] for s in st.session_state.strategies], index=0
        )
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st.session_state.capital = st.number_input("Capital (₹)", min_value=1000.0, step=1000.0, value=float(st.session_state.capital))
        with cc2:
            st.session_state.risk_per_trade_pct = st.number_input("Risk per Trade (%)", min_value=0.1, max_value=10.0, step=0.1, value=float(st.session_state.risk_per_trade_pct))
        with cc3:
            st.session_state.max_trades = st.number_input("Max Trades", min_value=1, max_value=20, step=1, value=int(st.session_state.max_trades))

        btn1, btn2, btn3 = st.columns(3)
        with btn1:
            if st.button("▶ Start (Paper)"):
                st.session_state.live_running = True
                st.success("Paper trading started (demo)")
        with btn2:
            if st.button("⏸ Stop"):
                st.session_state.live_running = False
                st.warning("Stopped.")
        with btn3:
            if st.button("⚙ Send Test Alert"):
                add_trade_log("NIFTY", "BUY", 50, 250.5, 0.0)
                st.info("Test alert -> trade log added.")

        st.subheader("Live Market Data (Demo)")
        syms = ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "INFY"]
        prices = np.round(np.random.normal(100, 10, len(syms)), 2)
        live_df = pd.DataFrame({"Symbol": syms, "LTP": prices, "Change%": np.round(np.random.normal(0.2, 0.5, len(syms)), 2)})
        st.dataframe(live_df, use_container_width=True)

    with right:
        st.subheader("Chart (Demo)")
        # Create fake ohlc for demo
        dt = pd.date_range(datetime.now() - timedelta(hours=4), periods=32, freq="7min")
        base = 100 + np.cumsum(np.random.normal(0, 0.5, len(dt)))
        dfc = pd.DataFrame({
            "Datetime": dt,
            "Open": base + np.random.normal(0, 0.5, len(dt)),
            "High": base + np.random.uniform(0.2, 1.2, len(dt)),
            "Low": base - np.random.uniform(0.2, 1.2, len(dt)),
            "Close": base + np.random.normal(0, 0.5, len(dt)),
        })
        plot_candles(dfc, title=f"{st.session_state.live_strategy} – Intraday")

    st.divider()

    st.subheader("Trade Logs")
    st.dataframe(st.session_state.trade_logs, use_container_width=True)

# ------------------------------------------------------------
# Products / Pricing
# ------------------------------------------------------------
elif MENU == "Products":
    st.title("Products & Pricing")

    cols = st.columns(3)
    for i, plan in enumerate(st.session_state.pricing):
        with cols[i]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"### {plan['name']}")
            st.markdown(f"#### ₹{plan['price']}/month")
            for feat in plan['features']:
                st.write(f"• {feat}")
            st.button(f"Subscribe {plan['name']}")
            st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.subheader("What you get")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.write("Automation")
    with c2: st.write("Backtesting")
    with c3: st.write("Paper Trading")
    with c4: st.write("Live Alerts")

# ------------------------------------------------------------
# Support
# ------------------------------------------------------------
elif MENU == "Support":
    st.title("Support & Resources")

    st.subheader("Documentation")
    st.write("• Getting Started  • Strategy Guide  • API Setup  • FAQ")

    st.subheader("FAQ")
    for q, a in st.session_state.faq:
        with st.expander(q):
            st.write(a)

    st.subheader("Contact Us")
    with st.form("contact_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        message = st.text_area("Message")
        submitted = st.form_submit_button("Send")
        if submitted:
            st.success("Thanks! We'll get back to you shortly (demo).")

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.markdown("---")
st.caption("© 2025 Your Brand • This is a demo UI. Replace demo handlers with your live logic, APIs, and secure storage.")
