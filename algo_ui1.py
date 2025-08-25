import time
from datetime import datetime, timedelta
from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------
# Page Config & Global Theming
# ------------------------------------------------------------
st.set_page_config(
    page_title="Algo Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 🎨 Modern colorful theme (light/dark aware)
BASE_CSS = """
<style>
/* App background */
.stApp {
  background: linear-gradient(135deg, #eef2ff 0%, #ffffff 60%);
}
[data-theme="dark"] .stApp, .stApp[theme="dark"] {
  background: linear-gradient(135deg, #0b1220 0%, #111827 60%);
}

:root {
  --blue: #2563eb;    /* indigo-600 */
  --amber: #f59e0b;   /* amber-500 */
  --purple: #9333ea;  /* purple-600 */
  --teal: #14b8a6;    /* teal-500 */
  --green: #16a34a;   /* green-600 */
  --red: #dc2626;     /* red-600 */
  --card-bg: rgba(255,255,255,0.75);
  --card-border: rgba(0,0,0,0.08);
}
[data-theme="dark"] :root, .stApp[theme="dark"] :root {
  --card-bg: rgba(17,24,39,0.65);
  --card-border: rgba(255,255,255,0.08);
}

/* Generic colorful card */
.card {
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: 18px;
  padding: 18px;
  box-shadow: 0 6px 22px rgba(0,0,0,0.08);
}
.card h3 { margin-top: 0 !important; }

/* KPI tiles */
.kpi-card { text-align:center; border-radius:16px; padding:16px; }
.kpi-title { margin:0; font-weight:700; font-size:14px; opacity:.9; }
.kpi-value { font-size:30px; font-weight:800; margin-top:6px; }

/* Buttons */
.stButton > button {
  background: linear-gradient(135deg, var(--blue), #3b82f6);
  color: #fff; border: none; border-radius: 10px; padding: 10px 18px; font-weight:700;
  box-shadow: 0 6px 18px rgba(37,99,235,0.3);
}
.stButton > button:hover { filter: brightness(1.05); }

/* Secondary variants via data-color attr */
.button-amber > button { background: linear-gradient(135deg, var(--amber), #fbbf24); box-shadow: 0 6px 18px rgba(245,158,11,.35); }
.button-purple > button { background: linear-gradient(135deg, var(--purple), #a855f7); box-shadow: 0 6px 18px rgba(147,51,234,.35); }
.button-teal > button { background: linear-gradient(135deg, var(--teal), #2dd4bf); box-shadow: 0 6px 18px rgba(20,184,166,.35); }

/* Badges */
.badge { display:inline-block; padding:4px 10px; border-radius:9999px; font-size:12px; font-weight:700; }
.badge.success { background: rgba(22,163,74,.15); color: var(--green); }
.badge.danger { background: rgba(220,38,38,.15); color: var(--red); }

/* Dataframes */
[data-testid="stDataFrame"] div .row_heading, [data-testid="stDataFrame"] div .blank {background: transparent;}

/* Dividers */
hr { border-top: 1px solid rgba(0,0,0,.08); }
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
            {"name": "Doctor Strategy 1.0", "short": "BB 20 SMA breakout with IV filter", "metrics": {"CAGR": 18.3, "Win%": 64.8, "MaxDD%": 12.6}},
            {"name": "ORB (Opening Range Breakout)", "short": "Range breakout after first 15m", "metrics": {"CAGR": 14.1, "Win%": 57.4, "MaxDD%": 10.9}},
            {"name": "EMA20 + Volume", "short": "Momentum confirmation with volume push", "metrics": {"CAGR": 11.7, "Win%": 55.0, "MaxDD%": 9.8}},
        ],
        selected_strategy="Doctor Strategy 1.0",
        pricing=[
            {"name": "Basic", "price": 699, "features": ["1 live strategy","Backtests & charts","Telegram alerts","Email support"]},
            {"name": "Pro", "price": 1499, "features": ["3 live strategies","Automation (paper/live)","Custom risk settings","Priority support"]},
            {"name": "Enterprise", "price": 3999, "features": ["Unlimited strategies","Broker API integration","SLA & onboarding","Dedicated manager"]},
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
# Utility: Colors & Components
# ------------------------------------------------------------

STRAT_COLORS: Dict[str, str] = {
    "Doctor Strategy 1.0": "var(--blue)",
    "ORB (Opening Range Breakout)": "var(--amber)",
    "EMA20 + Volume": "var(--teal)",
}


def kpi_card(title: str, value: str, color_css: str):
    st.markdown(
        f"""
        <div class='kpi-card' style='background: linear-gradient(135deg, {color_css}15, transparent); border:1px solid {color_css}35'>
            <div class='kpi-title' style='color:{color_css}'>{title}</div>
            <div class='kpi-value' style='color:{color_css}'>{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def strategy_card(name: str, desc: str, metrics: Dict[str, float]):
    color = STRAT_COLORS.get(name, "var(--purple)")
    st.markdown(
        f"""
        <div class='card' style='border-left:6px solid {color};'>
            <h3 style='color:{color}'>{name}</h3>
            <p style='margin:6px 0 14px 0'>{desc}</p>
            <div style='display:flex;gap:16px;flex-wrap:wrap'>
                <div class='badge' style='background:{color}15;color:{color}'><b>CAGR</b>&nbsp;{metrics.get('CAGR',0):.1f}%</div>
                <div class='badge' style='background:{color}15;color:{color}'><b>Win%</b>&nbsp;{metrics.get('Win%',0):.1f}%</div>
                <div class='badge' style='background:{color}15;color:{color}'><b>Max DD</b>&nbsp;{metrics.get('MaxDD%',0):.1f}%</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_equity_curve(pnl_series: pd.Series, title: str = "Equity Curve"):
    if pnl_series is None or len(pnl_series) == 0:
        st.info("Upload backtest CSV to see equity curve.")
        return
    cum = pnl_series.cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=cum, mode='lines', name='Equity', line=dict(width=3)))
    fig.update_layout(
        height=360,
        title=title,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_candles(df: pd.DataFrame, title: str = "Candlestick Chart"):
    req = {"Datetime", "Open", "High", "Low", "Close"}
    if not req.issubset(df.columns):
        st.warning("Candlestick requires columns: Datetime, Open, High, Low, Close")
        return
    fig = go.Figure(data=[go.Candlestick(x=df['Datetime'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
    fig.update_layout(
        height=420,
        title=title,
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
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
    # Mark attribute for CSS targeting
    st.markdown(f"<div style='display:none' data-theme={'dark' if st.session_state.theme_dark else 'light'}></div>", unsafe_allow_html=True)

    st.image(
        "https://assets-global.website-files.com/5e0a1f0d3a9f1b6f7f1b6f34/5e0a1f63a4f62a5534b5f5f9_finance-illustration.png",
        use_container_width=True,
    )

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
    st.write("Automate your trades with smart, auditable strategies. Connect your broker, choose a strategy, and manage risk — all from one dashboard.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### What is Algo Trading?")
        st.write("Algorithmic trading executes orders using pre-defined rules for entries, exits, and risk management.")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Why Choose Us?")
        st.write("Colorful, clean UI, safer defaults, backtests, paper trading, and live automation with popular brokers.")
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
        kpi_card("Capital", f"₹{st.session_state.capital:,.0f}", "var(--blue)")
    with k2:
        kpi_card("Risk / Trade", f"{st.session_state.risk_per_trade_pct:.1f}%", "var(--amber)")
    with k3:
        kpi_card("Max Trades", str(st.session_state.max_trades), "var(--purple)")
    with k4:
        connected = st.session_state.connected_broker or "—"
        color = "var(--green)" if connected != "—" else "var(--red)"
        kpi_card("Broker", connected, color)

    st.divider()
    st.info("Use the sidebar to explore Strategies, connect Broker APIs, and run the live Dashboard.")

# ------------------------------------------------------------
# Strategies
# ------------------------------------------------------------
elif MENU == "Strategies":
    st.title("Strategies Library")

    colA, colB = st.columns([2, 1])
    with colA:
        names = [s["name"] for s in st.session_state.strategies]
        st.session_state.selected_strategy = st.selectbox("Select strategy", names, index=0)
    with colB:
        st.caption("Filter")
        _min_win = st.slider("Min Win%", 0, 100, 50)

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
            with st.container():
                st.markdown("<div class='button-teal'>", unsafe_allow_html=True)
                if st.button("Connect Zerodha"):
                    st.session_state.api_status["Zerodha"] = True
                    st.session_state.connected_broker = "Zerodha"
                    st.success("Zerodha connected (demo)")
                st.markdown("</div>", unsafe_allow_html=True)
        elif sel == "Fyers":
            st.text_input("Client ID", key="f_id")
            st.text_input("Secret Key", type="password", key="f_secret")
            st.markdown("<div class='button-amber'>", unsafe_allow_html=True)
            if st.button("Connect Fyers"):
                st.session_state.api_status["Fyers"] = True
                st.session_state.connected_broker = "Fyers"
                st.success("Fyers connected (demo)")
            st.markdown("</div>", unsafe_allow_html=True)
        elif sel == "AliceBlue":
            st.text_input("User ID", key="a_id")
            st.text_input("API Key", type="password", key="a_key")
            st.markdown("<div class='button-purple'>", unsafe_allow_html=True)
            if st.button("Connect AliceBlue"):
                st.session_state.api_status["AliceBlue"] = True
                st.session_state.connected_broker = "AliceBlue"
                st.success("AliceBlue connected (demo)")
            st.markdown("</div>", unsafe_allow_html=True)

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

    # Top colorful KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Capital", f"₹{st.session_state.capital:,.0f}", "var(--blue)")
    with k2:
        kpi_card("Risk / Trade", f"{st.session_state.risk_per_trade_pct:.1f}%", "var(--amber)")
    with k3:
        kpi_card("Max Trades", str(st.session_state.max_trades), "var(--purple)")
    with k4:
        broker = st.session_state.connected_broker or "—"
        kpi_card("Broker", broker, "var(--green)" if broker != "—" else "var(--red)")

    st.divider()

    left, right = st.columns([1.25, 1])
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

        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("▶ Start (Paper)"):
                st.session_state.live_running = True
                st.success("Paper trading started (demo)")
        with b2:
            if st.button("⏸ Stop"):
                st.session_state.live_running = False
                st.warning("Stopped.")
        with b3:
            if st.button("⚙ Send Test Alert"):
                add_trade_log("NIFTY", "BUY", 50, 250.5, 0.0)
                st.info("Test alert → trade log added.")

        st.subheader("Live Market Data (Demo)")
        syms = ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "INFY"]
        prices = np.round(np.random.normal(100, 10, len(syms)), 2)
        live_df = pd.DataFrame({
            "Symbol": syms,
            "LTP": prices,
            "Change%": np.round(np.random.normal(0.2, 0.5, len(syms)), 2)
        })
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
st.caption("© 2025 Your Brand • This is a colorful demo UI. Replace demo handlers with your live logic, APIs, and secure storage.")
import time
from datetime import datetime, timedelta
from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------
# Page Config & Global Theming
# ------------------------------------------------------------
st.set_page_config(
    page_title="Algo Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 🎨 Modern colorful theme (light/dark aware)
BASE_CSS = """
<style>
/* App background */
.stApp {
  background: linear-gradient(135deg, #eef2ff 0%, #ffffff 60%);
}
[data-theme="dark"] .stApp, .stApp[theme="dark"] {
  background: linear-gradient(135deg, #0b1220 0%, #111827 60%);
}

:root {
  --blue: #2563eb;    /* indigo-600 */
  --amber: #f59e0b;   /* amber-500 */
  --purple: #9333ea;  /* purple-600 */
  --teal: #14b8a6;    /* teal-500 */
  --green: #16a34a;   /* green-600 */
  --red: #dc2626;     /* red-600 */
  --card-bg: rgba(255,255,255,0.75);
  --card-border: rgba(0,0,0,0.08);
}
[data-theme="dark"] :root, .stApp[theme="dark"] :root {
  --card-bg: rgba(17,24,39,0.65);
  --card-border: rgba(255,255,255,0.08);
}

/* Generic colorful card */
.card {
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: 18px;
  padding: 18px;
  box-shadow: 0 6px 22px rgba(0,0,0,0.08);
}
.card h3 { margin-top: 0 !important; }

/* KPI tiles */
.kpi-card { text-align:center; border-radius:16px; padding:16px; }
.kpi-title { margin:0; font-weight:700; font-size:14px; opacity:.9; }
.kpi-value { font-size:30px; font-weight:800; margin-top:6px; }

/* Buttons */
.stButton > button {
  background: linear-gradient(135deg, var(--blue), #3b82f6);
  color: #fff; border: none; border-radius: 10px; padding: 10px 18px; font-weight:700;
  box-shadow: 0 6px 18px rgba(37,99,235,0.3);
}
.stButton > button:hover { filter: brightness(1.05); }

/* Secondary variants via data-color attr */
.button-amber > button { background: linear-gradient(135deg, var(--amber), #fbbf24); box-shadow: 0 6px 18px rgba(245,158,11,.35); }
.button-purple > button { background: linear-gradient(135deg, var(--purple), #a855f7); box-shadow: 0 6px 18px rgba(147,51,234,.35); }
.button-teal > button { background: linear-gradient(135deg, var(--teal), #2dd4bf); box-shadow: 0 6px 18px rgba(20,184,166,.35); }

/* Badges */
.badge { display:inline-block; padding:4px 10px; border-radius:9999px; font-size:12px; font-weight:700; }
.badge.success { background: rgba(22,163,74,.15); color: var(--green); }
.badge.danger { background: rgba(220,38,38,.15); color: var(--red); }

/* Dataframes */
[data-testid="stDataFrame"] div .row_heading, [data-testid="stDataFrame"] div .blank {background: transparent;}

/* Dividers */
hr { border-top: 1px solid rgba(0,0,0,.08); }
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
            {"name": "Doctor Strategy 1.0", "short": "BB 20 SMA breakout with IV filter", "metrics": {"CAGR": 18.3, "Win%": 64.8, "MaxDD%": 12.6}},
            {"name": "ORB (Opening Range Breakout)", "short": "Range breakout after first 15m", "metrics": {"CAGR": 14.1, "Win%": 57.4, "MaxDD%": 10.9}},
            {"name": "EMA20 + Volume", "short": "Momentum confirmation with volume push", "metrics": {"CAGR": 11.7, "Win%": 55.0, "MaxDD%": 9.8}},
        ],
        selected_strategy="Doctor Strategy 1.0",
        pricing=[
            {"name": "Basic", "price": 699, "features": ["1 live strategy","Backtests & charts","Telegram alerts","Email support"]},
            {"name": "Pro", "price": 1499, "features": ["3 live strategies","Automation (paper/live)","Custom risk settings","Priority support"]},
            {"name": "Enterprise", "price": 3999, "features": ["Unlimited strategies","Broker API integration","SLA & onboarding","Dedicated manager"]},
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
# Utility: Colors & Components
# ------------------------------------------------------------

STRAT_COLORS: Dict[str, str] = {
    "Doctor Strategy 1.0": "var(--blue)",
    "ORB (Opening Range Breakout)": "var(--amber)",
    "EMA20 + Volume": "var(--teal)",
}


def kpi_card(title: str, value: str, color_css: str):
    st.markdown(
        f"""
        <div class='kpi-card' style='background: linear-gradient(135deg, {color_css}15, transparent); border:1px solid {color_css}35'>
            <div class='kpi-title' style='color:{color_css}'>{title}</div>
            <div class='kpi-value' style='color:{color_css}'>{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def strategy_card(name: str, desc: str, metrics: Dict[str, float]):
    color = STRAT_COLORS.get(name, "var(--purple)")
    st.markdown(
        f"""
        <div class='card' style='border-left:6px solid {color};'>
            <h3 style='color:{color}'>{name}</h3>
            <p style='margin:6px 0 14px 0'>{desc}</p>
            <div style='display:flex;gap:16px;flex-wrap:wrap'>
                <div class='badge' style='background:{color}15;color:{color}'><b>CAGR</b>&nbsp;{metrics.get('CAGR',0):.1f}%</div>
                <div class='badge' style='background:{color}15;color:{color}'><b>Win%</b>&nbsp;{metrics.get('Win%',0):.1f}%</div>
                <div class='badge' style='background:{color}15;color:{color}'><b>Max DD</b>&nbsp;{metrics.get('MaxDD%',0):.1f}%</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_equity_curve(pnl_series: pd.Series, title: str = "Equity Curve"):
    if pnl_series is None or len(pnl_series) == 0:
        st.info("Upload backtest CSV to see equity curve.")
        return
    cum = pnl_series.cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=cum, mode='lines', name='Equity', line=dict(width=3)))
    fig.update_layout(
        height=360,
        title=title,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_candles(df: pd.DataFrame, title: str = "Candlestick Chart"):
    req = {"Datetime", "Open", "High", "Low", "Close"}
    if not req.issubset(df.columns):
        st.warning("Candlestick requires columns: Datetime, Open, High, Low, Close")
        return
    fig = go.Figure(data=[go.Candlestick(x=df['Datetime'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
    fig.update_layout(
        height=420,
        title=title,
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
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
    # Sidebar (render only once)
    st.sidebar.title("⚡ Algo Trading")
    st.sidebar.toggle("Dark Theme", key="dark_theme_toggle")
    
    MENU = st.sidebar.radio(
        "Navigate",
        ["Home", "Strategies", "Broker API", "Dashboard", "Products", "Support"],
        index=0,
        key="main_menu_radio"
    )

    MENU = st.radio(
        "Navigate",
        ["Home", "Strategies", "Broker API", "Dashboard", "Products", "Support"],
        index=0,
    key="main_menu_radio"
    )

# ------------------------------------------------------------
# Home
# ------------------------------------------------------------
if MENU == "Home":
    st.title("Welcome to Your Algo Trading Platform")
    st.write("Automate your trades with smart, auditable strategies. Connect your broker, choose a strategy, and manage risk — all from one dashboard.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### What is Algo Trading?")
        st.write("Algorithmic trading executes orders using pre-defined rules for entries, exits, and risk management.")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Why Choose Us?")
        st.write("Colorful, clean UI, safer defaults, backtests, paper trading, and live automation with popular brokers.")
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
        kpi_card("Capital", f"₹{st.session_state.capital:,.0f}", "var(--blue)")
    with k2:
        kpi_card("Risk / Trade", f"{st.session_state.risk_per_trade_pct:.1f}%", "var(--amber)")
    with k3:
        kpi_card("Max Trades", str(st.session_state.max_trades), "var(--purple)")
    with k4:
        connected = st.session_state.connected_broker or "—"
        color = "var(--green)" if connected != "—" else "var(--red)"
        kpi_card("Broker", connected, color)

    st.divider()
    st.info("Use the sidebar to explore Strategies, connect Broker APIs, and run the live Dashboard.")

# ------------------------------------------------------------
# Strategies
# ------------------------------------------------------------
elif MENU == "Strategies":
    st.title("Strategies Library")

    colA, colB = st.columns([2, 1])
    with colA:
        names = [s["name"] for s in st.session_state.strategies]
        st.session_state.selected_strategy = st.selectbox("Select strategy", names, index=0)
    with colB:
        st.caption("Filter")
        _min_win = st.slider("Min Win%", 0, 100, 50)

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
            with st.container():
                st.markdown("<div class='button-teal'>", unsafe_allow_html=True)
                if st.button("Connect Zerodha"):
                    st.session_state.api_status["Zerodha"] = True
                    st.session_state.connected_broker = "Zerodha"
                    st.success("Zerodha connected (demo)")
                st.markdown("</div>", unsafe_allow_html=True)
        elif sel == "Fyers":
            st.text_input("Client ID", key="f_id")
            st.text_input("Secret Key", type="password", key="f_secret")
            st.markdown("<div class='button-amber'>", unsafe_allow_html=True)
            if st.button("Connect Fyers"):
                st.session_state.api_status["Fyers"] = True
                st.session_state.connected_broker = "Fyers"
                st.success("Fyers connected (demo)")
            st.markdown("</div>", unsafe_allow_html=True)
        elif sel == "AliceBlue":
            st.text_input("User ID", key="a_id")
            st.text_input("API Key", type="password", key="a_key")
            st.markdown("<div class='button-purple'>", unsafe_allow_html=True)
            if st.button("Connect AliceBlue"):
                st.session_state.api_status["AliceBlue"] = True
                st.session_state.connected_broker = "AliceBlue"
                st.success("AliceBlue connected (demo)")
            st.markdown("</div>", unsafe_allow_html=True)

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

    # Top colorful KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Capital", f"₹{st.session_state.capital:,.0f}", "var(--blue)")
    with k2:
        kpi_card("Risk / Trade", f"{st.session_state.risk_per_trade_pct:.1f}%", "var(--amber)")
    with k3:
        kpi_card("Max Trades", str(st.session_state.max_trades), "var(--purple)")
    with k4:
        broker = st.session_state.connected_broker or "—"
        kpi_card("Broker", broker, "var(--green)" if broker != "—" else "var(--red)")

    st.divider()

    left, right = st.columns([1.25, 1])
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

        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("▶ Start (Paper)"):
                st.session_state.live_running = True
                st.success("Paper trading started (demo)")
        with b2:
            if st.button("⏸ Stop"):
                st.session_state.live_running = False
                st.warning("Stopped.")
        with b3:
            if st.button("⚙ Send Test Alert"):
                add_trade_log("NIFTY", "BUY", 50, 250.5, 0.0)
                st.info("Test alert → trade log added.")

        st.subheader("Live Market Data (Demo)")
        syms = ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "INFY"]
        prices = np.round(np.random.normal(100, 10, len(syms)), 2)
        live_df = pd.DataFrame({
            "Symbol": syms,
            "LTP": prices,
            "Change%": np.round(np.random.normal(0.2, 0.5, len(syms)), 2)
        })
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
st.caption("© 2025 Your Brand • This is a colorful demo UI. Replace demo handlers with your live logic, APIs, and secure storage.")
