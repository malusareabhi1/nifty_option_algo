import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import time
import os
import plotly.express as px

st.set_page_config(page_title="F&O Market Dashboard", layout="wide")
st.title("📊 F&O Market Dashboard with Trends")

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
session = requests.Session()
session.get("https://www.nseindia.com", headers=HEADERS)

INDEX_SYMBOLS = ["NIFTY", "BANKNIFTY"]
FNO_SYMBOLS = ["RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK","SBIN",
               "HDFC","LT","AXISBANK","KOTAKBANK","MARUTI","ITC",
               "TATASTEEL","SUNPHARMA","BAJFINANCE","ULTRACEMCO"]

DATA_FILE = "FnO_History.xlsx"

# --- Fetch F&O Data ---
import requests
import json
#import time

# Persistent session
session = requests.Session()

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nseindia.com/",
    "Accept-Language": "en-US,en;q=0.9",
}

INDEX_SYMBOLS = ["NIFTY", "BANKNIFTY"]

def safe_get_json(url, retries=3, delay=2):
    """Helper function to safely fetch and parse JSON from NSE."""
    for _ in range(retries):
        try:
            resp = session.get(url, headers=HEADERS, timeout=10)
            if resp.status_code == 200 and resp.text.strip().startswith("{"):
                return resp.json()
        except Exception:
            pass
        time.sleep(delay)
    return None

def fetch_data(symbol):
    data = {"Symbol": symbol, "Futures Price": None, "Call OI": None, "Put OI": None, "PCR": None, "Direction": None}
    try:
        # --- Fetch Futures Data ---
        fut_url = f"https://www.nseindia.com/api/quote-derivative?symbol={symbol}"
        fut_resp = safe_get_json(fut_url)
        if fut_resp and "stocks" in fut_resp:
            data["Futures Price"] = fut_resp["stocks"][0]["marketDeptOrderBook"]["tradeInfo"]["lastPrice"]
        else:
            data["Direction"] = "Error: Futures data unavailable"
            return data

        # --- Fetch Option Chain Data ---
        if symbol in INDEX_SYMBOLS:
            opt_url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
        else:
            opt_url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"

        opt_resp = safe_get_json(opt_url)
        if not opt_resp or "records" not in opt_resp:
            data["Direction"] = "Error: Option chain data unavailable"
            return data

        records = opt_resp["records"]["data"]
        call_oi = sum(d["CE"]["openInterest"] for d in records if "CE" in d)
        put_oi = sum(d["PE"]["openInterest"] for d in records if "PE" in d)
        data["Call OI"] = call_oi
        data["Put OI"] = put_oi
        data["PCR"] = round(put_oi / call_oi, 2) if call_oi else None

        # --- Direction Logic ---
        pcr = data["PCR"]
        if pcr is None:
            data["Direction"] = "Data Error"
        elif pcr > 1.3:
            data["Direction"] = "Bullish (Overbought)"
        elif 0.7 <= pcr <= 1.3:
            data["Direction"] = "Bullish / Stable"
        elif pcr < 0.7:
            data["Direction"] = "Bearish / Weak"
        else:
            data["Direction"] = "Sideways"

    except Exception as e:
        data["Direction"] = f"Error: {str(e)}"

    return data


# --- Fetch Live Data ---
st.info("Fetching live F&O data... may take 30-60 sec")
index_data = [fetch_data(sym) for sym in INDEX_SYMBOLS]
stock_data = [fetch_data(sym) for sym in FNO_SYMBOLS]

df_today = pd.DataFrame(index_data + stock_data)
df_today["Date"] = datetime.now().strftime("%Y-%m-%d")

# --- Save Historical Data ---
if os.path.exists(DATA_FILE):
    df_history = pd.read_excel(DATA_FILE)
    df_history = pd.concat([df_history, df_today], ignore_index=True)
else:
    df_history = df_today.copy()

df_history.to_excel(DATA_FILE, index=False)

# --- Display Index Summary ---
st.subheader("📈 Major Indices")
st.dataframe(pd.DataFrame(index_data)[["Symbol","Futures Price","PCR","Direction"]])

# --- Display Top 5 Bullish/Bearish ---
st.subheader("🔥 Top 5 Bullish Stocks")
st.dataframe(pd.DataFrame(stock_data).sort_values(by="PCR", ascending=False).head(5)[["Symbol","Futures Price","PCR","Direction"]])

st.subheader("❄️ Top 5 Bearish Stocks")
st.dataframe(pd.DataFrame(stock_data).sort_values(by="PCR").head(5)[["Symbol","Futures Price","PCR","Direction"]])

# --- Dynamic Trend Chart ---
st.subheader("📊 Stock/Index PCR & Futures Price Trend")

symbols_all = [*INDEX_SYMBOLS, *FNO_SYMBOLS]
selected_symbol = st.selectbox("Select Symbol:", symbols_all)

df_symbol = df_history[df_history["Symbol"]==selected_symbol]
if not df_symbol.empty:
    # PCR Trend
    fig_pcr = px.line(df_symbol, x="Date", y="PCR", title=f"{selected_symbol} PCR Trend (Last {len(df_symbol)} Days)")
    st.plotly_chart(fig_pcr, use_container_width=True)

    # Futures Price Trend
    fig_price = px.line(df_symbol, x="Date", y="Futures Price", title=f"{selected_symbol} Futures Price Trend")
    st.plotly_chart(fig_price, use_container_width=True)
else:
    st.warning("No historical data for selected symbol yet.")

# --- Download Full Excel ---
st.subheader("💾 Download Full F&O History")
with open(DATA_FILE, "rb") as f:
    st.download_button(
        label="📥 Download Excel Report",
        data=f,
        file_name=DATA_FILE,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
