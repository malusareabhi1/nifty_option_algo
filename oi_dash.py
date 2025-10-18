import streamlit as st
import requests
import json
import pandas as pd
import time

st.set_page_config(page_title="NIFTY OI & PCR Dashboard", layout="centered")

st.title("📊 NIFTY / BANKNIFTY OI, PCR & Market Direction")

# NSE request setup
session = requests.Session()
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nseindia.com/",
    "Accept-Language": "en-US,en;q=0.9",
}

# Helper function to safely fetch JSON
def safe_get_json(url, retries=3, delay=2):
    for _ in range(retries):
        try:
            resp = session.get(url, headers=HEADERS, timeout=10)
            if resp.status_code == 200 and resp.text.strip().startswith("{"):
                return resp.json()
        except Exception:
            pass
        time.sleep(delay)
    return None

# Function to fetch OI, PCR, direction
def fetch_oi_data(symbol):
    data = {"Symbol": symbol, "Call OI": None, "Put OI": None, "PCR": None, "Direction": None}
    try:
        opt_url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
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

        # Direction logic
        pcr = data["PCR"]
        if pcr is None:
            data["Direction"] = "No Data"
        elif pcr > 1.3:
            data["Direction"] = "📈 Bullish (Overbought)"
        elif 0.7 <= pcr <= 1.3:
            data["Direction"] = "⚖️ Neutral / Stable"
        elif pcr < 0.7:
            data["Direction"] = "📉 Bearish / Weak"
        else:
            data["Direction"] = "Sideways"
    except Exception as e:
        data["Direction"] = f"Error: {str(e)}"

    return data


# --- Streamlit UI ---
symbol = st.selectbox("Select Index", ["NIFTY", "BANKNIFTY"])
if st.button("Fetch Live Data"):
    with st.spinner("Fetching live data from NSE..."):
        result = fetch_oi_data(symbol)
        time.sleep(1)
        st.success("Data fetched successfully ✅")

        df = pd.DataFrame([result])
        st.table(df)

        # Optional: Display summary text
        st.subheader("📊 Summary")
        st.markdown(f"""
        - **Symbol:** {symbol}  
        - **Total Call OI:** {result['Call OI']:,}  
        - **Total Put OI:** {result['Put OI']:,}  
        - **PCR:** {result['PCR']}  
        - **Market Direction:** {result['Direction']}  
        """)

