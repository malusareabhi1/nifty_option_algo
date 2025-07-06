import streamlit as st
from strategy import get_atm_strike
from backtest import backtest_straddle
from paper_trade import paper_trade, get_trade_log
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime

st.title("🔁 Nifty Option Algo: Backtest + Paper Trade")

symbol = "^NSEI"
expiry = st.text_input("Expiry (e.g., 11JUL24):", "11JUL24")

if st.button("Get ATM Strike"):
    atm_strike, spot = get_atm_strike(symbol)
    #st.write(f"Spot: ₹{spot:.2f}, ATM Strike: {atm_strike}")
    if spot is not None and atm_strike is not None:
        st.write(f"Spot: ₹{spot:.2f}, ATM Strike: {atm_strike}")
    else:
        st.error("❌ Unable to fetch spot price or calculate ATM strike.")


    if st.button("Backtest ATM Straddle"):
        result = backtest_straddle("NIFTY", expiry, atm_strike)
        if result is not None:
            st.write(result)
            st.success(f"Total P&L: ₹{result['P&L'].iloc[-1]:.2f}")
        else:
            st.error("Option data not available.")

    qty = st.number_input("Lot Size", value=50)
    sl = st.number_input("Stoploss %", value=0.2)
    tgt = st.number_input("Target %", value=0.4)
    entry_time = st.time_input("Paper Trade Entry Time", value=datetime.strptime("09:30:00", "%H:%M:%S").time())

    if st.button("Paper Buy CE & PE"):
        now = datetime.now().time()
        if now >= entry_time:
            df1 = paper_trade(f"NIFTY{expiry}{atm_strike}CE", "Buy", 100, qty, sl, tgt)
            df2 = paper_trade(f"NIFTY{expiry}{atm_strike}PE", "Buy", 100, qty, sl, tgt)
            df = pd.concat([df1, df2])
            st.dataframe(df)
        else:
            st.warning(f"Current time {now.strftime('%H:%M:%S')} is before entry time {entry_time.strftime('%H:%M:%S')}")

    if st.button("Refresh Trade Log"):
        st.dataframe(get_trade_log())

    trade_df = get_trade_log()
    if not trade_df.empty:
        csv = trade_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Trade Log", csv, "trade_log.csv", "text/csv")

    # Chart for CE price
    if expiry and st.button("📈 Show CE Price Chart"):
        ce_symbol = f"NIFTY{expiry}{atm_strike}CE.NS"
        ce_data = yf.download(ce_symbol, period="1d", interval="5m")
        if not ce_data.empty:
            fig = go.Figure(data=[go.Candlestick(x=ce_data.index,
                                                 open=ce_data['Open'],
                                                 high=ce_data['High'],
                                                 low=ce_data['Low'],
                                                 close=ce_data['Close'])])
            fig.update_layout(title=f"{ce_symbol} Price Chart", xaxis_title="Time", yaxis_title="Price")
            st.plotly_chart(fig)
        else:
            st.warning("No CE data available.")
