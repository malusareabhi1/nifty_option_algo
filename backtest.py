import pandas as pd
import yfinance as yf

def backtest_straddle(symbol, expiry, strike, start_time="09:30", end_time="15:00"):
    ce = f"{symbol}{expiry}{strike}CE.NS"
    pe = f"{symbol}{expiry}{strike}PE.NS"

    ce_data = yf.download(ce, period="1d", interval="5m")
    pe_data = yf.download(pe, period="1d", interval="5m")

    if ce_data.empty or pe_data.empty:
        return None

    ce_data = ce_data.between_time(start_time, end_time)
    pe_data = pe_data.between_time(start_time, end_time)

    ce_entry = ce_data['Open'].iloc[0]
    pe_entry = pe_data['Open'].iloc[0]
    ce_exit = ce_data['Close'].iloc[-1]
    pe_exit = pe_data['Close'].iloc[-1]

    pnl = (ce_exit + pe_exit) - (ce_entry + pe_entry)

    result = pd.DataFrame({
        "Option": ["CE", "PE"],
        "Entry": [ce_entry, pe_entry],
        "Exit": [ce_exit, pe_exit],
        "P&L": [ce_exit - ce_entry, pe_exit - pe_entry]
    })
    result.loc["Total"] = ["-", "-", "-", pnl]
    return result
