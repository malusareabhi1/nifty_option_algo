import pandas as pd
from datetime import datetime

trade_log = []

def paper_trade(symbol, side, price, qty, sl_pct=0.2, target_pct=0.4):
    stoploss = price * (1 - sl_pct / 100)
    target = price * (1 + target_pct / 100)

    trade = {
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Symbol": symbol,
        "Side": side,
        "Price": price,
        "Qty": qty,
        "Stoploss": round(stoploss, 2),
        "Target": round(target, 2),
        "P&L": 0
    }
    trade_log.append(trade)
    return pd.DataFrame(trade_log)

def get_trade_log():
    return pd.DataFrame(trade_log)
