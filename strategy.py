import yfinance as yf

def get_atm_strike(symbol="^NSEI"):
    data = yf.download(symbol, period="1d", interval="5m")
    spot_price = data['Close'].iloc[-1]
    atm_strike = round(spot_price / 50) * 50
    return atm_strike, spot_price

def get_spot_price():
    try:
        data = yf.download("^NSEI", period="1d", interval="1m")
        return data['Close'].iloc[-1]
    except:
        return None

