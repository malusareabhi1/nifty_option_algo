import yfinance as yf

def get_atm_strike(symbol="^NSEI"):
    data = yf.download(symbol, period="1d", interval="5m")
    spot_price = data['Close'].iloc[-1]
    atm_strike = round(spot_price / 50) * 50
    return atm_strike, spot_price
