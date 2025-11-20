import time
from datetime import datetime, timedelta
from typing import Dict


import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import yfinance as yf

#st.sidebar.image("shree.jpg",width=15)  # Correct parameter
# ------------------------------------------------------------
# Page Config & Global Theming
# ------------------------------------------------------------
st.set_page_config(
    page_title="Algo Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
def compute_performance(signal_df, brokerage_per_trade=20, gst_rate=0.18, stamp_duty_rate=0.00015, starting_capital=0):
    """
    Compute performance summary from signal log with PnL and include daily costs.
    
    Returns:
    - summary_df: Overall performance summary (including Total Expense)
    - pnl_per_day: Daily PnL with Total PnL, Net Expense, and Net PnL
    """
    import pandas as pd
    
    total_trades = len(signal_df)
    winning_trades = signal_df[signal_df['PnL'] > 0]
    losing_trades = signal_df[signal_df['PnL'] <= 0]
    
    total_pnl = signal_df['PnL'].sum()
    avg_pnl = signal_df['PnL'].mean() if total_trades > 0 else 0
    max_pnl = signal_df['PnL'].max() if total_trades > 0 else 0
    min_pnl = signal_df['PnL'].min() if total_trades > 0 else 0
    
    win_pct = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
    loss_pct = len(losing_trades) / total_trades * 100 if total_trades > 0 else 0
    
    # Group by date and calculate daily PnL
    pnl_per_day = signal_df.groupby('Date').agg({'PnL': 'sum', 'Quantity': 'sum'}).reset_index()
    
    cost_per_trade_list = []
    net_pnl_list = []
    capital_needed_list = []
    capital_after_list = []
    
    current_capital = starting_capital  # Initialize overall capital tracker
    
    for idx, row in pnl_per_day.iterrows():
        day_trades = signal_df[signal_df['Date'] == row['Date']]
        day_expense = 0
        day_capital_needed = 0  # Initialize daily capital before summing
        
        for _, trade in day_trades.iterrows():
            turnover = trade['Buy Premium'] * trade['Quantity'] * 2  # Buy + Sell
            brokerage = brokerage_per_trade
            gst = brokerage * gst_rate
            stamp_duty = turnover * stamp_duty_rate
            total_cost = brokerage + gst + stamp_duty
            day_expense += total_cost

            # Calculate capital needed for each trade
            trade_capital = trade['Buy Premium'] * trade['Quantity']
            day_capital_needed += trade_capital
        
        # Update capital after daily PnL
        current_capital += row['PnL'] - day_expense
        
        cost_per_trade_list.append(round(day_expense, 2))
        net_pnl_list.append(round(row['PnL'] - day_expense, 2))
        capital_needed_list.append(round(day_capital_needed, 2))
        capital_after_list.append(round(current_capital, 2))
    
    pnl_per_day['Total PnL'] = pnl_per_day['PnL'].round(2)
    pnl_per_day['Net Expense'] = cost_per_trade_list
    pnl_per_day['Net PnL'] = net_pnl_list
    pnl_per_day['Capital Needed'] = capital_needed_list
    pnl_per_day['Capital After'] = capital_after_list
    
    pnl_per_day = pnl_per_day[['Date', 'Total PnL', 'Net Expense', 'Net PnL', 'Capital Needed', 'Capital After']]
    
    total_expense = sum(cost_per_trade_list)
    
    summary = {
        "Total Trades": total_trades,
        "Winning Trades": len(winning_trades),
        "Losing Trades": len(losing_trades),
        "Win %": round(win_pct, 2),
        "Loss %": round(loss_pct, 2),
        "Total PnL": round(total_pnl, 2),
        "Average PnL": round(avg_pnl, 2),
        "Max PnL": round(max_pnl, 2),
        "Min PnL": round(min_pnl, 2),
        "Total Expense": round(total_expense, 2),
        "Net PnL (After Expenses)": round(sum(net_pnl_list), 2),
        "Final Capital": round(current_capital, 2)
    }
    
    summary_df = pd.DataFrame([summary])
    return summary_df, pnl_per_day

def compute_trade_pnl(signal_log_df, df):
    """
    Compute PnL and exit reason for each signal in signal_log_df based on price data in df.
    Returns updated DataFrame with Sell Price, PnL, and Exit Reason.
    """
    trade_results = []

    for _, row in signal_log_df.iterrows():
        day = row['Date']
        entry_time = row['Entry Time']
        exit_time = row['Time Exit (16 mins after entry)']
        buy_premium = row['Buy Premium']
        qty = row['Quantity']
        stoploss = row['Stoploss (Trailing 10%)']
        take_profit = row['Take Profit (10% rise)']
        option_type = row['Option Selected']

        # Filter df for the trading day and after entry
        day_df = df[df['Datetime'].dt.date == day]
        day_after_entry = day_df[(day_df['Datetime'] >= entry_time) & (day_df['Datetime'] <= exit_time)].sort_values('Datetime')

        sell_price = None
        actual_exit_time = exit_time
        exit_reason = "Time Exit"

        for _, candle in day_after_entry.iterrows():
            price = candle['Close_^NSEI']  # Spot price used for simulation; replace with option price if available
            
            # Check Take Profit for CALL or PUT
            if take_profit and (
                (option_type.upper() == "CE" and price >= take_profit) or
                (option_type.upper() == "PE" and price <= take_profit)
            ):
                sell_price = take_profit
                exit_reason = "Take Profit"
                actual_exit_time = candle['Datetime']
                break

            # Check Stoploss
            elif stoploss and (
                (option_type.upper() == "CE" and price <= stoploss) or
                (option_type.upper() == "PE" and price >= stoploss)
            ):
                sell_price = stoploss
                exit_reason = "Stoploss"
                actual_exit_time = candle['Datetime']
                break

        # If neither TP nor SL hit, exit at last available price (time exit)
        if sell_price is None:
            sell_price = day_after_entry['Close_^NSEI'].iloc[-1]

        # Compute PnL
        pnl = (sell_price - buy_premium) * qty if option_type.upper() == "CE" else (buy_premium - sell_price) * qty

        trade_results.append({
            **row.to_dict(),
            "Sell Price": sell_price,
            "Exit Reason": exit_reason,
            "Actual Exit Time": actual_exit_time,
            "PnL": pnl
        })

    return pd.DataFrame(trade_results)

def calculate_trade_cost(buy_price, sell_price, quantity, option_type="CE", brokerage_type="fixed"):
    """
    Calculate total cost/charges per trade.
    
    Params:
    - buy_price: Entry price per unit
    - sell_price: Exit price per unit
    - quantity: Number of units
    - option_type: "CE" or "PE"
    - brokerage_type: "fixed" or "percentage"
    
    Returns total charges
    """
    turnover = (buy_price + sell_price) * quantity

    # Brokerage
    if brokerage_type == "fixed":
        brokerage = 20  # assume ₹20 per trade
    else:  # percentage
        brokerage = turnover * 0.0003  # 0.03%

    # Exchange Transaction Charges
    exchange_charges = turnover * 0.0000325  # 0.00325%

    # GST on brokerage (18%)
    gst = 0.18 * brokerage

    # SEBI Charges (approx)
    sebi_charges = turnover * 0.000001

    # Stamp Duty (approx)
    stamp_duty = turnover * 0.00003

    total_charges = brokerage + exchange_charges + gst + sebi_charges + stamp_duty
    return total_charges

def compute_trade_pnl_with_costs(signal_log_df, df):
    """
    Compute PnL, exit reason, and capital impact per trade.
    """
    trade_results = []

    capital = 0  # running capital (cumulative PnL)
    

    for _, row in signal_log_df.iterrows():
        day = row['Date']
        entry_time = row['Entry Time']
        exit_time = row['Time Exit (16 mins after entry)']
        buy_premium = row['Buy Premium']
        qty = row['Quantity']
        stoploss = row['Stoploss (Trailing 10%)']
        take_profit = row['Take Profit (10% rise)']
        option_type = row['Option Selected']
        # Capital needed for this trade (premium × quantity)
        capital_needed = buy_premium * qty

        day_df = df[df['Datetime'].dt.date == day]
        day_after_entry = day_df[day_df['Datetime'] >= entry_time].sort_values('Datetime')

        sell_price = None
        exit_reason = "Time Exit"

        for _, candle in day_after_entry.iterrows():
            price = candle['Close_^NSEI']  # Option price if available
            if take_profit and price >= take_profit:
                sell_price = take_profit
                exit_reason = "Take Profit"
                exit_time = candle['Datetime']
                break
            elif stoploss and price <= stoploss:
                sell_price = stoploss
                exit_reason = "Stoploss"
                exit_time = candle['Datetime']
                break

        if sell_price is None:
            sell_price = day_after_entry['Close_^NSEI'].iloc[0]  # fallback

        raw_pnl = (sell_price - buy_premium) * qty if option_type.upper() == "CE" else (buy_premium - sell_price) * qty

        # Calculate brokerage & charges
        total_charges = calculate_trade_cost(buy_premium, sell_price, qty, option_type)
        
        net_pnl = raw_pnl - total_charges
        capital += net_pnl

        trade_results.append({
            **row.to_dict(),
            "Sell Price": sell_price,
            "Exit Reason": exit_reason,
            "Actual Exit Time": exit_time,
            "Raw PnL": raw_pnl,
            "Total Charges": total_charges,
            "Net PnL": net_pnl,
            "Capital Needed": capital_needed,  # ✅ Added column
            "Capital After Trade": capital
        })

    return pd.DataFrame(trade_results)
def display_3pm_candle_info(df, day):
    """
    Display the 3PM candle Open and Close prices for a given day (datetime.date).
    
    Parameters:
    - df: DataFrame with 'Datetime' column (timezone-aware datetime)
    - day: datetime.date object representing the trading day
    
    Returns:
    - (open_price, close_price) tuple or (None, None) if candle not found
    """
    candle = df[(df['Datetime'].dt.date == day) &
                (df['Datetime'].dt.hour == 15) &
                (df['Datetime'].dt.minute == 0)]
    
    if candle.empty:
        st.warning(f"No 3:00 PM candle found for {day}")
        return None, None
    
    open_price = candle.iloc[0]['Open_^NSEI']
    close_price = candle.iloc[0]['Close_^NSEI']
    
    #st.info(f"3:00 PM Candle for {day}: Open = {open_price}, Close = {close_price}")
    #st.write(f"🔵 3:00 PM Open for {day}: {open_price}")
    #st.write(f"🔴 3:00 PM Close for {day}: {close_price}")
    
    return open_price, close_price


def generate_trade_log_from_option(result, trade_signal):
    if result is None or trade_signal is None:
        return None
    # Determine exit reason and price
    stoploss_hit = False
    target_hit = False
    
    #exit_time = pd.to_datetime(exit_time)
    #result.index = pd.to_datetime(result.index)

    
    
    

    option = result['option_data']
    qty = result['total_quantity']

    condition = trade_signal['condition']
    entry_time = trade_signal['entry_time']
    message = trade_signal['message']

    buy_price = option.get('lastPrice', trade_signal.get('buy_price'))
    expiry = option.get('expiryDate', trade_signal.get('expiry'))
    option_type = option.get('optionType', trade_signal.get('option_type'))

    stoploss = buy_price * 0.9
    take_profit = buy_price * 1.10
    partial_qty = qty // 2
    time_exit = entry_time + timedelta(minutes=16)

    

    

    trade_log = {
        "Condition": condition,
        "Option Type": option_type,
        "Strike Price": option.get('strikePrice'),
        #"Exit Price": exit_price,  # ✅ new column
        "Buy Premium": buy_price,
        "Stoploss (Trailing 10%)": stoploss,
        "Take Profit (10% rise)": take_profit,
        "Quantity": qty,
        "Partial Profit Booking Qty (50%)": partial_qty,
        "Expiry Date": expiry.strftime('%Y-%m-%d') if hasattr(expiry, 'strftime') else expiry,
        "Entry Time": entry_time.strftime('%Y-%m-%d %H:%M:%S') if hasattr(entry_time, 'strftime') else entry_time,
        "Time Exit (16 mins after entry)": time_exit.strftime('%Y-%m-%d %H:%M:%S'),
        "Trade Message": message
        #"Trade Details": row["Trade Details"],
        #"Exit Reason": reason
    }

    # Add condition-specific details
    if condition == 1:
        trade_log["Trade Details"] = (
            "Buy nearest ITM CALL option. Stoploss trailing 10% below buy premium. "
            "Book 50% qty profit when premium rises 10%. "
            "Time exit after 16 minutes if no target hit."
        )
    elif condition == 2:
        trade_log["Trade Details"] = (
            "Major gap down. Buy nearest ITM PUT option when next candle crosses low of 9:30 candle. "
            "Stoploss trailing 10% below buy premium."
        )
    elif condition == 3:
        trade_log["Trade Details"] = (
            "Major gap up. Buy nearest ITM CALL option. Stoploss trailing 10% below buy premium. "
            "Book 50% qty profit when premium rises 10%. "
            "Time exit after 16 minutes if no target hit."
        )
    elif condition == 4:
        trade_log["Trade Details"] = (
            "Buy nearest ITM PUT option. Stoploss trailing 10% below buy premium. "
            "Book 50% qty profit when premium rises 10%. "
            "Time exit after 16 minutes if no target hit."
        )
    else:
        trade_log["Trade Details"] = "No specific trade details available."

    trade_log_df = pd.DataFrame([trade_log])
    return trade_log_df
def option_chain_finder(option_chain_df, spot_price, option_type, lots=10, lot_size=75):
    """
    Find nearest ITM option in option chain DataFrame.

    Parameters:
    - option_chain_df: pd.DataFrame with columns including ['strikePrice', 'expiryDate', 'optionType', ...]
    - spot_price: float, current underlying price
    - option_type: str, 'CE' for Call or 'PE' for Put
    - lots: int, number of lots to trade (default 10)
    - lot_size: int, lot size per option contract (default 75)

    Returns:
    - dict with keys:
        'strikePrice', 'expiryDate', 'optionType', 'total_quantity', 'option_data' (pd.Series row)
    """

    # Ensure expiryDate is datetime
    if not pd.api.types.is_datetime64_any_dtype(option_chain_df['expiryDate']):
        option_chain_df['expiryDate'] = pd.to_datetime(option_chain_df['expiryDate'])

    today = pd.Timestamp.today().normalize()

    # Find nearest expiry on or after today
    expiries = option_chain_df.loc[option_chain_df['expiryDate'] >= today, 'expiryDate'].unique()
    if len(expiries) == 0:
        raise ValueError("No expiry dates found on or after today.")
    nearest_expiry = min(expiries)

    # Filter for nearest expiry and option type
    df_expiry = option_chain_df[
        (option_chain_df['expiryDate'] == nearest_expiry) &
        (option_chain_df['optionType'] == option_type)
    ]

    if df_expiry.empty:
        raise ValueError(f"No options found for expiry {nearest_expiry.date()} and type {option_type}")

    # Find nearest ITM strike
    if option_type == "CALL":
        itm_strikes = df_expiry[df_expiry['strikePrice'] <= spot_price]
        if itm_strikes.empty:
            # fallback to minimum strike (OTM)
            nearest_strike = df_expiry['strikePrice'].min()
        else:
            nearest_strike = itm_strikes['strikePrice'].max()
    else:  # 'PE'
        itm_strikes = df_expiry[df_expiry['strikePrice'] >= spot_price]
        if itm_strikes.empty:
            # fallback to maximum strike (OTM)
            nearest_strike = df_expiry['strikePrice'].max()
        else:
            nearest_strike = itm_strikes['strikePrice'].min()

    # Get option row
    option_row = df_expiry[df_expiry['strikePrice'] == nearest_strike].iloc[0]

    total_qty = lots * lot_size

    return {
        'strikePrice': nearest_strike,
        'expiryDate': nearest_expiry,
        'optionType': option_type,
        'total_quantity': total_qty,
        'option_data': option_row
    }
def find_nearest_itm_option():
    import nsepython
    from nsepython import nse_optionchain_scrapper


    option_chain = nse_optionchain_scrapper('NIFTY')
    df = []
    
    for item in option_chain['records']['data']:
        strike = item['strikePrice']
        expiry = item['expiryDate']
        if 'CE' in item:
            ce = item['CE']
            ce['strikePrice'] = strike
            ce['expiryDate'] = expiry
            ce['optionType'] = 'CE'
            df.append(ce)
        if 'PE' in item:
            pe = item['PE']
            pe['strikePrice'] = strike
            pe['expiryDate'] = expiry
            pe['optionType'] = 'PE'
            df.append(pe)
    
    #import pandas as pd
    option_chain_df = pd.DataFrame(df)
    option_chain_df['expiryDate'] = pd.to_datetime(option_chain_df['expiryDate'])
    #st.write(option_chain_df.head())
    return  option_chain_df


def get_nearest_weekly_expiry(today):
    """
    Placeholder: implement your own logic to find nearest weekly expiry date
    For demo, returns today + 7 days (Saturday)
    """
    return today + pd.Timedelta(days=7)
    
def plot_nifty_multiday(df, trading_days):
    """
    Plots Nifty 15-min candles for multiple trading days with each previous day's 3PM Open/Close
    marked only on the next trading day and extending only until 3PM candle.
    
    Parameters:
    - df : DataFrame with columns ['Datetime', 'Open_^NSEI', 'High_^NSEI', 'Low_^NSEI', 'Close_^NSEI']
    - trading_days : list of sorted trading dates (datetime.date)
    """
    
    fig = go.Figure()
    
    for i in range(1, len(trading_days)):
        day0 = trading_days[i-1]  # Previous day (for Base Zone)
        day1 = trading_days[i]    # Current day
        
        # Filter data for current day only
        df_day1 = df[df['Datetime'].dt.date == day1]
        
        # Add candlestick trace for current day
        fig.add_trace(go.Candlestick(
            x=df_day1['Datetime'],
            open=df_day1['Open_^NSEI'],
            high=df_day1['High_^NSEI'],
            low=df_day1['Low_^NSEI'],
            close=df_day1['Close_^NSEI'],
            name=f"{day1}"
        ))
        
        # Get 3 PM candle of previous day (Base Zone)
        candle_3pm = df[df['Datetime'].dt.date == day0]
        candle_3pm = candle_3pm[(candle_3pm['Datetime'].dt.hour == 15) &
                                (candle_3pm['Datetime'].dt.minute == 0)]
        
        if not candle_3pm.empty:
            open_3pm = candle_3pm.iloc[0]['Open_^NSEI']
            close_3pm = candle_3pm.iloc[0]['Close_^NSEI']
            
            # Get day1 3PM candle time for line end
            day1_3pm_candle = df_day1[(df_day1['Datetime'].dt.hour == 15) &
                                       (df_day1['Datetime'].dt.minute == 0)]
            if not day1_3pm_candle.empty:
                x_end = day1_3pm_candle['Datetime'].iloc[0]
                x_start = df_day1['Datetime'].min()
                
                # Horizontal line for Open
                fig.add_shape(
                    type="line",
                    x0=x_start,
                    x1=x_end,
                    y0=open_3pm,
                    y1=open_3pm,
                    line=dict(color="blue", width=1, dash="dot"),
                )
                
                # Horizontal line for Close
                fig.add_shape(
                    type="line",
                    x0=x_start,
                    x1=x_end,
                    y0=close_3pm,
                    y1=close_3pm,
                    line=dict(color="red", width=1, dash="dot"),
                )
    
    # Layout adjustments
    fig.update_layout(
        title="Nifty 15-min Candles with Previous Day 3PM Open/Close Lines (to next day 3PM)",
        xaxis_rangeslider_visible=False,
        xaxis=dict(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),          # Hide weekends
                dict(bounds=[15.5, 9.25], pattern="hour")  # Hide off-hours
            ]
        )
    )
    
    return fig   

##############################################################################




def trading_signal_all_conditions1(df, quantity=10*75, return_all_signals=False):
    """
    Evaluate trading conditions based on Base Zone strategy with modified stop loss logic:
    - CALL stop loss = recent swing low (before entry)
    - PUT stop loss = recent swing high (before entry)
    """

    signals = []
    spot_price = df['Close_^NSEI'].iloc[-1]
    df = df.copy()
    df['Date'] = df['Datetime'].dt.date
    unique_days = sorted(df['Date'].unique())
    if len(unique_days) < 2:
        return None

    day0 = unique_days[-2]
    day1 = unique_days[-1]

    candle_3pm = df[(df['Date'] == day0) &
                    (df['Datetime'].dt.hour == 15) &
                    (df['Datetime'].dt.minute == 0)]
    if candle_3pm.empty:
        return None

    base_open = candle_3pm.iloc[0]['Open_^NSEI']
    base_close = candle_3pm.iloc[0]['Close_^NSEI']
    base_low = min(base_open, base_close)
    base_high = max(base_open, base_close)

    candle_915 = df[(df['Date'] == day1) &
                    (df['Datetime'].dt.hour == 9) &
                    (df['Datetime'].dt.minute == 15)]
    if candle_915.empty:
        return None

    H1 = candle_915.iloc[0]['High_^NSEI']
    L1 = candle_915.iloc[0]['Low_^NSEI']
    C1 = candle_915.iloc[0]['Close_^NSEI']
    entry_time = candle_915.iloc[0]['Datetime']

    expiry = get_nearest_weekly_expiry(pd.to_datetime(day1))
    day1_after_915 = df[(df['Date'] == day1) & (df['Datetime'] > entry_time)].sort_values('Datetime')

    # Helper to find recent swing high and low before a given time
    def get_recent_swing(df, current_time):
        recent_data = df[(df['Datetime'] < current_time)].tail(10)  # last 10 candles before entry
        swing_high = recent_data['High_^NSEI'].max()
        swing_low = recent_data['Low_^NSEI'].min()
        return swing_high, swing_low

    # Get recent swing values before 09:15
    swing_high, swing_low = get_recent_swing(df[df['Date'] == day1], entry_time)

    # Condition 1: CALL breakout
    if (L1 < base_high and H1 > base_low) and (C1 > base_high):
        sig = {
            'condition': 1,
            'option_type': 'CALL',
            'buy_price': H1,
            'stoploss': swing_low,             # Updated stoploss
            'take_profit': H1 * 1.10,
            'quantity': quantity,
            'expiry': expiry,
            'entry_time': entry_time,
            'message': 'Condition 1: Bullish breakout above Base Zone → Buy CALL above H1',
            'spot_price': spot_price
        }
        signals.append(sig)
        if not return_all_signals:
            return sig

    if C1 < base_low:
        for _, next_candle in day1_after_915.iterrows():
            swing_high, swing_low = get_recent_swing(df[df['Date'] == day1], next_candle['Datetime'])

            # Condition 2: PUT continuation
            if next_candle['Low_^NSEI'] < L1:
                sig = {
                    'condition': 2,
                    'option_type': 'PUT',
                    'buy_price': L1,
                    'stoploss': swing_high,        # Updated stoploss
                    'take_profit': L1 * 0.90,
                    'quantity': quantity,
                    'expiry': expiry,
                    'entry_time': next_candle['Datetime'],
                    'message': 'Condition 2: Gap down confirmed → Buy PUT below L1',
                    'spot_price': spot_price
                }
                signals.append(sig)
                if not return_all_signals:
                    return sig

            # Flip rule 2.7: CALL
            if next_candle['Close_^NSEI'] > base_high:
                ref_high = next_candle['High_^NSEI']
                sig_flip = {
                    'condition': 2.7,
                    'option_type': 'CALL',
                    'buy_price': ref_high,
                    'stoploss': swing_low,        # Updated stoploss
                    'take_profit': ref_high * 1.10,
                    'quantity': quantity,
                    'expiry': expiry,
                    'entry_time': next_candle['Datetime'],
                    'message': 'Condition 2 Flip: Later candle closed above Base Zone → Buy CALL above Candle 2 high',
                    'spot_price': spot_price
                }
                signals.append(sig_flip)
                if not return_all_signals:
                    return sig_flip

    if C1 > base_high:
        for _, next_candle in day1_after_915.iterrows():
            swing_high, swing_low = get_recent_swing(df[df['Date'] == day1], next_candle['Datetime'])

            # Condition 3: CALL continuation
            if next_candle['High_^NSEI'] > H1:
                sig = {
                    'condition': 3,
                    'option_type': 'CALL',
                    'buy_price': H1,
                    'stoploss': swing_low,        # Updated stoploss
                    'take_profit': H1 * 1.10,
                    'quantity': quantity,
                    'expiry': expiry,
                    'entry_time': next_candle['Datetime'],
                    'message': 'Condition 3: Gap up confirmed → Buy CALL above H1',
                    'spot_price': spot_price
                }
                signals.append(sig)
                if not return_all_signals:
                    return sig

            # Flip rule 3.7: PUT
            if next_candle['Close_^NSEI'] < base_low:
                ref_low = next_candle['Low_^NSEI']
                sig_flip = {
                    'condition': 3.7,
                    'option_type': 'PUT',
                    'buy_price': ref_low,
                    'stoploss': swing_high,       # Updated stoploss
                    'take_profit': ref_low * 0.90,
                    'quantity': quantity,
                    'expiry': expiry,
                    'entry_time': next_candle['Datetime'],
                    'message': 'Condition 3 Flip: Later candle closed below Base Zone → Buy PUT below Candle 3 low',
                    'spot_price': spot_price
                }
                signals.append(sig_flip)
                if not return_all_signals:
                    return sig_flip

    # Condition 4: PUT breakdown
    if (L1 < base_high and H1 > base_low) and (C1 < base_low):
        sig = {
            'condition': 4,
            'option_type': 'PUT',
            'buy_price': L1,
            'stoploss': swing_high,             # Updated stoploss
            'take_profit': L1 * 0.90,
            'quantity': quantity,
            'expiry': expiry,
            'entry_time': entry_time,
            'message': 'Condition 4: Bearish breakdown below Base Zone → Buy PUT below L1',
            'spot_price': spot_price
        }
        signals.append(sig)
        if not return_all_signals:
            return sig

    return signals if signals else None
    

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
            {"name": "3PM  Strategy 1.0", "short": "3PM 15min  with  last day breakout with today", "metrics": {"CAGR": 18.3, "Win%": 85.8, "MaxDD%": 15.6}},
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

    #st.image(
        #"https://assets-global.website-files.com/5e0a1f0d3a9f1b6f7f1b6f34/5e0a1f63a4f62a5534b5f5f9_finance-illustration.png",
        #use_container_width=True,
    #)

    MENU = st.radio(
        "Navigate",
        ["Home", "Strategies", "Zerodha Broker API","Groww Broker API", "Dashboard","Backtest", "Products", "Support"],
        index=0,
    )

# ------------------------------------------------------------
# Home
# ------------------------------------------------------------
if MENU == "Home":
    st.title("Welcome to SHREE Algo Trading Platform")
    st.write("Automate your trades with smart, auditable strategies. Connect your broker, choose a strategy, and manage risk — all from one dashboard.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### What is Algo Trading?")
        st.markdown(
        r"""
        Algorithmic Trading (**Algo Trading**) means using **computer programs** to automatically place
        **buy/sell orders** based on predefined, rule‑based logic. Instead of clicking buttons manually,
        algorithms monitor data streams (price, volume, indicators) and execute trades **fast**, **consistently**,
        and **without emotions**.
        
        
        ---
        
        
        ### 🔑 Why Traders Use It
        - **Automation**: Executes your plan 24×7 (where markets allow) exactly as written.
        - **Speed**: Milliseconds matter for entries, exits, and order routing.
        - **Backtesting**: Test your ideas on **historical data** before going live.
        - **Scalability**: Watch dozens of instruments simultaneously.
        
        
        ### ⚠️ Risks to Respect
        - **Bad logic = fast losses** (garbage in, garbage out).
        - **Overfitting**: Great on the past, weak in live markets.
        - **Operational**: Data glitches, API limits, slippage, latency.
        
        
        > **TL;DR**: Algo Trading = *rules → code → automated execution*.
        """
        )
        st.write("Algorithmic trading executes orders using pre-defined rules for entries, exits, and risk management.")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Why Choose Us?")
        st.markdown("""
        At **Shree Software**, we are committed to delivering **high-quality software solutions** that cater to your business needs. Our approach combines **innovation, reliability, and customer focus** to ensure your success.
        
        
        ---
        
        
        ### 🔑 Key Reasons to Choose Us
        
        
        1. **Expert Team**: Experienced developers and designers who understand your business challenges.
        2. **Customized Solutions**: Tailored software to match your specific requirements.
        3. **On-Time Delivery**: We value your time and ensure project timelines are met.
        4. **Affordable Pricing**: Competitive pricing without compromising on quality.
        5. **24/7 Support**: Dedicated support team to help you whenever you need assistance.
        
        
        > Our mission is to empower businesses through technology, making processes **efficient, reliable, and scalable**.
        
        
        ### 🎯 Our Approach
        - **Consultation & Analysis**: Understanding your business goals.
        - **Design & Development**: Building robust and scalable software.
        - **Testing & Quality Assurance**: Ensuring flawless performance.
        - **Deployment & Maintenance**: Smooth launch and continuous support.
        
        
        We combine the best of **technology, strategy, and creativity** to ensure your project stands out.
        """)
        
        
        # Optional HTML Styling for Highlight
        st.markdown(
        """
        <div style='padding:12px;border-radius:12px;background:#f0f8ff;border:1px solid #cce0ff;'>
        <h4 style='margin:0 0 6px 0;'>Client Commitment</h4>
        <p style='margin:0;'>We focus on delivering solutions that <b>drive growth, efficiency, and innovation</b> for our clients.</p>
        </div>
        """,
        unsafe_allow_html=True
        )
        
        
        st.success("Learn more about our services by contacting us today!")
        st.write("Colorful, clean UI, safer defaults, backtests, paper trading, and live automation with popular brokers.")
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Risk Disclaimer")
        st.write("Algorithmic trading executes orders using pre-defined rules for entries, exits, and risk management.Trading involves risk. Past performance does not guarantee future returns. Trade responsibly.")
        
        st.markdown("""
        ---
        ### ⚠️ Risk Disclaimer
        
        1. **Informational Purpose Only**  
           All content, services, or solutions provided by **Shree Software** are for **informational and educational purposes only**. They do not constitute financial, legal, or professional advice.
        
        2. **No Guaranteed Outcomes**  
           While we aim to provide accurate and timely information, **we do not guarantee any specific results, profits, or success** from using our services.
        
        3. **User Responsibility**  
           Clients and users must exercise **due diligence**, make **informed decisions**, and consult qualified professionals when necessary before acting on any information or solutions provided.
        
        4. **Limitation of Liability**  
           **Shree Software shall not be liable** for any direct, indirect, or consequential loss or damage resulting from the use of our services, content, or advice.
        
        5. **Third-Party Dependencies**  
           We may provide data, tools, or links from third-party sources. **We are not responsible for the accuracy, completeness, or outcomes** associated with such third-party information.
        
        6. **Market / Technology Risks** *(if applicable)*  
           For financial or technical solutions, market conditions, system failures, or software limitations may impact results. Users must acknowledge these **inherent risks**.
        ---
        """)
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
# Backtest Strategies
# ------------------------------------------------------------
elif MENU == "Backtest":
    st.title("Backtest Strategies")

    colA, colB = st.columns([2, 1])
    with colA:
        # Get strategy names
        names = [s["name"] for s in st.session_state.strategies]

        # Selectbox for strategies
        st.session_state.selected_strategy = st.selectbox(
            "Select strategy",
            names,
            index=0
        )

        # Find selected strategy details
        selected = next(
            (s for s in st.session_state.strategies if s["name"] == st.session_state.selected_strategy),
            None
        )
    st.divider()
    # Show description in markdown
    if selected and "description" in selected:
        st.markdown(f"### Strategy Info\n{selected['description']}")
            # Ask for inputs only when strategy is selected
        st.subheader("Backtest Parameters")

        equity = st.text_input("Enter Equity Symbol (e.g. TCS, INFY, NIFTY)")
        start_date = st.date_input("Start Date")
        end_date = st.date_input("End Date")

        capital = st.number_input("Initial Capital (₹)", value=100000, step=1000)
        risk = st.slider("Risk per Trade (%)", 1, 10, 2)

        if st.button("Run Backtest"):
            st.success(f"Running backtest for **{equity}** using strategy **{selected['name']}** from {start_date} to {end_date}")

    data_source = st.radio("Select Data Source", ["Yahoo Finance", "Upload CSV"])
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            start_date = df['Datetime'].min().date()
            end_date = df['Datetime'].max().date()
            
        else:
            st.stop()
    else:
           start_date = st.date_input("Select Start Date", value=datetime.today() - timedelta(days=15))
           end_date = st.date_input("Select End Date", value=datetime.today())
    
    if start_date >= end_date:
        st.warning("End date must be after start date")
        st.stop()
    
    # ✅ Download full data for range (start-1 day to end)
    download_start = start_date - timedelta(days=1)  # To include previous day for first day
    df = yf.download("^NSEI", start=download_start, end=end_date + timedelta(days=1), interval="15m")
    if df.empty:
        st.warning("No data for selected range")
        st.stop()
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Datetime'}, inplace=True)  # Ensure proper name
    #st.write(df.columns)
    #st.write(df.columns.tolist())
    
    # ✅ Normalize columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
    
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    if df['Datetime'].dt.tz is None:
        df['Datetime'] = df['Datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
    else:
        df['Datetime'] = df['Datetime'].dt.tz_convert('Asia/Kolkata')
    
    # ✅ Filter only NSE trading hours
    df = df[(df['Datetime'].dt.time >= datetime.strptime("09:15", "%H:%M").time()) &
            (df['Datetime'].dt.time <= datetime.strptime("15:30", "%H:%M").time())]
    
    # ✅ Get all unique trading days
    unique_days = sorted(df['Datetime'].dt.date.unique())
    
    # ✅ Filter for user-selected range
    unique_days = [d for d in unique_days if start_date <= d <= end_date]
    
    if len(unique_days) < 2:
        st.warning("Not enough trading days in the selected range")
        st.stop()
    
    # ✅ Initialize combined trade log
    combined_trade_log = []
    # trading_days = list of unique trading days in selected range
    
    trading_days = sorted([d for d in df['Datetime'].dt.date.unique() if start_date <= d <= end_date])
    
    fig = plot_nifty_multiday(df, trading_days)
    st.plotly_chart(fig, use_container_width=True)
    ########################################################################################

    # Initialize empty list to store signals
    signal_log_list = []
    # ✅ Loop through each day (starting from 2nd day in range)
    for i in range(1, len(unique_days)):
        day0 = unique_days[i-1]
        day1 = unique_days[i]
    
        day_df = df[df['Datetime'].dt.date.isin([day0, day1])]
    
        # Call your trading signal function
        signal = trading_signal_all_conditions1(day_df)
        

##############################################################################################################

    if signal:
        #st.write(f"### {day1} → Signal detected: {signal['message']}")
        #st.table(pd.DataFrame([signal]))

        # Get option chain and trade log
        result_chain = find_nearest_itm_option()
        spot_price = signal['spot_price']
        ot = "CE" if signal["option_type"].upper() == "CALL" else "PE"
        result = option_chain_finder(result_chain, spot_price, option_type=ot, lots=10, lot_size=75)
        
        # Extract the option selected info
        option_data = result['option_data']
        strike_price = option_data.get('strikePrice')
        buy_premium = option_data.get('lastPrice')
        identifier = option_data.get('identifier')

        # Construct signal log dictionary
        sig_log = {
            "Date": day1,  # Add the trading day
            "Condition Type": signal['condition'],
            "Entry Time": signal['entry_time'],
            "Spot Price": spot_price,
            "Option Selected": ot,
            "Identifier": identifier,
            "Strike Price": strike_price,
            "Buy Premium": buy_premium,
            "Stoploss (Trailing 10%)": buy_premium * 0.9 if buy_premium else None,
            "Take Profit (10% rise)": buy_premium * 1.1 if buy_premium else None,
            "Quantity": signal['quantity'],
            "Partial Profit Booking Qty (50%)": signal['quantity'] / 2,
            "Expiry Date": signal['expiry'],
            "Time Exit (16 mins after entry)": signal['entry_time'] + pd.Timedelta(minutes=16)
        }
        # Append to list
        signal_log_list.append(sig_log)
        trade_log_df = generate_trade_log_from_option(result, signal)
        
        # Drop Trade details column
        if 'Trade details' in trade_log_df.columns:
            trade_log_df = trade_log_df.drop(columns=['Trade details'])
        
        #st.table(trade_log_df)

        # Append to combined trade log
        combined_trade_log.append(trade_log_df)

        #df_plot = df[df['Datetime'].dt.date == selected_date]
        # Get all unique trading days in the data within the selected range
        unique_days = sorted(df['Datetime'].dt.date.unique())
        trading_days = [d for d in unique_days if start_date <= d <= end_date]
        
        # Loop through each day
        for i in range(1, len(trading_days)):
            day0 = trading_days[i-1]  # Previous day (for Base Zone)
            day1 = trading_days[i]    # Current day (for signals)
            
            df_plot = df[df['Datetime'].dt.date.isin([day0, day1])]
            
            # Now you can use df_plot safely
            open_3pm, close_3pm = display_3pm_candle_info(df_plot, day0)  

        ##################################################################################
        # Convert to DataFrame
        if signal_log_list:
            signal_log_df = pd.DataFrame(signal_log_list)
            st.write("### Signal Log")
            st.dataframe(signal_log_df)
            
            # Optional: download as CSV
            csv = signal_log_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Signal Log CSV", data=csv, file_name="signal_log.csv", mime="text/csv")
        else:
            st.write("No trade signals detected for the selected period.")
        
        ##################################################################################


        # ✅ Combine all logs into one DataFrame
        if combined_trade_log:
            all_trades = pd.concat(combined_trade_log, ignore_index=True)
            st.write("### Combined Trade Log for Selected Period")
            st.dataframe(all_trades)
        
            # ✅ Download button
            csv = all_trades.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Trade Log CSV", data=csv, file_name="multi_day_trade_log.csv", mime="text/csv")
        else:
            st.write("No trade signals found for the selected period.")

##################################################################################

        if signal_log_list:
            signal_log_df_with_costs = compute_trade_pnl_with_costs(signal_log_df, df)
            
            st.write("### Signal Log with PnL, Costs & Capital")
            st.dataframe(signal_log_df_with_costs)
        
            # Download CSV
            csv = signal_log_df_with_costs.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Trade Log with Costs CSV", data=csv, file_name="trade_log_with_costs.csv", mime="text/csv")

##################################################################################

        if signal_log_list:
            # Compute PnL and Exit Reason first
            signal_log_df_with_pnl = compute_trade_pnl(signal_log_df, df)
            
            # Performance summary
            perf_summary_df, pnl_per_day = compute_performance(signal_log_df_with_pnl)
            
            
            
            st.write("### PnL Per Day")
            st.table(pnl_per_day)
        
            st.write("### Performance Summary")
            st.table(perf_summary_df)
            
            # Optional: download CSV
            csv_perf = perf_summary_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Performance Summary CSV", data=csv_perf, file_name="performance_summary.csv", mime="text/csv")
            
            csv_daily = pnl_per_day.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Daily PnL CSV", data=csv_daily, file_name="pnl_per_day.csv", mime="text/csv")
        




       

 
########################################################################################################



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
        #####################################################################################################
       


        ######################################################################################################
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
# Zerodha Broker API Broker API
# ------------------------------------------------------------
elif MENU == "Zerodha Broker API":
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
# Groww Broker API
# ------------------------------------------------------------
elif MENU == "Groww Broker API":
    import os
    import json
    import streamlit as st
    from groww import GrowwHttpClient

    st.title("Groww Broker API")
    st.write("Connect your Groww account for live/paper trading. This is a demo integration.")

    gcol1, gcol2 = st.columns(2)

    # Init session state
    if "groww_status" not in st.session_state:
        st.session_state.groww_status = False
        st.session_state.groww_client = None

    with gcol1:
        st.subheader("Groww Login")
        st.text_input("Groww Client ID", key="g_client_id")
        st.text_input("Groww Password", type="password", key="g_password")
        st.text_input("Groww PIN", type="password", key="g_pin")

        if st.button("Connect Groww"):
            try:
                client = GrowwHttpClient()

                login_res = client.login(
                    st.session_state.g_client_id,
                    st.session_state.g_password,
                    st.session_state.g_pin
                )

                if login_res.get("success"):
                    st.session_state.groww_status = True
                    st.session_state.groww_client = client
                    st.success("Groww connected successfully!")
                else:
                    st.error("Groww login failed. Check credentials.")

            except Exception as e:
                st.error(f"Error: {e}")

    with gcol2:
        st.subheader("Connection Status")
        status = st.session_state.groww_status
        badge = (
            "<span class='badge success'>Connected</span>"
            if status else
            "<span class='badge danger'>Not Connected</span>"
        )
        st.markdown(f"**Groww**: {badge}", unsafe_allow_html=True)

        st.caption("Login via Groww API (demo). Store tokens securely for production.")


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

        
        ###################################################################################
        st.subheader("Live NIFTY 50 Market Data")

        # Predefined NIFTY50 stock list (you can fetch dynamically from NSE if needed)
        nifty50_symbols = ["^NSEI","^NSEBANK" ]
        
        # Fetch data
        data = yf.download(nifty50_symbols, period="1d", interval="1m")["Close"].iloc[-1]
        
        # Convert to DataFrame
        df = pd.DataFrame(data).reset_index()
        df.columns = ["Symbol", "LTP"]
        
        # Calculate % Change from previous close
        prev_close = yf.download(nifty50_symbols, period="2d", interval="1d")["Close"].iloc[-2]
        df["Change%"] = ((df["LTP"] - prev_close.values) / prev_close.values) * 100
        
        # Display in Streamlit
        #st.dataframe(df.style.format({"LTP": "{:.2f}", "Change%": "{:.2f}"}), use_container_width=True)

            # Function to apply color
        def color_change(val):
            if val > 0:
                return "color: green;"
            elif val < 0:
                return "color: red;"
            else:
                return "color: black;"

                # Custom color function for Styler
        def color_positive_negative(val):
            color = 'green' if val > 0 else 'red'
            return f'color: {color}; font-weight: bold;'
        
        # Apply colors to all columns
        styled_df = df.style.format({"LTP": "{:.2f}", "Change%": "{:.2f}"}).applymap(color_positive_negative, subset=["LTP", "Change%"]).applymap(
            lambda x: 'color: black; font-weight: bold;', subset=["Symbol"]
        )
        
        # Apply style
        #styled_df = df.style.format({"LTP": "{:.2f}", "Change%": "{:.2f}"}).applymap(color_change, subset=["Change%"])
        
        # Display in Streamlit
        st.dataframe(styled_df, use_container_width=True)
        #######################################################################################

    

    with right:     
        #st.subheader("NIFTY 15-Minute(Today + Previous Day)")
        # Fetch NIFTY 50 index data
# Fetch NIFTY 50 index data
        # Fetch NIFTY 50 index data
        ticker = "^NSEI"  # NIFTY Index symbol for Yahoo Finance
        end = datetime.now()
        start = end - timedelta(days=2)
        
        # Download data
        df = yf.download(ticker, start=start, end=end, interval="15m")
        
        # Ensure data is available
        if df.empty:
            st.error("⚠️ No 15-min data fetched from Yahoo Finance. Market may be closed or ticker invalid.")
        else:
            # If multi-index, flatten it
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join(col).strip() for col in df.columns.values]
            # Reset index
            df = df.reset_index()
            # Convert to IST
           # Ensure Datetime is in IST
            if df['Datetime'].dt.tz is None:  
                # naive → localize to UTC first
                df['Datetime'] = df['Datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
            else:
                # already tz-aware → just convert
                df['Datetime'] = df['Datetime'].dt.tz_convert('Asia/Kolkata')

            df = df.reset_index()

            # Filter only market hours (09:15 - 15:30)
            market_open = pd.to_datetime("09:15:00").time()
            market_close = pd.to_datetime("15:30:00").time()
            df = df[df['Datetime'].dt.time.between(market_open, market_close)]
            #st.write(df)
            # Remove timezone if exists
            #df['Datetime'] = df['Datetime'].dt.tz_localize(None)
            #st.write(df)
            # Extract date
            df['Date'] = df['Datetime'].dt.date
            unique_days = sorted(df['Date'].unique())
        
            # Filter last 2 days
            if len(unique_days) >= 2:
                filtered_df = df[df['Date'].isin(unique_days[-2:])]
            else:
                filtered_df = df
        
            # Plot candlestick chart
            def plot_candles(df, title="Candlestick Chart"):
                fig = go.Figure(data=[go.Candlestick(
                    x=df['Datetime'],
                    open=df['Open_^NSEI'],
                    high=df['High_^NSEI'],
                    low=df['Low_^NSEI'],
                    close=df['Close_^NSEI'],
                    name='candlestick'
                )])
                # Hide non-trading gaps on x-axis
                fig.update_xaxes(rangebreaks=[
                    dict(bounds=["sat", "mon"]),  # hide weekends
                    dict(bounds=[16, 9], pattern="hour"),  # hide non-market hours (after 15:30 until 09:15)
                ])
                # --- Find 3 PM candles ---
               # --- Find 3 PM candles ---
                three_pm = df[(df['Datetime'].dt.hour == 15) & (df['Datetime'].dt.minute == 0)]
                
                for _, row in three_pm.iterrows():
                    start_time = row['Datetime']
                    end_time   = start_time + timedelta(minutes=15)
                
                    open_price  = row['Open_^NSEI']
                    close_price = row['Close_^NSEI']
                
                    # Line for Open
                    fig.add_shape(
                        type="line",
                        x0=start_time, x1=end_time,
                        y0=open_price, y1=open_price,
                        line=dict(color="blue", width=1, dash="dot"),
                    )
                
                    # Line for Close
                    fig.add_shape(
                        type="line",
                        x0=start_time, x1=end_time,
                        y0=close_price, y1=close_price,
                        line=dict(color="red", width=1, dash="dot"),
                    )
                fig.update_layout(title=title, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
        
            plot_candles(filtered_df, title="NIFTY 15-Min Candlestick (Last 2 Days)")

    st.divider()
    
    if st.session_state.api_status.get("Zerodha"):

        funds, holdings, positions, orders = fetch_zerodha_data()
    
        st.subheader("Zerodha Account Overview")
    
        colA, colB = st.columns(2)
    
        with colA:
            st.metric("Total Funds", f"₹{funds:,.0f}")
    
        with colB:
            st.metric("Open Positions", len(positions))
    
        st.markdown("### Holdings")
        st.dataframe(holdings, use_container_width=True, height=200)
    
        st.markdown("### Positions")
        st.dataframe(positions, use_container_width=True, height=200)
    
        st.markdown("### Orders")
        st.dataframe(orders, use_container_width=True, height=200)


    

#############################################################################################################################st.subheader("Trade Logs")
    #st.dataframe(st.session_state.trade_logs, use_container_width=True)
    













############################################################################################################################

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
st.caption("© 2025 Shree Software • This is a colorful demo UI. Replace demo handlers with your live logic, APIs, and secure storage.")
