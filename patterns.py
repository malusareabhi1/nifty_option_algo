"""
Streamlit app: Multi-Timeframe Chart Pattern Detector + Trend & Trade Signal

Features:
- Fetch data via yfinance or upload CSV
- Analyze multiple timeframes (user-selectable list)
- Detect simple chart patterns: Double Top/Bottom, Head & Shoulders, Symmetric Triangle
- Compute ATR for stoploss sizing
- Aggregate signals across timeframes and suggest trade (LONG / SHORT / NO TRADE)
- Interactive Plotly candlestick with pattern markers

Notes / Limitations:
- Pattern detection uses heuristic rules (fast prototyping). For production, replace with more robust algorithms or ML models.
- yfinance interval support: '1m','2m','5m','15m','30m','60m','90m','1h','1d','1wk','1mo' depending on symbol and date range.

Author: Generated for user
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ----------------------- Utility functions -----------------------

def calculate_atr(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=1).mean()
    return atr


def is_local_peak(series, idx, window=5, rel_threshold=0.002):
    # Check if value at idx is max in window around it and noticeably greater than neighbors
    left = max(0, idx - window)
    right = min(len(series) - 1, idx + window)
    segment = series[left:right+1]
    val = series.iloc[idx]
    if val != segment.max():
        return False
    # relative threshold ensures it is not a tiny spike
    if (val - segment.median()) / (segment.median() + 1e-9) < rel_threshold:
        return False
    return True


def find_peaks_and_troughs(df, window=5, rel_threshold=0.002):
    highs = df['High']
    lows = df['Low']
    peak_idxs = []
    trough_idxs = []
    for i in range(len(df)):
        if is_local_peak(highs, i, window=window, rel_threshold=rel_threshold):
            peak_idxs.append(i)
        if is_local_peak(-lows, i, window=window, rel_threshold=rel_threshold):
            trough_idxs.append(i)
    return peak_idxs, trough_idxs


def detect_double_top_bottom(df, peaks, troughs, tolerance_pct=0.03, lookback=60):
    # Double Top: two peaks at roughly same level, with a trough between them
    # Double Bottom: two troughs at same level with peak between
    signals = []
    # peaks
    for i in range(len(peaks) - 1):
        p1, p2 = peaks[i], peaks[i+1]
        if p2 - p1 > lookback:
            continue
        price1 = df['High'].iloc[p1]
        price2 = df['High'].iloc[p2]
        if abs(price1 - price2) / max(price1, price2) <= tolerance_pct:
            # ensure trough between
            middle = df['Low'].iloc[p1:p2+1].min()
            signals.append({'type': 'double_top', 'idxs': (p1, p2), 'level': (price1+price2)/2, 'neckline': middle})
    for i in range(len(troughs) - 1):
        t1, t2 = troughs[i], troughs[i+1]
        if t2 - t1 > lookback:
            continue
        price1 = df['Low'].iloc[t1]
        price2 = df['Low'].iloc[t2]
        if abs(price1 - price2) / max(abs(price1), abs(price2)) <= tolerance_pct:
            middle = df['High'].iloc[t1:t2+1].max()
            signals.append({'type': 'double_bottom', 'idxs': (t1, t2), 'level': (price1+price2)/2, 'neckline': middle})
    return signals


def detect_head_and_shoulders(df, peaks, lookback=90, shoulder_tolerance=0.06):
    signals = []
    # Look for three peaks in sequence: left shoulder, head (higher), right shoulder
    for i in range(len(peaks)-2):
        p1, p2, p3 = peaks[i], peaks[i+1], peaks[i+2]
        if not (p1 < p2 < p3):
            continue
        # Ensure distances reasonable
        if p3 - p1 > lookback:
            continue
        h1 = df['High'].iloc[p1]
        h2 = df['High'].iloc[p2]
        h3 = df['High'].iloc[p3]
        # head higher than shoulders
        if h2 > h1 and h2 > h3:
            # shoulders roughly equal
            if abs(h1 - h3) / max(h1, h3) <= shoulder_tolerance:
                neckline = min(df['Low'].iloc[p1:p3+1])
                signals.append({'type': 'head_and_shoulders', 'idxs': (p1, p2, p3), 'neckline': neckline})
    # inverse H&S on troughs
    # find three troughs
    troughs = []
    lows = df['Low']
    for i in range(len(lows)):
        if is_local_peak(-lows, i, window=5):
            troughs.append(i)
    for i in range(len(troughs)-2):
        t1, t2, t3 = troughs[i], troughs[i+1], troughs[i+2]
        if not (t1 < t2 < t3):
            continue
        if t3 - t1 > lookback:
            continue
        l1 = df['Low'].iloc[t1]
        l2 = df['Low'].iloc[t2]
        l3 = df['Low'].iloc[t3]
        if l2 < l1 and l2 < l3 and abs(l1 - l3) / max(l1, l3) <= shoulder_tolerance:
            neckline = max(df['High'].iloc[t1:t3+1])
            signals.append({'type': 'inverse_head_and_shoulders', 'idxs': (t1,t2,t3), 'neckline': neckline})
    return signals


def detect_symmetric_triangle(df, peaks, troughs, lookback=120, slope_tol=0.0005):
    # Use linear regression to estimate slope of recent highs and lows
    signals = []
    import numpy as np
    from numpy.linalg import lstsq
    n = len(df)
    if n < 20:
        return signals
    win = min(lookback, n)
    x = np.arange(win)
    highs = df['High'].iloc[-win:].values
    lows = df['Low'].iloc[-win:].values
    # regress highs and lows
    A = np.vstack([x, np.ones(len(x))]).T
    m_high, c_high = lstsq(A, highs, rcond=None)[0]
    m_low, c_low = lstsq(A, lows, rcond=None)[0]
    # check slopes have opposite signs and magnitude decreasing (converging)
    if abs(m_high) < 1 and abs(m_low) < 1 and m_high * m_low < 0 and abs(m_high) < slope_tol and abs(m_low) < slope_tol:
        # further check recent range contracted
        recent_range = (df['High'].iloc[-win:] - df['Low'].iloc[-win:]).mean()
        long_range = (df['High'].iloc[-2*win//3:] - df['Low'].iloc[-2*win//3:]).mean()
        if recent_range < long_range * 1.05:
            signals.append({'type': 'symmetric_triangle', 'start_idx': n-win, 'end_idx': n-1})
    return signals


def generate_signals_for_df(df):
    peak_idxs, trough_idxs = find_peaks_and_troughs(df, window=5, rel_threshold=0.003)
    dt_signals = detect_double_top_bottom(df, peak_idxs, trough_idxs)
    hs_signals = detect_head_and_shoulders(df, peak_idxs)
    tri_signals = detect_symmetric_triangle(df, peak_idxs, trough_idxs)
    # compile
    signals = dt_signals + hs_signals + tri_signals
    return signals


def score_pattern_signal(pattern):
    # Map pattern to bullish/bearish and score
    typ = pattern['type']
    if typ in ['double_top', 'head_and_shoulders']:
        return {'direction': 'bearish', 'score': 1}
    if typ in ['double_bottom', 'inverse_head_and_shoulders']:
        return {'direction': 'bullish', 'score': 1}
    if typ == 'symmetric_triangle':
        # triangle can breakout either way - neutral but with continuation bias -> use last slope
        return {'direction': 'neutral', 'score': 0.5}
    return {'direction': 'neutral', 'score': 0}

# ----------------------- Streamlit UI -----------------------

st.set_page_config(page_title="Multi-Timeframe Pattern Detector", layout="wide")
st.title("Multi-Timeframe Chart Pattern Detector — Trend & Trade Suggestion")

with st.sidebar:
    st.header("Inputs")
    symbol = st.text_input("Ticker symbol (yfinance)", value="AAPL")
    use_csv = st.checkbox("Upload OHLCV CSV instead of yfinance", value=False)
    uploaded_file = None
    if use_csv:
        uploaded_file = st.file_uploader("Upload CSV with columns Date,Open,High,Low,Close,Volume", type=['csv'])
    tf_choices = st.multiselect("Timeframes to analyze (multiple)", options=['1m','5m','15m','30m','60m','1d','1wk','1mo'], default=['15m','1h','1d'])
    if '1h' in tf_choices:
        # yfinance expects '60m'
        pass
    start_date = st.date_input("Start date", value=(datetime.now() - timedelta(days=60)).date())
    end_date = st.date_input("End date", value=datetime.now().date())
    analyze_btn = st.button("Analyze")
    atr_period = st.number_input("ATR period", value=14, min_value=1)
    risk_pct = st.number_input("Risk per trade (% of capital)", value=1.0, min_value=0.1, step=0.1)
    capital = st.number_input("Account capital (for position sizing)", value=100000.0, step=1000.0)

# helper to normalize timeframe name for yfinance

def normalize_interval(tf):
    mapping = {'1h':'60m'}
    return mapping.get(tf, tf)

# main analyze
if analyze_btn:
    if use_csv and uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file, parse_dates=['Date'])
        df_raw = df_raw.rename(columns={c: c.capitalize() for c in df_raw.columns})
        df_raw.set_index('Date', inplace=True)
        st.success("CSV loaded — using uploaded data for single timeframe analysis")
        # We will run analyze only for the single timeframe represented
        # assume the CSV's frequency is whatever the user uploaded
        dfs = { 'uploaded': df_raw }
        tf_labels = ['uploaded']
    else:
        if use_csv and uploaded_file is None:
            st.error("Please upload a CSV file or uncheck 'Upload CSV' to use yfinance.")
            st.stop()
        if len(tf_choices) == 0:
            st.error("Please choose at least one timeframe to analyze.")
            st.stop()
        dfs = {}
        tf_labels = []
        for tf in tf_choices:
            yf_interval = normalize_interval(tf)
            try:
                # yfinance can have issues for intraday far history; user must choose reasonable start range
                data = yf.download(symbol, start=start_date, end=(end_date + timedelta(days=1)), interval=yf_interval, progress=False, threads=False)
            except Exception as e:
                st.error(f"Error downloading {symbol} {tf}: {e}")
                continue
            if data is None or data.empty:
                st.warning(f"No data for {symbol} at {tf}")
                continue
            data = data.rename(columns={
                'Open':'Open','High':'High','Low':'Low','Close':'Close','Volume':'Volume'
            })
            data = data[['Open','High','Low','Close','Volume']].dropna()
            dfs[tf] = data
            tf_labels.append(tf)

    if len(dfs) == 0:
        st.error("No data to analyze — try other timeframes or symbol.")
        st.stop()

    # Container for multi-timeframe results
    overall_votes = {'bullish':0, 'bearish':0, 'neutral':0}
    details = {}

    for tf, df in dfs.items():
        st.subheader(f"Analysis — {symbol} — Timeframe: {tf}")
        df = df.copy().reset_index()
        # Ensure datetime index
        if 'Date' in df.columns:
            df = df.rename(columns={'Date':'Datetime'})
        if 'Datetime' not in df.columns and hasattr(df.index, 'name'):
            df['Datetime'] = df.index
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True)
        # compute ATR
        df['ATR'] = calculate_atr(df, period=atr_period)
        signals = generate_signals_for_df(df)
        st.write(f"Detected {len(signals)} pattern(s) in {tf}")
        # Score
        tf_score = {'bullish':0,'bearish':0,'neutral':0}
        for s in signals:
            sc = score_pattern_signal(s)
            dir = sc['direction']
            if dir == 'bullish':
                tf_score['bullish'] += sc['score']
            elif dir == 'bearish':
                tf_score['bearish'] += sc['score']
            else:
                tf_score['neutral'] += sc['score']
        # Trend by moving averages
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()
        ma_signal = 'neutral'
        if df['SMA20'].iloc[-1] > df['SMA50'].iloc[-1]:
            ma_signal = 'bullish'
            tf_score['bullish'] += 0.5
        elif df['SMA20'].iloc[-1] < df['SMA50'].iloc[-1]:
            ma_signal = 'bearish'
            tf_score['bearish'] += 0.5
        details[tf] = {'signals': signals, 'ma_signal': ma_signal, 'tf_score': tf_score, 'df': df}
        # pick winner for tf
        winner = max(tf_score, key=lambda k: tf_score[k])
        overall_votes[winner] += 1

    # Aggregate multi-timeframe decision
    st.header("Multi-Timeframe Aggregation & Trade Suggestion")
    st.write("Votes:", overall_votes)
    agg_winner = max(overall_votes, key=lambda k: overall_votes[k])

    # derive recommended trade
    if agg_winner == 'bullish':
        recommendation = 'LONG'
    elif agg_winner == 'bearish':
        recommendation = 'SHORT'
    else:
        recommendation = 'NO CLEAR TREND'

    st.subheader(f"Recommendation: {recommendation}")

    # Show detailed cards per timeframe with plots
    cols = st.columns(len(details))
    for i, (tf, info) in enumerate(details.items()):
        with cols[i]:
            st.markdown(f"**{tf} — MA: {info['ma_signal']}**")
            df = info['df']
            signals = info['signals']
            # Build candlestick
            fig = go.Figure(data=[go.Candlestick(x=df.index,
                                                 open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='price')])
            # mark patterns
            for s in signals[:10]:
                typ = s['type']
                if typ == 'double_top':
                    x0 = df.index[s['idxs'][0]]
                    x1 = df.index[s['idxs'][1]]
                    y = s['level']
                    fig.add_scatter(x=[x0,x1], y=[y,y], mode='markers+lines', name='Double Top', marker={'symbol':'x','size':9})
                elif typ == 'double_bottom':
                    x0 = df.index[s['idxs'][0]]
                    x1 = df.index[s['idxs'][1]]
                    y = s['level']
                    fig.add_scatter(x=[x0,x1], y=[y,y], mode='markers+lines', name='Double Bottom', marker={'symbol':'circle','size':8})
                elif typ == 'head_and_shoulders':
                    p1,p2,p3 = s['idxs']
                    fig.add_scatter(x=[df.index[p1],df.index[p2],df.index[p3]], y=[df['High'].iloc[p1],df['High'].iloc[p2],df['High'].iloc[p3]], mode='markers+lines', name='H&S')
                elif typ == 'inverse_head_and_shoulders':
                    t1,t2,t3 = s['idxs']
                    fig.add_scatter(x=[df.index[t1],df.index[t2],df.index[t3]], y=[df['Low'].iloc[t1],df['Low'].iloc[t2],df['Low'].iloc[t3]], mode='markers+lines', name='Inverse H&S')
                elif typ == 'symmetric_triangle':
                    start = s['start_idx']; end = s['end_idx']
                    fig.add_vrect(x0=df.index[start], x1=df.index[end], fillcolor="LightSalmon", opacity=0.1, line_width=0)
            fig.update_layout(height=350, margin={'l':0,'r':0,'t':20,'b':20})
            st.plotly_chart(fig, use_container_width=True)

    # Trade sizing & stops using latest timeframe's ATR
    # choose highest timeframe df for conservative ATR (prefer daily if available)
    prefer_order = ['1mo','1wk','1d','60m','30m','15m','5m','1m','uploaded']
    chosen_df = None
    for k in prefer_order:
        if k in details:
            chosen_df = details[k]['df']
            break
    if chosen_df is None:
        # pick any
        chosen_df = list(details.values())[0]['df']
    last_close = chosen_df['Close'].iloc[-1]
    atr = chosen_df['ATR'].iloc[-1]
    st.write(f"Latest close: {last_close:.2f}, ATR: {atr:.4f}")

    if recommendation in ['LONG','SHORT']:
        # simple position sizing: risk_pct of capital to ATR-based stop
        stop_distance = atr * 1.5
        if recommendation == 'LONG':
            entry = last_close
            stop = entry - stop_distance
            target1 = entry + 2 * stop_distance
            direction = 'LONG'
        else:
            entry = last_close
            stop = entry + stop_distance
            target1 = entry - 2 * stop_distance
            direction = 'SHORT'
        risk_amount = capital * (risk_pct / 100.0)
        qty = int(max(1, (risk_amount / (abs(entry - stop) + 1e-9))))
        st.markdown("**Suggested Trade**")
        st.write(f"Direction: {direction}")
        st.write(f"Entry: {entry:.2f}")
        st.write(f"Stop: {stop:.2f} (distance {stop_distance:.2f})")
        st.write(f"Target: {target1:.2f}")
        st.write(f"Position size (approx): {qty} units (based on {risk_pct}% risk of capital={capital})")
    else:
        st.info("No clear trade recommendation from multi-timeframe analysis. Wait for confirmation or adjust timeframes.")

    st.success("Analysis complete.")

else:
    st.write("Enter inputs on the left and click **Analyze**. You may upload CSV or use yfinance to fetch data.")


# EOF
