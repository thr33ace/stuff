"""Streamlit app that displays the Forex predictor with BUY/SELL/HOLD signal."""

import datetime as dt

import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from ta.volatility import AverageTrueRange


# ----------------------------- Data & Indicators ----------------------------- #
@st.cache_data(ttl=600)
def download_data(ticker: str, start: str = "2000-01-01") -> pd.DataFrame:
    end_date = dt.date.today().strftime("%Y-%m-%d")
    df = yf.download(
        tickers=ticker,
        start=start,
        end=end_date,
        interval="1d",
        progress=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close, high, low = out["Close"], out["High"], out["Low"]
    out["SMA_50"] = SMAIndicator(close=close, window=50).sma_indicator()
    out["SMA_200"] = SMAIndicator(close=close, window=200).sma_indicator()
    out["RSI"] = RSIIndicator(close=close, window=14).rsi()
    out["ATR"] = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
    return out.dropna()


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Target"] = (out["Close"].shift(-1) > out["Close"]).astype(int)
    return out.iloc[:-1].dropna()


def train_model(df: pd.DataFrame):
    feature_cols = ["SMA_50", "SMA_200", "RSI", "ATR", "Close"]
    X = df[feature_cols]
    y = df["Target"]
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model, feature_cols


def add_tp_sl(df: pd.DataFrame, signal: str, sl_mult=1.5, tp_mult=3.0) -> pd.DataFrame:
    """Add Stop Loss / Take Profit columns based on ATR and a single signal."""
    out = df.copy()
    close = out["Close"]
    atr = out["ATR"]
    sl_dist = sl_mult * atr
    tp_dist = tp_mult * atr
    is_buy = signal == "BUY"
    out["Stop_Loss"] = (close - sl_dist) if is_buy else (close + sl_dist)
    out["Take_Profit"] = (close + tp_dist) if is_buy else (close - tp_dist)
    return out


# ----------------------------- Sidebar ----------------------------- #
st.set_page_config(page_title="Forex Predictor", page_icon="💱", layout="wide")

st.title("💱 Forex Predictor Dashboard")

currency_pairs = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "USDCAD=X",
}

selected_label = st.sidebar.selectbox("Select Currency Pair", list(currency_pairs.keys()))
ticker = currency_pairs[selected_label]

# ----------------------------- Main Content ----------------------------- #
with st.spinner(f"Loading data for {selected_label}..."):
    raw_df = download_data(ticker)
    if raw_df.empty:
        st.error("No data returned from Yahoo Finance.")
        st.stop()

    df = add_indicators(raw_df)
    df = create_target(df)

    model, feature_cols = train_model(df)

    # Latest row features
    latest = df.iloc[-1:]
    X_latest = latest[feature_cols]
    pred = int(model.predict(X_latest)[0])
    proba = float(model.predict_proba(X_latest)[0].max())

    signal = "BUY" if pred == 1 else "SELL"

    # Add TP/SL to the entire dataframe (latest row will be used for display)
    df_with_levels = add_tp_sl(latest, signal)
    latest_row = df_with_levels.iloc[-1]

# Line chart of closing prices
st.subheader(f"{selected_label} Closing Price")
st.line_chart(raw_df["Close"])

# Most recent data table
st.subheader("Most Recent Data")
st.table(latest_row.to_frame().T)

# Big bold signal
if signal == "BUY":
    st.markdown(
        f"<h1 style='text-align: center; color: green; font-weight: bold;'>📈 BUY ({proba:.0%} confidence)</h1>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"<h1 style='text-align: center; color: red; font-weight: bold;'>📉 SELL ({proba:.0%} confidence)</h1>",
        unsafe_allow_html=True,
    )

# TP/SL levels
col1, col2 = st.columns(2)
with col1:
    st.metric("Take Profit", f"{latest_row['Take_Profit']:.5f}")
with col2:
    st.metric("Stop Loss", f"{latest_row['Stop_Loss']:.5f}")

st.caption(f"Data source: Yahoo Finance • Model trained on {len(df)} rows • Signal based on latest bar")
