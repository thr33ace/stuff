"""Add technical-indicator columns (SMA 50/200, RSI, ATR) to a EUR/USD dataframe.

The `ta` library is used for the indicator calculations. The helper handles both
the multi-level column layout returned by ``yfinance.download`` (Close/Ticker/Date)
and plain single-level OHLC frames.
"""

from __future__ import annotations

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from ta.volatility import AverageTrueRange


def _get_price_columns(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (close, high, low) series from an OHLC dataframe.

    Works with both flat and yfinance-style MultiIndex columns.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten so we can look up "Close", "High", "Low" by level-0 name.
        flat = df.copy()
        flat.columns = [c[0] for c in flat.columns]
    else:
        flat = df

    missing = {"Close", "High", "Low"} - set(flat.columns)
    if missing:
        raise KeyError(f"DataFrame is missing required price columns: {sorted(missing)}")

    return flat["Close"], flat["High"], flat["Low"]


def add_indicators(
    df: pd.DataFrame,
    sma_short: int = 50,
    sma_long: int = 200,
    rsi_window: int = 14,
    atr_window: int = 14,
) -> pd.DataFrame:
    """Add SMA-50, SMA-200, RSI-14 and ATR-14 columns, then drop NaN rows.

    Parameters
    ----------
    df : pd.DataFrame
        Daily OHLC data for EUR/USD (as returned by ``yfinance.download``).
    sma_short / sma_long : int
        Windows for the simple moving averages (default 50 / 200 days).
    rsi_window / atr_window : int
        Lookback windows for RSI and ATR (default 14 days).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with four new indicator columns; rows containing NaN
        values (warm-up periods) are dropped.
    """
    out = df.copy()
    close, high, low = _get_price_columns(out)

    # --- Trend: Simple Moving Averages ------------------------------------
    sma_50 = SMAIndicator(close=close, window=sma_short).sma_indicator()
    sma_200 = SMAIndicator(close=close, window=sma_long).sma_indicator()

    # --- Momentum: Relative Strength Index --------------------------------
    rsi = RSIIndicator(close=close, window=rsi_window).rsi()

    # --- Volatility: Average True Range -----------------------------------
    atr = AverageTrueRange(high=high, low=low, close=close,
                           window=atr_window).average_true_range()

    out[f"SMA_{sma_short}"] = sma_50.values
    out[f"SMA_{sma_long}"] = sma_200.values
    out[f"RSI_{rsi_window}"] = rsi.values
    out[f"ATR_{atr_window}"] = atr.values

    # Drop warm-up rows that still contain NaN values.
    out = out.dropna()

    return out


if __name__ == "__main__":
    import datetime as dt

    import yfinance as yf

    raw = yf.download("EURUSD=X", start="2000-01-01",
                      end=dt.date.today().strftime("%Y-%m-%d"),
                      interval="1d", progress=False)

    enriched = add_indicators(raw)

    print("First 5 indicator-enriched rows:")
    print(enriched.head())
    print()
    print(f"Shape before dropping NaNs: {raw.shape}")
    print(f"Shape after  dropping NaNs: {enriched.shape}")
