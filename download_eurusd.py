"""Download daily historical data for EUR/USD (EURUSD=X) from Jan 1, 2000 to today."""

import datetime as dt

import yfinance as yf


def main() -> None:
    end_date = dt.date.today().strftime("%Y-%m-%d")

    df = yf.download(
        tickers="EURUSD=X",
        start="2000-01-01",
        end=end_date,
        interval="1d",
        progress=False,
    )

    print(f"First 5 rows of EUR/USD daily data ({df.shape[0]} rows total):")
    print(df.head())
    print()
    print("Dataframe shape:", df.shape)


if __name__ == "__main__":
    main()
