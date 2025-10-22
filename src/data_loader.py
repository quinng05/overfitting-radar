import pandas as pd
from typing import List, Optional
from pathlib import Path

BLOOMBERG_COL_MAP = {
    "Security": "ticker",
    "Date": "date",
    "PX_OPEN": "open",
    "PX_HIGH": "high",
    "PX_LOW": "low",
    "PX_LAST": "close",
    "PX_ADJ_CLOSE": "adj_close",
    "VOLUME": "volume",
}

def load_bloomberg_csv(path: str | Path) -> pd.DataFrame:
    """
    Load a Bloomberg Excel/CSV export (daily OHLCV) and normalize columns.
    """
    df = pd.read_csv(path)
    # rename headers
    cols = {c: BLOOMBERG_COL_MAP.get(c, c) for c in df.columns}
    df = df.rename(columns=cols)

    # basic checks
    required = {"ticker", "date", "open", "high", "low", "close", "adj_close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Check Bloomberg export fields.")

    # types and ordering
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df[["ticker","date","open","high","low","close","adj_close","volume"]]

def load_yfinance_prices(
    tickers: List[str],
    start: str,
    end: str,
    provider: str = "yfinance",
) -> pd.DataFrame:
    """
    Temporary fallback for when Bloomberg data unavailable (uses yfinance instead).
    """
    if provider != "yfinance":
        raise ValueError("Only yfinance fallback is implemented for now.")
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError("Install yfinance or use Bloomberg CSV.") from e

    frames = []
    for t in tickers:
        data = yf.download(t, start=start, end=end, progress=False, auto_adjust=False)
        if data.empty:
            continue
        # Flatten MultiIndex columns to simple names
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        data = data.rename(
            columns={
                "Open":"open","High":"high","Low":"low",
                "Close":"close","Adj Close":"adj_close","Volume":"volume"
            }
        )
        data["ticker"] = t
        data["date"] = data.index.tz_localize(None)
        frames.append(data.reset_index(drop=True)[
            ["ticker","date","open","high","low","close","adj_close","volume"]
        ])
    if not frames:
        raise ValueError("No data downloaded. Check tickers or dates.")
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["ticker","date"]).reset_index(drop=True)
    return df
