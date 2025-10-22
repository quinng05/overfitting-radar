# src/features.py
import pandas as pd

def _ensure_unique_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate-named columns (keep first)."""
    if not pd.Index(df.columns).is_unique:
        df = df.loc[:, ~df.columns.duplicated()]
    return df

def _series_col(df: pd.DataFrame, name: str) -> pd.Series:
    """Return a single Series for a column name even if duplicates exist."""
    obj = df[name]
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]  # take the first if duplicates slipped in
    return obj

def add_basic_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["ticker", "date"]).copy()
    out = _ensure_unique_cols(out)

    s = _series_col(out, "adj_close")
    g = s.groupby(out["ticker"])

    # pct_change on SeriesGroupBy returns a Series; specify fill_method=None to avoid deprecation
    out["ret1"]  = g.pct_change(1, fill_method=None)
    out["ret5"]  = g.pct_change(5, fill_method=None)
    out["ret10"] = g.pct_change(10, fill_method=None)
    return out

def add_realized_vtlty(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    out = _ensure_unique_cols(df.copy())
    r1 = _series_col(out, "ret1")
    out["rv10"] = (
        r1.groupby(out["ticker"])
          .rolling(window)
          .std()
          .reset_index(level=0, drop=True)
    )
    return out

def add_rsi(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    out = df.sort_values(["ticker", "date"]).copy()
    out = _ensure_unique_cols(out)

    s = _series_col(out, "adj_close")  # ensure Series
    delta = s.groupby(out["ticker"]).diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta.clip(upper=0))

    avg_gain = (
        gain.groupby(out["ticker"])
            .rolling(window)
            .mean()
            .reset_index(level=0, drop=True)
    )
    avg_loss = (
        loss.groupby(out["ticker"])
            .rolling(window)
            .mean()
            .reset_index(level=0, drop=True)
    )

    rs = avg_gain / (avg_loss.replace(0, 1e-9))
    out[f"rsi{window}"] = 100 - (100 / (1 + rs))
    return out

def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    x = _ensure_unique_cols(df)
    x = add_basic_returns(x)
    x = add_realized_vtlty(x, window=10)
    x = add_rsi(x, window=5)
    x = x.dropna().reset_index(drop=True)
    return x
