import pandas as pd

def _pick_price_series(df: pd.DataFrame) -> pd.Series:
    # Prefer adj_close; fallback to close; error if neither present.
    if "adj_close" in df.columns:
        s = df["adj_close"]
    elif "close" in df.columns:
        s = df["close"]
    else:
        raise KeyError(f"Expected 'adj_close' or 'close' in columns, got: {list(df.columns)}")

    # If duplicates somehow produced a DataFrame, take first column.
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s

def add_labels(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    # Sort and de-duplicate columns to avoid DataFrame math
    out = df.sort_values(["ticker", "date"]).reset_index(drop=True).copy()
    if not pd.Index(out.columns).is_unique:
        out = out.loc[:, ~out.columns.duplicated()]

    price = _pick_price_series(out)  # 1-D Series
    # Forward price per ticker
    fwd_price = price.groupby(out["ticker"]).shift(-horizon)

    # Forward return as 1-D Series, aligned to out.index
    fwd_ret = (fwd_price / price) - 1.0
    col = f"y_ret_{horizon}"

    # Assign positionally to avoid any index alignment surprises
    out[col] = fwd_ret.to_numpy()
    out[f"y_up_{horizon}"] = (out[col] > 0).astype("int8")

    # Keep rows where forward label is available (drop tail of each ticker)
    out = out[out[col].notna()].reset_index(drop=True)
    return out
