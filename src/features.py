import pandas as pd

def add_basic_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret1"]  = out.groupby("ticker")["adj_close"].pct_change(1)
    out["ret5"]  = out.groupby("ticker")["adj_close"].pct_change(5)
    out["ret10"] = out.groupby("ticker")["adj_close"].pct_change(10)
    return out

def add_realized_vtlty(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    out = df.copy()
    out["rv10"] = out.groupby("ticker")["ret1"].rolling(window).std().reset_index(level=0, drop=True)
    return out

def add_rsi(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    out = df.copy()
    delta = out.groupby("ticker")["adj_close"].diff()
    gain = delta.clip(lower=0).groupby(out["ticker"]).rolling(window).mean().reset_index(level=0, drop=True)
    loss = (-delta.clip(upper=0)).groupby(out["ticker"]).rolling(window).mean().reset_index(level=0, drop=True)
    rs = gain / (loss.replace(0, 1e-9))
    out[f"rsi{window}"] = 100 - (100 / (1 + rs))
    return out

def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    x = add_basic_returns(df)
    x = add_realized_vtlty(x, window=10)
    x = add_rsi(x, window=5)
    x = x.dropna().reset_index(drop=True)
    return x
