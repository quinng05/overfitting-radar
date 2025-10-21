import pandas as pd

def add_labels(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    out = df.copy()
    fwd = out.groupby("ticker")["adj_close"].shift(-horizon) / out["adj_close"] - 1.0
    out[f"y_ret_{horizon}"] = fwd
    out[f"y_up_{horizon}"]  = (out[f"y_ret_{horizon}"] > 0).astype(int)
    return out.dropna(subset=[f"y_ret_{horizon}"])