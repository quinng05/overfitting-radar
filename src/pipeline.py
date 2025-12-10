from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import pandas as pd

from src.config import (
    CLASS_COL,
    DATE_COL,
    DEFAULT_FEATURE_COLS,
    DEFAULT_START,
    DEFAULT_TICKERS,
    TARGET_COL,
)
from src.data_loader import load_yfinance_prices
from src.features import build_feature_table
from src.labels import add_labels


def ensure_nonempty(df: pd.DataFrame, stage: str) -> None:
    if df is None or df.empty:
        raise ValueError(f"{stage}: got empty DataFrame")


def ensure_columns(df: pd.DataFrame, required: Iterable[str], stage: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{stage}: missing columns {missing}")


def build_model_table(
    tickers: Sequence[str] | None = None, start: str | None = None
) -> Tuple[pd.DataFrame, list[str], list[str]]:
    tickers = list(tickers) if tickers is not None else DEFAULT_TICKERS
    start = start or DEFAULT_START

    df = load_yfinance_prices(tickers, start=start, end=None)
    ensure_nonempty(df, "after load_yfinance_prices")
    ensure_columns(df, ["ticker", DATE_COL, "close"], "after load_yfinance_prices")

    df = build_feature_table(df)
    ensure_nonempty(df, "after build_feature_table")
    ensure_columns(df, DEFAULT_FEATURE_COLS[:5], "after build_feature_table")

    df = add_labels(df, horizon=1)
    ensure_nonempty(df, "after add_labels")
    label_cols = [TARGET_COL, CLASS_COL]
    ensure_columns(df, label_cols, "after add_labels")

    df = df.dropna(subset=DEFAULT_FEATURE_COLS + label_cols)
    if len(df) < 200:
        raise ValueError(
            f"Not enough rows after cleaning ({len(df)}), need at least ~200 for CV and backtest"
        )

    return df, DEFAULT_FEATURE_COLS, label_cols
