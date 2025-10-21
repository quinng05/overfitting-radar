import numpy as np
import pandas as pd
from typing import Iterator, Tuple

def expanding_time_splits(dates: pd.Series, n_splits: int = 5, min_train_days: int = 252) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    uniq = np.array(pd.Index(dates).sort_values().unique())
    if len(uniq) <= min_train_days + 1:
        yield np.where(dates <= uniq[min_train_days])[0], np.where(dates > uniq[min_train_days])[0]
        return
    cut_idx = np.linspace(min_train_days, len(uniq) - 2, n_splits, dtype=int)
    step = max(1, (len(uniq) - min_train_days)//n_splits)
    for c in cut_idx:
        train_end = uniq[c]
        test_end = uniq[min(c + step, len(uniq) - 1)]
        tr = np.where(dates <= train_end)[0]
        te = np.where((dates > train_end) & (dates <= test_end))[0]
        if len(te) == 0:
            te = np.where(dates > train_end)[0]
        yield tr, te
