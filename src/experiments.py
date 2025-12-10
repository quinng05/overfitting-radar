from __future__ import annotations
from src.config import DATE_COL, DEFAULT_N_SPLITS, DEFAULT_TC_BPS, TARGET_COL
from src.models import (
    ridge_walkforward_backtest,
    rf_vol_scaled_walkforward_backtest,
)

from typing import Any

import pandas as pd

from src.config import DATE_COL, DEFAULT_N_SPLITS, DEFAULT_TC_BPS, TARGET_COL
from src.models import (
    gb_walkforward_backtest,
    rf_vol_scaled_walkforward_backtest,
    rf_walkforward_backtest,
    ridge_walkforward_backtest,
    tree_walkforward_backtest,
)


def make_model_experiments(
    df: pd.DataFrame, feature_cols: list[str]
) -> list[dict[str, Any]]:
    """
    Build the standard set of walk-forward experiments used in notebooks/run.py.
    """
    base_features = list(feature_cols)
    regime_features = base_features
    if "regime_km" in df.columns and "regime_km" not in base_features:
        regime_features = base_features + ["regime_km"]

    common_kwargs = {
        "df": df,
        "feature_cols": base_features,
        "target_col": TARGET_COL,
        "date_col": DATE_COL,
        "n_splits": DEFAULT_N_SPLITS,
        "transaction_cost_bps": DEFAULT_TC_BPS,
    }

    experiments: list[dict[str, Any]] = []

    experiments.append({
        "name": "ridge",
        "backtest_fn": ridge_walkforward_backtest,
        "kwargs": {
            **common_kwargs,
            "alpha": 1.0,
            "long_threshold": 0.0,
            "short_threshold": None,
        },
    })

    for depth in (3, 5):
        experiments.append({
            "name": f"tree_depth_{depth}",
            "backtest_fn": tree_walkforward_backtest,
            "kwargs": {
                **common_kwargs,
                "max_depth": depth,
                "long_threshold": 0.0,
                "short_threshold": None,
            },
        })

    rf_settings = [
        ("rf_depth_4_n200", {"maxDepth": 4, "nEstimators": 200}),
        ("rf_depth_6_n400", {"maxDepth": 6, "nEstimators": 400}),
    ]
    for name, rf_params in rf_settings:
        experiments.append({
            "name": name,
            "backtest_fn": rf_walkforward_backtest,
            "kwargs": {
                **common_kwargs,
                "rf_params": rf_params,
                "long_threshold": 0.0,
                "short_threshold": None,
            },
        })

    experiments.append({
        "name": "gb_lr_0_05",
        "backtest_fn": gb_walkforward_backtest,
        "kwargs": {
            **common_kwargs,
            "gb_params": {
                "learningRate": 0.05,
                "nEstimators": 400,
                "maxDepth": 3,
            },
            "long_threshold": 0.0,
            "short_threshold": None,
        },
    })

    experiments.append({
        "name": "rf_vol_scaled",
        "backtest_fn": rf_vol_scaled_walkforward_backtest,
        "kwargs": {
            **common_kwargs,
            "feature_cols": regime_features,
            "vol_col": "ret1_std_20",
            "rf_params": {"maxDepth": 5, "nEstimators": 400},
            "max_leverage": 1.0,
        },
    })

    return experiments

def make_feature_ablation_experiments(
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """
    For each feature set, build walk-forward experiments
    for a couple of representative models (ridge + RF vol-scaled)
    """
    experiments: list[dict[str, Any]] = []

    for fsName, fsCols in feature_sets.items():
        commonKwargs = {
            "df": df,
            "feature_cols": fsCols,
            "target_col": TARGET_COL,
            "date_col": DATE_COL,
            "n_splits": DEFAULT_N_SPLITS,
            "transaction_cost_bps": DEFAULT_TC_BPS,
        }

        experiments.append({
            "name": f"ridge_fs_{fsName}",
            "feature_set": fsName,
            "model": "ridge",
            "backtest_fn": ridge_walkforward_backtest,
            "kwargs": {
                **commonKwargs,
                "alpha": 1.0,
                "long_threshold": 0.0,
                "short_threshold": None,
            },
        })

        experiments.append({
            "name": f"rf_vol_scaled_fs_{fsName}",
            "feature_set": fsName,
            "model": "rf_vol_scaled",
            "backtest_fn": rf_vol_scaled_walkforward_backtest,
            "kwargs": {
                **commonKwargs,
                "vol_col": "ret1_std_20",
                "rf_params": {"maxDepth": 5, "nEstimators": 400},
                "max_leverage": 1.0,
            },
        })

    return experiments