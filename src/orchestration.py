from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.plots import plot_equity_curve, plot_walkforward_sharpe


def run_walkforward_block(
    name: str,
    backtest_fn: Callable[..., Any],
    backtest_kwargs: dict,
    paths: dict,
):
    """
    Call a *_walkforward_backtest, print metrics, save per-fold metrics,
    save equity curve CSV, and save Sharpe/equity plots using the paths
    dict with keys 'backtests' and 'plots'.
    Return (fold_metrics, bt_global).
    """
    fold_metrics, bt_global, dates_wf, equity_wf = backtest_fn(**backtest_kwargs)

    backtests_dir = Path(paths["backtests"])
    plots_dir = Path(paths["plots"])
    backtests_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    fold_csv = backtests_dir / f"{name}_walkforward_metrics.csv"
    fold_metrics.to_csv(fold_csv, index=False)

    dates_series = pd.Series(pd.to_datetime(dates_wf)).reset_index(drop=True)
    equity_series = pd.Series(equity_wf).reset_index(drop=True)
    equity_df = pd.DataFrame({
        "date": dates_series,
        "equity": equity_series,
    })
    equity_csv = backtests_dir / f"{name}_walkforward_equity_curve.csv"
    equity_df.to_csv(equity_csv, index=False)

    plot_equity_curve(
        dates=equity_df["date"],
        equity=equity_df["equity"],
        title=f"{name} walk-forward equity curve",
        output_path=plots_dir / f"{name}_equity_walkforward.png",
    )

    plot_walkforward_sharpe(
        fold_metrics=fold_metrics,
        output_path=plots_dir / f"{name}_walkforward_sharpe.png",
        model_name=name,
    )

    print(f"{name}: Sharpe {bt_global.sharpe:.3f}, MaxDD {bt_global.max_drawdown:.3f}, Turnover {bt_global.turnover:.3f}")
    print(f"Saved metrics -> {fold_csv}")
    print(f"Saved equity curve -> {equity_csv}")

    return fold_metrics, bt_global
