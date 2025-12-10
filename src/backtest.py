from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    strat_returns: pd.Series
    sharpe: float
    max_drawdown: float
    turnover: float


def preds_to_signals(
    preds: pd.Series | np.ndarray,
    long_threshold: float = 0.0,
    short_threshold: Optional[float] = None,
) -> pd.Series:
    """
    Turn predictions into basic trading signals

    Long-only version: go long when the model looks positive, otherwise stay flat
    Long-short version: long on big positives, short on big negatives, chill in between
    """
    preds = pd.Series(preds)

    if short_threshold is None:
        # long if the model thinks returns are above some bar
        signals = (preds > long_threshold).astype(int)
    else:
        # long-short setup
        signals = pd.Series(0, index=preds.index, dtype=int)
        signals[preds > long_threshold] = 1
        signals[preds < short_threshold] = -1

    return signals


def compute_strategy_returns(
    realized_returns: pd.Series,
    signals: pd.Series,
    transaction_cost_bps: float = 0.0,
) -> pd.Series:
    """
    Apply positions to actual returns to see how the strategy would’ve done

    Use yesterday’s position to avoid cheating
    If trading costs are on, subtract a tiny penalty when we flip positions
    """
    realized_returns, signals = realized_returns.align(signals, join="inner")

    # don’t use future information — stick to lagged signals
    shifted = signals.shift(1).fillna(0)

    strat_ret = shifted * realized_returns

    if transaction_cost_bps > 0.0:
        # every time we switch direction, it costs a bit
        changes = shifted.diff().abs().fillna(0)
        cost_rate = transaction_cost_bps / 1e4
        strat_ret = strat_ret - changes * cost_rate

    return strat_ret


def compute_equity_curve(
    strat_returns: pd.Series,
    initial_capital: float = 1.0,
) -> pd.Series:
    """
    Roll returns forward into an account value
    """
    return (1 + strat_returns).cumprod() * initial_capital


def compute_sharpe(
    strat_returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    Quick Sharpe ratio estimate
    """
    mu = strat_returns.mean()
    sigma = strat_returns.std()

    if sigma == 0 or np.isnan(sigma):
        return np.nan

    return np.sqrt(periods_per_year) * mu / sigma


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Worst drop from a high point to a low point
    """
    peak = equity_curve.cummax()
    dd = equity_curve / peak - 1
    return dd.min()


def backtest_strategy(
    realized_returns: pd.Series,
    preds: pd.Series | np.ndarray,
    long_threshold: float = 0.0,
    short_threshold: Optional[float] = None,
    transaction_cost_bps: float = 0.0,
    periods_per_year: int = 252,
) -> BacktestResult:
    """
    Full pipeline: preds -> signals -> PnL -> equity -> stats
    """
    signals = preds_to_signals(preds, long_threshold, short_threshold)
    strat_returns = compute_strategy_returns(realized_returns, signals, transaction_cost_bps)
    equity_curve = compute_equity_curve(strat_returns)
    sharpe = compute_sharpe(strat_returns, periods_per_year)
    max_dd = compute_max_drawdown(equity_curve)

    # average size of position flips; basic turnover estimate
    turnover = signals.shift(1).fillna(0).diff().abs().mean()

    return BacktestResult(
        equity_curve=equity_curve,
        strat_returns=strat_returns,
        sharpe=sharpe,
        max_drawdown=max_dd,
        turnover=float(turnover),
    )

def backtest_with_signals(
    realized_returns: pd.Series,
    signals: pd.Series,
    transaction_cost_bps: float = 0.0,
    periods_per_year: int = 252,
) -> BacktestResult:
    """
    Variant of backtest_strategy that accepts *signals directly*.
    This lets us do position sizing like w_t = f(pred_t, vol_t).
    """
    strat_returns = compute_strategy_returns(
        realized_returns=realized_returns,
        signals=signals,
        transaction_cost_bps=transaction_cost_bps,
    )
    equity_curve = compute_equity_curve(strat_returns)
    sharpe = compute_sharpe(strat_returns, periods_per_year)
    max_dd = compute_max_drawdown(equity_curve)
    turnover = signals.shift(1).fillna(0).diff().abs().mean()

    return BacktestResult(
        equity_curve=equity_curve,
        strat_returns=strat_returns,
        sharpe=sharpe,
        max_drawdown=max_dd,
        turnover=float(turnover),
    )

