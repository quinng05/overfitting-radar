import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def plot_equity_curve(
    dates: pd.Series,
    equity: pd.Series,
    title: str,
    output_path: str,
) -> None:
    """
    make and save a simple equity curve plot
    """

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(dates, equity)
    ax.set_title(title)
    ax.set_xlabel("date")
    ax.set_ylabel("equity")
    ax.grid(alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_walkforward_sharpe(
    fold_metrics: pd.DataFrame,
    output_path: Path,
    model_name: str = "model",
):
    """
    Bar chart of Sharpe by fold for a walk-forward backtest.

    model_name controls the title, e.g. 'ridge', 'random forest', etc.
    """
    plt.figure(figsize=(6, 4))
    plt.bar(fold_metrics["fold"], fold_metrics["sharpe"])
    plt.xlabel("fold id")
    plt.ylabel("sharpe")
    plt.title(f"{model_name} walk-forward sharpe by fold")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
