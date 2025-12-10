from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


BASE_PROCESSED = Path("data/processed")
CV_DIR = BASE_PROCESSED / "cv"
BACKTESTS_DIR = BASE_PROCESSED / "backtests"
FI_DIR = BASE_PROCESSED / "feature_importance"
VISUALS_DIR = BASE_PROCESSED / "visuals"


# Pretty labels for plots (so slides don’t show raw code names)
MODEL_LABELS = {
    "ridge": "Ridge (α=1)",
    "tree_depth_3": "Tree (depth=3)",
    "tree_depth_5": "Tree (depth=5)",
    "rf_depth_4_n200": "Random forest (d=4, n=200)",
    "rf_depth_6_n400": "Random forest (d=6, n=400)",
    "gb_lr_0_05": "Gradient boosting (lr=0.05)",
    "rf_vol_scaled": "Random forest (vol-scaled)",
}

FEATURE_SET_LABELS = {
    "all": "All features",
    "core": "Core features",
    "core_plus_20": "Core + 20 extra",
}

FEATURE_NAME_LABELS = {
    "ret1": "1-day return",
    "ret5": "5-day return",
    "ret10": "10-day return",
    "rv10": "Realized vol (10-day)",
    "ret1_mean_20": "Mean ret1 (20-day)",
    "ret1_mean_60": "Mean ret1 (60-day)",
    "ret1_std_20": "Vol ret1 (20-day)",
    "ret1_std_60": "Vol ret1 (60-day)",
    "rsi5": "RSI (5-day)",
    # any others will just fall back to their raw name
}


def ensure_dirs() -> None:
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)


def plot_global_sharpe_ranking() -> None:
    """
    Scoreboard for all walk-forward models: Sharpe ratio on the
    out-of-sample test period.
    """
    path = BACKTESTS_DIR / "global_walkforward_comparison.csv"
    df = pd.read_csv(path)

    # Expect columns: model, sharpe, max_drawdown, turnover
    df["nice_model"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
    df_sorted = df.sort_values("sharpe", ascending=True)

    plt.figure(figsize=(7, 4))
    plt.barh(df_sorted["nice_model"], df_sorted["sharpe"])
    plt.xlabel("Sharpe ratio (higher is better)")
    plt.ylabel("Model")
    plt.title("Walk-forward performance: Sharpe by model")
    plt.tight_layout()

    out_path = VISUALS_DIR / "global_sharpe_ranking.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved -> {out_path}")


def plot_tree_overfitting_curve() -> None:
    """
    Train vs test RMSE as tree depth increases (bias–variance story).
    Uses tree_depth_cv.csv.
    """
    path = CV_DIR / "tree_depth_cv.csv"
    df = pd.read_csv(path)

    # Drop the "unlimited depth" row (max_depth = -1) for clarity in the plot
    df = df.copy()
    df = df[df["max_depth"] > 0]

    grouped = (
        df.groupby("max_depth", as_index=False)[["mean_train_rmse", "mean_test_rmse"]]
        .mean()
    )

    depths = grouped["max_depth"].to_numpy()
    train_rmse = grouped["mean_train_rmse"].to_numpy()
    test_rmse = grouped["mean_test_rmse"].to_numpy()

    plt.figure(figsize=(7, 4))
    plt.plot(depths, train_rmse, marker="o", label="Train RMSE")
    plt.plot(depths, test_rmse, marker="o", label="Test RMSE")
    plt.xlabel("Tree maximum depth (model complexity)")
    plt.ylabel("RMSE (lower is better)")
    plt.title("Decision tree complexity vs error (time-series CV)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = VISUALS_DIR / "tree_depth_overfitting.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved -> {out_path}")


def plot_feature_ablation_summary() -> None:
    """
    Bar chart: feature set vs Sharpe, grouped by model (ridge vs rf_vol_scaled).
    Uses feature_ablation_comparison.csv.
    """
    path = BACKTESTS_DIR / "feature_ablation_comparison.csv"
    df = pd.read_csv(path)

    # Be defensive about column names: allow either 'features' or 'feature_set'
    if "feature_set" in df.columns:
        feature_col = "feature_set"
    elif "features" in df.columns:
        feature_col = "features"
    else:
        raise ValueError(
            "Expected a 'feature_set' or 'features' column in "
            "feature_ablation_comparison.csv"
        )

    # Pivot to model columns, feature_set rows
    pivot = df.pivot_table(
        index=feature_col,
        columns="model",
        values="sharpe",
        aggfunc="mean",
    )

    feature_sets = pivot.index.to_list()
    models = pivot.columns.to_list()

    x = np.arange(len(feature_sets))
    width = 0.35 if len(models) == 2 else 0.8 / max(len(models), 1)

    plt.figure(figsize=(7, 4))
    for i, model in enumerate(models):
        offsets = x + (i - (len(models) - 1) / 2) * width
        label = MODEL_LABELS.get(model, model)
        plt.bar(offsets, pivot[model], width=width, label=label)

    pretty_feature_labels = [
        FEATURE_SET_LABELS.get(fs, fs) for fs in feature_sets
    ]

    plt.xticks(x, pretty_feature_labels)
    plt.ylabel("Sharpe ratio (walk-forward)")
    plt.xlabel("Feature set")
    plt.title("Feature ablation: Sharpe by feature set and model")
    plt.legend()
    plt.tight_layout()

    out_path = VISUALS_DIR / "feature_ablation_sharpe.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved -> {out_path}")


def plot_ridge_feature_importance(top_n: int = 10) -> None:
    """
    Horizontal bar chart of top-N features by mean importance for Ridge
    (time-series feature importance).
    Uses feature_importance/ridge_feature_importance.csv.
    """
    path = FI_DIR / "ridge_feature_importance.csv"
    df = pd.read_csv(path)

    # Expect columns: feature, mean_importance, std_importance
    df_sorted = df.sort_values("mean_importance", ascending=False).head(top_n)

    # Map feature codes to nicer labels for the plot
    nice_features = [
        FEATURE_NAME_LABELS.get(f, f) for f in df_sorted["feature"]
    ]

    plt.figure(figsize=(7, 4))
    plt.barh(nice_features, df_sorted["mean_importance"])
    plt.gca().invert_yaxis()  # most important at top
    plt.xlabel("Mean absolute coefficient (|β|)")
    plt.ylabel("Feature")
    plt.title(f"Ridge: top {top_n} predictive features")
    plt.tight_layout()

    out_path = VISUALS_DIR / "ridge_feature_importance_top10.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved -> {out_path}")

def plot_equity_overlay_intro() -> None:
    """
    Overlay walk-forward equity curves for several models.
    This is mainly for a visually interesting 'markets are noisy' intro slide.
    """
    # Pick a small set of representative models so the plot is busy but readable
    model_files = {
        "Ridge (α=1)": "ridge_walkforward_equity_curve.csv",
        "Tree (depth=3)": "tree_depth_3_walkforward_equity_curve.csv",
        "Tree (depth=5)": "tree_depth_5_walkforward_equity_curve.csv",
        "RF (depth=4, n=200)": "rf_depth_4_n200_walkforward_equity_curve.csv",
        "GB (lr=0.05)": "gb_lr_0_05_walkforward_equity_curve.csv",
        "RF (vol-scaled)": "rf_vol_scaled_walkforward_equity_curve.csv",
    }

    plt.figure(figsize=(7, 4))

    for label, filename in model_files.items():
        path = BACKTESTS_DIR / filename
        if not path.exists():
            # Skip silently if some variant wasn't generated
            continue

        df = pd.read_csv(path)
        # Expect columns: date, equity
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        plt.plot(
            df["date"],
            df["equity"],
            label=label,
            linewidth=1.6,
            alpha=0.9,
        )

    plt.xlabel("Date")
    plt.ylabel("Cumulative equity (normalized)")
    plt.title("Walk-forward equity curves for different models")
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()

    out_path = VISUALS_DIR / "equity_overlay_intro.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved -> {out_path}")



def main() -> None:
    ensure_dirs()
    plot_global_sharpe_ranking()
    plot_tree_overfitting_curve()
    plot_feature_ablation_summary()
    plot_ridge_feature_importance(top_n=10)
    plot_equity_overlay_intro()


if __name__ == "__main__":
    main()
