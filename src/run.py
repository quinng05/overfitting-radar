import pandas as pd
from pathlib import Path
from src.backtest import backtest_strategy
from src.experiments import make_model_experiments, make_feature_ablation_experiments
from src.models import (run_all_and_save, 
                        ridge_alpha_sweep, 
                        ridge_backtest_split, 
                        tree_backtest_split, tree_depth_sweep, 
                        rf_depth_sweep, 
                        rf_trees_sweep, 
                        gb_learning_rate_sweep,
                        ridge_feature_importance_timeseries,
                        tree_feature_importance_timeseries,
                        rf_feature_importance_timeseries,
                        gb_feature_importance_timeseries)
from src.orchestration import run_walkforward_block
from src.pipeline import build_model_table
from src.plots import plot_equity_curve
from src.regimes import add_kmeans_regime
from src.config import (
    FEATURE_SETS,
    FEATURE_IMPORTANCE_DIR,
)
from sklearn.model_selection import TimeSeriesSplit


# light sanity checks so we fail fast if something is off
def ensure_nonempty(df: pd.DataFrame, stage: str) -> None:
    if df is None or df.empty:
        raise ValueError(f"{stage}: got empty DataFrame")

def ensure_columns(df: pd.DataFrame, required: list[str], stage: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{stage}: missing columns {missing}")

def main():
    # unified output folders
    PATH_PROCESSED = Path("data/processed")
    PATH_CV = PATH_PROCESSED / "cv"
    PATH_BACKTESTS = PATH_PROCESSED / "backtests"
    PATH_PLOTS = PATH_PROCESSED / "plots"
    PATH_FEATURE_IMPORTANCE = Path(FEATURE_IMPORTANCE_DIR)
    PATH_CV.mkdir(parents=True, exist_ok=True)
    PATH_BACKTESTS.mkdir(parents=True, exist_ok=True)
    PATH_PLOTS.mkdir(parents=True, exist_ok=True)

    tickers = ["NVDA", "AVGO", "QCOM", "AMD"]
    start = "2018-01-01"

    df, feature_cols, label_cols = build_model_table(tickers=tickers, start=start)
    
    regime_features_for_clustering = [
        "rv10", "rsi5", "ret1_mean_20", "ret1_std_20"
    ]

    df, kmeans_model, regime_scaler = add_kmeans_regime(
        df,
        feature_cols=regime_features_for_clustering,
        n_clusters=3,
        random_state=0,
    )

    print("Running baselines (Ridge & Logistic; expanding CV)")
    scores = run_all_and_save(df)
    print(scores.to_string(index=False))
    print("Saved -> data/processed/baseline_cv_scores.csv")

    print("\nRunning Ridge alpha sweep (time-series CV)")
    alpha_summary = ridge_alpha_sweep(
        df,
        feature_cols=feature_cols,
        target_col="y_ret_1",
        alphas=[0.01, 0.1, 1.0, 10.0, 100.0],
        n_splits=5,
        output_path=PATH_CV / "ridge_alpha_cv.csv",
    )
    print("Ridge alpha CV summary:")
    print(alpha_summary.to_string(index=False))
    print(f"Saved -> {PATH_CV / 'ridge_alpha_cv.csv'}")

    print("\nRunning simple ridge backtest")
    dates_test, realized_test, preds_test = ridge_backtest_split(
        df, alpha=1.0, train_frac=0.7
    )

    # small check that test slice is not empty and vectors align
    if len(realized_test) == 0 or len(preds_test) == 0:
        raise ValueError("Backtest split produced empty test data")
    if len(realized_test) != len(preds_test):
        raise ValueError(
            f"Backtest mismatch: realized_test len {len(realized_test)} vs preds_test len {len(preds_test)}"
        )

    bt = backtest_strategy(
        realized_returns=realized_test,
        preds=preds_test,
        long_threshold=0.0,
        short_threshold=None,
        transaction_cost_bps=5.0,
    )

    print("\nBacktest results:")
    print(f"Sharpe:       {bt.sharpe:.3f}")
    print(f"Max drawdown: {bt.max_drawdown:.3f}")
    print(f"Turnover:     {bt.turnover:.3f}")

    # store summary metrics for later analysis
    metrics = pd.DataFrame([{
        "model": "ridge",
        "alpha": 1.0,
        "tickers": ",".join(tickers),
        "train_frac": 0.7,
        "sharpe": bt.sharpe,
        "max_drawdown": bt.max_drawdown,
        "turnover": bt.turnover,
        "start_test": dates_test.min(),
        "end_test": dates_test.max(),
    }])
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    metrics.to_csv(PATH_BACKTESTS / "backtest_metrics_ridge.csv", index=False)
    print(f"Saved -> {PATH_BACKTESTS / 'backtest_metrics_ridge.csv'}")

    out = pd.DataFrame({
        "date": dates_test,
        "equity": bt.equity_curve.values
    })
    out.to_csv(PATH_BACKTESTS / "ridge_equity_curve.csv", index=False)
    print(f"Saved -> {PATH_BACKTESTS / 'ridge_equity_curve.csv'}")

    plot_equity_curve(
        dates=pd.to_datetime(out["date"]),
        equity=out["equity"],
        title="ridge simple holdout equity curve",
        output_path=PATH_PLOTS / "ridge_equity_simple.png",
    )
    print(f"Saved -> {PATH_PLOTS / 'ridge_equity_simple.png'}")

    print("\nRunning decision tree depth sweep (time-series CV)")

    tree_depths = [2, 3, 4, 6, 8, None]
    tree_cv = tree_depth_sweep(
        df,
        feature_cols=feature_cols,
        target_col="y_ret_1",
        date_col="date",
        depths=tree_depths,
        n_splits=5,
        output_path=PATH_CV / "tree_depth_cv.csv",
    )
    print("Tree depth CV summary:")
    print(tree_cv.to_string(index=False))
    print(f"Saved -> {PATH_CV / 'tree_depth_cv.csv'}")

    print("\nRunning simple decision tree backtest (max_depth=4)")
    dates_test_tree, realized_test_tree, preds_test_tree = tree_backtest_split(
        df,
        feature_cols=feature_cols,
        target_col="y_ret_1",
        date_col="date",
        max_depth=4,
        train_frac=0.7,
    )

    bt_tree = backtest_strategy(
        realized_returns=realized_test_tree,
        preds=preds_test_tree,
        long_threshold=0.0,
        short_threshold=None,
        transaction_cost_bps=5.0,
    )

    print("\nDecision tree backtest results (simple split):")
    print(f"Sharpe:       {bt_tree.sharpe:.3f}")
    print(f"Max drawdown: {bt_tree.max_drawdown:.3f}")
    print(f"Turnover:     {bt_tree.turnover:.3f}")

    print("\nRunning Random Forest depth and tree-count sweeps (time-series CV)")

    X = df[feature_cols].values
    y = df["y_ret_1"].values

    # Use the same TimeSeriesSplit for all RF/GB experiments
    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X))

    rf_depths = [2, 3, 4, 6, 8, None]
    rf_depth_cv = rf_depth_sweep(
        X=X,
        y=y,
        splits=splits,
        depths=rf_depths,
    )
    rf_depth_cv.to_csv(PATH_CV / "rf_depth_cv.csv", index=False)
    print("RF depth CV summary:")
    print(
        rf_depth_cv.groupby("max_depth")[["train_rmse", "test_rmse"]]
        .mean()
        .reset_index()
        .to_string(index=False)
    )
    print(f"Saved -> {PATH_CV / 'rf_depth_cv.csv'}")

    rf_n_estimators = [50, 100, 200, 400]
    rf_trees_cv = rf_trees_sweep(
        X=X,
        y=y,
        splits=splits,
        nEstimatorsList=rf_n_estimators,
        maxDepth=4,  # pick a reasonable depth from rf_depth_cv
    )
    rf_trees_cv.to_csv(PATH_CV / "rf_trees_cv.csv", index=False)
    print("RF n_estimators CV summary:")
    print(
        rf_trees_cv.groupby("n_estimators")[["train_rmse", "test_rmse"]]
        .mean()
        .reset_index()
        .to_string(index=False)
    )
    print(f"Saved -> {PATH_CV / 'rf_trees_cv.csv'}")

    print("\nRunning Gradient Boosting learning-rate sweep (time-series CV)")

    gb_lrs = [0.01, 0.05, 0.1, 0.2, 0.3]
    gb_cv = gb_learning_rate_sweep(
        X=X,
        y=y,
        splits=splits,
        learningRates=gb_lrs,
        nEstimators=300,
        maxDepth=3,
    )
    gb_cv.to_csv(PATH_CV / "gb_learning_rate_cv.csv", index=False)
    print("GB learning-rate CV summary:")
    print(
        gb_cv.groupby("learning_rate")[["train_rmse", "test_rmse"]]
        .mean()
        .reset_index()
        .to_string(index=False)
    )
    print(f"Saved -> {PATH_CV / 'gb_learning_rate_cv.csv'}")

    paths = {
        "cv": PATH_CV,
        "backtests": PATH_BACKTESTS,
        "plots": PATH_PLOTS,
    }

    experiments = make_model_experiments(df, feature_cols)
    bt_globals = {}
    for exp in experiments:
        name = exp["name"]
        print(f"\nRunning {name} walk-forward backtest")
        fold_metrics, bt_global = run_walkforward_block(
            name=name,
            backtest_fn=exp["backtest_fn"],
            backtest_kwargs=exp["kwargs"],
            paths=paths,
        )
        bt_globals[name] = bt_global

    comparison_rows = [
        {
            "model": name,
            "sharpe": bt.sharpe,
            "max_drawdown": bt.max_drawdown,
            "turnover": bt.turnover,
        }
        for name, bt in bt_globals.items()
    ]
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(PATH_BACKTESTS / "global_walkforward_comparison.csv", index=False)
    print(f"Saved -> {PATH_BACKTESTS / 'global_walkforward_comparison.csv'}")

    print("\n=== Feature ablation experiments ===")

    ablationExps = make_feature_ablation_experiments(df, FEATURE_SETS)

    ablationRows = []

    for exp in ablationExps:
        name = exp["name"]
        modelName = exp["model"]
        featureSet = exp["feature_set"]

        print(f"\nRunning {name} (model={modelName}, features={featureSet})")

        foldMetrics, btGlobal = run_walkforward_block(
            name=name,
            backtest_fn=exp["backtest_fn"],
            backtest_kwargs=exp["kwargs"],
            paths=paths,   # same dict you already use: {"backtests": ..., "plots": ...}
        )

        ablationRows.append({
            "model": modelName,
            "feature_set": featureSet,
            "sharpe": btGlobal.sharpe,
            "max_drawdown": btGlobal.max_drawdown,
            "turnover": btGlobal.turnover,
        })

    # compact comparison table for ablation
    if ablationRows:
        ablationDf = pd.DataFrame(ablationRows)
        outPath = PATH_BACKTESTS / "feature_ablation_comparison.csv"
        ablationDf.to_csv(outPath, index=False)
        print(f"Saved -> {outPath}")

    print("\n=== Feature importance experiments ===")

    PATH_FEATURE_IMPORTANCE.mkdir(parents=True, exist_ok=True)

    fi_ridge = ridge_feature_importance_timeseries(
        df=df,
        feature_cols=feature_cols,
        target_col="y_ret_1",
        date_col="date",
        alpha=1.0,
        n_splits=5,
    )
    ridge_fi_path = PATH_FEATURE_IMPORTANCE / "ridge_feature_importance.csv"
    fi_ridge.to_csv(ridge_fi_path, index=False)
    print(f"Saved -> {ridge_fi_path}")

    fi_tree = tree_feature_importance_timeseries(
        df=df,
        feature_cols=feature_cols,
        target_col="y_ret_1",
        date_col="date",
        max_depth=4,
        n_splits=5,
    )
    tree_fi_path = PATH_FEATURE_IMPORTANCE / "tree_feature_importance.csv"
    fi_tree.to_csv(tree_fi_path, index=False)
    print(f"Saved -> {tree_fi_path}")

    fi_rf = rf_feature_importance_timeseries(
        df=df,
        feature_cols=feature_cols,
        target_col="y_ret_1",
        date_col="date",
        max_depth=4,
        n_estimators=200,
        n_splits=5,
    )
    rf_fi_path = PATH_FEATURE_IMPORTANCE / "rf_feature_importance.csv"
    fi_rf.to_csv(rf_fi_path, index=False)
    print(f"Saved -> {rf_fi_path}")

    fi_gb = gb_feature_importance_timeseries(
        df=df,
        feature_cols=feature_cols,
        target_col="y_ret_1",
        date_col="date",
        learning_rate=0.05,
        max_depth=3,
        n_estimators=400,
        n_splits=5,
    )
    gb_fi_path = PATH_FEATURE_IMPORTANCE / "gb_feature_importance.csv"
    fi_gb.to_csv(gb_fi_path, index=False)
    print(f"Saved -> {gb_fi_path}")

if __name__ == "__main__":
    main()
