import numpy as np
import pandas as pd
from pathlib import Path
from typing import Iterable
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from src.splits import expanding_time_splits
from src.backtest import backtest_strategy, backtest_with_signals

FEATURES = ["ret1","ret5","ret10","rv10","rsi5"]
REG_LABEL = "y_ret_1"
CLF_LABEL = "y_up_1"
DATE_COL  = "date"


def ridge_baseline(df: pd.DataFrame, alphas: Iterable[float] = (0.1, 1.0, 10.0)) -> pd.DataFrame:
    X = df[FEATURES].values
    y = df[REG_LABEL].values
    dates = df[DATE_COL]
    rows = []
    for alpha in alphas:   # Ridge α == λ (Lecture 7)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge",  Ridge(alpha=alpha))
        ])
        fold_rmse = []
        for tr, te in expanding_time_splits(dates, n_splits=5, min_train_days=60):
            pipe.fit(X[tr], y[tr])
            pred = pipe.predict(X[te])
            fold_rmse.append(mean_squared_error(y[te], pred) ** 0.5)
        rows.append({"model":"Ridge","alpha":alpha,
                     "rmse_mean":float(np.mean(fold_rmse)),"rmse_std":float(np.std(fold_rmse))})
    return pd.DataFrame(rows)

def logreg_baseline(df: pd.DataFrame, Cs: Iterable[float] = (1.0, 0.5, 0.1), calibrate: bool = True) -> pd.DataFrame:
    X = df[FEATURES].values
    y = df[CLF_LABEL].values
    dates = df[DATE_COL]
    rows = []
    for C in Cs:  # Logistic C = 1/λ (Lecture 7)
        base = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=C, solver="lbfgs"))
        ])
        def fit_model(Xtr, ytr):
            if not calibrate:
                return base.fit(Xtr, ytr)
            return CalibratedClassifierCV(base, method="sigmoid", cv=3).fit(Xtr, ytr)  # calibrate on train only
        accs, aucs, losses = [], [], []
        for tr, te in expanding_time_splits(dates, n_splits=5, min_train_days=60):
            m = fit_model(X[tr], y[tr])
            prob = m.predict_proba(X[te])[:,1]
            pred = (prob >= 0.5).astype(int)  # threshold tuning optional (on train/val)
            accs.append(accuracy_score(y[te], pred))
            try:
                aucs.append(roc_auc_score(y[te], prob))
            except ValueError:
                pass
            losses.append(log_loss(y[te], prob, labels=[0,1]))
        rows.append({"model":"LogReg","C":C,"calibrated":calibrate,
                     "acc_mean":float(np.mean(accs)),
                     "auc_mean":float(np.mean(aucs)) if aucs else np.nan,
                     "logloss_mean":float(np.mean(losses))})
    return pd.DataFrame(rows)

def run_all_and_save(df: pd.DataFrame, outdir: str = "data/processed/cv") -> pd.DataFrame:
    Path(outdir).mkdir(parents=True, exist_ok=True)
    ridge_df = ridge_baseline(df, alphas=(0.1,1.0,10.0))
    log_df   = logreg_baseline(df, Cs=(1.0,0.5,0.1), calibrate=True)
    out = pd.concat([ridge_df, log_df], ignore_index=True)
    out.to_csv(f"{outdir}/baseline_cv_scores.csv", index=False)
    return out

def ridge_alpha_sweep(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "y_ret_1",
    alphas: list[float] | None = None,
    n_splits: int = 5,
    output_path: str = "data/processed/ridge_alpha_cv.csv",
) -> pd.DataFrame:
    """
    Time-series cross-validation over a grid of Ridge alphas.

    For each alpha and each time-series split, fit on past data and
    evaluate on the next chunk. Returns a tidy DataFrame with
    per-fold and aggregated metrics, and writes to CSV.

    This is directly analogous to the cross-validation diagnostics
    in the Overfitting & Regularization notebook, just adapted to
    time-ordered financial data.
    """

    if alphas is None:
        # reasonable starting grid; you can tweak later
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0]

    # basic guards
    if df.empty:
        raise ValueError("ridge_alpha_sweep: got empty DataFrame")

    missing = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"ridge_alpha_sweep: missing columns {missing}")

    X = df[feature_cols].values
    y = df[target_col].values

    tscv = TimeSeriesSplit(n_splits=n_splits)

    records: list[dict] = []

    for alpha in alphas:
        fold_idx = 0
        for train_idx, test_idx in tscv.split(X):
            fold_idx += 1
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if len(np.unique(y_train)) == 1:
                # if target is constant in this train slice, model is meaningless here
                continue

            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_rmse = _rmse(y_train, y_train_pred)
            test_rmse  = _rmse(y_test,  y_test_pred)

            records.append({
                "alpha": alpha,
                "fold": fold_idx,
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            })

    if not records:
        raise ValueError("ridge_alpha_sweep: no valid folds produced any records")

    df_cv = pd.DataFrame.from_records(records)

    # aggregate across folds so you can quickly see the sweet spot
    agg = (
        df_cv
        .groupby("alpha")
        .agg(
            mean_train_rmse=("train_rmse", "mean"),
            std_train_rmse=("train_rmse", "std"),
            mean_test_rmse=("test_rmse", "mean"),
            std_test_rmse=("test_rmse", "std"),
            n_folds=("fold", "nunique"),
        )
        .reset_index()
        .sort_values("alpha")
    )

    agg["rmse_gap"] = agg["mean_test_rmse"] - agg["mean_train_rmse"]
    agg["overfit_ratio"] = agg["mean_test_rmse"] / agg["mean_train_rmse"]

    # combine detailed + summary into a single CSV
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# per-fold results\n")
        df_cv.to_csv(f, index=False)
        f.write("\n# aggregated over folds\n")
        agg.to_csv(f, index=False)

    return agg

def ridge_backtest_split(
    df: pd.DataFrame,
    alpha: float = 1.0,
    train_frac: float = 0.7,
) -> tuple[pd.Series, pd.Series, np.ndarray]:
    """
    Train a single ridge model on the first chunk of time
    Return dates, realized returns, and predictions for the unseen chunk
    """
    # sort by date so the split is chronological
    df_sorted = df.sort_values(DATE_COL).reset_index(drop=True)

    X = df_sorted[FEATURES].values
    y = df_sorted[REG_LABEL].values
    dates = df_sorted[DATE_COL]

    n = len(df_sorted)
    split_idx = int(n * train_frac)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_test = dates.iloc[split_idx:]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=alpha)),
    ])

    pipe.fit(X_train, y_train)
    preds_test = pipe.predict(X_test)

    # for the backtest we treat y_test as the realized 1-day-ahead returns
    realized_test = pd.Series(y_test, index=dates_test, name=REG_LABEL)
    preds_test_series = pd.Series(preds_test, index=dates_test, name="y_pred")

    return dates_test, realized_test, preds_test_series

def ridge_walkforward_backtest(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "y_ret_1",
    date_col: str = "date",
    alpha: float = 1.0,
    n_splits: int = 5,
    long_threshold: float = 0.0,
    short_threshold: float | None = None,
    transaction_cost_bps: float = 5.0,
):
    """
    walk-forward backtest for Ridge regression using expanding
    time-series splits.
    """

    if df.empty:
        raise ValueError("ridge_walkforward_backtest: got empty DataFrame")

    missing = [c for c in feature_cols + [target_col, date_col] if c not in df.columns]
    if missing:
        raise ValueError(f"ridge_walkforward_backtest: missing columns {missing}")

    # sort by date so time ordering is clean
    df_sorted = df.sort_values(by=date_col).reset_index(drop=True)

    # sklearn wants numpy for fitting
    X = df_sorted[feature_cols].values

    # backtest_strategy wants pandas series
    y = df_sorted[target_col]
    dates = pd.to_datetime(df_sorted[date_col])

    n = len(df_sorted)
    if n < 300:
        raise ValueError(f"ridge_walkforward_backtest: need at least ~300 rows, got {n}")

    tscv = TimeSeriesSplit(n_splits=n_splits)

    # storage for global out-of-sample predictions
    all_preds = pd.Series(index=df_sorted.index, dtype=float)
    all_is_test = np.zeros(n, dtype=bool)

    fold_records: list[dict] = []
    fold_id = 0

    for train_idx, test_idx in tscv.split(X):
        fold_id += 1

        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        dates_test = dates.iloc[test_idx]

        # avoid degenerate folds where target is constant
        if len(np.unique(y_train.to_numpy())) <= 1:
            continue

        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train.to_numpy())
        preds_test = model.predict(X_test)

        # store into the global prediction containers
        all_preds.iloc[test_idx] = preds_test
        all_is_test[test_idx] = True

        # make preds a pandas series aligned with y_test
        preds_test_series = pd.Series(preds_test, index=y_test.index)

        # per-fold backtest
        bt_fold = backtest_strategy(
            realized_returns=y_test,
            preds=preds_test_series,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            transaction_cost_bps=transaction_cost_bps,
        )

        fold_records.append({
            "fold": fold_id,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "start_test": dates_test.min(),
            "end_test": dates_test.max(),
            "sharpe": bt_fold.sharpe,
            "max_drawdown": bt_fold.max_drawdown,
            "turnover": bt_fold.turnover,
        })

    if not fold_records:
        raise ValueError("ridge_walkforward_backtest: no valid folds produced metrics")

    fold_metrics = pd.DataFrame(fold_records)

    # select only rows that were test rows in at least one fold
    test_mask = all_is_test
    realized_all = y[test_mask]
    preds_all = all_preds[test_mask]
    dates_all_test = dates[test_mask]

    if len(realized_all) == 0:
        raise ValueError("ridge_walkforward_backtest: global test set is empty")

    # global walk-forward backtest
    bt_global = backtest_strategy(
        realized_returns=realized_all,
        preds=preds_all,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        transaction_cost_bps=transaction_cost_bps,
    )

    equity_curve_global = bt_global.equity_curve

    return fold_metrics, bt_global, dates_all_test, equity_curve_global


def _rmse(y_true, y_pred) -> float:
    # RMSE helper
    mse = mean_squared_error(y_true, y_pred)
    return float(np.sqrt(mse))

def _compute_feature_importance_timeseries(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    date_col: str,
    model_factory,
    n_splits: int = 5,
    importance_attr: str | None = "feature_importances_",
    use_abs_coef: bool = False,
) -> pd.DataFrame:
    """
    Time-series CV feature importance:
    - Sorts by date_col
    - Uses TimeSeriesSplit with expanding windows
    - Fits model_factory() on each fold
    - Extracts feature importance per fold
    - Returns per-feature mean/std importance across folds
    """
    df_sorted = df.sort_values(by=date_col).reset_index(drop=True)
    X = df_sorted[feature_cols].to_numpy()
    y = df_sorted[target_col].to_numpy()
    dates = df_sorted[date_col]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, y_train = X[train_idx], y[train_idx]
        model = model_factory()
        model.fit(X_train, y_train)

        if use_abs_coef:
            if not hasattr(model, "coef_"):
                raise ValueError("use_abs_coef=True but model has no coef_ attribute")
            importances = np.abs(model.coef_).ravel()
        else:
            if importance_attr is None or not hasattr(model, importance_attr):
                raise ValueError(f"Model has no attribute '{importance_attr}' for feature importances")
            importances = getattr(model, importance_attr)

        if len(importances) != len(feature_cols):
            raise ValueError("Length of feature importances does not match feature_cols")

        for feat, imp in zip(feature_cols, importances):
            rows.append(
                {
                    "fold": fold_idx,
                    "feature": feat,
                    "importance": float(imp),
                }
            )

    imp_df = pd.DataFrame(rows)
    agg = (
        imp_df
        .groupby("feature", as_index=False)["importance"]
        .agg(mean_importance="mean", std_importance="std")
    )
    agg = agg.sort_values("mean_importance", ascending=False).reset_index(drop=True)
    return agg

def ridge_feature_importance_timeseries(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    date_col: str,
    alpha: float = 1.0,
    n_splits: int = 5,
) -> pd.DataFrame:
    def model_factory():
        return Ridge(alpha=alpha)

    return _compute_feature_importance_timeseries(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        date_col=date_col,
        model_factory=model_factory,
        n_splits=n_splits,
        importance_attr=None,
        use_abs_coef=True,
    )

def tree_feature_importance_timeseries(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    date_col: str,
    max_depth: int | None = 4,
    n_splits: int = 5,
) -> pd.DataFrame:
    def model_factory():
        return DecisionTreeRegressor(max_depth=max_depth, random_state=0)

    return _compute_feature_importance_timeseries(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        date_col=date_col,
        model_factory=model_factory,
        n_splits=n_splits,
        importance_attr="feature_importances_",
        use_abs_coef=False,
    )

def rf_feature_importance_timeseries(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    date_col: str,
    max_depth: int | None = 4,
    n_estimators: int = 200,
    n_splits: int = 5,
) -> pd.DataFrame:
    def model_factory():
        return RandomForestRegressor(
            max_depth=max_depth,
            n_estimators=n_estimators,
            random_state=0,
            n_jobs=-1,
        )

    return _compute_feature_importance_timeseries(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        date_col=date_col,
        model_factory=model_factory,
        n_splits=n_splits,
        importance_attr="feature_importances_",
        use_abs_coef=False,
    )

def gb_feature_importance_timeseries(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    date_col: str,
    learning_rate: float = 0.05,
    max_depth: int = 3,
    n_estimators: int = 400,
    n_splits: int = 5,
) -> pd.DataFrame:
    def model_factory():
        return GradientBoostingRegressor(
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            random_state=0,
        )

    return _compute_feature_importance_timeseries(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        date_col=date_col,
        model_factory=model_factory,
        n_splits=n_splits,
        importance_attr="feature_importances_",
        use_abs_coef=False,
    )


def tree_depth_sweep(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "y_ret_1",
    date_col: str = "date",
    depths: list[int | None] = (2, 3, 4, 6, 8, None),
    n_splits: int = 5,
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    time-series cv for decision tree regression with different max_depth values.
    this mirrors ridge_alpha_sweep but uses tree depth as the capacity knob.
    """

    if df.empty:
        raise ValueError("tree_depth_sweep: got empty dataframe")

    missing = [c for c in feature_cols + [target_col, date_col] if c not in df.columns]
    if missing:
        raise ValueError(f"tree_depth_sweep: missing columns {missing}")

    df_sorted = df.sort_values(by=date_col).reset_index(drop=True)

    X = df_sorted[feature_cols].values
    y = df_sorted[target_col].values

    n = len(df_sorted)
    if n < 300:
        raise ValueError(f"tree_depth_sweep: need at least ~300 rows, got {n}")

    tscv = TimeSeriesSplit(n_splits=n_splits)

    records: list[dict] = []

    for depth in depths:
        train_rmses = []
        test_rmses = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = DecisionTreeRegressor(
                max_depth=depth,
                random_state=0,
            )
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_rmses.append(_rmse(y_train, y_train_pred))
            test_rmses.append(_rmse(y_test, y_test_pred))

        records.append({
            "max_depth": depth if depth is not None else -1,
            "mean_train_rmse": float(np.mean(train_rmses)),
            "std_train_rmse": float(np.std(train_rmses)),
            "mean_test_rmse": float(np.mean(test_rmses)),
            "std_test_rmse": float(np.std(test_rmses)),
            "n_folds": n_splits,
        })

    out = pd.DataFrame(records)

    metric_cols = ("mean_test_rmse", "mean_train_rmse")
    if not set(metric_cols).issubset(out.columns):
        metric_cols = ("test_rmse", "train_rmse")
    test_col, train_col = metric_cols
    out["rmse_gap"] = out[test_col] - out[train_col]
    out["overfit_ratio"] = out[test_col] / out[train_col]

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False)

    return out

def tree_backtest_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "y_ret_1",
    date_col: str = "date",
    max_depth: int | None = 3,
    train_frac: float = 0.7,
):
    """
    single train/test split backtest for a decision tree regressor.
    returns (dates_test, realized_test, preds_test).
    """

    if df.empty:
        raise ValueError("tree_backtest_split: got empty dataframe")

    missing = [c for c in feature_cols + [target_col, date_col] if c not in df.columns]
    if missing:
        raise ValueError(f"tree_backtest_split: missing columns {missing}")

    df_sorted = df.sort_values(by=date_col).reset_index(drop=True)

    n = len(df_sorted)
    n_train = int(n * train_frac)
    if n_train <= 0 or n_train >= n:
        raise ValueError(f"tree_backtest_split: bad train_frac {train_frac} for n={n}")

    train = df_sorted.iloc[:n_train]
    test = df_sorted.iloc[n_train:]

    X_train = train[feature_cols].values
    y_train = train[target_col].values
    X_test = test[feature_cols].values
    y_test = test[target_col].values

    model = DecisionTreeRegressor(
        max_depth=max_depth,
        random_state=0,
    )
    model.fit(X_train, y_train)
    preds_test = model.predict(X_test)

    dates_test = pd.to_datetime(test[date_col])
    realized_test = pd.Series(y_test, index=test.index)
    preds_test_series = pd.Series(preds_test, index=test.index)

    return dates_test, realized_test, preds_test_series

def tree_walkforward_backtest(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "y_ret_1",
    date_col: str = "date",
    max_depth: int | None = 3,
    n_splits: int = 5,
    long_threshold: float = 0.0,
    short_threshold: float | None = None,
    transaction_cost_bps: float = 5.0,
):
    """
    walk-forward backtest for a decision tree regressor using expanding
    time-series splits. mirrors ridge_walkforward_backtest but uses max_depth
    as the capacity knob.
    """

    if df.empty:
        raise ValueError("tree_walkforward_backtest: got empty dataframe")

    missing = [c for c in feature_cols + [target_col, date_col] if c not in df.columns]
    if missing:
        raise ValueError(f"tree_walkforward_backtest: missing columns {missing}")

    df_sorted = df.sort_values(by=date_col).reset_index(drop=True)

    X = df_sorted[feature_cols].values
    y = df_sorted[target_col]
    dates = pd.to_datetime(df_sorted[date_col])

    n = len(df_sorted)
    if n < 300:
        raise ValueError(f"tree_walkforward_backtest: need at least ~300 rows, got {n}")

    tscv = TimeSeriesSplit(n_splits=n_splits)

    all_preds = pd.Series(index=df_sorted.index, dtype=float)
    all_is_test = np.zeros(n, dtype=bool)

    fold_records: list[dict] = []
    fold_id = 0

    for train_idx, test_idx in tscv.split(X):
        fold_id += 1

        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        dates_test = dates.iloc[test_idx]

        if len(np.unique(y_train.to_numpy())) <= 1:
            continue

        model = DecisionTreeRegressor(
            max_depth=max_depth,
            random_state=0,
        )
        model.fit(X_train, y_train.to_numpy())
        preds_test = model.predict(X_test)

        all_preds.iloc[test_idx] = preds_test
        all_is_test[test_idx] = True

        preds_test_series = pd.Series(preds_test, index=y_test.index)

        bt_fold = backtest_strategy(
            realized_returns=y_test,
            preds=preds_test_series,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            transaction_cost_bps=transaction_cost_bps,
        )

        fold_records.append({
            "fold": fold_id,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "start_test": dates_test.min(),
            "end_test": dates_test.max(),
            "sharpe": bt_fold.sharpe,
            "max_drawdown": bt_fold.max_drawdown,
            "turnover": bt_fold.turnover,
        })

    if not fold_records:
        raise ValueError("tree_walkforward_backtest: no valid folds produced metrics")

    fold_metrics = pd.DataFrame(fold_records)

    test_mask = all_is_test
    realized_all = y[test_mask]
    preds_all = all_preds[test_mask]
    dates_all_test = dates[test_mask]

    if len(realized_all) == 0:
        raise ValueError("tree_walkforward_backtest: global test set is empty")

    bt_global = backtest_strategy(
        realized_returns=realized_all,
        preds=preds_all,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        transaction_cost_bps=transaction_cost_bps,
    )

    equity_curve_global = bt_global.equity_curve

    return fold_metrics, bt_global, dates_all_test, equity_curve_global

def evaluate_model_over_splits(model_factory, X, y, splits) -> pd.DataFrame:
    """
    Given a model factory, features X, targets y, and an iterable of
    (train_idx, test_idx) splits, fit a fresh model on each split and
    record train/test RMSE.

    Returns a tidy DataFrame with one row per split.
    """
    records: list[dict] = []

    for split_id, (train_idx, test_idx) in enumerate(splits, start=1):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])

        y_train_pred = model.predict(X[train_idx])
        y_test_pred  = model.predict(X[test_idx])

        records.append({
            "fold": split_id,
            "train_rmse": _rmse(y[train_idx], y_train_pred),
            "test_rmse":  _rmse(y[test_idx],  y_test_pred),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        })

    return pd.DataFrame.from_records(records)


def make_random_forest(maxDepth=None, nEstimators=200, randomState=0):
    return RandomForestRegressor(
        n_estimators=nEstimators,
        max_depth=maxDepth,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=randomState,
    )

def rf_depth_sweep(X, y, splits, depths):
    """
    For each depth, run time-series CV and record train/test RMSE.
    X, y: numpy arrays aligned with your df index.
    splits: iterable of (train_idx, test_idx).
    depths: list of max_depth values to try.
    """
    rows = []
    for maxDepth in depths:
        def factory():
            return make_random_forest(maxDepth=maxDepth)

        dfMetrics = evaluate_model_over_splits(factory, X, y, splits)
        dfMetrics["max_depth"] = maxDepth
        rows.append(dfMetrics)

    results = pd.concat(rows, ignore_index=True)
    metric_cols = ("mean_test_rmse", "mean_train_rmse")
    if not set(metric_cols).issubset(results.columns):
        metric_cols = ("test_rmse", "train_rmse")
    test_col, train_col = metric_cols
    results["rmse_gap"] = results[test_col] - results[train_col]
    results["overfit_ratio"] = results[test_col] / results[train_col]
    return results

def rf_trees_sweep(X, y, splits, nEstimatorsList, maxDepth):
    rows = []
    for nEstimators in nEstimatorsList:
        def factory():
            return make_random_forest(
                maxDepth=maxDepth,
                nEstimators=nEstimators
            )

        dfMetrics = evaluate_model_over_splits(factory, X, y, splits)
        dfMetrics["n_estimators"] = nEstimators
        dfMetrics["max_depth"]    = maxDepth
        rows.append(dfMetrics)

    results = pd.concat(rows, ignore_index=True)
    metric_cols = ("mean_test_rmse", "mean_train_rmse")
    if not set(metric_cols).issubset(results.columns):
        metric_cols = ("test_rmse", "train_rmse")
    test_col, train_col = metric_cols
    results["rmse_gap"] = results[test_col] - results[train_col]
    results["overfit_ratio"] = results[test_col] / results[train_col]
    return results

def rf_walkforward_predictions(X, y, splits, rfParams=None):
    """
    Train RF on each train fold, predict on test fold.
    Return a DataFrame aligned with your original index with columns:
      'y_true', 'y_pred', 'fold'
    """
    if rfParams is None:
        rfParams = {}

    records = []
    for splitId, (trainIdx, testIdx) in enumerate(splits):
        model = make_random_forest(**rfParams)
        model.fit(X[trainIdx], y[trainIdx])
        yPred = model.predict(X[testIdx])
        for i, idx in enumerate(testIdx):
            records.append({
                "row": int(idx),
                "fold": splitId,
                "y_true": float(y[idx]),
                "y_pred": float(yPred[i]),
            })

    dfPred = pd.DataFrame(records).set_index("row").sort_index()
    return dfPred

def rf_walkforward_backtest(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "y_ret_1",
    date_col: str = "date",
    rf_params: dict | None = None,
    n_splits: int = 5,
    long_threshold: float = 0.0,
    short_threshold: float | None = None,
    transaction_cost_bps: float = 5.0,
):
    """
    Walk-forward backtest for a Random Forest regressor using expanding
    time-series splits. Same shape as ridge_walkforward_backtest.
    """

    if rf_params is None:
        rf_params = {}

    if df.empty:
        raise ValueError("rf_walkforward_backtest: got empty DataFrame")

    missing = [c for c in feature_cols + [target_col, date_col] if c not in df.columns]
    if missing:
        raise ValueError(f"rf_walkforward_backtest: missing columns {missing}")

    df_sorted = df.sort_values(by=date_col).reset_index(drop=True)

    X = df_sorted[feature_cols].values
    y = df_sorted[target_col]
    dates = pd.to_datetime(df_sorted[date_col])

    n = len(df_sorted)
    if n < 300:
        raise ValueError(f"rf_walkforward_backtest: need at least ~300 rows, got {n}")

    tscv = TimeSeriesSplit(n_splits=n_splits)

    all_preds = pd.Series(index=df_sorted.index, dtype=float)
    all_is_test = np.zeros(n, dtype=bool)

    fold_records: list[dict] = []
    fold_id = 0

    for train_idx, test_idx in tscv.split(X):
        fold_id += 1

        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        dates_test = dates.iloc[test_idx]

        # skip degenerate folds
        if len(np.unique(y_train.to_numpy())) <= 1:
            continue

        model = make_random_forest(**rf_params)
        model.fit(X_train, y_train.to_numpy())
        preds_test = model.predict(X_test)

        all_preds.iloc[test_idx] = preds_test
        all_is_test[test_idx] = True

        preds_test_series = pd.Series(preds_test, index=y_test.index)

        bt_fold = backtest_strategy(
            realized_returns=y_test,
            preds=preds_test_series,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            transaction_cost_bps=transaction_cost_bps,
        )

        fold_records.append({
            "fold": fold_id,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "start_test": dates_test.min(),
            "end_test": dates_test.max(),
            "sharpe": bt_fold.sharpe,
            "max_drawdown": bt_fold.max_drawdown,
            "turnover": bt_fold.turnover,
        })

    if not fold_records:
        raise ValueError("rf_walkforward_backtest: no valid folds produced metrics")

    fold_metrics = pd.DataFrame(fold_records)

    test_mask = all_is_test
    realized_all = y[test_mask]
    preds_all = all_preds[test_mask]

    if len(realized_all) == 0:
        raise ValueError("rf_walkforward_backtest: global test set is empty")

    bt_global = backtest_strategy(
        realized_returns=realized_all,
        preds=preds_all,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        transaction_cost_bps=transaction_cost_bps,
    )

    equity_curve_global = bt_global.equity_curve

    return fold_metrics, bt_global, dates[test_mask], equity_curve_global

def make_gradient_boosting(
    nEstimators=200,
    learningRate=0.1,
    maxDepth=3,
    randomState=0
):
    return GradientBoostingRegressor(
        n_estimators=nEstimators,
        learning_rate=learningRate,
        max_depth=maxDepth,
        random_state=randomState,
    )

def gb_learning_rate_sweep(X, y, splits, learningRates, nEstimators=300, maxDepth=3):
    rows = []
    for lr in learningRates:
        def factory():
            return make_gradient_boosting(
                nEstimators=nEstimators,
                learningRate=lr,
                maxDepth=maxDepth,
            )

        dfMetrics = evaluate_model_over_splits(factory, X, y, splits)
        dfMetrics["learning_rate"] = lr
        dfMetrics["n_estimators"]  = nEstimators
        dfMetrics["max_depth"]     = maxDepth
        rows.append(dfMetrics)

    results = pd.concat(rows, ignore_index=True)
    metric_cols = ("mean_test_rmse", "mean_train_rmse")
    if not set(metric_cols).issubset(results.columns):
        metric_cols = ("test_rmse", "train_rmse")
    test_col, train_col = metric_cols
    results["rmse_gap"] = results[test_col] - results[train_col]
    results["overfit_ratio"] = results[test_col] / results[train_col]
    return results

def gb_walkforward_backtest(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "y_ret_1",
    date_col: str = "date",
    gb_params: dict | None = None,
    n_splits: int = 5,
    long_threshold: float = 0.0,
    short_threshold: float | None = None,
    transaction_cost_bps: float = 5.0,
):
    """
    Walk-forward backtest for a Gradient Boosting regressor using expanding
    time-series splits. Same shape as ridge/tree/rf walk-forward backtests.
    """

    if gb_params is None:
        gb_params = {}

    if df.empty:
        raise ValueError("gb_walkforward_backtest: got empty DataFrame")

    missing = [c for c in feature_cols + [target_col, date_col] if c not in df.columns]
    if missing:
        raise ValueError(f"gb_walkforward_backtest: missing columns {missing}")

    df_sorted = df.sort_values(by=date_col).reset_index(drop=True)

    X = df_sorted[feature_cols].values
    y = df_sorted[target_col]
    dates = pd.to_datetime(df_sorted[date_col])

    n = len(df_sorted)
    if n < 300:
        raise ValueError(f"gb_walkforward_backtest: need at least ~300 rows, got {n}")

    tscv = TimeSeriesSplit(n_splits=n_splits)

    all_preds = pd.Series(index=df_sorted.index, dtype=float)
    all_is_test = np.zeros(n, dtype=bool)

    fold_records: list[dict] = []
    fold_id = 0

    for train_idx, test_idx in tscv.split(X):
        fold_id += 1

        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        dates_test = dates.iloc[test_idx]

        if len(np.unique(y_train.to_numpy())) <= 1:
            continue

        model = make_gradient_boosting(**gb_params)
        model.fit(X_train, y_train.to_numpy())
        preds_test = model.predict(X_test)

        all_preds.iloc[test_idx] = preds_test
        all_is_test[test_idx] = True

        preds_test_series = pd.Series(preds_test, index=y_test.index)

        bt_fold = backtest_strategy(
            realized_returns=y_test,
            preds=preds_test_series,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            transaction_cost_bps=transaction_cost_bps,
        )

        fold_records.append({
            "fold": fold_id,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "start_test": dates_test.min(),
            "end_test": dates_test.max(),
            "sharpe": bt_fold.sharpe,
            "max_drawdown": bt_fold.max_drawdown,
            "turnover": bt_fold.turnover,
        })

    if not fold_records:
        raise ValueError("gb_walkforward_backtest: no valid folds produced metrics")

    fold_metrics = pd.DataFrame(fold_records)

    test_mask = all_is_test
    realized_all = y[test_mask]
    preds_all = all_preds[test_mask]

    if len(realized_all) == 0:
        raise ValueError("gb_walkforward_backtest: global test set is empty")

    bt_global = backtest_strategy(
        realized_returns=realized_all,
        preds=preds_all,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        transaction_cost_bps=transaction_cost_bps,
    )

    equity_curve_global = bt_global.equity_curve

    return fold_metrics, bt_global, dates[test_mask], equity_curve_global

def rf_vol_scaled_walkforward_backtest(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "y_ret_1",
    vol_col: str = "ret1_std_20",   # or "rv10"
    date_col: str = "date",
    rf_params: dict | None = None,
    n_splits: int = 5,
    max_leverage: float = 1.0,
    transaction_cost_bps: float = 5.0,
):
    """
    Walk-forward RF where position size is pred / rolling-volatility.
    """

    if rf_params is None:
        rf_params = {}

    df_sorted = df.sort_values(by=date_col).reset_index(drop=True)

    X = df_sorted[feature_cols].values
    y = df_sorted[target_col]
    vol = df_sorted[vol_col]
    dates = pd.to_datetime(df_sorted[date_col])

    n = len(df_sorted)
    if n < 300:
        raise ValueError("rf_vol_scaled_walkforward_backtest: need at least ~300 rows")

    tscv = TimeSeriesSplit(n_splits=n_splits)

    all_signals = pd.Series(index=df_sorted.index, dtype=float)
    all_is_test = np.zeros(n, dtype=bool)

    fold_records: list[dict] = []
    fold_id = 0

    for train_idx, test_idx in tscv.split(X):
        fold_id += 1

        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        vol_test = vol.iloc[test_idx]
        dates_test = dates.iloc[test_idx]

        if len(np.unique(y_train.to_numpy())) <= 1:
            continue

        model = make_random_forest(**rf_params)
        model.fit(X_train, y_train.to_numpy())
        preds_test = model.predict(X_test)

        # position sizing: prediction divided by volatility, clipped
        raw_pos = preds_test / (vol_test.to_numpy() + 1e-8)
        raw_pos = np.clip(raw_pos, -max_leverage, max_leverage)

        signals = pd.Series(raw_pos, index=y_test.index)
        all_signals.iloc[test_idx] = signals
        all_is_test[test_idx] = True

        bt_fold = backtest_with_signals(
            realized_returns=y_test,
            signals=signals,
            transaction_cost_bps=transaction_cost_bps,
        )

        fold_records.append({
            "fold": fold_id,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "start_test": dates_test.min(),
            "end_test": dates_test.max(),
            "sharpe": bt_fold.sharpe,
            "max_drawdown": bt_fold.max_drawdown,
            "turnover": bt_fold.turnover,
        })

    fold_metrics = pd.DataFrame(fold_records)

    mask = all_is_test
    realized_all = y[mask]
    signals_all = all_signals[mask]

    bt_global = backtest_with_signals(
        realized_returns=realized_all,
        signals=signals_all,
        transaction_cost_bps=transaction_cost_bps,
    )

    return fold_metrics, bt_global, dates[mask], bt_global.equity_curve
