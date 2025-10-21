import numpy as np
import pandas as pd
from pathlib import Path
from typing import Iterable
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, log_loss
from .splits import expanding_time_splits

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
        for tr, te in expanding_time_splits(dates, n_splits=5, min_train_days=252):
            pipe.fit(X[tr], y[tr])
            pred = pipe.predict(X[te])
            fold_rmse.append(mean_squared_error(y[te], pred, squared=False))
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
        for tr, te in expanding_time_splits(dates, n_splits=5, min_train_days=252):
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

def run_all_and_save(df: pd.DataFrame, outdir: str = "data/processed") -> pd.DataFrame:
    Path(outdir).mkdir(parents=True, exist_ok=True)
    ridge_df = ridge_baseline(df, alphas=(0.1,1.0,10.0))
    log_df   = logreg_baseline(df, Cs=(1.0,0.5,0.1), calibrate=True)
    out = pd.concat([ridge_df, log_df], ignore_index=True)
    out.to_csv(f"{outdir}/baseline_cv_scores.csv", index=False)
    return out
