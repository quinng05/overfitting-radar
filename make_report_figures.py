# make_report_figures.py
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# --------------------------------------------------
# Paths
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
CV_DIR = DATA_DIR / "cv"
BT_DIR = DATA_DIR / "backtests"

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

print("ROOT:", ROOT)
print("Saving figures to:", FIG_DIR)


# --------------------------------------------------
# Figure 1 – Overfitting Radar workflow diagram
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 2.5))
ax.set_axis_off()

nodes = [
    "Raw price data",
    "Feature\nengineering",
    "Expanding-window\nCV",
    "Walk-forward\nbacktests",
    "Diagnostics &\nplots",
]

for i, label in enumerate(nodes):
    x = i
    y = 0.5

    rect = FancyBboxPatch(
        (x - 0.4, y - 0.2),
        0.8, 0.4,
        boxstyle="round,pad=0.02",
        edgecolor="black",
        facecolor="white",
        linewidth=1.0,
    )
    ax.add_patch(rect)
    ax.text(x, y, label, ha="center", va="center", fontsize=8)

    if i < len(nodes) - 1:
        ax.annotate(
            "",
            xy=(x + 0.6, y),
            xytext=(x + 0.4, y),
            arrowprops=dict(arrowstyle="->", lw=1),
        )

ax.set_xlim(-0.5, len(nodes) - 0.0)
ax.set_ylim(0, 1)

plt.title("Overfitting Radar workflow", fontsize=12)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig1_workflow_overfitting_radar.png",
            dpi=300, bbox_inches="tight")
plt.close()


# --------------------------------------------------
# Figure 2 – Ridge CV RMSE vs alpha
# (for the “CV RMSE for Ridge across alphas” figure)
# --------------------------------------------------
baseline_cv = pd.read_csv(CV_DIR / "baseline_cv_scores.csv")
ridge_cv = baseline_cv[baseline_cv["model"] == "Ridge"].copy()

plt.figure(figsize=(4, 3))
plt.plot(ridge_cv["alpha"], ridge_cv["rmse_mean"], marker="o")
plt.fill_between(
    ridge_cv["alpha"],
    ridge_cv["rmse_mean"] - ridge_cv["rmse_std"],
    ridge_cv["rmse_mean"] + ridge_cv["rmse_std"],
    alpha=0.2,
)
plt.xscale("log")
plt.xlabel("Ridge alpha")
plt.ylabel("Mean RMSE (expanding CV)")
plt.title("Ridge expanding-CV performance")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig2_ridge_cv_rmse.png", dpi=300,
            bbox_inches="tight")
plt.close()


# --------------------------------------------------
# Figure 3 – Sharpe comparison across all models
# (global walk-forward comparison)
# --------------------------------------------------
cmp = pd.read_csv(BT_DIR / "global_walkforward_comparison.csv")
cmp_sorted = cmp.sort_values("sharpe", ascending=False)

plt.figure(figsize=(6, 3))
plt.bar(cmp_sorted["model"], cmp_sorted["sharpe"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Walk-forward Sharpe")
plt.title("Walk-forward Sharpe across models")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig3_global_sharpe_comparison.png",
            dpi=300, bbox_inches="tight")
plt.close()


# --------------------------------------------------
# Figure 4 – Walk-forward equity curves: Ridge vs RF vol-scaled
# --------------------------------------------------
ridge_eq = pd.read_csv(BT_DIR / "ridge_walkforward_equity_curve.csv")
rf_vol_eq = pd.read_csv(BT_DIR / "rf_vol_scaled_walkforward_equity_curve.csv")

ridge_eq["date"] = pd.to_datetime(ridge_eq["date"])
rf_vol_eq["date"] = pd.to_datetime(rf_vol_eq["date"])

plt.figure(figsize=(6, 3))
plt.plot(ridge_eq["date"], ridge_eq["equity"], label="Ridge walk-forward")
plt.plot(rf_vol_eq["date"], rf_vol_eq["equity"],
         label="RF vol-scaled walk-forward")
plt.xticks(rotation=45)
plt.ylabel("Equity")
plt.title("Walk-forward equity curves: Ridge vs RF vol-scaled")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "fig4_equity_ridge_vs_rfvol.png",
            dpi=300, bbox_inches="tight")
plt.close()


# --------------------------------------------------
# Figure 5 – Feature ablation: Sharpe by feature set (Ridge only)
# --------------------------------------------------
ablation = pd.read_csv(BT_DIR / "feature_ablation_comparison.csv")
ridge_ablation = ablation[ablation["model"] == "ridge"].copy()

plt.figure(figsize=(4, 3))
plt.bar(ridge_ablation["feature_set"], ridge_ablation["sharpe"])
plt.ylabel("Walk-forward Sharpe")
plt.title("Feature ablation – Ridge Sharpe by feature set")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig5_feature_ablation_ridge.png",
            dpi=300, bbox_inches="tight")
plt.close()

print("Done. Figures written to:", FIG_DIR)
