DATA_DIR = "data"
PROCESSED_DIR = "data/processed"
CV_DIR = "data/processed/cv"
BACKTESTS_DIR = "data/processed/backtests"
PLOTS_DIR = "data/processed/plots"
FEATURE_IMPORTANCE_DIR = "data/processed/feature_importance"

DEFAULT_TICKERS = ["NVDA", "AVGO", "QCOM", "AMD"]
DEFAULT_START = "2018-01-01"

DEFAULT_FEATURE_COLS = [
    "ret1",
    "ret5",
    "ret10",
    "rv10",
    "rsi5",
    "ret1_mean_20",
    "ret1_std_20",
    "ret1_mean_60",
    "ret1_std_60",
]
TARGET_COL = "y_ret_1"
CLASS_COL = "y_up_1"
DATE_COL = "date"

DEFAULT_N_SPLITS = 5
DEFAULT_TC_BPS = 5.0

FEATURE_SETS = {
    "core": [
        "ret1", "ret5", "ret10", "rv10", "rsi5",
    ],
    "core_plus_20": [
        "ret1", "ret5", "ret10", "rv10", "rsi5",
        "ret1_mean_20", "ret1_std_20",
    ],
    "all": DEFAULT_FEATURE_COLS,
}
