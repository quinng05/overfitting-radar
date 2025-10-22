from pathlib import Path
from src.data_loader import load_yfinance_prices
from src.features import build_feature_table
from src.labels import add_labels
from src.models import run_all_and_save

def main():
    tickers = ["NVDA","AVGO","QCOM","AMD"]
    start   = "2018-01-01"  # shorter = faster; extend later
    print("Downloading prices")
    df = load_yfinance_prices(tickers, start=start, end=None)
    print("Building features")
    df = build_feature_table(df)
    print("Adding labels")
    print("Columns before labels:", list(df.columns))
    df = add_labels(df, horizon=1)
    print("Running baselines (Ridge & Logistic; expanding CV)")
    scores = run_all_and_save(df)  # writes data/processed/baseline_cv_scores.csv
    print(scores.to_string(index=False))
    print("Saved → data/processed/baseline_cv_scores.csv")

if __name__ == "__main__":
    main()
