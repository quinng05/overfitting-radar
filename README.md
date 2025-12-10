## Installation, Setup, and Execution

### Prerequisites
- Python **3.10 or 3.11**
- `git`
- Internet access (required on first run for `yfinance` to download historical prices)

### 1. Clone the Repository
```bash
git clone https://github.com/quinng05/overfitting-radar.git
cd overfitting-radar
```

### 2. (Recommended) Create a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
This installs: numpy, pandas, scikit-learn>=1.3, matplotlib, yfinance, joblib, tqdm.

### 4. Run the Full Pipeline
```bash
python -m src.run
```
This command automatically:
1. Loads tickers, feature sets, date ranges, and paths from `src/config.py`.
2. Downloads daily OHLCV data via `yfinance`.
3. Builds technical features (returns, volatility, RSI, rolling stats).
4. Creates next-day regression and classification labels.
5. Constructs expanding time-series CV splits.
6. Trains Ridge, Logistic, Decision Tree, Random Forest, and Gradient Boosting models.
7. Runs multiple walk-forward trading backtests.
8. Computes feature importances for all models.
9. Saves all outputs under `data/processed/` (CV metrics, backtests, plots, feature importances).

### 5. Output Directory Structure
```
data/processed/
├── cv/                     # Cross-validation metrics (CSV)
├── backtests/              # Walk-forward metrics + equity curves (CSV)
├── feature_importance/     # Feature importance summaries (CSV)
└── plots/                  # Sharpe plots, equity curves, overfitting visuals (PNG)
```

### 6. Example Notebook
An interactive notebook is provided at:
```
notebooks/overfitting_radar_demo.ipynb
```
It visualizes:
- Cross-validation results  
- Global Sharpe ranking  
- Tree depth overfitting curve  
- Feature ablation results  
- Example equity curves  

Launch with:
```bash
pip install notebook   # if needed
jupyter notebook
```
Then open the notebook and run all cells.

**Important:** Run the pipeline first so the notebook has data:
```bash
python -m src.run
```

### 7. Reproducibility Notes
- All configuration lives in `src/config.py`.
- You can fully regenerate the project with:
```bash
pip install -r requirements.txt
python -m src.run
```
