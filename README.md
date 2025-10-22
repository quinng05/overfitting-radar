Overfitting Radar runs a simple, end-to-end trading ML pipeline: load prices, build technical features,
create next-day labels, train Ridge (regression) and Logistic (classification) with expanding
time-series CV, and save baseline metrics to `data/processed/baseline_cv_scores.csv`.


**Reqs:** Python 3.10+, 
`pip install -r requirements.txt`
(pandas, numpy, scikit-learn, yfinance)

**Run baseline pipeline**
```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m src.run
