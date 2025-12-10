import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def add_kmeans_regime(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_clusters: int = 3,
    random_state: int = 0,
):
    """
    K-Means regime labels, in the same spirit as the Lecture 11 code:
    - standardize selected features
    - run KMeans
    - attach integer regime label per row
    """
    df_sorted = df.sort_values(["date"]).copy()

    # only cluster where all features are present
    mask = df_sorted[feature_cols].notna().all(axis=1)
    X = df_sorted.loc[mask, feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",   # matches modern sklearn examples
    )
    labels = km.fit_predict(X_scaled)

    df_sorted.loc[mask, "regime_km"] = labels.astype(int)

    return df_sorted, km, scaler