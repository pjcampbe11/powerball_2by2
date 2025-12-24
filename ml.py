import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

DEFAULT_FEATURES = [
    "frequency", "weighted_score", "chi_square",
    "score_global", "score_365d", "score_90d", "score_30d",
    "delta_365d", "delta_90d", "delta_30d",
    "volatility"
]

def run_dbscan(feature_df: pd.DataFrame, eps: float, min_samples: int, features=None):
    if features is None:
        features = DEFAULT_FEATURES

    # Ensure columns exist
    missing = [c for c in features if c not in feature_df.columns]
    if missing:
        raise ValueError(f"Missing ML feature columns: {missing}")

    X = feature_df[features].fillna(0.0).astype(float).values
    Xs = StandardScaler().fit_transform(X)

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(Xs)

    out = feature_df.copy()
    out["cluster"] = labels
    return out, features

def cluster_summary(clustered_df: pd.DataFrame):
    # -1 is noise
    df = clustered_df.copy()
    grp = df.groupby("cluster", dropna=False)

    summary = grp.agg(
        count=("pair", "count"),
        avg_score_global=("score_global", "mean"),
        avg_delta_30d=("delta_30d", "mean"),
        avg_volatility=("volatility", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_chi=("chi_square", "mean"),
    ).reset_index().sort_values(["cluster", "count"], ascending=[True, False])

    return summary
