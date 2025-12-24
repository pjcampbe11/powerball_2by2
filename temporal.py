import pandas as pd
from metrics import pair_frequencies, weighted_scores, chi_square_per_pair
from scoring import build_score_table

WINDOWS = [365, 90, 30]

def get_window_df(df: pd.DataFrame, days: int | None):
    if days is None:
        return df
    cutoff = df["date"].max() - pd.Timedelta(days=days)
    return df[df["date"] >= cutoff].copy()

def window_score_df(df: pd.DataFrame, color: str, days: int | None, decay: float):
    sub = get_window_df(df, days)
    red_freq, white_freq = pair_frequencies(sub)
    freq = red_freq if color == "red" else white_freq
    w = weighted_scores(sub, decay=decay)
    chi = chi_square_per_pair(freq, len(sub))
    return build_score_table(color, freq.keys(), freq, w, chi)

def build_temporal_features(df: pd.DataFrame, color: str, decay: float):
    global_scores = window_score_df(df, color, None, decay)[["pair", "composite_score"]].rename(
        columns={"composite_score": "score_global"}
    )

    merged = global_scores
    for days in WINDOWS:
        win = window_score_df(df, color, days, decay)[["pair", "composite_score"]].rename(
            columns={"composite_score": f"score_{days}d"}
        )
        merged = merged.merge(win, on="pair", how="outer")

    merged = merged.fillna(0.0)

    for days in WINDOWS:
        merged[f"delta_{days}d"] = merged[f"score_{days}d"] - merged["score_global"]

    score_cols = ["score_global"] + [f"score_{d}d" for d in WINDOWS]
    merged["volatility"] = merged[score_cols].std(axis=1)

    def trend(row):
        if row["delta_30d"] > 0.15:
            return "EMERGING"
        if row["delta_30d"] < -0.15:
            return "FADING"
        return "STABLE"

    merged["trend"] = merged.apply(trend, axis=1)
    merged["color"] = color
    return merged
