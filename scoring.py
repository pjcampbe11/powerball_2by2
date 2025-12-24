import pandas as pd

def normalize(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx == mn:
        return series * 0.0
    return (series - mn) / (mx - mn)

def build_score_table(color: str, pairs, freq_dict, weighted_dict, chi_dict):
    rows = []
    for pair in pairs:
        rows.append({
            "color": color,
            "pair": pair,
            "frequency": freq_dict.get(pair, 0),
            "weighted_score": weighted_dict.get((color, pair), 0.0),
            "chi_square": chi_dict.get(pair, 0.0),
        })

    df = pd.DataFrame(rows)

    df["freq_norm"] = normalize(df["frequency"])
    df["weighted_norm"] = normalize(df["weighted_score"])
    df["chi_norm"] = normalize(df["chi_square"])

    # Tunable weights (kept simple and stable)
    df["composite_score"] = (
        0.4 * df["freq_norm"] +
        0.4 * df["weighted_norm"] +
        0.2 * df["chi_norm"]
    )

    df["confidence"] = pd.cut(
        df["composite_score"],
        bins=[-1, 0.33, 0.66, 1.01],
        labels=["LOW", "MEDIUM", "HIGH"]
    )

    return df.sort_values(["composite_score", "frequency"], ascending=[False, False]).reset_index(drop=True)
