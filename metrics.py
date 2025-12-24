from collections import Counter, defaultdict
from itertools import combinations
import pandas as pd

def all_pairs_support():
    return list(combinations(range(1, 27), 2))  # 325 pairs

def pair_frequencies(df: pd.DataFrame):
    red = Counter()
    white = Counter()
    for _, r in df.iterrows():
        red.update([tuple(sorted((int(r.r1), int(r.r2))))])
        white.update([tuple(sorted((int(r.w1), int(r.w2))))])
    return red, white

def weighted_scores(df: pd.DataFrame, decay: float = 0.98):
    scores = defaultdict(float)
    max_date = df["date"].max()
    for _, r in df.iterrows():
        age = (max_date - r["date"]).days
        w = decay ** age
        scores[("red", tuple(sorted((int(r.r1), int(r.r2)))))] += w
        scores[("white", tuple(sorted((int(r.w1), int(r.w2)))))] += w
    return scores

def chi_square_per_pair(counter: Counter, total_draws: int):
    expected = total_draws / 325.0
    return {k: ((v - expected) ** 2) / expected for k, v in counter.items()}

def cooccurrence_matrix(df: pd.DataFrame, color="red"):
    matrix = defaultdict(Counter)
    cols = ("r1", "r2") if color == "red" else ("w1", "w2")

    for _, r in df.iterrows():
        a, b = int(r[cols[0]]), int(r[cols[1]])
        matrix[a][b] += 1
        matrix[b][a] += 1

    nums = list(range(1, 27))
    out = pd.DataFrame(0, index=nums, columns=nums, dtype=int)
    for a in matrix:
        for b, v in matrix[a].items():
            out.loc[a, b] = int(v)
    return out
