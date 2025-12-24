import argparse
import os
import pandas as pd

from parser import fetch_and_parse
from metrics import (
    pair_frequencies, weighted_scores, chi_square_per_pair,
    cooccurrence_matrix, all_pairs_support
)
from scoring import build_score_table
from temporal import build_temporal_features, get_window_df
from rules import load_rules, evaluate_rules
from stats import window_stats
from viz import ensure_dir, plot_heatmap, plot_bar, plot_hist, plot_scatter
from ml import run_dbscan, cluster_summary

def enrich_with_facts(df_draws, color, decay):
    red_freq, white_freq = pair_frequencies(df_draws)
    freq = red_freq if color == "red" else white_freq

    weighted = weighted_scores(df_draws, decay=decay)
    chi = chi_square_per_pair(freq, len(df_draws))

    score_df = build_score_table(color, freq.keys(), freq, weighted, chi)

    # Temporal features (scores/deltas/volatility/trend)
    temporal_df = build_temporal_features(df_draws, color, decay)

    # Merge: score_df has detailed norms/confidence; temporal has window scores/deltas
    merged = score_df.merge(temporal_df, on=["color", "pair"], how="left")
    merged = merged.fillna(0.0)

    return merged, freq, weighted, chi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", default="output")
    ap.add_argument("--rules", default="rules.yaml")
    ap.add_argument("--decay", type=float, default=0.98)

    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--plot-dir", default="plots")
    ap.add_argument("--top-n", type=int, default=10)

    ap.add_argument("--ml", action="store_true")
    ap.add_argument("--eps", type=float, default=0.8)
    ap.add_argument("--min-samples", type=int, default=8)

    args = ap.parse_args()

    df = fetch_and_parse(args.url)
    print(f"[+] Parsed draws: {len(df)} (min={df.date.min().date()} max={df.date.max().date()})")

    rules = load_rules(args.rules)

    all_outputs = []
    all_rule_hits = []
    all_stats_rows = []

    # Step 5 window-level stats
    support = all_pairs_support()
    for days in [None, 365, 90, 30]:
        sub = get_window_df(df, days)
        red_freq, white_freq = pair_frequencies(sub)
        for color, freq in [("red", red_freq), ("white", white_freq)]:
            s = window_stats(freq, support)
            all_stats_rows.append({
                "window": "global" if days is None else f"{days}d",
                "color": color,
                **s
            })

    stats_windows_df = pd.DataFrame(all_stats_rows)
    stats_windows_df.to_csv(f"{args.out}_stats_windows.csv", index=False)

    # Build pair-level features for both colors
    for color in ["red", "white"]:
        merged, freq, weighted, chi = enrich_with_facts(df, color, decay=args.decay)

        # Rule hits (Step 1) using raw facts
        for _, row in merged.iterrows():
            facts = {
                "frequency": float(row["frequency"]),
                "weighted_score": float(row["weighted_score"]),
                "chi_square": float(row["chi_square"])
            }
            all_rule_hits.extend(evaluate_rules(color, row["pair"], facts, rules))

        merged.to_csv(f"{args.out}_{color}_pair_features.csv", index=False)
        all_outputs.append(merged)

        # Convenience exports
        merged.sort_values("composite_score", ascending=False).head(25)\
            .to_csv(f"{args.out}_{color}_top25_global.csv", index=False)
        merged.sort_values("delta_30d", ascending=False).head(25)\
            .to_csv(f"{args.out}_{color}_top25_emerging.csv", index=False)
        merged.sort_values("delta_30d", ascending=True).head(25)\
            .to_csv(f"{args.out}_{color}_top25_fading.csv", index=False)

        # Plots (Step 4)
        if args.plots:
            ensure_dir(args.plot_dir)

            # Heatmaps from raw draws
            heat = cooccurrence_matrix(df, color=color)
            plot_heatmap(heat, f"{color.upper()} pair co-occurrence heatmap", os.path.join(args.plot_dir, f"heatmap_{color}.png"))

            # Top global pairs (by frequency)
            top_freq = merged.sort_values("frequency", ascending=False)[["pair", "frequency"]].rename(columns={"frequency": "value"})
            plot_bar(top_freq, f"Top {args.top_n} {color.upper()} pairs by frequency (global)", os.path.join(args.plot_dir, f"top_{color}_frequency.png"), top_n=args.top_n)

            # Emerging (delta_30d)
            top_delta = merged.sort_values("delta_30d", ascending=False)[["pair", "delta_30d"]].rename(columns={"delta_30d": "value"})
            plot_bar(top_delta, f"Top {args.top_n} {color.upper()} EMERGING pairs (delta_30d)", os.path.join(args.plot_dir, f"top_{color}_emerging_delta30d.png"), top_n=args.top_n)

            # Delta distribution
            plot_hist(merged["delta_30d"], f"{color.upper()} delta_30d distribution", os.path.join(args.plot_dir, f"{color}_delta30d_hist.png"))

            # Rank change: global score rank vs 30d score rank
            tmp = merged.copy()
            tmp["rank_global"] = tmp["score_global"].rank(ascending=False, method="min")
            tmp["rank_30d"] = tmp["score_30d"].rank(ascending=False, method="min")
            plot_scatter(tmp["rank_global"], tmp["rank_30d"],
                         f"{color.upper()} rank change: global vs 30d",
                         os.path.join(args.plot_dir, f"{color}_rank_change_global_vs_30d.png"),
                         "Global rank (lower=better)", "30d rank (lower=better)")

        # ML (Step 6): DBSCAN clustering on engineered feature space
        if args.ml:
            clustered, used_features = run_dbscan(merged, eps=args.eps, min_samples=args.min_samples)
            clustered.to_csv(f"{args.out}_{color}_ml_dbscan.csv", index=False)

            summary = cluster_summary(clustered)
            summary.to_csv(f"{args.out}_{color}_ml_clusters_summary.csv", index=False)

    # Combined exports
    all_features = pd.concat(all_outputs, ignore_index=True)
    all_features.to_csv(f"{args.out}_ALL_pair_features.csv", index=False)

    pd.DataFrame(all_rule_hits).to_csv(f"{args.out}_rule_hits.csv", index=False)

    print("[+] Done.")
    print(f"    - Window stats: {args.out}_stats_windows.csv")
    print(f"    - Rule hits:    {args.out}_rule_hits.csv")
    print(f"    - Pair feats:   {args.out}_red_pair_features.csv, {args.out}_white_pair_features.csv")
    if args.ml:
        print(f"    - ML outputs:   {args.out}_*_ml_dbscan.csv and {args.out}_*_ml_clusters_summary.csv")
    if args.plots:
        print(f"    - Plots dir:    {args.plot_dir}/")

if __name__ == "__main__":
    main()
