# main.py
import argparse
import os
import pandas as pd

from data_parser import parse_file, parse_glob
from metrics import (
    pair_frequencies, weighted_scores, chi_square_per_pair,
    cooccurrence_matrix, all_pairs_support
)
from scoring import build_score_table
from temporal import build_temporal_features, get_window_df
from rules import load_rules, evaluate_rules
from stats import window_stats
from ml import run_dbscan, cluster_summary


def enrich_with_facts(df_draws, color, decay):
    red_freq, white_freq = pair_frequencies(df_draws)
    freq = red_freq if color == "red" else white_freq

    weighted = weighted_scores(df_draws, decay=decay)
    chi = chi_square_per_pair(freq, len(df_draws))

    score_df = build_score_table(color, freq.keys(), freq, weighted, chi)

    temporal_df = build_temporal_features(df_draws, color, decay)

    merged = score_df.merge(temporal_df, on=["color", "pair"], how="left")

    # Fill numeric NaNs only (avoid categorical 'confidence' blowing up)
    num_cols = merged.select_dtypes(include=["number"]).columns
    merged[num_cols] = merged[num_cols].fillna(0.0)

    # Optional: fill trend/confidence if missing (should rarely happen)
    if "trend" in merged.columns:
        merged["trend"] = merged["trend"].fillna("STABLE")
    if "confidence" in merged.columns:
        merged["confidence"] = merged["confidence"].cat.add_categories(["UNKNOWN"]).fillna("UNKNOWN")

    return merged, freq, weighted, chi



def main():
    ap = argparse.ArgumentParser(description="Powerball 2by2 analysis pipeline (HTML mode)")
    ap.add_argument("--out", default="output")
    ap.add_argument("--rules", default="rules.yaml")
    ap.add_argument("--decay", type=float, default=0.98)

    # HTML input modes
    ap.add_argument("--html-file", help="Path to a saved 2by2 results HTML page")
    ap.add_argument("--html-glob", help=r'Glob for saved HTML pages, e.g. "C:\...\2by2_pg_*.html"')

    # Plots
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--plot-dir", default="plots")
    ap.add_argument("--top-n", type=int, default=10)

    # ML
    ap.add_argument("--ml", action="store_true")
    ap.add_argument("--eps", type=float, default=0.8)
    ap.add_argument("--min-samples", type=int, default=8)

    args = ap.parse_args()

    # --- Load data ---
    if args.html_glob:
        df = parse_glob(args.html_glob)
    elif args.html_file:
        df = parse_file(args.html_file)
    else:
        raise SystemExit(
            "You must provide --html-glob or --html-file.\n"
            "Example:\n"
            '  python main.py --html-glob "C:\\Users\\12242\\powerball_2by2\\2by2_pg_*.html" --out run1'
        )

    # Normalize + de-dupe (important when you stitch many pages)
    if "date" not in df.columns:
        raise RuntimeError("Parsed dataframe missing 'date' column.")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date", "r1", "r2", "w1", "w2"]).reset_index(drop=True)

    print(f"[+] Parsed draws: {len(df)} (min={df.date.min().date()} max={df.date.max().date()})")

    # --- Rules ---
    try:
        rules = load_rules(args.rules)
    except FileNotFoundError:
        print(f"[!] rules file not found: {args.rules} (continuing with zero rules)")
        rules = []

    all_outputs = []
    all_rule_hits = []
    all_stats_rows = []

    # --- Step 5: window-level stats ---
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

    # --- Pair-level features for red + white ---
    for color in ["red", "white"]:
        merged, freq, weighted, chi = enrich_with_facts(df, color, decay=args.decay)

        # Rule hits (using raw facts)
        if rules:
            for _, row in merged.iterrows():
                facts = {
                    "frequency": float(row["frequency"]),
                    "weighted_score": float(row["weighted_score"]),
                    "chi_square": float(row["chi_square"]),
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

        # --- Plots ---
        if args.plots:
            from viz import ensure_dir, plot_heatmap, plot_bar, plot_hist, plot_scatter
            ensure_dir(args.plot_dir)

            heat = cooccurrence_matrix(df, color=color)
            plot_heatmap(
                heat,
                f"{color.upper()} pair co-occurrence heatmap",
                os.path.join(args.plot_dir, f"heatmap_{color}.png")
            )

            top_freq = merged.sort_values("frequency", ascending=False)[["pair", "frequency"]]\
                             .rename(columns={"frequency": "value"})
            plot_bar(
                top_freq,
                f"Top {args.top_n} {color.upper()} pairs by frequency (global)",
                os.path.join(args.plot_dir, f"top_{color}_frequency.png"),
                top_n=args.top_n
            )

            top_delta = merged.sort_values("delta_30d", ascending=False)[["pair", "delta_30d"]]\
                              .rename(columns={"delta_30d": "value"})
            plot_bar(
                top_delta,
                f"Top {args.top_n} {color.upper()} EMERGING pairs (delta_30d)",
                os.path.join(args.plot_dir, f"top_{color}_emerging_delta30d.png"),
                top_n=args.top_n
            )

            plot_hist(
                merged["delta_30d"],
                f"{color.upper()} delta_30d distribution",
                os.path.join(args.plot_dir, f"{color}_delta30d_hist.png")
            )

            tmp = merged.copy()
            tmp["rank_global"] = tmp["score_global"].rank(ascending=False, method="min")
            tmp["rank_30d"] = tmp["score_30d"].rank(ascending=False, method="min")
            plot_scatter(
                tmp["rank_global"], tmp["rank_30d"],
                f"{color.upper()} rank change: global vs 30d",
                os.path.join(args.plot_dir, f"{color}_rank_change_global_vs_30d.png"),
                "Global rank (lower=better)", "30d rank (lower=better)"
            )

        # --- ML (DBSCAN) ---
        if args.ml:
            clustered, used_features = run_dbscan(merged, eps=args.eps, min_samples=args.min_samples)
            clustered.to_csv(f"{args.out}_{color}_ml_dbscan.csv", index=False)

            summary = cluster_summary(clustered)
            summary.to_csv(f"{args.out}_{color}_ml_clusters_summary.csv", index=False)

    # --- Combined exports ---
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
