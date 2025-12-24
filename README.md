# Powerball 2by2 Analytics Pipeline 

This project parses Powerball 2by2 historical draws and builds:
- Step 1: Rule engine (YARA-style rules from YAML)
- Step 2: Normalized composite scoring
- Step 3: Temporal drift + volatility + trends
- Step 4: Visualization (heatmaps, deltas, rank change)
- Step 5: Statistical modeling (entropy, KL/JS divergence, chi2/G-test per window)
- Step 6: Unsupervised ML discovery (DBSCAN clustering)

## powerball_2by2
- This project leverages historical 2by2 winning numbers and uses a 6-step process to arrive at possible winning sets.
- Uses ->rules → scoring → drift → plots → deeper stats → and ML clustering to produce possible winning number pairs

This system is analytical, not predictive.

---

## Setup

### 1) Create virtual environment
```
python3 -m venv venv
source venv/bin/activate
```
## Install requirements

```
pip install -r requirements.txt
```

```
python main.py \
  --url "https://www.powerball.com/previous-results?gc=2by2&sd=2020-01-06&ed=2025-12-23" \
  --out run1
```

## Outputs
```

run1_stats_windows.csv

run1_rule_hits.csv

run1_red_pair_features.csv

run1_white_pair_features.csv

run1_ALL_pair_features.csv

run1_red_top25_global.csv / run1_white_top25_global.csv

run1_red_top25_emerging.csv / run1_white_top25_emerging.csv

run1_red_top25_fading.csv / run1_white_top25_fading.csv
```
## Run with plots (Step 4)
```
python main.py \
  --url "https://www.powerball.com/previous-results?gc=2by2&sd=2020-01-06&ed=2025-12-23" \
  --out run1 \
  --plots \
  --plot-dir plots \
  --top-n 10
```
## Creates PNGs in ./plots
```
heatmap_red.png / heatmap_white.png

top_red_frequency.png / top_white_frequency.png

top_red_emerging_delta30d.png / top_white_emerging_delta30d.png

red_delta30d_hist.png / white_delta30d_hist.png

red_rank_change_global_vs_30d.png / white_rank_change_global_vs_30d.png
```

## Run with ML clustering (Step 6)
```
python main.py \
  --url "https://www.powerball.com/previous-results?gc=2by2&sd=2020-01-06&ed=2025-12-23" \
  --out run1 \
  --ml \
  --eps 0.8 \
  --min-samples 8
```
## Outputs
```
run1_red_ml_dbscan.csv

run1_red_ml_clusters_summary.csv

run1_white_ml_dbscan.csv

run1_white_ml_clusters_summary.csv
```
## Notes

### DBSCAN cluster = -1 means noise / outlier

### eps / min-samples are tunable; start with eps=0.8 and min-samples=8

## Rules
```
Edit rules.yaml to change detection logic without code changes.
```
## Interpreting Step 5 stats (window-level)
```
In *_stats_windows.csv:

entropy_bits lower = more concentrated distribution (less uniform)

KL/JS higher = farther from uniform randomness

chi2_p_approx smaller = window looks less consistent with uniform (use for trending)

Why this is built like a detection pipeline

Facts → Scores → Drift → Visuals → Statistical divergence → Unsupervised discovery
```

## Quick usage examples
### 1) Full pipeline + plots + ML, all at once
```
python main.py \
  --url "https://www.powerball.com/previous-results?gc=2by2&sd=2020-01-06&ed=2025-12-23" \
  --out fullrun \
  --plots --plot-dir plots \
  --ml --eps 0.85 --min-samples 10
```
### 2) Just get top emerging pairs (fast)
```
Run once, then open:

fullrun_red_top25_emerging.csv

fullrun_white_top25_emerging.csv

::contentReference[oaicite:0]{index=0}
```
