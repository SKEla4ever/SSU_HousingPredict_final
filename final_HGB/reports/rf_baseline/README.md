# Random Forest baseline

The tree experiments provide historical benchmarks for the two HGB models in Final v3. The release submissions come from the frozen HGB pair. The tree study compared a single Decision Tree, Bagging and Random Forest under the same chronological splits. Within that study, development selected RF for Off-market and Bagging for List-aware.

| Protocol | Track | Homes | RF MdAPE | RF MAPE | Final v3 HGB MdAPE |
|---|---|---:|---:|---:|---:|
| Walk-forward, Dec 2025–May 2026 | Off-market | 10,069 | 15.70% | 20.72% | 7.64% |
| Walk-forward, Dec 2025–May 2026 | List-aware | 10,069 | 4.75% | 5.94% | 4.29% |
| Fixed-origin, Feb–May 2026 | Off-market | 7,773 | 15.50% | 20.36% | 8.24% |
| Fixed-origin, Feb–May 2026 | List-aware | 7,773 | 4.75% | 5.99% | 4.24% |

The comparison uses pooled errors on unrounded prices. Walk-forward refits at each sale-month boundary. Fixed-origin trains once on sales before February 1, 2026. RF uses 300 bootstrap trees, maximum depth 12, a per-split feature fraction of 0.7, seed 42 and minimum leaf sizes of 20 for Off-market and 50 for List-aware. Both tracks learn `log(sold_price)` and exponentiate the mean tree prediction. The four categorical fields are one-hot encoded within each training fold. List-aware adds `list_price` to the common structured features.

## Reproduce the baseline comparison

Use the same Python environment and official data snapshot as the [HGB workflow](../../README.md). Run the following commands from the package root. The loader downloads and caches the organizer's Parquet files by default.

```sh
unset DATA_DIR
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
python analysis/run_tree_baselines.py --phase holdout
python analysis/run_tree_baselines.py --phase horizon
```

If you already have all three original Parquet files locally, replace `unset DATA_DIR` in the block above with the following line.

```sh
export DATA_DIR="/absolute/path/to/competition_data"
```

The commands replay the tree families in the frozen comparison and write run-specific predictions, metrics and diagnostics under `outputs/tree_baselines/`. The holdout phase also scores the organizer's starter baselines. The [frozen settings](../../outputs/tree_baselines/frozen_config.json) preserve the original selection record and use file sizes and SHA-256 digests to identify the data snapshot. The development phase can repeat selection in a separate copy with no existing frozen configuration.

## Archived reference metrics

- [Development comparison](development_metrics.csv) for June–November 2025.
- [Walk-forward metrics](holdout_metrics.csv) and [fixed-origin metrics](horizon_metrics.csv).
- [RF walk-forward subgroup metrics](rf_holdout_grouped_metrics.csv) and [selected tree models' subgroup metrics](selected_holdout_grouped_metrics.csv).
- [HGB walk-forward metrics](../validation_report.json) and [HGB fixed-origin metrics](../horizon_report.json).
