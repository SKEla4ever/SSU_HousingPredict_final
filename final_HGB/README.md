# SSU Data Dash 2026 · Final HGB package

The Final v3 release uses **HGB v3 aged-system** for Off-market and **HGB v1** for List-aware. The **Random Forest** results are a separate baseline comparison. The two submission CSVs were generated from the [frozen HGB configurations](config/frozen_models.json).

| Track | Submission | Rows | Walk-forward MdAPE |
|---|---|---:|---:|
| Off-market | [submission_off_market.csv](submissions/submission_off_market.csv) | 3,974 | 7.64% |
| List-aware | [submission_list_aware.csv](submissions/submission_list_aware.csv) | 4,064 | 4.29% |

## Run the official sequence with the final models

From this folder on macOS or Linux, use Python 3.12 and the pinned dependencies.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
export DATA_DIR="$PWD/data"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1

python run.py validate
python run.py report
python run.py predict
python run.py check
shasum -a 256 -c submissions/SHA256SUMS.txt
```

The local handoff includes the three original competition Parquet files for competition use. You can also point `DATA_DIR` to another copy of those files. If running from a GitHub checkout without the Parquet files, leave `DATA_DIR` unset so the loader can download the [organizer's dataset](https://huggingface.co/datasets/Baliyang/SSU_data_dash). The final-model workflow verifies the Parquet SHA-256 hashes.

- `validate` refits both tracks for the December 2025–May 2026 walk-forward evaluation and saves the February–May 2026 fixed-origin backtest.
- `report` computes metrics and the price-decile chart from those backtest predictions.
- `predict` fits on all 150,000 sales through May 2026 and writes the CSVs and fitted models.
- `check` verifies the two-column format, complete and unique IDs, and positive finite prices. The SHA-256 check compares both CSVs against the packaged release.

These commands write to `outputs/`, `reports/`, `figures/` and `submissions/`. The reproducibility report includes commands for rerunning in a temporary copy.

## Included material

- [Official starter kit, handout, rubric and report template](official/README.md)
- [Final v3 reproducibility report in Chinese](reports/V3_REPRODUCIBILITY_REPORT.md)
- [Final v3 slides brief in English](SLIDES_BRIEF.md)
- [Walk-forward results](reports/validation_report.md) and [fixed-origin results](reports/horizon_report.json)
- [HGB model-selection records](reports/hgb_selection/) from the archived experiments referenced in the brief
- [Random Forest baseline reproduction and results](reports/rf_baseline/README.md)
- [Round 1 report draft PDF](report/report_draft.pdf) and [LaTeX source](report/report_draft.tex)
