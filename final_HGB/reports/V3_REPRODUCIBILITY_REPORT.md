# SSU Data Dash Final v3 Model Reproducibility Report

Final v3 is the release for both competition tracks. **Off-market** uses `hgb_v3_aged_system` to estimate the sale price from the price per square foot. **List-aware** uses `hgb_v1` to estimate the sale price from the asking price and the predicted sold-to-list ratio. Both use scikit-learn's `HistGradientBoostingRegressor` with chronological splits for training and backtesting. [frozen_models.json](../config/frozen_models.json) records the release configuration, input file checksums, and model version for each track.

## 1. Data and training cutoff

The data comes from the synthetic Toronto sales provided by the SSU Data Dash competition. The final training set includes all 150,000 completed sales from 2021-01-01 through 2026-05-31. The final cutoff is **2026-06-01**. Each track's model is fitted once to predict the hidden test set for June–September 2026. The table lists the SHA-256 hashes for the three Parquet files.

| File | Rows | SHA-256 |
|---|---:|---|
| `listings.parquet` | 150,000 | `18cf87fa122b544018fccbc15040403b0fed37080c44a75d1b55deca96d2c4bf` |
| `test_off_market.parquet` | 3,974 | `d078d1bd652550a8bf33bbaa42af21e9a7ea6c42dfd53c57c083e5597165b1ad` |
| `test_list_aware.parquet` | 4,064 | `ef1b82d16d547199b93a4e1a27d1ff3946a327854c320956a32f437b81469b35` |

Training uses `sold_price` to construct the target and `date_sold` to determine when a record becomes eligible for training. During validation, the pipeline removes `sold_price`, `date_sold`, `days_on_market`, and the two `external_avm_*` columns from the prediction inputs. It also removes `list_price` for Off-market. The models use only the fields listed in the next section. The pipeline uses `ml_num` to align predictions with submission rows. See [data.py](../data.py) and [run.py](../run.py).

## 2. Feature engineering

| Feature group | Processing shared by both tracks |
|---|---|
| Area | Convert a single `house_area` value directly to `square_feet`. For a range, use the midpoint `(low + high) / 2`. If the upper bound cannot be parsed, use the lower bound in its place. `area_is_range = int("-" in str(house_area))`. |
| Property configuration | Keep `bedrooms`, `washrooms`, `parking_total`, and `garage`. Add `garage_gt_parking = int(garage > parking_total)`. |
| Lot | Keep `lot_front` and `lot_depth`. Add `lot_area = lot_front × lot_depth`. Set `lot_missing = 1` if either dimension is missing. |
| Location and time | Keep `latitude` and `longitude`. `months_since_2021 = (date_listed - 2021-01-01).days / 30.44`. |
| Categorical | Use `house_type_name`, `house_style`, `municipality`, and `community`. Learn the category levels for each fold from its training records only. |

**Off-market** converts the four categorical features to the pandas `category` dtype so HGB can handle them natively. Categories unseen during training are treated as missing at inference. The final input has **29 columns** ordered as 14 shared numeric features, 4 categorical features, and 11 remarks flags. See [features.py](../features.py) and [models.py](../models.py).

**List-aware** appends `list_price` to the shared numeric features. It encodes the four categorical features with `OneHotEncoder(categories="auto", drop=None, handle_unknown="ignore", sparse_output=False, dtype=float64)`. An unseen category produces an all-zero one-hot block for that field. The `ColumnTransformer` passes through the numeric columns and uses `sparse_threshold=0.0` and `remainder="drop"`. The four categorical fields have 6, 16, 5, and 145 levels, respectively, in the final training set. HGB therefore receives **187 columns**, consisting of 15 numeric features and 172 one-hot columns. Both tracks pass numeric features directly to HGB, which handles `NaN` values. The configuration sets imputation and scaling to `null`. See [models.py](../models.py) and [frozen_models.json](../config/frozen_models.json).

### Off-market's 11 remarks flags

Replace missing `remarks` with an empty string and convert the text to lowercase. Each fixed regex produces `1` if it matches at least once and `0` otherwise, with output dtype `int8`. Each flag indicates whether the text mentions the corresponding attribute. The table lists the flags in model input order and gives example matches.

| Flag | Example matches |
|---|---|
| `remark_renovated` | `renovated`, `turnkey`, `upgraded` |
| `remark_needs_work` | `as-is`, `fixer-upper`, `needs repairs`, `dated kitchen` |
| `remark_updated_systems` | `new windows`, `replaced roof`, `upgraded hvac` |
| `remark_finished_basement` | `finished basement`, `renovated bsmt` |
| `remark_separate_entrance` | `separate entrance`, `private side entrance` |
| `remark_view` | `panoramic views`, `lake view` |
| `remark_corner_or_split_layout` | `corner unit`, `split-bedroom` |
| `remark_building_amenities` | `concierge`, `guest suites`, `fitness centre` |
| `remark_upper_floor` | `high floor`, `penthouse` |
| `remark_walkout_or_open_plan` | `walk-out`, `open concept` |
| `remark_aged_system` | `older furnace`, `original wiring`, `outdated plumbing` |

`remark_aged_system` uses the regex below. All 11 patterns are recorded under `preprocessing.remarks.flags` and `preprocessing.condition_remarks.flags` in the frozen configuration. The final version sets `market_trend=false`, weights training samples equally, and excludes local comparables from the feature set.

```text
\b(?:older|old|original|dated|aging|ageing|outdated) (?:furnace|roof|windows|electrical|wiring|plumbing|hvac|mechanicals)\b
```

See [hgb_v3_candidate.py](../analysis/hgb_v3_candidate.py), [run_hgb_condition_flags.py](../analysis/run_hgb_condition_flags.py), and [frozen_models.json](../config/frozen_models.json) for the implementation.

## 3. Training algorithm and frozen parameters

Let `P` be the actual sale price, `A` the parsed `square_feet`, `L` the `list_price`, and `f(x)` the HGB prediction on the transformed target scale. `ln` denotes the natural logarithm.

| | Off-market | List-aware |
|---|---|---|
| Release version | `hgb_v3_aged_system` | `hgb_v1` |
| Estimator target | `ln(P) - ln(A)`, or `ln(P / A)` | `ln(P) - ln(L)`, or `ln(P / L)` |
| Conversion to sale price | `exp(f(x) + ln(A))` | `exp(f(x) + ln(L))` |
| `max_iter` | `1600` | `400` |
| `max_leaf_nodes` | `127` | `63` |
| `min_samples_leaf` | `20` | `50` |

HGB first groups numeric feature values into histogram bins, then starts gradient boosting from the mean of the transformed target. In round `m`, tree `h_m` fits the negative gradient at the current prediction. Under squared-error loss, the negative gradient is `z - F_(m-1)(x)`, where `z` is the target in the table. The update is `F_m(x) = F_(m-1)(x) + 0.1 × h_m(x)`. Tree splits use binned statistics. Splits on native categorical features partition their levels into sets. In each fold, the pipeline fits preprocessing and all boosting rounds from scratch. Training minimizes squared error on the transformed target, while the report computes MdAPE and MAPE after converting predictions back to CAD prices.

| Estimator settings shared by both tracks | Frozen values |
|---|---|
| Objective and shrinkage | `loss="squared_error"`, `learning_rate=0.1`, `l2_regularization=0.0` |
| Binning and feature selection | `max_bins=255`, `max_features=1.0`, `categorical_features="from_dtype"` |
| Tree constraints | `max_depth=None`, `interaction_cst=None`, `monotonic_cst=None` |
| Fit controls | `random_state=42`, `early_stopping=False`, `warm_start=False`, `verbose=0` |
| Other serialized settings | `quantile=None`, `n_iter_no_change=10`, `scoring="loss"`, `tol=1e-7`, `validation_fraction=0.1` |

With `early_stopping=False`, the model fits on the entire training fold and `max_iter` determines the number of rounds. For Off-market, `options.target_mode="log_price"` is an argument to the base class constructor, while `candidate.price_target="log_ppsf"` determines the target used for fitting. Both `sold_price` and the corresponding offset input must be positive and finite. Backtests keep the original floating-point predictions. The pipeline applies `np.round` only when writing the final submissions. See [run.py](../run.py), [models.py](../models.py), and [hgb_v3_candidate.py](../analysis/hgb_v3_candidate.py).

## 4. Temporal backtests and expected metrics

**Walk-forward** splits the records into months by `date_sold`. With the first day of each month as the cutoff, the pipeline refits both tracks on records with `date_sold < cutoff` and predicts the sale prices of homes sold that month. It pools individual errors across months before computing the metrics.

| Validation month | Training rows | Validation rows per track |
|---|---:|---:|
| 2025-12 | 139,931 | 1,238 |
| 2026-01 | 141,169 | 1,058 |
| 2026-02 | 142,227 | 1,442 |
| 2026-03 | 143,669 | 1,816 |
| 2026-04 | 145,485 | 2,224 |
| 2026-05 | 147,709 | 2,291 |

**Fixed-origin** reuses the fit with the 2026-02-01 cutoff. Each track is fitted once on 142,227 earlier sales, then predicts 7,773 sales from February through May 2026. The fitted models and category levels stay fixed over the four months, mirroring the single fit used for the hidden test. See [run.py](../run.py).

For each sale, `signed percentage error = 100 × (prediction - sold_price) / sold_price`. `APE` is its absolute value. `MdAPE` is the median of the individual APEs, and `MAPE` is their mean. Brackets show 95% percentile bootstrap intervals. For each interval, the bootstrap uses seed `0` and 1,000 resamples of individual homes with replacement, then takes the 2.5th and 97.5th percentiles.

| Protocol | Track | N | MdAPE | MAPE | Median signed error |
|---|---|---:|---:|---:|---:|
| Walk-forward, 2025-12 to 2026-05 | Off-market | 10,069 | 7.64% [7.45%, 7.84%] | 10.27% [10.08%, 10.49%] | −0.12% |
| Walk-forward, 2025-12 to 2026-05 | List-aware | 10,069 | 4.29% [4.18%, 4.40%] | 5.36% [5.28%, 5.45%] | +0.62% |
| Fixed-origin, 2026-02 to 2026-05 | Off-market | 7,773 | 8.24% [8.02%, 8.50%] | 10.66% [10.45%, 10.89%] | −2.84% |
| Fixed-origin, 2026-02 to 2026-05 | List-aware | 7,773 | 4.24% [4.13%, 4.38%] | 5.42% [5.32%, 5.53%] | −0.57% |

On the same 7,773 homes sold from February through May, Off-market MdAPE rises from 7.64% under walk-forward to 8.24% under fixed-origin, while List-aware MdAPE changes from 4.39% to 4.24%. Both historical windows were examined during development, and the fixed-origin backtest uses a single recent origin. The bootstrap samples individual homes, so the intervals omit shared market fluctuations among homes sold in the same month. See the [walk-forward metrics](validation_report.json) and [fixed-origin metrics](horizon_report.json).

These splits measure future-sale prediction. Of the 10,069 walk-forward homes, 6,465 (64.2%) were listed before their respective sale-month cutoffs. To measure valuation performance on the day of listing, the training split must use the records available at each home's valuation time.

### Selection of the aged-system revision

In the development comparison, the original v3 leaf20 model already used log PPSF and 10 remarks flags. Adding `remark_aged_system` alone lowered rolling MdAPE from 7.98% to 7.91% for June–November 2025. In the pooled fixed-origin evaluation, which forecast four months from each of the 2025-04-01 and 2025-08-01 origins, MdAPE rose from 8.08% to 8.11%. For the 766 homes whose remarks mentioned aged systems, MdAPE fell from 9.85% to 8.96%, and median signed error fell from +6.01% to +2.84%. The release retained the flag, with gains concentrated in this subgroup and the rolling development results. See the [model selection record](../MODEL_OVERVIEW.md).

## 5. Reproduction from scratch and verification

Use **Python 3.12** and install the pinned dependencies in `requirements.txt`. The original Off-market run used macOS arm64 and Python 3.12.5. The core versions are `scikit-learn==1.7.2`, `numpy==2.2.6`, `pandas==2.3.3`, `scipy==1.18.1`, `pyarrow==22.0.0`, `joblib==1.6.0`, and `matplotlib==3.10.9`. Set `OMP_NUM_THREADS=4` and `OPENBLAS_NUM_THREADS=1`, and preserve the original row order in the Parquet files. See [requirements.txt](../requirements.txt) and [README.md](../README.md).

Run the following commands from the `final_HGB` root. Retraining in a temporary copy preserves the CSVs, models, and backtest reports in the handoff package. By default, the loader downloads and caches the three Parquet files from the [organizer's dataset](https://huggingface.co/datasets/Baliyang/SSU_data_dash).

```sh
(
    set -e
    WORK_DIR="$(mktemp -d)"
    cp -R "$PWD" "$WORK_DIR/final_HGB"
    cd "$WORK_DIR/final_HGB"
    printf 'Reproduction directory %s\n' "$PWD"

    python3.12 -m venv "$WORK_DIR/venv"
    . "$WORK_DIR/venv/bin/activate"
    python -m pip install -r requirements.txt
    unset DATA_DIR
    export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1

    python run.py validate
    python run.py report
    python run.py predict
    python run.py check
    shasum -a 256 -c submissions/SHA256SUMS.txt
)
```

If you already have all three original Parquet files locally, replace `unset DATA_DIR` inside the block above with:

```sh
export DATA_DIR="/absolute/path/to/competition_data"
```

See the [RF baseline](rf_baseline/README.md) for the Random Forest comparison configuration, results, and reproduction commands.

| Command | Outputs and verification |
|---|---|
| `predict` | Verifies the SHA-256 hashes of the three input files, trains both models from scratch, and writes `outputs/fitted_models/{off_market,list_aware}.joblib` and the two `submissions/submission_*.csv` files. Before writing to the release paths, it reloads the Joblib models and compares rounded predictions row by row. |
| `check` | Both CSVs should show `OK`. It verifies the `ml_num,predicted_price` columns, complete and unique ID coverage, and positive, finite predictions. |
| `shasum` | Both files should show `OK`. The `check` command verifies submission format and ID coverage. The SHA-256 check compares file contents against [SHA256SUMS.txt](../submissions/SHA256SUMS.txt). |
| `validate` | Verifies the training Parquet's SHA-256 and generates `outputs/validation_predictions.csv` and `outputs/horizon_predictions.csv`. The files should contain 20,138 and 15,546 data rows, respectively, across both tracks. |
| `report` | Reads the predictions from both backtests, recomputes `reports/validation_report.json` and `reports/horizon_report.json`, and generates the walk-forward metrics table and `figures/error_by_price_range.png`. Compare the metrics with Section 4. |

### Final v3 release reproduction results

Both tracks were refitted on all 150,000 sales using macOS arm64, Python 3.12.5, and the pinned dependencies. Both submission CSVs passed the submission and SHA-256 checks and matched the release byte for byte. The original and refitted models also produced identical unrounded predictions on all 8,038 hidden-test inputs.

After rerunning `validate` and `report`, both backtest prediction CSVs and both metrics JSON files matched the release byte for byte. The regenerated Joblib files had different SHA-256 hashes. Reproduction is verified using prediction values and submission files.
