# SSU Data Dash 2026 · Final v3 slides brief

The five-minute story is built around two valuation scenarios, then uses the temporal split and the errors on high-priced homes to explain when the model can be trusted.

## 1. Two questions, two use cases · 0:00–0:40

| Track | User's question | Actual model output |
|---|---|---|
| **Off-market** | **What is this home worth?** Before setting an asking price, the seller wants to know roughly how much the home could sell for. | Use the property, location, date, and description provided by the competition to predict the home's sale price. The model learns log price per square foot, then multiplies the resulting price per square foot by the area. |
| **List-aware** | **What will this listing close at?** After seeing the asking price, the buyer wants to estimate the likely sale price. | Use the asking price as an anchor and combine it with property, location, and date information to predict the sold-to-list ratio. Then multiply the ratio by the asking price. |

Both tracks produce point estimates. Off-market means valuing the home without the asking price while still using the competition-provided `date_listed` and `remarks`. A real pre-listing deployment would require a valuation date and a property description available at that time. The competition data consist of synthetic Toronto sales, so the geographic findings apply only to this dataset. Hidden-test accuracy is worth 20 points for Off-market and 10 points for List-aware.

Optional opening example. Use the condo whose actual sale price was at the median of the six-month validation set (`SYN-0045873`) to illustrate the two sets of inputs. The Off-market estimate is CAD 680,308. With an asking price of CAD 816,200, the List-aware estimate is CAD 772,456. The actual sale price was CAD 805,300. This home illustrates the outputs. See Section 4 for overall performance.

## 2. Features and the two model targets · 0:40–1:40

| Feature block | Fields used |
|---|---|
| Home size and features | `square_feet` (midpoint for an area range), `area_is_range`, `bedrooms`, `washrooms`, `parking_total`, `garage`, `garage_gt_parking` |
| Lot | `lot_front`, `lot_depth`, `lot_area = lot_front × lot_depth`, `lot_missing` |
| Location and time | `latitude`, `longitude`, `months_since_2021` (calculated from `date_listed`) |
| Categorical | `house_type_name`, `house_style`, `municipality`, `community` |
| Off-market only | 11 binary regex flags extracted from lowercased `remarks` |
| List-aware only | `list_price`, used both as a feature and as the anchor for recovering the sale price |

The 11 Off-market flags are renovated, needs work, updated systems, finished basement, separate entrance, view, corner or split layout, building amenities, upper floor, walkout or open plan, and aged system. For example, “original wiring” triggers the aged-system flag. A flag of 0 means the rule found no matching phrase. Off-market has 29 model inputs. List-aware has 19 input fields before one-hot encoding, and the final fitted model uses 187 columns.

Both tracks use `HistGradientBoostingRegressor`. It fits trees sequentially to capture nonlinear interactions among size, property type, location, and condition, and it handles missing numeric values directly. The category vocabulary is fitted only on the training set for the current period. Unseen categories are treated as missing in Off-market. In List-aware, an unseen category is represented by an all-zero one-hot block for that field. Both feature whitelists exclude `ml_num`, `date_sold`, `days_on_market`, and external AVM. The final Off-market model uses home area as the offset, and List-aware uses the asking price as the offset.

| Frozen parameters | Off-market · `hgb_v3_aged_system` | List-aware · `hgb_v1` |
|---|---|---|
| Target | `log(sold_price / square_feet)` | `log(sold_price / list_price)` |
| Dollar prediction | `square_feet × exp(model_output)` | `list_price × exp(model_output)` |
| Categorical encoding | Native categories | Dense one-hot |
| `max_iter` | 1600 | 400 |
| `max_leaf_nodes` | 127 | 63 |
| `min_samples_leaf` | 20 | 50 |
| Shared settings | Squared error on transformed target, learning rate 0.10, max bins 255, L2 0, max depth unset, all features available at splits, random seed 42, early stopping off | Same |

See the [frozen config](config/frozen_models.json) for the full parameters and regex patterns.

## 3. Why we chose these two models · 1:40–2:20

Development walk-forward used Jun–Nov 2025. Off-market also used Apr 1 and Aug 1 as fixed origins, forecasting four months from each.

1. **Start with simple baselines for comparison.** Nearest earlier sales had an Off-market MdAPE of 20.84%, and the list-price-only baseline had an MdAPE of 7.26%. On the same Dec–May validation homes, the final HGB choice scored 4.29%. The comparison covers the whole pipeline, including feature representation, target, and estimator.
2. **Choose the encoding and target for each track.** The HGB development comparisons supported native categories for Off-market and one-hot encoding for List-aware. Using the same one-hot HGB parameters to compare List-aware targets, log sold-to-list ratio had an MdAPE of 4.264%, while directly predicting log sale price had an MdAPE of 4.317%. With the asking price available, the model mainly learns which kinds of homes sell above or below ask.
3. **Increase Off-market model capacity, then add condition information and area normalization.** V2 retained a larger HGB (127 leaves, 1600 rounds). Explicit local comparables, recency weighting, and the subsequent 90-day market trend did not improve both rolling and fixed-horizon MdAPE in their respective matched ablations. After v3 added 10 remarks flags, development rolling/fixed MdAPE dropped from 8.63%/8.89% to 8.05%/8.20%. Switching to the log PPSF target brought it down to 7.98%/8.08%. Area already provides the price scale, so the trees can focus on learning how price per square foot changes with property type, location, home features, and condition.
4. **Freeze the aged-system revision.** In fixed-horizon development, MdAPE for homes whose remarks mentioned aged systems (n=766) dropped from 9.85% to 8.96%, and median overprediction fell from +6.01% to +2.84%. With this flag added, overall development rolling MdAPE improved (7.98%→7.91%) and fixed-horizon MdAPE worsened slightly (8.08%→8.11%). The choice was based on the improvement for this subgroup and the final version decision. The previously examined Dec–May rolling and Feb–May fixed results changed from 7.72% to 7.64% and from 8.33% to 8.24%, respectively, and should be labelled as reused validation.

## 4. How validation error predicts hidden-test error · 2:20–3:25

**Official reporting protocol.** Monthly walk-forward from Dec 2025 to May 2026. Each month's fit uses only sales completed before the first day of that month and generates out-of-sample predictions for the month. For each track, take the median of the pooled APEs for all 10,069 homes across the six months. `APE = 100 × abs(predicted_price − sold_price) / sold_price`. MdAPE is the median and MAPE is the mean. A median signed error below 0 indicates a tendency to underpredict.

| Protocol | Homes per track | Off-market MdAPE / MAPE | List-aware MdAPE / MAPE |
|---|---:|---:|---:|
| Official walk-forward · Dec–May | 10,069 | **7.64% / 10.27%** | **4.29% / 5.36%** |
| Walk-forward · same Feb–May homes | 7,773 | 7.64% | 4.39% |
| Fixed origin · one fit before Feb 1, covering Feb–May | 7,773 | **8.24% / 10.66%** | **4.24% / 5.42%** |

**Hidden-test proxy.** The final fit uses the 150,000 historical sales through May 31 to predict Jun–Sep 2026, with no new labels during those four months. The Feb–May fixed-origin evaluation is closer to this deployment cadence. After freezing the fit, Off-market MdAPE rose by 0.60 percentage points and median signed error shifted from −0.52% to −2.84%. In the same window, List-aware MdAPE was less sensitive to whether the model was updated, which is consistent with the asking price providing an anchor. If the presentation needs an approximate hidden-test error, **about 8.2% / 4.2% is the historical reference closest to the submission cadence**. Hidden labels have yet to be released. The backtest has only one recent four-month origin, and the team has already examined this validation period. Changes in the market and sample mix may cause the actual scores to differ.

**Timing boundary.** The design sets training cutoffs by sale month. In Dec–May, 64.2% of homes were already listed before that month's training cutoff, so some historical sales known at training time may have occurred after those homes' individual listing dates. These results measure performance under the competition's future-sale protocol. A strict listing-day valuation evaluation would need to split the data at each home's actual valuation time.

## 5. Where does MdAPE fail? · 3:25–4:35

### Sold-price decile

[Decile figure ready to use in slides](report/error_by_price_decile.png)

| From lowest to highest price | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Off-market MdAPE | 6.39% | 5.65% | 5.51% | 5.90% | 7.07% | 7.67% | 8.33% | 9.65% | 10.60% | **14.08%** |
| List-aware MdAPE | 3.72% | 3.56% | 3.64% | 3.71% | 3.97% | 4.66% | 4.66% | 4.80% | 5.54% | **5.47%** |

- **The high-price tail is the main Off-market weakness.** D10 starts at $1.761M (n=1,007), with an MdAPE of 14.08%, a median signed error of −8.56%, and a 90th-percentile APE of 32.26%. D1 (n=1,007) tends toward overprediction, with a median signed error of +5.09%. These directions are consistent with predictions shrinking toward typical prices. Possible reasons include uncommon combinations of property characteristics and high-end premiums that the descriptions do not fully convey. The existing comparisons cannot distinguish between these two mechanisms.
- **Errors also vary by neighbourhood.** Across the 90 communities with at least 50 sales each, the 10th–90th percentile of Off-market community MdAPE was 6.06%–9.71%. Examples include Cypressglen at 12.74% (n=54) and Hazelfield at 12.32% (n=68). In all 36 communities with at least 8 D10 sales and 20 D2–D9 sales, D10 MdAPE was higher. The median gap was 5.96 percentage points. This group comparison shows that the high-price issue occurs across communities, and property mix may still account for part of it.
- **Property type.** Off-market MdAPE was 10.13% for Detached (n=3,123) and 6.30% for Condo Apt (n=4,843). Within Detached, D10 still had an MdAPE of 13.67% (n=775), compared with 9.18% for D2–D9 (n=2,347).
- **The List-aware high-price tail is much milder.** D10 MdAPE was 5.47%, median signed error was −0.01%, and 90th-percentile APE was still 15.28%. Across the full sample, 12.16% of Off-market homes had an APE ≥20%, compared with 1.27% for List-aware. MdAPE describes only the middle error. These additional metrics are needed to convey tail risk.

These slices group homes retrospectively by actual `sold_price` for diagnosis. For users, estimates for high-priced or unusual homes can help set initial expectations. Pricing decisions still need to incorporate comparable sales and human judgment.

## 6. Closing · 4:35–5:00

> We answer two questions with two anchors. Without an asking price, we estimate value per square foot from the home, its location and its condition. With an asking price, we estimate the closing-price adjustment. A four-month backtest gives us a realistic error benchmark, and the luxury tail is where we would be most cautious.

When making the slides, prioritize a single visual for both tracks' inputs and formulas, a timeline for rolling and fixed-origin validation, and a decile line chart for the high-price tail. All error figures in the charts come from the team's own validation.

## Evidence and reproduction

- `python run.py validate` reruns both temporal splits. The generated `outputs/validation_predictions.csv` and `outputs/horizon_predictions.csv` contain `ml_num`, `community`, property type, actual, prediction, month, and cutoff, which can be used to recompute the grouped results. See the [README](README.md) for the commands.
- The [archived HGB experiment records](reports/hgb_selection/) include the Off-market [release revision](reports/hgb_selection/hgb_v3/current_revision.json) and [backtest](reports/hgb_selection/hgb_v3_aged_system/backtest_summary.json), plus the [List-aware configuration and results](reports/hgb_selection/hgb_v1/). The [v3 selection record](reports/hgb_selection/hgb_v3/frozen_config.json) and [condition-flag comparisons](reports/hgb_selection/hgb_v3_condition_flags/) document the Off-market model selection.
