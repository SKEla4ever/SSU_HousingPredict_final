# SSU Data Dash 2026 · Final v3 slides brief

五分钟叙事围绕两个估价场景展开，再用时间切分和高价房的误差回答什么时候可以信任模型。

## 1. 两个问题，两种使用场景 · 0:00–0:40

| Track | 用户的问题 | 模型实际输出 |
|---|---|---|
| **Off-market** | **What is this home worth?** 卖家在确定 asking price 前想知道大概能卖多少。 | 以竞赛提供的房屋、位置、日期和描述为输入，预测这套房的 sale price。模型学习 log price per square foot，再乘回面积。 |
| **List-aware** | **What will this listing close at?** 买家看到 asking price 后，想判断可能的成交价。 | 以 asking price 为 anchor，结合房屋、位置和日期预测 sold-to-list ratio，再把 ratio 乘回 asking price。 |

两条 track 都是 point estimates。Off-market 的含义是估价时不用 asking price，仍使用竞赛提供的 `date_listed` 和 `remarks`。上线到真实 pre-listing 场景时，需要当时可得的估价日期与房屋描述。竞赛数据为 synthetic Toronto sales，地理结果只解释这份数据。Off-market 的 hidden-test accuracy 权重为 20 分，List-aware 为 10 分。

可选开场例子。以六个月 validation 中实际成交价处在 median 的 condo（`SYN-0045873`）说明两种输入。Off-market 估 CAD 680,308。给出 CAD 816,200 的 asking price 后，List-aware 估 CAD 772,456。实际成交 CAD 805,300。这一套房用来说明输出，整体效果见第 4 节。

## 2. Features 和两个模型的 targets · 0:40–1:40

| Feature block | 实际使用的字段 |
|---|---|
| 房屋面积与配置 | `square_feet`（面积区间取 midpoint）、`area_is_range`、`bedrooms`、`washrooms`、`parking_total`、`garage`、`garage_gt_parking` |
| Lot | `lot_front`、`lot_depth`、`lot_area = lot_front × lot_depth`、`lot_missing` |
| 位置与时间 | `latitude`、`longitude`、`months_since_2021`（从 `date_listed` 算） |
| Categorical | `house_type_name`、`house_style`、`municipality`、`community` |
| Off-market 专用 | 11 个 binary regex flags，从 lowercase `remarks` 提取 |
| List-aware 专用 | `list_price`，同时作为 feature 和恢复成交价的 anchor |

Off-market 的 11 个 flags 是 renovated、needs work、updated systems、finished basement、separate entrance、view、corner or split layout、building amenities、upper floor、walkout or open plan、aged system。例如 “original wiring” 会触发 aged system。Flag 为 0 表示规则没有匹配到词组。Off-market 共 29 个 model inputs。List-aware 在 one-hot encoding 前有 19 个输入字段，最终训练得到 187 列。

两条 track 都使用 `HistGradientBoostingRegressor`。它用逐步拟合的 trees 表达 size、property type、location 和 condition 的 nonlinear interactions，能直接处理 numeric missing values。Category vocabulary 只从当期训练集拟合。Off-market 的 unseen categories 按 missing 处理，List-aware 的 unseen category 在该字段的 one-hot block 中全为 0。两条 track 的 feature whitelist 排除了 `ml_num`、`date_sold`、`days_on_market` 和 external AVM。最终 Off-market 使用房屋面积作 offset，List-aware 使用 asking price 作 offset。

| Frozen parameters | Off-market · `hgb_v3_aged_system` | List-aware · `hgb_v1` |
|---|---|---|
| Target | `log(sold_price / square_feet)` | `log(sold_price / list_price)` |
| Dollar prediction | `square_feet × exp(model_output)` | `list_price × exp(model_output)` |
| Categorical encoding | Native categories | Dense one-hot |
| `max_iter` | 1600 | 400 |
| `max_leaf_nodes` | 127 | 63 |
| `min_samples_leaf` | 20 | 50 |
| Shared settings | Squared error on transformed target, learning rate 0.10, max bins 255, L2 0, max depth unset, all features available at splits, random seed 42, early stopping off | Same |

完整参数和 regex patterns 见 [frozen config](config/frozen_models.json)。

## 3. 为什么选这两个模型 · 1:40–2:20

Development 的 walk-forward 用 Jun–Nov 2025，Off-market 另用 Apr 1 和 Aug 1 两个 fixed origins，各预测四个月。

1. **从简单 baseline 建立参照。** Nearest earlier sales 的 Off-market MdAPE 为 20.84%，List-price-only 为 7.26%。在相同的 Dec–May validation homes 上，最后选用 4.29% 的 HGB。比较的是整套 pipeline，包含 feature representation、target 和 estimator。
2. **按 track 选择 encoding 和 target。** HGB 的 development 对比支持 Off-market 用 native categories、List-aware 用 one-hot。用同一套 one-hot HGB 参数比较 List-aware 的 targets，log sold-to-list ratio 的 MdAPE 为 4.264%，直接预测 log sale price 为 4.317%。已知 asking price 后，模型主要学习什么样的房屋会高于或低于 ask 成交。
3. **Off-market 先补充 model capacity，再补了 condition 和 area normalization。** v2 保留更大的 HGB（127 leaves、1600 rounds）。显式 local comparables、recency weighting 和后续 90-day market trend 在相应的 matched ablations 中没有同时改善 rolling 与 fixed-horizon MdAPE。v3 加入 10 个 remarks flags 后，development 的 rolling/fixed MdAPE 从 8.63%/8.89% 降到 8.05%/8.20%。再改用 log PPSF target，降到 7.98%/8.08%。面积已提供价格尺度，树可以专注学习每平方英尺价格如何随类型、位置、房屋配置和 condition 改变。
4. **冻结 aged-system revision。** 在 fixed-horizon development 中，提及老旧设备的房屋（n=766）MdAPE 从 9.85% 降至 8.96%，median overprediction 从 +6.01% 降至 +2.84%。加入这一个 flag 后，整体 development 的 rolling MdAPE 改善（7.98%→7.91%），fixed 略退（8.08%→8.11%）。选择它是基于该 subgroup 的改善和最终版本决策。此前已观察过的 Dec–May rolling / Feb–May fixed 结果分别为 7.72%→7.64% 和 8.33%→8.24%，应标为 reused validation。

## 4. Validation error 如何预测 hidden-test error · 2:20–3:25

**官方报告口径。** Dec 2025–May 2026 逐月 walk-forward。每月只用该月第一天前已成交的销售记录拟合，逐月产生 out-of-sample predictions。每条 track 将六个月共 10,069 套房的 APE 合并后取 median。`APE = 100 × abs(predicted_price − sold_price) / sold_price`，MdAPE 是 median，MAPE 是 mean。Median signed error 低于 0 表示倾向低估。

| Protocol | 样本数 / track | Off-market MdAPE / MAPE | List-aware MdAPE / MAPE |
|---|---:|---:|---:|
| 官方 walk-forward · Dec–May | 10,069 | **7.64% / 10.27%** | **4.29% / 5.36%** |
| Walk-forward · 相同的 Feb–May homes | 7,773 | 7.64% | 4.39% |
| Fixed origin · Feb 1 前 fit 一次，覆盖 Feb–May | 7,773 | **8.24% / 10.66%** | **4.24% / 5.42%** |

**Hidden-test proxy。** 最终用截至 May 31 的 150,000 笔历史成交一次 fit，预测 Jun–Sep 2026，四个月内没有新 labels。Feb–May fixed origin 更接近这个部署节奏。冻结 fit 后，Off-market 的 MdAPE 上升 0.60 percentage points，median signed error 从 −0.52% 变为 −2.84%。同一窗口内，List-aware 的 MdAPE 对是否更新模型更不敏感，与 asking price 提供 anchor 的解释一致。若台上需要给 hidden-test error 的量级，可以说 **约 8.2% / 4.2% 是最接近提交节奏的历史参考**。Hidden labels 尚未公开。回测只有一个最近的四个月 origin，这段 validation 也被团队观察过，市场和样本 mix 的变化都可能使实际分数偏离。

**Timing 边界。** 此设计按 sale month 控制训练 cutoff。Dec–May 中 64.2% 的 homes 在当月训练 cutoff 前已挂牌，因此训练时已知的部分历史成交可能晚于这些房屋各自的 listing date。上述结果衡量比赛的 future-sale protocol。严格评估 listing-day valuation 需要以每套房的实际估价时间切分。

## 5. Where does MdAPE fail? · 3:25–4:35

### Sold-price decile

[可直接用于 slides 的 decile figure](report/error_by_price_decile.png)

| 从最低价到最高价 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Off-market MdAPE | 6.39% | 5.65% | 5.51% | 5.90% | 7.07% | 7.67% | 8.33% | 9.65% | 10.60% | **14.08%** |
| List-aware MdAPE | 3.72% | 3.56% | 3.64% | 3.71% | 3.97% | 4.66% | 4.66% | 4.80% | 5.54% | **5.47%** |

- **Off-market 的首要弱点是高价尾部。** D10 从 $1.761M 起（n=1,007），MdAPE 14.08%，median signed error −8.56%，第 90 percentile APE 32.26%。D1（n=1,007）则倾向高估，median signed error +5.09%。这种方向与预测向典型价位收缩一致。可能原因包括少见的房屋组合与描述未充分表达的高端溢价，现有对比无法分离这两种机制。
- **Neighbourhood 也有差异。** 在 90 个各有至少 50 笔成交的 communities 里，Off-market community MdAPE 的 10th–90th percentile 为 6.06%–9.71%。例如 Cypressglen 12.74%（n=54），Hazelfield 12.32%（n=68）。在有至少 8 笔 D10、20 笔 D2–D9 的 36 个 communities 中，36 个的 D10 MdAPE 都更高，差距的 median 为 5.96 percentage points。这个分组对照表明高价问题跨社区出现，property mix 仍可能解释其中一部分。
- **Property type。** Off-market Detached 的 MdAPE 10.13%（n=3,123），Condo Apt 为 6.30%（n=4,843）。在 Detached 内，D10 仍有 13.67% 的 MdAPE（n=775），D2–D9 为 9.18%（n=2,347）。
- **List-aware 的高价尾部温和得多。** D10 MdAPE 5.47%，median signed error −0.01%，第 90 percentile APE 仍为 15.28%。全样本中 Off-market 有 12.16% 的 homes APE ≥20%，List-aware 为 1.27%。MdAPE 只描述中间那笔误差，尾部风险要用这些补充指标表达。

这些切片使用实际 `sold_price` 事后分组，作为 diagnosis。面向用户时，高价或特殊房屋的 estimate 可用来形成初步预期，定价决策还需要结合 comps 和人工判断。

## 6. 收尾 · 4:35–5:00

> We answer two questions with two anchors. Without an asking price, we estimate value per square foot from the home, its location and its condition. With an asking price, we estimate the closing-price adjustment. A four-month backtest gives us a realistic error benchmark, and the luxury tail is where we would be most cautious.

制作 slides 时，优先把两条 track 的输入和公式放在一张图、把 rolling 与 fixed-origin 放在一张时间轴、用一张 decile 折线图呈现高价尾部。图表上的误差数字均来自团队自己的 validation。

## Evidence and reproduction

- `python run.py validate` 重跑两种时间切分。生成的 `outputs/validation_predictions.csv` 和 `outputs/horizon_predictions.csv` 含 `ml_num`、`community`、property type、actual、prediction、month、cutoff，可复算分组结果。命令详见 [README](README.md)。
- 原始实验位于工作区 `ssu_competition_personal/SSU-Data-Dash/outputs/hgb_baselines/`。最终 Off-market 的来源为 `hgb_v3/current_revision.json` 与 `hgb_v3_aged_system/backtest_summary.json`。List-aware 来源为 `hgb_v1/{frozen_config,backtest_summary,horizon_summary}.json`。Model selection 对照见 `hgb_v1/development_summary.json`、`hgb_v3/frozen_config.json` 和 `hgb_v3_condition_flags/*/summary.json`。
