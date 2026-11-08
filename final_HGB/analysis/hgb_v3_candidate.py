"""Off-market HGB v3 ablations on text, log PPSF and local market trends."""
from time import perf_counter

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.exceptions import NotFittedError

from analysis.hgb_local_comps import TIERS
from analysis.hgb_market_trend import LocalMarketTrend, TREND_FEATURES, TREND_MIN_COMPS, TREND_WINDOW_DAYS
from features import square_feet
from models import HistPriceModel


REMARK_PATTERNS = {
    "remark_renovated": r"\b(?:renovated|turnkey|upgraded)\b",
    "remark_needs_work": (
        r"\b(?:as[- ]is|fixer[- ]upper|tlc|needs? (?:work|updating|repairs?)|"
        r"needs? (?:a |some )?(?:full )?(?:redo|renovation)|"
        r"dated (?:kitchen|bath(?:room)?|finishes)|worn (?:floors?|flooring)|cosmetic upgrades)\b"
    ),
    "remark_updated_systems": r"\b(?:new|replaced|upgraded)\s+(?:windows|roof|furnace|hvac)\b",
    "remark_finished_basement": r"\b(?:finished|renovated)\s+(?:basement|bsmt)\b",
    "remark_separate_entrance": r"\b(?:separate|side|private)\s+(?:side\s+)?entrance\b",
    "remark_view": r"\b(?:unobstructed|panoramic|skyline|lake)\s+views?\b",
    "remark_corner_or_split_layout": r"\b(?:corner\s+(?:unit|suite|lot)|split[- ]bedroom)\b",
    "remark_building_amenities": r"\b(?:concierge|guest suites|rooftop terrace|indoor pool|fitness centre|gym)\b",
    "remark_upper_floor": r"\b(?:high floor|upper floor|penthouse|top floor)\b",
    "remark_walkout_or_open_plan": r"\b(?:walk[- ]?out|open[- ]concept)\b",
}
REMARK_FEATURES = tuple(REMARK_PATTERNS)


def remark_features(homes):
    """Extract the fixed pilot cues without fitting a vocabulary."""
    text = homes["remarks"].fillna("").str.lower()
    return pd.DataFrame({name: text.str.contains(pattern, regex=True).astype("int8")
                         for name, pattern in REMARK_PATTERNS.items()}, index=homes.index)


class HGBV3Candidate(HistPriceModel):
    """Keep the native HGB pipeline and add only the selected ablation blocks."""

    def __init__(self, options, cutoff, *, remarks=False, price_target="log_price", market_trend=False):
        super().__init__(**options)
        if (self.track, self.encoding, self.target_mode) != ("off_market", "native", "log_price"):
            raise ValueError("HGBV3Candidate requires native Off-market log-price options")
        self.cutoff = pd.Timestamp(cutoff)
        if pd.isna(self.cutoff) or self.cutoff.tzinfo is not None:
            raise ValueError("A valid timezone-naive training cutoff is required")
        if price_target not in ("log_price", "log_ppsf"):
            raise ValueError(f"Unknown v3 price target: {price_target}")
        self.remarks = bool(remarks)
        self.price_target = price_target
        self.market_trend = bool(market_trend)
        self.trend_index_ = None

    def configuration(self):
        config = super().configuration()
        schema = config["schema"]
        if self.remarks:
            schema["raw_features"].append("remarks")
            schema["numeric_features"].extend(REMARK_FEATURES)
            schema["input_feature_order"].extend(REMARK_FEATURES)
            config["preprocessing"]["remarks"] = {
                "extractor": "analysis.hgb_v3_candidate.remark_features",
                "normalization": "fill missing with empty string and lowercase",
                "flags": dict(REMARK_PATTERNS),
                "dtype": "int8",
            }
        if self.market_trend:
            schema["numeric_features"].extend(TREND_FEATURES)
            schema["input_feature_order"].extend(TREND_FEATURES)
            config["preprocessing"]["market_trend"] = {
                "class": "analysis.hgb_market_trend.LocalMarketTrend",
                "reference_source": "fitted model's training fold only",
                "query_identity_column": "ml_num",
                "as_of": "min(date_listed, training cutoff)",
                "sale_availability": "date_sold strictly before as_of",
                "window_days_each": TREND_WINDOW_DAYS,
                "minimum_comps_in_each_window": TREND_MIN_COMPS,
                "trend": "log(median recent PPSF) - log(median prior PPSF)",
                "tiers": [list(keys) for keys in TIERS],
            }
        if self.price_target == "log_ppsf":
            config["target"] = "natural_log_sold_price_minus_natural_log_square_feet"
            config["prediction_log_offset"] = "log(square_feet(house_area))"
        config["candidate"] = {
            "class": "analysis.hgb_v3_candidate.HGBV3Candidate",
            "cutoff": self.cutoff.isoformat(),
            "remarks": self.remarks,
            "price_target": self.price_target,
            "market_trend": self.market_trend,
        }
        return config

    def prepare_features(self, homes, fit=False):
        matrix = super().prepare_features(homes, fit=fit)
        extra = []
        if self.remarks:
            extra.append(remark_features(homes))
        if self.market_trend:
            if self.trend_index_ is None:
                raise NotFittedError("Fit the training-fold market trend index before transforming homes")
            trend = self.trend_index_.transform(homes, self.cutoff)
            if tuple(trend.columns) != TREND_FEATURES or not trend.index.equals(homes.index):
                raise ValueError("Market trend features must match input row order and feature schema")
            extra.append(trend)
        if extra:
            matrix = pd.concat([matrix, *extra], axis=1)
        if fit:
            # The base HGB preprocessor still uses its own native feature_order_.
            self.model_feature_order_ = matrix.columns.tolist()
        return matrix

    def _log_offset(self, homes):
        if self.price_target == "log_price":
            return 0.0
        area = square_feet(homes["house_area"]).to_numpy(dtype=float, na_value=np.nan)
        valid = np.isfinite(area) & (area > 0)
        if not valid.all():
            raise ValueError("square_feet must be positive and finite for log_ppsf; invalid row indices: "
                             f"{homes.index[~valid].tolist()[:10]}")
        return np.log(area)

    def fit(self, train):
        if train.empty:
            raise ValueError("HGB training table must not be empty")
        if is_numeric_dtype(train["date_sold"]):
            raise ValueError("date_sold must contain dates")
        dates = pd.to_datetime(train["date_sold"])
        if dates.dt.tz is not None or dates.isna().any() or not dates.lt(self.cutoff).all():
            raise ValueError("Training sales must have valid dates strictly before the training cutoff")
        target = np.log(self._positive_prices(train, "sold_price")) - self._log_offset(train)
        started = perf_counter()
        if self.market_trend:
            self.trend_index_ = LocalMarketTrend(train)
        matrix = self.prepare_features(train, fit=True)
        self.fit_preprocessing_seconds_ = perf_counter() - started
        started = perf_counter()
        self.estimator.fit(matrix, target)
        self.fit_estimator_seconds_ = perf_counter() - started
        return self
