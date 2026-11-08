"""Off-market HGB with optional historical comparables and recency weights."""
from time import perf_counter

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from analysis.hgb_local_comps import MIN_COMPS, SITE, SITE_TYPES, TIERS, ComparableSales, feature_names
from models import HistPriceModel


class OffMarketCandidate(HistPriceModel):
    """Use the frozen HGB model interface with an explicit training cutoff.

    ``options`` are the native Off-market log-price HistPriceModel options.
    An optional comparable index may be shared across candidates for the same
    training fold. A fitted instance retains that index for standalone inference.
    """

    def __init__(self, options, cutoff, comp_mode="none", half_life_months=None):
        super().__init__(**options)
        if (self.track, self.encoding, self.target_mode) != ("off_market", "native", "log_price"):
            raise ValueError("OffMarketCandidate requires native Off-market log-price options")
        self.cutoff = pd.Timestamp(cutoff)
        if pd.isna(self.cutoff) or self.cutoff.tzinfo is not None:
            raise ValueError("A valid timezone-naive training cutoff is required")
        self.comp_mode = comp_mode
        self.comp_feature_names = () if comp_mode == "none" else tuple(feature_names(comp_mode))
        if half_life_months is not None:
            half_life_months = float(half_life_months)
            if not np.isfinite(half_life_months) or half_life_months <= 0:
                raise ValueError("half_life_months must be positive and finite")
        self.half_life_months = half_life_months
        self.comp_index_ = None

    def configuration(self):
        config = super().configuration()
        numeric = [*config["schema"]["numeric_features"], *self.comp_feature_names]
        config["schema"]["numeric_features"] = numeric
        config["schema"]["input_feature_order"] = numeric + self.categorical_features
        config["candidate"] = {
            "class": "analysis.hgb_off_market_candidate.OffMarketCandidate",
            "cutoff": self.cutoff.isoformat(),
            "comp_mode": self.comp_mode,
            "half_life_months": self.half_life_months,
        }
        if self.comp_feature_names:
            comparable_config = {
                "class": "analysis.hgb_local_comps.ComparableSales",
                "feature_names": list(self.comp_feature_names),
                "reference_source": "fitted model's training fold only",
                "query_identity_column": "ml_num",
                "as_of": "min(date_listed, training cutoff)",
                "sale_window": "[as_of minus 12 calendar months, as_of)",
                "price_statistic": "median(sold_price / square_feet(house_area))",
                "age_statistic": "days from latest qualifying sale to as_of",
                "minimum_count_for_price_and_age": MIN_COMPS,
                "tiers": [list(group) for group in TIERS],
            }
            if self.comp_mode == "community_and_site":
                comparable_config["site_keys"] = list(SITE)
                comparable_config["site_types"] = list(SITE_TYPES)
            config["preprocessing"]["comparable_sales"] = comparable_config
        config["sample_weight"] = ("equal" if self.half_life_months is None else {
            "age_days": "(training cutoff - date_sold) in days",
            "formula": "exp2(-age_days / (half_life_months * 365.25 / 12))",
            "normalization": "divide by mean weight within training fold",
        })
        return config

    def _weights(self, dates):
        if self.half_life_months is None:
            return None
        age_days = (self.cutoff - dates).dt.total_seconds().to_numpy(dtype=float) / 86400
        half_life_days = self.half_life_months * 365.25 / 12
        weights = np.exp2(-age_days / half_life_days)
        if not (np.isfinite(weights) & (weights > 0)).all():
            raise ValueError("Recency weights must be positive and finite")
        return weights / weights.mean()

    def _comps(self, homes, comp_features):
        if comp_features is None:
            comp_features = self.comp_index_.transform(homes, self.cutoff, mode=self.comp_mode)
        if (not isinstance(comp_features, pd.DataFrame)
                or tuple(comp_features.columns) != self.comp_feature_names
                or not comp_features.index.equals(homes.index)):
            raise ValueError("Comparable features must match the home's row index, order, and feature schema")
        cutoff = self.cutoff.isoformat()
        if (comp_features.attrs.get("comp_mode", self.comp_mode) != self.comp_mode
                or comp_features.attrs.get("comp_cutoff", cutoff) != cutoff):
            raise ValueError("Comparable features were computed for a different mode or cutoff")
        if (not all(is_numeric_dtype(dtype) for dtype in comp_features.dtypes)
                or np.isinf(comp_features.to_numpy(dtype=float, na_value=np.nan)).any()):
            raise ValueError("Comparable features must be numeric and contain no infinities")
        return comp_features

    def prepare_features(self, homes, fit=False, *, comp_features=None):
        if not self.comp_feature_names and comp_features is not None:
            raise ValueError("comp_features requires an enabled comp_mode")
        base = super().prepare_features(homes, fit=fit)
        if self.comp_feature_names:
            comps = self._comps(homes, comp_features)
            order = [*self.numeric_features, *self.comp_feature_names, *self.categorical_features]
            matrix = pd.concat([base, comps], axis=1).loc[:, order]
        else:
            matrix = base
        if fit:
            # HistPriceModel uses feature_order_ internally for its base transform.
            self.model_feature_order_ = matrix.columns.tolist()
        return matrix

    def fit(self, train, *, comp_index=None, train_comp_features=None):
        if train.empty:
            raise ValueError("HGB training table must not be empty")
        if is_numeric_dtype(train["date_sold"]):
            raise ValueError("date_sold must contain dates")
        dates = pd.to_datetime(train["date_sold"])
        if dates.dt.tz is not None or dates.isna().any() or not dates.lt(self.cutoff).all():
            raise ValueError("Training sales must have valid dates strictly before the training cutoff")
        if not self.comp_feature_names and (comp_index is not None or train_comp_features is not None):
            raise ValueError("Comparable inputs require an enabled comp_mode")

        target = np.log(self._positive_prices(train, "sold_price"))
        weights = self._weights(dates)
        started = perf_counter()
        if self.comp_feature_names:
            if comp_index is None:
                comp_index = ComparableSales(train)
            if (not comp_index.source_ids.equals(pd.Index(train["ml_num"]))
                    or comp_index.latest_sale_date != dates.max()):
                raise ValueError("Comparable reference must match this exact training fold and row order")
            self.comp_index_ = comp_index
        matrix = self.prepare_features(train, fit=True, comp_features=train_comp_features)
        self.fit_preprocessing_seconds_ = perf_counter() - started
        started = perf_counter()
        if weights is None:
            self.estimator.fit(matrix, target)
        else:
            self.estimator.fit(matrix, target, sample_weight=weights)
        self.fit_estimator_seconds_ = perf_counter() - started
        return self
