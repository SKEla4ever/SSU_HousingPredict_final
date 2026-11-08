"""Historical comparable-sale features using only labels available at valuation time."""

from datetime import date

import numpy as np
import pandas as pd

from features import square_feet


MIN_COMPS = 3
TYPE = "house_type_name"
TIERS = ((TYPE, "community"), (TYPE, "municipality"), (TYPE,))
SITE = (TYPE, "latitude", "longitude")
SITE_TYPES = ("Condo Apt", "Condo Townhouse")
COMMUNITY_FEATURES = ("comp_median_price_per_sqft", "comp_count", "comp_age_days", "comp_tier")
SITE_FEATURES = ("site_comp_median_price_per_sqft", "site_comp_count", "site_comp_age_days")


def feature_names(mode):
    """Return the ordered numeric feature schema for a comparable-sale mode."""
    if mode == "community":
        return COMMUNITY_FEATURES
    if mode == "community_and_site":
        return COMMUNITY_FEATURES + SITE_FEATURES
    raise ValueError(f"Unknown comparable-sale mode: {mode}")


def _ids(frame):
    ids = pd.Index(frame["ml_num"])
    if ids.hasnans or not ids.is_unique:
        raise ValueError("ml_num must be nonmissing and unique")
    return ids


def _dates(frame, column):
    values = frame[column]
    if pd.api.types.is_numeric_dtype(values):
        raise ValueError(f"{column} must contain dates")
    parsed = pd.to_datetime(values, errors="raise")
    if (not pd.api.types.is_datetime64_any_dtype(parsed)
            or isinstance(parsed.dtype, pd.DatetimeTZDtype) or parsed.isna().any()):
        raise ValueError(f"{column} must contain nonmissing timezone-naive dates")
    return parsed.to_numpy(dtype="datetime64[ns]")


def _cutoff_date(cutoff):
    if not isinstance(cutoff, (str, date, np.datetime64)):
        raise ValueError("cutoff must be a timezone-naive date")
    value = pd.Timestamp(cutoff)
    if pd.isna(value) or value.tzinfo is not None:
        raise ValueError("cutoff must be a nonmissing timezone-naive date")
    return value.as_unit("ns").to_datetime64()


def _site_keys(frame):
    keys = frame.loc[:, list(SITE)].copy()
    for column, limit in (("latitude", 90), ("longitude", 180)):
        values = pd.to_numeric(keys[column], errors="raise")
        keys[column] = values.where(values.between(-limit, limit))
    return keys


def _groups(frame, keys):
    by = keys[0] if len(keys) == 1 else list(keys)
    return frame.groupby(by, sort=False, observed=True, dropna=True).indices


def _index_groups(frame, keys, dates, prices_per_sqft):
    # The input is already in date order. Each group retains the same order.
    return {key: (dates[positions], prices_per_sqft[positions])
            for key, positions in _groups(frame, keys).items()}


def _window_stats(index, query, keys, start, end):
    """Summarize sales in [start, end) for each row, reusing identical windows."""
    result = np.full((len(query), 3), np.nan)
    result[:, 1] = 0
    for key, positions in _groups(query, keys).items():
        if key not in index:
            continue
        dates, prices = index[key]
        left = np.searchsorted(dates, start[positions], side="left")
        right = np.searchsorted(dates, end[positions], side="left")
        counts = right - left
        result[positions, 1] = counts
        supported = counts >= MIN_COMPS
        selected = positions[supported]
        medians = {}
        for row, lo, hi in zip(selected, left[supported], right[supported]):
            window = (lo, hi)
            if window not in medians:
                medians[window] = np.median(prices[lo:hi])
            result[row, 0] = medians[window]
        result[selected, 2] = (end[selected] - dates[right[supported] - 1]) / np.timedelta64(1, "D")
    return result


class ComparableSales:
    """Index training-fold sales and compute trailing comparable-sale features.

    Windows start 12 calendar months before ``min(date_listed, cutoff)`` and
    exclude sales at the end timestamp. Each count refers to sales with valid
    square footage. Community tiers select the first group with at least three
    comparables: community and type (1), municipality and type (2), then type
    alone (3). Tier 0 has count zero and missing price and age.

    Optional site features keep the exact-coordinate and type statistics in a
    separate block. They apply to Condo Apt and Condo Townhouse only. For an
    eligible home with fewer than three comps, the site count is available but
    the median price and age are missing. Ineligible homes and eligible homes
    without valid coordinates have missing site features throughout.
    """

    def __init__(self, sales):
        source_ids = _ids(sales)
        listed = _dates(sales, "date_listed")
        sold = _dates(sales, "date_sold")
        if (sold < listed).any():
            raise ValueError("date_sold cannot precede date_listed in comparable sales")
        price = pd.to_numeric(sales["sold_price"], errors="raise").to_numpy(dtype=float, na_value=np.nan)
        if not (np.isfinite(price) & (price > 0)).all():
            raise ValueError("sold_price must be positive and finite in comparable sales")
        self._listing_dates = pd.Series(listed, index=source_ids)
        self.latest_sale_date = pd.Timestamp(sold.max()) if len(sold) else pd.NaT

        area = square_feet(sales["house_area"]).to_numpy(dtype=float, na_value=np.nan)
        valid_area = np.flatnonzero(np.isfinite(area) & (area > 0))
        order = valid_area[np.argsort(sold[valid_area], kind="stable")]
        ordered = sales.iloc[order]
        dates = sold[order]
        prices_per_sqft = price[order] / area[order]
        if not np.isfinite(prices_per_sqft).all():
            raise ValueError("Comparable sold price per square foot must be finite")
        self._tiers = [_index_groups(ordered, keys, dates, prices_per_sqft) for keys in TIERS]
        self._site = None
        if set(SITE).issubset(sales.columns):
            site = _site_keys(ordered)
            eligible = site[TYPE].isin(SITE_TYPES).to_numpy()
            self._site = _index_groups(site.iloc[eligible], SITE, dates[eligible], prices_per_sqft[eligible])

    @property
    def source_ids(self):
        """The source IDs in the input training order."""
        return self._listing_dates.index.copy(deep=True)

    def transform(self, homes, cutoff, mode="community"):
        """Return the numeric comparables frame in the input row order and index."""
        columns = feature_names(mode)
        cutoff = _cutoff_date(cutoff)
        query_ids = _ids(homes)
        listed = _dates(homes, "date_listed")
        known_listed = self._listing_dates.reindex(query_ids).to_numpy()
        if ((~np.isnat(known_listed)) & (known_listed != listed)).any():
            raise ValueError("A comparable-sales source ml_num cannot change its date_listed")
        # For a source row, chronology and this upper bound exclude its own sale.
        end = np.minimum(listed, cutoff)
        start = (pd.DatetimeIndex(end) - pd.DateOffset(months=12)).to_numpy(dtype="datetime64[ns]")

        values = np.full((len(homes), len(columns)), np.nan)
        values[:, (1, 3)] = 0
        pending = np.arange(len(homes))
        community_keys = homes.loc[:, [TYPE, "community", "municipality"]]
        for tier, (keys, index) in enumerate(zip(TIERS, self._tiers), start=1):
            if not len(pending):
                break
            stats = _window_stats(index, community_keys.iloc[pending], keys, start[pending], end[pending])
            supported = stats[:, 1] >= MIN_COMPS
            values[pending[supported], :3] = stats[supported]
            values[pending[supported], 3] = tier
            pending = pending[~supported]

        if mode == "community_and_site":
            if self._site is None:
                raise ValueError("The site mode needs latitude and longitude in the training sales")
            site_keys = _site_keys(homes)
            eligible = (site_keys[TYPE].isin(SITE_TYPES)
                        & site_keys["latitude"].notna() & site_keys["longitude"].notna()).to_numpy()
            values[eligible, 4:] = _window_stats(self._site, site_keys.iloc[eligible], SITE,
                                                  start[eligible], end[eligible])
        frame = pd.DataFrame(values, index=homes.index, columns=columns)
        frame.attrs.update(comp_mode=mode, comp_cutoff=pd.Timestamp(cutoff).isoformat())
        return frame
