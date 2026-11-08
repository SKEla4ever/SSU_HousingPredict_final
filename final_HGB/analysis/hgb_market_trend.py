"""Past-only local price trends from two adjacent 90-day sale windows."""

import numpy as np
import pandas as pd

from analysis.hgb_local_comps import (
    TYPE,
    TIERS,
    ComparableSales,
    _cutoff_date,
    _dates,
    _ids,
    _window_stats,
)


TREND_FEATURES = ("trend_log_ppsf_ratio", "trend_recent_count", "trend_prior_count", "trend_tier")
TREND_MIN_COMPS = 5
TREND_WINDOW_DAYS = 90


class LocalMarketTrend(ComparableSales):
    """Index known sales and measure how their median price per sqft has moved.

    Anchor both windows to the home's as-of date, capped by the fold cutoff.
    The recent window is [as_of - 90 days, as_of), and the prior window is
    [as_of - 180 days, as_of - 90 days). Each tier must have five valid sales
    in both windows. Tiers 1 to 3 are type and community, type and municipality,
    and type only. An unsupported home has a missing ratio and zeros for counts
    and tier. Counts refer to the selected tier and its two windows.
    """

    def __init__(self, sales):
        # Use the shared grouped PPSF index without constructing its site groups.
        columns = ["ml_num", "date_listed", "date_sold", "sold_price", "house_area",
                   TYPE, "community", "municipality"]
        super().__init__(sales.loc[:, columns])

    def transform(self, homes, cutoff):
        """Compute log median PPSF change and its support in the input row order."""
        cutoff = _cutoff_date(cutoff)
        query_ids = _ids(homes)
        listed = _dates(homes, "date_listed")
        known_listed = self._listing_dates.reindex(query_ids).to_numpy()
        if ((~np.isnat(known_listed)) & (known_listed != listed)).any():
            raise ValueError("A market-trend source ml_num cannot change its date_listed")

        # The exclusive end and source chronology also exclude each row's own sale.
        end = np.minimum(listed, cutoff)
        as_of = pd.DatetimeIndex(end)
        middle = (as_of - pd.Timedelta(days=TREND_WINDOW_DAYS)).to_numpy()
        start = (as_of - pd.Timedelta(days=2 * TREND_WINDOW_DAYS)).to_numpy()
        values = np.zeros((len(homes), len(TREND_FEATURES)))
        values[:, 0] = np.nan
        pending = np.arange(len(homes))
        group_keys = homes.loc[:, [TYPE, "community", "municipality"]]

        for tier, (keys, index) in enumerate(zip(TIERS, self._tiers), start=1):
            if not len(pending):
                break
            query = group_keys.iloc[pending]
            recent = _window_stats(index, query, keys, middle[pending], end[pending])
            prior = _window_stats(index, query, keys, start[pending], middle[pending])
            supported = ((recent[:, 1] >= TREND_MIN_COMPS) & (prior[:, 1] >= TREND_MIN_COMPS)
                         & (recent[:, 0] > 0) & (prior[:, 0] > 0))
            selected = pending[supported]
            values[selected, 0] = np.log(recent[supported, 0]) - np.log(prior[supported, 0])
            values[selected, 1] = recent[supported, 1]
            values[selected, 2] = prior[supported, 1]
            values[selected, 3] = tier
            pending = pending[~supported]

        frame = pd.DataFrame(values, index=homes.index, columns=TREND_FEATURES)
        frame.attrs["trend_cutoff"] = pd.Timestamp(cutoff).isoformat()
        return frame
