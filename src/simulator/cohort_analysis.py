from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


DEFAULT_ACTIVITY_EVENT_TYPES: tuple[str, ...] = (
    "visit",
    "page_view",
    "search",
    "add_to_cart",
    "remove_from_cart",
    "purchase",
    "support_contact",
    "coupon_open",
    "coupon_redeem",
)


def _month_number_from_string(month_str: pd.Series) -> pd.Series:
    text = month_str.astype(str)
    year = text.str.slice(0, 4).astype(int)
    month = text.str.slice(5, 7).astype(int)
    return year * 12 + month


def _normalize_event_types(event_types: Optional[Iterable[str]]) -> Optional[set[str]]:
    if event_types is None:
        return None
    normalized = {str(x) for x in event_types}
    return normalized or None


def build_cohort_retention(
    customers: pd.DataFrame,
    events: pd.DataFrame,
    periods: int = 7,
    end_date: Optional[str] = None,
    activity_event_types: Optional[Sequence[str]] = DEFAULT_ACTIVITY_EVENT_TYPES,
) -> pd.DataFrame:
    """
    Build a conventional monthly cohort-retention table.

    Design choices:
    - Period 0 is defined as 100% retention by cohort definition.
    - Periods that are not yet observable at the simulation end are marked as
      unobserved and receive NaN retention instead of 0, preventing right-censoring
      from artificially depressing recent cohorts.
    - Customer activity is defined by engagement events (default includes visit,
      page_view, search, cart, purchase, support, and coupon interactions).
    """
    columns = [
        "cohort_month",
        "period",
        "cohort_size",
        "retained_customers",
        "retention_rate",
        "observed",
    ]

    if customers.empty:
        return pd.DataFrame(columns=columns)

    if periods <= 0:
        raise ValueError("periods must be positive.")

    base = customers[["customer_id", "acquisition_month"]].copy()
    base["cohort_month"] = base["acquisition_month"].astype(str)
    base["cohort_month_num"] = _month_number_from_string(base["cohort_month"])
    base = base.drop_duplicates(subset=["customer_id"])

    event_type_filter = _normalize_event_types(activity_event_types)

    if events.empty:
        activity = pd.DataFrame(columns=["customer_id", "event_month_num"])
        inferred_end_month_num = int(base["cohort_month_num"].max())
    else:
        activity = events[["customer_id", "timestamp", "event_type"]].copy()
        if event_type_filter is not None:
            activity = activity[activity["event_type"].astype(str).isin(event_type_filter)].copy()

        if activity.empty:
            activity = pd.DataFrame(columns=["customer_id", "event_month_num"])
            inferred_end_month_num = int(base["cohort_month_num"].max())
        else:
            activity["event_time"] = pd.to_datetime(activity["timestamp"])
            activity["event_month_num"] = (
                activity["event_time"].dt.year * 12 + activity["event_time"].dt.month
            )
            activity = activity[["customer_id", "event_month_num"]].drop_duplicates()
            inferred_end_month_num = int(activity["event_month_num"].max())

    if end_date is not None:
        end_ts = pd.Timestamp(end_date)
        end_month_num = int(end_ts.year * 12 + end_ts.month)
    else:
        end_month_num = inferred_end_month_num

    merged = base.merge(activity, on="customer_id", how="left")
    merged["period"] = merged["event_month_num"] - merged["cohort_month_num"]
    merged = merged[(merged["period"] >= 0) & (merged["period"] < periods)].copy()

    cohort_sizes = base.groupby("cohort_month")["customer_id"].nunique()
    retained_counts = merged.groupby(["cohort_month", "period"])["customer_id"].nunique()
    observed_max_period = (end_month_num - cohort_sizes.index.to_series().pipe(_month_number_from_string)).astype(int)

    rows: list[dict] = []
    for cohort_month, cohort_size in cohort_sizes.items():
        max_observed = int(max(observed_max_period.get(cohort_month, 0), 0))
        cohort_size_int = int(cohort_size)

        for period in range(periods):
            is_observed = period <= max_observed
            if not is_observed:
                retained_customers = np.nan
                retention_rate = np.nan
            elif period == 0:
                retained_customers = cohort_size_int
                retention_rate = 1.0
            else:
                retained_customers = int(retained_counts.get((cohort_month, period), 0))
                retention_rate = retained_customers / max(cohort_size_int, 1)

            rows.append(
                {
                    "cohort_month": str(cohort_month),
                    "period": int(period),
                    "cohort_size": cohort_size_int,
                    "retained_customers": retained_customers,
                    "retention_rate": retention_rate,
                    "observed": bool(is_observed),
                }
            )

    result = pd.DataFrame(rows, columns=columns)
    return result.sort_values(["cohort_month", "period"]).reset_index(drop=True)


__all__ = ["DEFAULT_ACTIVITY_EVENT_TYPES", "build_cohort_retention"]
