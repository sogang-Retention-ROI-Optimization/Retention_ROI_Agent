from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import SimulationConfig
from .event_rules import (
    compute_add_to_cart_probability,
    compute_browse_probability,
    compute_coupon_open_probability,
    compute_coupon_redeem_probability,
    compute_purchase_probability,
    compute_remove_cart_probability,
    compute_search_probability,
    compute_visit_probability,
)
from .order_builder import build_orders
from .state_tracker import StateTracker


def _event_frame(
    customer_ids: np.ndarray,
    event_type: str,
    session_ids: np.ndarray,
    timestamps: np.ndarray,
    item_category: Optional[np.ndarray] = None,
    quantity: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    n = len(customer_ids)
    if n == 0:
        return pd.DataFrame(
            columns=[
                "event_id",
                "customer_id",
                "timestamp",
                "event_type",
                "session_id",
                "item_category",
                "quantity",
            ]
        )

    seed = pd.Series(session_ids).astype(str).str.replace("SES-", "", regex=False).fillna("0")
    event_suffix = np.arange(1, n + 1, dtype=int)
    return pd.DataFrame(
        {
            "event_id": [f"EVT-{event_type[:3].upper()}-{s}-{i:02d}" for s, i in zip(seed, event_suffix)],
            "customer_id": customer_ids.astype(int),
            "timestamp": pd.to_datetime(timestamps),
            "event_type": event_type,
            "session_id": session_ids.astype(str),
            "item_category": item_category if item_category is not None else None,
            "quantity": quantity if quantity is not None else None,
        }
    )


def _base_session_start_times(
    customers: pd.DataFrame,
    visit_mask: np.ndarray,
    date: pd.Timestamp,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.flatnonzero(visit_mask)
    if len(idx) == 0:
        return np.array([], dtype=object), np.array([], dtype="datetime64[ns]")

    visit_customers = customers.iloc[idx]
    device = visit_customers["device_type"].to_numpy()
    persona = visit_customers["persona"].to_numpy()

    base_minutes = np.full(len(idx), 14 * 60, dtype=int)
    base_minutes += np.where(device == "desktop", -150, 0)
    base_minutes += np.where(device == "mobile", 45, 0)
    base_minutes += np.where(device == "tablet", -20, 0)
    base_minutes += np.where(persona == "new_signup", 25, 0)
    base_minutes += np.where(persona == "vip_loyal", -10, 0)
    base_minutes += np.where(persona == "churn_progressing", 70, 0)
    base_minutes += np.where(date.weekday() >= 5, 60, -10)
    base_minutes += rng.normal(0, 110, size=len(idx)).round().astype(int)

    night_mask = (device == "mobile") & (rng.random(len(idx)) < 0.08)
    base_minutes[night_mask] = rng.integers(0, 5 * 60, size=int(night_mask.sum()))

    base_minutes = np.clip(base_minutes, 0, 23 * 60 + 50)
    session_ids = np.array([f"SES-{int(x)}" for x in rng.integers(10_000_000, 99_999_999, size=len(idx))], dtype=object)
    timestamps = pd.Timestamp(date.normalize()) + pd.to_timedelta(base_minutes, unit="m")
    return session_ids, timestamps.to_numpy()


def _session_lookup(customer_ids: np.ndarray, values: np.ndarray) -> dict[int, object]:
    return {int(cid): values[i] for i, cid in enumerate(customer_ids.astype(int))}


def _sample_page_view_counts(customers: pd.DataFrame, browse_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx = np.flatnonzero(browse_mask)
    if len(idx) == 0:
        return np.array([], dtype=int)
    browse_prob = customers.iloc[idx]["browse_prob_base"].to_numpy()
    search_prob = customers.iloc[idx]["search_prob_base"].to_numpy()
    mean_views = 1.4 + 3.2 * browse_prob + 1.1 * search_prob
    counts = rng.poisson(np.clip(mean_views, 1.0, 6.0))
    return np.clip(counts, 1, 9).astype(int)


def _repeat_page_view_events(
    customer_ids: np.ndarray,
    session_ids: np.ndarray,
    session_times: np.ndarray,
    page_view_counts: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if len(customer_ids) == 0:
        return pd.DataFrame(columns=["event_id", "customer_id", "timestamp", "event_type", "session_id", "item_category", "quantity"])

    records = []
    for cid, sid, base_ts, n_views in zip(customer_ids.astype(int), session_ids.astype(str), pd.to_datetime(session_times), page_view_counts.astype(int)):
        n_views = max(int(n_views), 1)
        offsets = np.sort(rng.integers(1, 4 * n_views + 3, size=n_views))
        for step, offset in enumerate(offsets, start=1):
            ts = base_ts + pd.Timedelta(minutes=int(offset))
            records.append(
                {
                    "event_id": f"EVT-PAG-{sid.replace('SES-', '')}-{step:02d}",
                    "customer_id": cid,
                    "timestamp": ts,
                    "event_type": "page_view",
                    "session_id": sid,
                    "item_category": None,
                    "quantity": None,
                }
            )
    return pd.DataFrame.from_records(records)


def simulate_events(
    customers: pd.DataFrame,
    assignments: pd.DataFrame,
    config: SimulationConfig,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Simulate the full customer journey and return raw tables:
    - events
    - orders
    - campaign_exposures
    - state_snapshots
    - final_state_metrics
    """
    rng = rng or np.random.default_rng(config.random_seed)

    sim = customers.merge(assignments, on="customer_id", how="left").sort_values("customer_id").reset_index(drop=True)
    tracker = StateTracker(n_customers=len(sim))
    coupon_cost_lookup = sim["coupon_cost"].to_numpy()

    event_frames: List[pd.DataFrame] = []
    order_frames: List[pd.DataFrame] = []
    exposure_frames: List[pd.DataFrame] = []
    snapshot_frames: List[pd.DataFrame] = []

    order_seq = 1
    exposure_seq = 1

    dates = pd.date_range(config.start_date, config.end_date, freq="D")
    signup_dates = pd.to_datetime(sim["signup_date"]).to_numpy()
    assigned_at_days = (pd.to_datetime(sim["assigned_at"]) - pd.Timestamp(config.start_date)).dt.days.to_numpy()

    customer_ids_all = sim["customer_id"].to_numpy().astype(int)

    for day_idx, date in enumerate(dates):
        tracker.start_day()
        active_mask = signup_dates <= np.datetime64(date)

        eligible_exposure = (
            active_mask
            & (sim["treatment_flag"].to_numpy() == 1)
            & (tracker.days_since_last_coupon >= config.coupon_cooldown_days)
            & (tracker.exposures_total < config.max_exposures_per_customer)
            & (
                (tracker.inactivity_days >= config.coupon_trigger_inactivity_days)
                | (assigned_at_days == day_idx)
            )
        )
        exposure_prob = np.clip(
            0.18 + 0.30 * sim["coupon_affinity"].to_numpy() + 0.10 * (tracker.inactivity_days >= config.coupon_trigger_inactivity_days),
            0.0,
            0.95,
        )
        exposure_mask = eligible_exposure & (rng.random(len(sim)) < exposure_prob)
        tracker.record_exposure(exposure_mask)

        if exposure_mask.any():
            exposure_ids = [f"EXP-{day_idx:03d}-{exposure_seq + i:07d}" for i in range(int(exposure_mask.sum()))]
            exposure_seq += int(exposure_mask.sum())
            exposure_times = pd.Timestamp(date.normalize()) + pd.to_timedelta(rng.integers(9 * 60, 21 * 60, size=int(exposure_mask.sum())), unit="m")
            exposure_frames.append(
                pd.DataFrame(
                    {
                        "exposure_id": exposure_ids,
                        "customer_id": sim.loc[exposure_mask, "customer_id"].to_numpy().astype(int),
                        "exposure_time": exposure_times.to_numpy(),
                        "campaign_type": sim.loc[exposure_mask, "campaign_type"].to_numpy(),
                        "coupon_cost": sim.loc[exposure_mask, "coupon_cost"].to_numpy().astype(int),
                    }
                )
            )

        coupon_open_prob = compute_coupon_open_probability(sim, exposure_mask, tracker)
        coupon_open_mask = exposure_mask & (rng.random(len(sim)) < coupon_open_prob)
        tracker.record_coupon_open(coupon_open_mask)

        visit_prob = compute_visit_probability(sim, tracker, active_mask, date)
        visit_mask = rng.random(len(sim)) < visit_prob
        tracker.record_visit(visit_mask, day_idx)

        browse_prob = compute_browse_probability(sim, visit_mask, tracker)
        browse_mask = visit_mask & (rng.random(len(sim)) < browse_prob)

        search_prob = compute_search_probability(sim, visit_mask, tracker)
        search_mask = visit_mask & (rng.random(len(sim)) < search_prob)

        add_cart_prob = compute_add_to_cart_probability(sim, browse_mask, search_mask, tracker)
        add_to_cart_mask = browse_mask & (rng.random(len(sim)) < add_cart_prob)
        tracker.record_cart_add(add_to_cart_mask)

        purchase_prob = compute_purchase_probability(sim, visit_mask, add_to_cart_mask, coupon_open_mask, tracker)
        purchase_mask = visit_mask & (rng.random(len(sim)) < purchase_prob)

        coupon_redeem_prob = compute_coupon_redeem_probability(sim, coupon_open_mask, purchase_mask)
        coupon_redeem_mask = coupon_open_mask & purchase_mask & (rng.random(len(sim)) < coupon_redeem_prob)
        tracker.record_coupon_redeem(coupon_redeem_mask)

        remove_cart_prob = compute_remove_cart_probability(sim, add_to_cart_mask, purchase_mask, tracker)
        remove_cart_mask = add_to_cart_mask & (rng.random(len(sim)) < remove_cart_prob)
        tracker.record_cart_remove(remove_cart_mask)

        support_prob = np.clip(
            sim["support_contact_propensity"].to_numpy() + 0.05 * remove_cart_mask.astype(float) + 0.03 * (tracker.inactivity_days > 20),
            0.0,
            0.55,
        )
        support_mask = visit_mask & (rng.random(len(sim)) < support_prob)

        visit_customer_ids = customer_ids_all[visit_mask]
        session_ids, session_start_times = _base_session_start_times(sim, visit_mask, date, rng)
        session_id_map = _session_lookup(visit_customer_ids, session_ids)
        session_time_map = _session_lookup(visit_customer_ids, session_start_times)

        browse_customer_ids = customer_ids_all[browse_mask]
        browse_session_ids = np.array([session_id_map[int(cid)] for cid in browse_customer_ids], dtype=object) if len(browse_customer_ids) else np.array([], dtype=object)
        browse_session_times = np.array([session_time_map[int(cid)] for cid in browse_customer_ids], dtype="datetime64[ns]") if len(browse_customer_ids) else np.array([], dtype="datetime64[ns]")
        page_view_counts = _sample_page_view_counts(sim, browse_mask, rng)

        search_customer_ids = customer_ids_all[search_mask]
        search_session_ids = np.array([session_id_map[int(cid)] for cid in search_customer_ids], dtype=object) if len(search_customer_ids) else np.array([], dtype=object)
        search_times = np.array([session_time_map[int(cid)] for cid in search_customer_ids], dtype="datetime64[ns]") if len(search_customer_ids) else np.array([], dtype="datetime64[ns]")
        if len(search_times):
            search_times = pd.to_datetime(search_times) + pd.to_timedelta(rng.integers(1, 6, size=len(search_times)), unit="m")

        add_customer_ids = customer_ids_all[add_to_cart_mask]
        add_session_ids = np.array([session_id_map[int(cid)] for cid in add_customer_ids], dtype=object) if len(add_customer_ids) else np.array([], dtype=object)
        add_times = np.array([session_time_map[int(cid)] for cid in add_customer_ids], dtype="datetime64[ns]") if len(add_customer_ids) else np.array([], dtype="datetime64[ns]")
        if len(add_times):
            add_times = pd.to_datetime(add_times) + pd.to_timedelta(rng.integers(2, 9, size=len(add_times)), unit="m")

        remove_customer_ids = customer_ids_all[remove_cart_mask]
        remove_session_ids = np.array([session_id_map[int(cid)] for cid in remove_customer_ids], dtype=object) if len(remove_customer_ids) else np.array([], dtype=object)
        remove_times = np.array([session_time_map[int(cid)] for cid in remove_customer_ids], dtype="datetime64[ns]") if len(remove_customer_ids) else np.array([], dtype="datetime64[ns]")
        if len(remove_times):
            remove_times = pd.to_datetime(remove_times) + pd.to_timedelta(rng.integers(6, 16, size=len(remove_times)), unit="m")

        support_customer_ids = customer_ids_all[support_mask]
        support_session_ids = np.array([session_id_map[int(cid)] for cid in support_customer_ids], dtype=object) if len(support_customer_ids) else np.array([], dtype=object)
        support_times = np.array([session_time_map[int(cid)] for cid in support_customer_ids], dtype="datetime64[ns]") if len(support_customer_ids) else np.array([], dtype="datetime64[ns]")
        if len(support_times):
            support_times = pd.to_datetime(support_times) + pd.to_timedelta(rng.integers(4, 12, size=len(support_times)), unit="m")

        purchase_customer_ids = customer_ids_all[purchase_mask]
        purchase_session_ids = np.array([session_id_map[int(cid)] for cid in purchase_customer_ids], dtype=object) if len(purchase_customer_ids) else np.array([], dtype=object)
        purchase_times = np.array([session_time_map[int(cid)] for cid in purchase_customer_ids], dtype="datetime64[ns]") if len(purchase_customer_ids) else np.array([], dtype="datetime64[ns]")
        if len(purchase_times):
            purchase_times = pd.to_datetime(purchase_times) + pd.to_timedelta(rng.integers(4, 14, size=len(purchase_times)), unit="m")

        coupon_open_customer_ids = customer_ids_all[coupon_open_mask]
        open_session_ids = np.array([session_id_map.get(int(cid), f"SES-{int(x)}") for cid, x in zip(coupon_open_customer_ids, rng.integers(10_000_000, 99_999_999, size=len(coupon_open_customer_ids)))], dtype=object) if len(coupon_open_customer_ids) else np.array([], dtype=object)
        open_base_times = []
        for cid in coupon_open_customer_ids:
            if int(cid) in session_time_map:
                open_base_times.append(pd.Timestamp(session_time_map[int(cid)]))
            else:
                minute = int(rng.integers(8 * 60, 21 * 60))
                open_base_times.append(pd.Timestamp(date.normalize()) + pd.Timedelta(minutes=minute))
        open_times = np.array(open_base_times, dtype="datetime64[ns]")
        if len(open_times):
            open_times = pd.to_datetime(open_times) + pd.to_timedelta(rng.integers(0, 8, size=len(open_times)), unit="m")

        coupon_redeem_customer_ids = customer_ids_all[coupon_redeem_mask]
        redeem_session_ids = np.array([session_id_map.get(int(cid), f"SES-{int(x)}") for cid, x in zip(coupon_redeem_customer_ids, rng.integers(10_000_000, 99_999_999, size=len(coupon_redeem_customer_ids)))], dtype=object) if len(coupon_redeem_customer_ids) else np.array([], dtype=object)
        redeem_base_times = []
        for cid in coupon_redeem_customer_ids:
            if int(cid) in session_time_map:
                redeem_base_times.append(pd.Timestamp(session_time_map[int(cid)]))
            else:
                minute = int(rng.integers(8 * 60, 21 * 60))
                redeem_base_times.append(pd.Timestamp(date.normalize()) + pd.Timedelta(minutes=minute))
        redeem_times = np.array(redeem_base_times, dtype="datetime64[ns]")
        if len(redeem_times):
            redeem_times = pd.to_datetime(redeem_times) + pd.to_timedelta(rng.integers(5, 16, size=len(redeem_times)), unit="m")

        event_frames.append(_event_frame(visit_customer_ids, "visit", session_ids, session_start_times))
        event_frames.append(_repeat_page_view_events(browse_customer_ids, browse_session_ids, browse_session_times, page_view_counts, rng))
        event_frames.append(_event_frame(search_customer_ids, "search", search_session_ids, search_times))
        event_frames.append(_event_frame(add_customer_ids, "add_to_cart", add_session_ids, add_times))
        event_frames.append(_event_frame(remove_customer_ids, "remove_from_cart", remove_session_ids, remove_times))
        event_frames.append(_event_frame(coupon_open_customer_ids, "coupon_open", open_session_ids, open_times))
        event_frames.append(_event_frame(coupon_redeem_customer_ids, "coupon_redeem", redeem_session_ids, redeem_times))
        event_frames.append(_event_frame(support_customer_ids, "support_contact", support_session_ids, support_times))
        event_frames.append(_event_frame(purchase_customer_ids, "purchase", purchase_session_ids, purchase_times))

        orders = build_orders(
            customers=sim,
            purchase_mask=purchase_mask,
            date=date,
            day_idx=day_idx,
            order_sequence_start=order_seq,
            coupon_open_mask=coupon_redeem_mask,
            coupon_cost_lookup=coupon_cost_lookup,
            rng=rng,
        )
        if not orders.empty:
            purchase_time_lookup = {int(cid): ts for cid, ts in zip(purchase_customer_ids, pd.to_datetime(purchase_times))}
            if "customer_id" in orders.columns:
                orders["order_time"] = orders["customer_id"].map(purchase_time_lookup).fillna(pd.Timestamp(date.normalize()) + pd.Timedelta(hours=12))
            order_seq += len(orders)
            order_frames.append(orders)
            amount_lookup = orders.set_index("customer_id")["net_amount"].to_dict()
            order_amounts = np.zeros(len(sim), dtype=float)
            purchase_idx = np.flatnonzero(purchase_mask)
            if len(purchase_idx):
                purchase_customer_ids_for_amount = customer_ids_all[purchase_idx]
                for arr_idx, cid in zip(purchase_idx, purchase_customer_ids_for_amount):
                    order_amounts[arr_idx] = amount_lookup.get(int(cid), 0.0)
            tracker.record_purchase(purchase_mask, order_amounts, day_idx)

        if (day_idx % config.snapshot_frequency_days == 0) or (day_idx == len(dates) - 1):
            snapshot_frames.append(
                tracker.to_snapshot(
                    customers=sim[["customer_id"]],
                    snapshot_date=date,
                    day_idx=day_idx,
                    dormant_threshold=config.dormant_inactivity_days,
                    churn_threshold=config.churn_inactivity_days,
                )
            )

    events = pd.concat([df for df in event_frames if not df.empty], ignore_index=True) if event_frames else pd.DataFrame()
    orders = pd.concat(order_frames, ignore_index=True) if order_frames else pd.DataFrame()
    exposures = pd.concat(exposure_frames, ignore_index=True) if exposure_frames else pd.DataFrame()
    state_snapshots = pd.concat(snapshot_frames, ignore_index=True) if snapshot_frames else pd.DataFrame()
    final_state = tracker.final_metrics(simulation_days=config.simulation_days)

    if not events.empty:
        events["timestamp"] = pd.to_datetime(events["timestamp"])
    if not orders.empty:
        orders["order_time"] = pd.to_datetime(orders["order_time"])
    if not exposures.empty:
        exposures["exposure_time"] = pd.to_datetime(exposures["exposure_time"])
    if not state_snapshots.empty:
        state_snapshots["snapshot_date"] = pd.to_datetime(state_snapshots["snapshot_date"])
        state_snapshots["last_visit_date"] = pd.to_datetime(state_snapshots["last_visit_date"])
        state_snapshots["last_purchase_date"] = pd.to_datetime(state_snapshots["last_purchase_date"])

    return events, orders, exposures, state_snapshots, final_state
