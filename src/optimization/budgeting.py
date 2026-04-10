from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd

from src.optimization.policy import build_intensity_action_candidates, normalize
from src.optimization.timing import load_survival_predictions


@dataclass
class OptimizationArtifacts:
    selected_customers: pd.DataFrame
    summary: Dict
    selected_path: str
    segment_path: str
    summary_path: str
    scenario_path: str


STRATEGY_BY_SEGMENT = {
    "High Value-Persuadables": {
        "strategy_name": "VIP concierge + personalized offer",
        "cost": 30000,
        "effect_multiplier": 1.15,
    },
    "High Value-Sure Things": {
        "strategy_name": "Loyalty touchpoint",
        "cost": 8000,
        "effect_multiplier": 0.15,
    },
    "High Value-Lost Causes": {
        "strategy_name": "Deep-dive outreach",
        "cost": 12000,
        "effect_multiplier": 0.10,
    },
    "Low Value-Persuadables": {
        "strategy_name": "Coupon campaign",
        "cost": 7000,
        "effect_multiplier": 0.85,
    },
    "Low Value-Lost Causes": {
        "strategy_name": "No Action",
        "cost": 0,
        "effect_multiplier": 0.0,
    },
    "Low Value-Sure Things": {
        "strategy_name": "Light reminder",
        "cost": 3000,
        "effect_multiplier": 0.05,
    },
    "New Customers": {
        "strategy_name": "Onboarding sequence",
        "cost": 5000,
        "effect_multiplier": 0.20,
    },
}


def _apply_strategy(df: pd.DataFrame, survival_predictions: pd.DataFrame | None = None) -> pd.DataFrame:
    mapping = pd.DataFrame.from_dict(STRATEGY_BY_SEGMENT, orient="index").reset_index().rename(columns={"index": "customer_segment"})
    mapping = mapping.rename(columns={"cost": "strategy_cost", "effect_multiplier": "strategy_effect_multiplier"})
    out = df.merge(mapping, on="customer_segment", how="left")
    enriched = build_intensity_action_candidates(out, survival_predictions=survival_predictions)
    enriched["optimization_score"] = enriched["expected_incremental_profit"] / enriched["coupon_cost"].where(enriched["coupon_cost"] > 0, 1.0)
    enriched["selection_score"] = 0.55 * enriched["priority_score"] + 0.45 * normalize(enriched["optimization_score"])
    return enriched


def _greedy_select(candidates: pd.DataFrame, budget: int) -> pd.DataFrame:
    if candidates.empty or budget <= 0:
        return candidates.head(0).copy()

    ranked = candidates[candidates["coupon_cost"] > 0].copy()
    ranked = ranked[ranked["expected_revenue"] > 0].copy()
    ranked = ranked[ranked["expected_incremental_profit"] > 0].copy()
    ranked = ranked.sort_values(
        [
            "selection_score",
            "priority_score",
            "optimization_score",
            "timing_urgency_score",
            "expected_revenue",
            "retention_priority_score",
            "customer_id",
            "coupon_cost",
        ],
        ascending=[False, False, False, False, False, False, True, True],
    )

    selected_rows = []
    used_customers: set[int] = set()
    spent = 0.0
    for row in ranked.itertuples(index=False):
        customer_id = int(getattr(row, "customer_id"))
        cost = float(getattr(row, "coupon_cost", 0.0))
        if customer_id in used_customers:
            continue
        if cost <= 0:
            continue
        if spent + cost > float(budget):
            continue
        selected_rows.append(row._asdict())
        used_customers.add(customer_id)
        spent += cost

    if not selected_rows:
        return ranked.head(0).copy()
    return pd.DataFrame(selected_rows)


def _segment_allocation(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame(columns=["customer_segment", "intervention_intensity", "customer_count", "allocated_budget", "expected_revenue", "expected_roi"])
    allocation = (
        selected.groupby(["customer_segment", "intervention_intensity"], as_index=False)
        .agg(
            customer_count=("customer_id", "nunique"),
            allocated_budget=("coupon_cost", "sum"),
            expected_revenue=("expected_revenue", "sum"),
        )
    )
    allocation["expected_roi"] = (allocation["expected_revenue"] - allocation["allocated_budget"]) / allocation["allocated_budget"].where(allocation["allocated_budget"] > 0, 1.0)
    allocation = allocation.sort_values(["allocated_budget", "expected_revenue"], ascending=[False, False])
    return allocation


def _scenario_rows(candidates: pd.DataFrame, budget: int) -> pd.DataFrame:
    rows = []
    for label, scenario_budget in [
        ("50%", int(budget * 0.5)),
        ("100%", int(budget)),
        ("200%", int(budget * 2.0)),
    ]:
        sel = _greedy_select(candidates, scenario_budget)
        spent = float(sel["coupon_cost"].sum()) if len(sel) else 0.0
        revenue = float(sel["expected_revenue"].sum()) if len(sel) else 0.0
        roi = ((revenue - spent) / spent) if spent > 0 else 0.0
        rows.append(
            {
                "scenario": label,
                "budget": int(scenario_budget),
                "spent": round(spent, 2),
                "remaining": round(scenario_budget - spent, 2),
                "num_targeted": int(len(sel)),
                "expected_revenue": round(revenue, 2),
                "expected_roi": round(roi, 6),
            }
        )
    return pd.DataFrame(rows)


def run_budget_optimization(result_dir: Path, budget: int) -> OptimizationArtifacts:
    segments = pd.read_csv(result_dir / "customer_segments.csv")
    survival_predictions = load_survival_predictions(result_dir)
    candidates = _apply_strategy(segments, survival_predictions=survival_predictions)
    selected = _greedy_select(candidates, budget)
    spent = float(selected["coupon_cost"].sum()) if len(selected) else 0.0
    revenue = float(selected["expected_revenue"].sum()) if len(selected) else 0.0
    profit = float(selected["expected_incremental_profit"].sum()) if len(selected) else 0.0
    roi = (profit / spent) if spent > 0 else 0.0

    segment_allocation = _segment_allocation(selected)
    summary = {
        "budget": int(budget),
        "spent": int(round(spent)),
        "remaining": int(round(budget - spent)),
        "num_targeted": int(len(selected)),
        "candidate_customers": int(candidates["customer_id"].nunique()) if len(candidates) else 0,
        "candidate_actions": int(len(candidates)),
        "expected_revenue": round(revenue, 2),
        "expected_incremental_profit": round(profit, 2),
        "overall_roi": round(roi, 6),
        "baseline_method": "Greedy multiple-choice selection over customer × timing × intensity actions",
        "objective": "Maximize Σ(Uplift_i × CLV_i × SurvivalTiming_i × IntensityEffect_i × Action_i)",
        "selected_intensity_counts": selected["intervention_intensity"].value_counts().to_dict() if len(selected) else {},
        "survival_enriched": bool(not survival_predictions.empty),
        "avg_timing_urgency_score": round(float(candidates["timing_urgency_score"].mean()), 6) if len(candidates) else 0.0,
    }

    selected_path = result_dir / "optimization_selected_customers.csv"
    segment_path = result_dir / "optimization_segment_budget.csv"
    summary_path = result_dir / "optimization_summary.json"
    scenario_path = result_dir / "optimization_what_if.csv"

    selected.sort_values(["priority_score", "expected_revenue"], ascending=[False, False]).to_csv(selected_path, index=False)
    segment_allocation.to_csv(segment_path, index=False)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _scenario_rows(candidates, budget).to_csv(scenario_path, index=False)

    return OptimizationArtifacts(
        selected_customers=selected,
        summary=summary,
        selected_path=str(selected_path),
        segment_path=str(segment_path),
        summary_path=str(summary_path),
        scenario_path=str(scenario_path),
    )
