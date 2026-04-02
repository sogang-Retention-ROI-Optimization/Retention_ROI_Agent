from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd
from scipy.stats import chi2_contingency, norm


@dataclass
class ABTestArtifacts:
    result_path: str
    report_path: str


def _power_sample_size(p1: float, p2: float, alpha: float = 0.05, power: float = 0.80) -> int:
    p_bar = (p1 + p2) / 2.0
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    numerator = (
        z_alpha * math.sqrt(2 * p_bar * (1 - p_bar))
        + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    denominator = max((p2 - p1) ** 2, 1e-9)
    return int(math.ceil(numerator / denominator))


def _two_proportion_z_test(x1: int, n1: int, x0: int, n0: int) -> Dict[str, float]:
    p1 = x1 / max(n1, 1)
    p0 = x0 / max(n0, 1)
    pooled = (x1 + x0) / max(n1 + n0, 1)
    se = math.sqrt(max(pooled * (1 - pooled) * (1 / max(n1, 1) + 1 / max(n0, 1)), 1e-12))
    z = (p1 - p0) / se
    p_value = 2 * (1 - norm.cdf(abs(z)))
    diff = p1 - p0
    se_ci = math.sqrt(max(p1 * (1 - p1) / max(n1, 1) + p0 * (1 - p0) / max(n0, 1), 1e-12))
    ci_low = diff - 1.96 * se_ci
    ci_high = diff + 1.96 * se_ci
    return {
        "treatment_rate": p1,
        "control_rate": p0,
        "difference": diff,
        "z_stat": z,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def run_ab_test_analysis(result_dir: Path) -> ABTestArtifacts:
    uplift = pd.read_csv(result_dir / "uplift_segmentation.csv")
    raw = pd.read_csv(result_dir.parent / "data" / "raw" / "customer_summary.csv") if (result_dir.parent / "data" / "raw" / "customer_summary.csv").exists() else pd.DataFrame()
    if "treatment_group" not in uplift.columns and not raw.empty:
        uplift = uplift.merge(raw[["customer_id", "treatment_group"]], on="customer_id", how="left")

    if "retained_60d" not in uplift.columns:
        raise ValueError("uplift_segmentation.csv must contain retained_60d for A/B testing.")

    treatment = uplift[uplift["treatment_group"] == "treatment"]
    control = uplift[uplift["treatment_group"] == "control"]

    n1 = int(len(treatment))
    n0 = int(len(control))
    x1 = int(pd.to_numeric(treatment["retained_60d"], errors="coerce").fillna(0).sum())
    x0 = int(pd.to_numeric(control["retained_60d"], errors="coerce").fillna(0).sum())

    ztest = _two_proportion_z_test(x1, n1, x0, n0)
    contingency = [[x1, n1 - x1], [x0, n0 - x0]]
    chi2, chi2_p, _, _ = chi2_contingency(contingency)
    required_n = _power_sample_size(ztest["control_rate"], ztest["treatment_rate"])

    result = {
        "metric": "60-day retention rate",
        "n_treatment": n1,
        "n_control": n0,
        "success_treatment": x1,
        "success_control": x0,
        "treatment_rate": round(ztest["treatment_rate"], 6),
        "control_rate": round(ztest["control_rate"], 6),
        "difference": round(ztest["difference"], 6),
        "p_value": round(ztest["p_value"], 8),
        "chi_square_p_value": round(float(chi2_p), 8),
        "z_stat": round(ztest["z_stat"], 6),
        "chi_square_stat": round(float(chi2), 6),
        "confidence_interval_95": [round(ztest["ci_low"], 6), round(ztest["ci_high"], 6)],
        "required_sample_size_per_group": int(required_n),
        "statistically_significant": bool(ztest["p_value"] < 0.05),
    }

    interpretation = [
        f"Treatment retention={result['treatment_rate']:.3f}, control retention={result['control_rate']:.3f}.",
        f"Two-sided z-test p-value={result['p_value']:.6f}, chi-square p-value={result['chi_square_p_value']:.6f}.",
        f"95% CI for treatment-control difference: {result['confidence_interval_95'][0]:.3f} to {result['confidence_interval_95'][1]:.3f}.",
        "p < 0.05 이면 통계적으로 유의하다고 판단한다." if result["statistically_significant"] else "현재 표본에서는 p < 0.05 기준을 충족하지 못한다.",
        f"동일한 효과 크기를 검출하려면 그룹당 약 {required_n}명이 필요하다.",
    ]

    result_path = result_dir / "ab_test_results.json"
    report_path = result_dir / "ab_test_report.md"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(
        "# A/B Test Report\n\n" + "\n".join(f"- {line}" for line in interpretation),
        encoding="utf-8",
    )

    return ABTestArtifacts(result_path=str(result_path), report_path=str(report_path))
