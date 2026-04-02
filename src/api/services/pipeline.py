from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from src.workflows.pipeline_runner import (
    ensure_simulation_outputs,
    run_churn_training_pipeline,
    run_optimize_pipeline,
    run_uplift_pipeline,
)


def bootstrap_data(data_dir: Path) -> None:
    required = [data_dir / 'customer_summary.csv', data_dir / 'cohort_retention.csv']
    if all(path.exists() for path in required):
        return
    ensure_simulation_outputs(data_dir=data_dir)


def run_mode(mode: str, data_dir: Path, model_dir: Path, result_dir: Path, budget: Optional[int] = None) -> Dict:
    bootstrap_data(data_dir)
    if mode == 'train':
        return run_churn_training_pipeline(data_dir=data_dir, model_dir=model_dir, result_dir=result_dir)
    if mode == 'uplift':
        return run_uplift_pipeline(data_dir=data_dir, result_dir=result_dir)
    if mode == 'optimize':
        return run_optimize_pipeline(data_dir=data_dir, result_dir=result_dir, budget=int(budget or 50000000))
    raise ValueError(f'Unsupported mode: {mode}')
