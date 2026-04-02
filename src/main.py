from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.workflows.pipeline_runner import (
    ensure_simulation_outputs,
    run_churn_training_pipeline,
    run_optimize_pipeline,
    run_uplift_pipeline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Retention ROI project entrypoint')
    parser.add_argument('--mode', required=True, choices=['train', 'uplift', 'optimize', 'simulate'])
    parser.add_argument('--budget', type=int, default=50000000)
    parser.add_argument('--data-dir', default='data/raw')
    parser.add_argument('--model-dir', default='models')
    parser.add_argument('--result-dir', default='results')
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    result_dir = Path(args.result_dir)

    if args.mode == 'simulate':
        ensure_simulation_outputs(data_dir)
        print(f'Simulation outputs are ready in {data_dir}')
        return 0
    if args.mode == 'train':
        result = run_churn_training_pipeline(data_dir, model_dir, result_dir)
    elif args.mode == 'uplift':
        result = run_uplift_pipeline(data_dir, result_dir)
    else:
        result = run_optimize_pipeline(data_dir, result_dir, budget=args.budget)

    print(f"Mode: {result['mode']}")
    if result.get('model_path'):
        print(f"Model saved to: {result['model_path']}")
    if result.get('metrics_path'):
        print(f"Metrics saved to: {result['metrics_path']}")
    if result.get('primary_result_path'):
        print(f"Primary result saved to: {result['primary_result_path']}")
    if result.get('extra_result_paths'):
        for path in result['extra_result_paths']:
            print(f"Additional result: {path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
