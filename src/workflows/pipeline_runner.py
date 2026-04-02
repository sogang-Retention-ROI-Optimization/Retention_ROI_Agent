from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.api.services.analytics import allocate_budget, budget_allocation_by_segment
from src.simulator.config import DEFAULT_CONFIG
from src.simulator.pipeline import run_simulation

RESULT_PREFIX = Path('results')
MODEL_PREFIX = Path('models')


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_simulation_outputs(data_dir: Path) -> Dict[str, pd.DataFrame]:
    data_dir = ensure_directory(data_dir)
    required = [data_dir / 'customer_summary.csv', data_dir / 'cohort_retention.csv']
    if all(p.exists() for p in required):
        return {
            'customer_summary': pd.read_csv(data_dir / 'customer_summary.csv'),
            'cohort_retention': pd.read_csv(data_dir / 'cohort_retention.csv'),
        }
    return run_simulation(config=DEFAULT_CONFIG, export=True, output_dir=str(data_dir), file_format='csv')


def load_customer_summary(data_dir: Path) -> pd.DataFrame:
    ensure_simulation_outputs(data_dir)
    return pd.read_csv(data_dir / 'customer_summary.csv')


def _build_training_label(df: pd.DataFrame) -> pd.Series:
    return ((df['inactivity_days'] >= 45) | (df['churn_probability'] >= 0.55)).astype(int)


def _select_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_features = [
        'recency_days', 'frequency', 'monetary', 'visits_last_7', 'visits_prev_7',
        'visit_change_rate', 'purchase_last_30', 'purchase_prev_30', 'purchase_change_rate',
        'coupon_cost', 'coupon_exposure_count', 'coupon_redeem_count', 'inactivity_days',
        'price_sensitivity', 'coupon_affinity', 'basket_size_preference', 'support_contact_propensity',
    ]
    categorical_features = ['persona', 'acquisition_month', 'region', 'device_type', 'acquisition_channel', 'treatment_group']
    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]
    return numeric_features, categorical_features


def run_churn_training_pipeline(data_dir: Path, model_dir: Path, result_dir: Path) -> Dict:
    model_dir = ensure_directory(model_dir)
    result_dir = ensure_directory(result_dir)
    df = load_customer_summary(data_dir)

    y = _build_training_label(df)
    numeric_features, categorical_features = _select_feature_columns(df)
    feature_columns = numeric_features + categorical_features
    X = df[feature_columns].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
            ]), categorical_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=220,
        max_depth=10,
        min_samples_leaf=8,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced_subsample',
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model),
    ])
    pipeline.fit(X_train, y_train)

    y_score = pipeline.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, y_score))

    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc_plot_path = result_dir / 'churn_auc_roc.png'
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Churn Model AUC-ROC')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(auc_plot_path, dpi=160)
    plt.close()

    sample_size = min(300, len(X_test))
    X_test_sample = X_test.sample(n=sample_size, random_state=42) if len(X_test) > sample_size else X_test.copy()
    transformed_sample = pipeline.named_steps['preprocessor'].transform(X_test_sample)
    estimator = pipeline.named_steps['model']
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(transformed_sample)
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    shap_array = shap_values[1] if isinstance(shap_values, list) else shap_values
    shap_plot_path = result_dir / 'churn_shap_summary.png'
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_array, transformed_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(shap_plot_path, dpi=160, bbox_inches='tight')
    plt.close()

    model_path = model_dir / 'churn_model.joblib'
    joblib.dump(pipeline, model_path)

    metrics_path = result_dir / 'churn_metrics.json'
    metrics = {
        'auc_roc': auc,
        'positive_rate': float(y.mean()),
        'train_rows': int(len(X_train)),
        'test_rows': int(len(X_test)),
        'feature_columns': feature_columns,
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')

    return {
        'mode': 'train',
        'model_path': str(model_path),
        'metrics_path': str(metrics_path),
        'primary_result_path': str(auc_plot_path),
        'extra_result_paths': [str(shap_plot_path)],
        'metadata': metrics,
    }


def run_uplift_pipeline(data_dir: Path, result_dir: Path) -> Dict:
    result_dir = ensure_directory(result_dir)
    df = load_customer_summary(data_dir).copy()
    output = df[[
        'customer_id', 'persona', 'uplift_score', 'uplift_segment', 'clv', 'churn_probability',
        'expected_incremental_profit', 'expected_roi'
    ]].sort_values(['uplift_score', 'clv'], ascending=False)
    out_path = result_dir / 'uplift_segmentation.csv'
    output.to_csv(out_path, index=False)
    segment_counts = output['uplift_segment'].value_counts().to_dict()
    meta = {'rows': int(len(output)), 'segment_counts': segment_counts}
    meta_path = result_dir / 'uplift_summary.json'
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    return {
        'mode': 'uplift',
        'model_path': None,
        'metrics_path': str(meta_path),
        'primary_result_path': str(out_path),
        'extra_result_paths': [],
        'metadata': meta,
    }


def run_optimize_pipeline(data_dir: Path, result_dir: Path, budget: int) -> Dict:
    result_dir = ensure_directory(result_dir)
    df = load_customer_summary(data_dir)
    selected, summary = allocate_budget(df, budget=budget)
    segment_allocation = budget_allocation_by_segment(selected)

    selected_path = result_dir / 'optimization_selected_customers.csv'
    segment_path = result_dir / 'optimization_segment_budget.csv'
    summary_path = result_dir / 'optimization_summary.json'

    selected.to_csv(selected_path, index=False)
    segment_allocation.to_csv(segment_path, index=False)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    return {
        'mode': 'optimize',
        'model_path': None,
        'metrics_path': str(summary_path),
        'primary_result_path': str(segment_path),
        'extra_result_paths': [str(selected_path)],
        'metadata': summary,
    }
