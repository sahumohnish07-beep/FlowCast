"""Model definition, training, evaluation, and inference pipelines for FlowCast."""

from src.models.baseline import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    compute_metrics,
    load_datasets,
    run_baseline_training,
    train_linear_model,
    train_random_forest,
    train_xgboost,
)

__all__ = [
    "FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "compute_metrics",
    "load_datasets",
    "train_linear_model",
    "train_random_forest",
    "train_xgboost",
    "run_baseline_training",
]
