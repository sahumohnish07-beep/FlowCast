"""Extra predictive modules for FlowCast: Congestion, Travel Time, and Accident Risk.

Modules:
    1. Congestion Level Classifier (Low/Medium/High) derived from corridor volume thresholds.
    2. Corridor Travel Time Estimator based on speed-flow physics and regression.
    3. Accident Risk Classifier with SMOTE class imbalance handling.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
from pathlib import Path
import sys

# Ensure FlowCast root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
)
from xgboost import XGBClassifier, XGBRegressor

logger = logging.getLogger("flowcast.models.extra_modules")

PREDICTOR_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
    "is_holiday_eve",
    "is_morning_rush",
    "is_evening_rush",
    "is_rush_hour",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "temp_c",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "is_raining",
    "is_snowing",
    "is_freezing",
    "weather_severity_index",
    "traffic_lag_1h",
    "traffic_lag_2h",
    "traffic_lag_24h",
    "traffic_rolling_mean_3h",
    "traffic_rolling_mean_24h",
]


def train_congestion_module(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    images_dir: Path,
) -> tuple[XGBClassifier, dict[str, float]]:
    """Train and evaluate 3-class road congestion classifier (Low, Medium, High)."""
    logger.info("Training Congestion Level Classifier (XGBClassifier)...")
    X_train = df_train[PREDICTOR_FEATURES]
    y_train = df_train["congestion_level"]

    X_test = df_test[PREDICTOR_FEATURES]
    y_test = df_test["congestion_level"]

    clf = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))
    precision = float(precision_score(y_test, y_pred, average="macro"))
    recall = float(recall_score(y_test, y_pred, average="macro"))
    roc_auc = float(roc_auc_score(y_test, y_prob, multi_class="ovr"))

    metrics = {
        "Accuracy": round(acc, 4),
        "Precision_Macro": round(precision, 4),
        "Recall_Macro": round(recall, 4),
        "F1_Macro": round(macro_f1, 4),
        "F1_Weighted": round(weighted_f1, 4),
        "ROC_AUC": round(roc_auc, 4),
    }

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Low (<2500)", "Medium (2.5k-4.5k)", "High (>4500)"]

    plt.figure(figsize=(7, 6), dpi=300)
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Congestion Level Classifier: Test Confusion Matrix", fontsize=11, fontweight="bold", pad=12)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=20)
    plt.yticks(tick_marks, labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontweight="bold")

    plt.xlabel("Predicted Congestion Level")
    plt.ylabel("True Congestion Level")
    plt.tight_layout()
    plt.savefig(images_dir / "congestion_confusion_matrix.png")
    plt.close()

    return clf, metrics


def train_travel_time_module(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    images_dir: Path,
) -> tuple[XGBRegressor, dict[str, float]]:
    """Train and evaluate corridor travel time estimator."""
    logger.info("Training Travel Time Estimator (XGBRegressor)...")
    X_train = df_train[PREDICTOR_FEATURES]
    y_train = df_train["travel_time_min"]

    X_test = df_test[PREDICTOR_FEATURES]
    y_test = df_test["travel_time_min"]

    model = XGBRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(root_mean_squared_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    metrics = {
        "MAE_Minutes": round(mae, 3),
        "RMSE_Minutes": round(rmse, 3),
        "R2": round(r2, 4),
    }

    # Plot Travel Time Predictions
    plt.figure(figsize=(10, 5), dpi=300)
    slice_1w = slice(0, 168)
    time_subset = df_test["date_time"].iloc[slice_1w]

    plt.plot(time_subset, y_test.iloc[slice_1w], label="Actual Travel Time (BPR Formula)", color="black", linewidth=2.0)
    plt.plot(time_subset, y_pred[slice_1w], label="Estimated Travel Time (Model)", color="#d62728", linestyle="--", linewidth=1.8)

    plt.title("15-Mile Corridor Travel Time: Actual vs. Estimated (1-Week Window)", fontsize=11, fontweight="bold", pad=10)
    plt.xlabel("Timestamp")
    plt.ylabel("Travel Time (Minutes)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(images_dir / "travel_time_predictions.png")
    plt.close()

    return model, metrics


def train_accident_risk_module(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    images_dir: Path,
    high_risk_threshold: float = 0.60,
) -> tuple[XGBClassifier, dict[str, float]]:
    """Train and evaluate high accident risk classifier handling class imbalance with SMOTE.

    Note on Data Provenance:
        High accident risk labels are synthetically formulated based on empirical
        crash risk dynamics: volume-to-capacity saturation, weather severity
        (precipitation, snow accumulation, freezing road temperatures), late-night
        visibility, and holiday travel surges. Binary classification target represents
        the upper ~19% most hazardous driving conditions.
    """
    logger.info("Training Accident Risk Classifier with SMOTE Resampling...")
    y_train = (df_train["accident_risk_index"] >= high_risk_threshold).astype(int)
    y_test = (df_test["accident_risk_index"] >= high_risk_threshold).astype(int)

    X_train = df_train[PREDICTOR_FEATURES]
    X_test = df_test[PREDICTOR_FEATURES]

    # Handle class imbalance using Synthetic Minority Over-sampling Technique (SMOTE)
    logger.info("Applying SMOTE on training split (original positives: %d / %d)...", y_train.sum(), len(y_train))
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    logger.info("Resampled training split balanced: %d total samples (%d positives).", len(y_resampled), y_resampled.sum())

    clf = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_resampled, y_resampled)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    precision = float(precision_score(y_test, y_pred))
    recall = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    roc_auc = float(roc_auc_score(y_test, y_prob))

    metrics = {
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "ROC_AUC": round(roc_auc, 4),
        "Positives_Test": int(y_test.sum()),
        "Total_Test": int(len(y_test)),
    }

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(7, 6), dpi=300)
    plt.plot(fpr, tpr, color="#e74c3c", linewidth=2.2, label=f"Accident Risk ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#7f8c8d", linestyle="--", linewidth=1.5, label="Random Guess (AUC = 0.50)")

    plt.title("High Accident Risk Classifier: ROC Curve (Test Set)", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall / Sensitivity)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(images_dir / "accident_risk_roc_curve.png")
    plt.close()

    return clf, metrics


def run_extra_modules_pipeline(
    processed_dir: str | Path = "data/processed",
    models_dir: str | Path = "models",
    docs_dir: str | Path = "docs",
) -> dict[str, dict[str, float]]:
    """Execute end-to-end training and evaluation of all extra prediction modules."""
    proc = Path(processed_dir).resolve()
    df_train = pd.read_parquet(proc / "train.parquet")
    df_test = pd.read_parquet(proc / "test.parquet")

    models_path = Path(models_dir).resolve()
    models_path.mkdir(parents=True, exist_ok=True)

    docs_path = Path(docs_dir).resolve()
    images_dir = docs_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # 1. Congestion Level Classifier
    congestion_clf, congestion_metrics = train_congestion_module(df_train, df_test, images_dir)
    joblib.dump(
        {
            "model": congestion_clf,
            "feature_names": PREDICTOR_FEATURES,
            "classes": ["Low", "Medium", "High"],
            "metrics": congestion_metrics,
            "trained_at": datetime.now().isoformat(),
        },
        models_path / "congestion_classifier.joblib",
    )

    # 2. Travel Time Estimator
    travel_time_model, travel_metrics = train_travel_time_module(df_train, df_test, images_dir)
    joblib.dump(
        {
            "model": travel_time_model,
            "feature_names": PREDICTOR_FEATURES,
            "target": "travel_time_min",
            "metrics": travel_metrics,
            "trained_at": datetime.now().isoformat(),
        },
        models_path / "travel_time_model.joblib",
    )

    # 3. Accident Risk Classifier
    accident_clf, accident_metrics = train_accident_risk_module(df_train, df_test, images_dir)
    joblib.dump(
        {
            "model": accident_clf,
            "feature_names": PREDICTOR_FEATURES,
            "target": "is_high_accident_risk",
            "threshold": 0.60,
            "imbalance_handling": "SMOTE",
            "metrics": accident_metrics,
            "trained_at": datetime.now().isoformat(),
        },
        models_path / "accident_risk_classifier.joblib",
    )

    # Save summary CSV
    summary_records = [
        {"Task": "Congestion Classification", "Primary_Metric": "F1_Macro", "Score": congestion_metrics["F1_Macro"], "Aux_Metric": "ROC_AUC", "Aux_Score": congestion_metrics["ROC_AUC"]},
        {"Task": "Corridor Travel Time Estimation", "Primary_Metric": "MAE_Minutes", "Score": travel_metrics["MAE_Minutes"], "Aux_Metric": "R2", "Aux_Score": travel_metrics["R2"]},
        {"Task": "Accident Risk Detection (SMOTE)", "Primary_Metric": "F1_Score", "Score": accident_metrics["F1"], "Aux_Metric": "ROC_AUC", "Aux_Score": accident_metrics["ROC_AUC"]},
    ]
    df_summary = pd.DataFrame(summary_records)
    csv_out = docs_path / "extra_modules_results.csv"
    df_summary.to_csv(csv_out, index=False)

    print("\n" + "=" * 80)
    print(" FlowCast Phase 5: Extra Prediction Modules Performance Summary")
    print("=" * 80)
    print("1. Congestion Level Classifier (Low/Medium/High):")
    for k, v in congestion_metrics.items():
        print(f"   - {k}: {v}")
    print("\n2. Corridor Travel Time Estimator:")
    for k, v in travel_metrics.items():
        print(f"   - {k}: {v}")
    print("\n3. High Accident Risk Detection (SMOTE Imbalance Handled):")
    for k, v in accident_metrics.items():
        print(f"   - {k}: {v}")
    print("=" * 80)
    print(f"Results CSV : {csv_out}")
    print(f"Artifacts   : {models_path}/[congestion_classifier, travel_time_model, accident_risk_classifier].joblib")
    print("=" * 80 + "\n")

    return {
        "congestion": congestion_metrics,
        "travel_time": travel_metrics,
        "accident_risk": accident_metrics,
    }


def main() -> None:
    """CLI entrypoint for extra prediction modules."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_extra_modules_pipeline()


if __name__ == "__main__":
    main()
