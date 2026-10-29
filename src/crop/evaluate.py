"""
Evaluation and Benchmarking Script for Crop Recommendation Model (Module C).
Evaluates baseline accuracy on original test split and tests sensitivity on district inputs.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.ensemble import RandomForestClassifier

from config.settings import CROP_DATA_CSV, CROP_MODEL_PKL

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def run_crop_evaluation(test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    """
    Performs empirical evaluation of the Random Forest baseline model.
    Confirms metrics against the 99.09% - 99.7% benchmarks reported in the project synopsis.
    """
    if not CROP_DATA_CSV.exists():
        raise FileNotFoundError(f"Crop dataset missing at {CROP_DATA_CSV}")

    df = pd.read_csv(CROP_DATA_CSV)
    X = df[FEATURES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)

    # 5-Fold Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")

    # Feature importances
    feature_imp = {feat: round(float(imp), 4) for feat, imp in zip(FEATURES, model.feature_importances_)}

    # Sensitivity Analysis on District-level Perturbations (+/- 10% on NPK, temp, rain)
    X_pert = X_test.copy()
    noise = np.random.RandomState(42).normal(1.0, 0.08, size=X_pert.shape)
    X_pert_vals = np.clip(X_pert.values * noise, a_min=0, a_max=None)
    pert_preds = model.predict(X_pert_vals)
    perturbed_acc = accuracy_score(y_test, pert_preds)

    results = {
        "dataset_rows": len(df),
        "crop_classes_count": int(df["label"].nunique()),
        "test_sample_count": len(y_test),
        "test_accuracy_pct": round(float(acc) * 100, 2),
        "macro_precision_pct": round(float(prec) * 100, 2),
        "macro_recall_pct": round(float(rec) * 100, 2),
        "macro_f1_pct": round(float(f1) * 100, 2),
        "cv_mean_accuracy_pct": round(float(cv_scores.mean()) * 100, 2),
        "cv_std_pct": round(float(cv_scores.std()) * 100, 2),
        "perturbed_district_estimate_accuracy_pct": round(float(perturbed_acc) * 100, 2),
        "feature_importances": feature_imp,
        "classes": sorted(df["label"].unique().tolist()),
    }
    return results


if __name__ == "__main__":
    report = run_crop_evaluation()
    print("=== CROP MODEL BENCHMARK REPORT ===")
    print(f"Dataset Size: {report['dataset_rows']} rows, {report['crop_classes_count']} classes")
    print(f"Test Accuracy: {report['test_accuracy_pct']}% (Target in Synopsis: ~99.09% - 99.7%)")
    print(f"Macro F1-Score: {report['macro_f1_pct']}%")
    print(f"5-Fold CV Mean: {report['cv_mean_accuracy_pct']}% +/- {report['cv_std_pct']}%")
    print(f"District Perturbation Tolerance: {report['perturbed_district_estimate_accuracy_pct']}%")
    print("Feature Importances:", report["feature_importances"])
