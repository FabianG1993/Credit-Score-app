"""Reproducible training for the CreditVision AI demonstrator.

The source dataset is intentionally not versioned. This script must only be run
with approved, non-production data until governance controls are in place.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
import sklearn
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import train_test_split

from api.config import FEATURE_COLUMNS

TARGET = "SeriousDlqin2yrs"


def dataset_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el dataset: {path}")
    frame = pl.read_csv(path).to_pandas()
    required = set(FEATURE_COLUMNS + [TARGET])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
    return frame


def preprocess(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    data = frame.loc[:, FEATURE_COLUMNS + [TARGET]].copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    # Domain caps prevent known sentinel/outlier values from dominating training.
    caps = {"RevolvingUtilizationOfUnsecuredLines": 1, "age": 100, "DebtRatio": 5,
            "MonthlyIncome": 1_000_000, "NumberOfOpenCreditLinesAndLoans": 100,
            "NumberRealEstateLoansOrLines": 20, "NumberOfDependents": 20}
    lateness = ["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTimes90DaysLate", "NumberOfTime60-89DaysPastDueNotWorse"]
    caps.update({column: 24 for column in lateness})
    for column, maximum in caps.items():
        data.loc[(data[column] < 0) | (data[column] > maximum), column] = np.nan
    for column in FEATURE_COLUMNS:
        data[column] = data[column].fillna(data[column].median())
    target = data.pop(TARGET)
    if not set(target.unique()).issubset({0, 1}):
        raise ValueError("La variable objetivo debe contener únicamente 0 y 1.")
    return data, target


def evaluate(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    fpr, tpr, _ = __import__("sklearn.metrics", fromlist=["roc_curve"]).roc_curve(y_true, probabilities)
    prob_true, prob_pred = calibration_curve(y_true, probabilities, n_bins=10, strategy="quantile")
    return {"roc_auc": roc_auc_score(y_true, probabilities), "pr_auc": average_precision_score(y_true, probabilities),
            "gini": 2 * roc_auc_score(y_true, probabilities) - 1, "ks": float(np.max(tpr - fpr)),
            "brier_score": brier_score_loss(y_true, probabilities), "threshold": threshold,
            "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
            "calibration_curve": {"mean_predicted": prob_pred.tolist(), "fraction_positive": prob_true.tolist()}}


def train(train_path: Path, model_path: Path, seed: int, threshold: float) -> dict:
    raw = load_data(train_path)
    features, target = preprocess(raw)
    x_train, x_valid, y_train, y_valid = train_test_split(features, target, test_size=0.2, random_state=seed, stratify=target)
    estimator = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_depth=5, random_state=seed)
    # Calibration is evaluated by Brier score and retained as part of this demo pipeline.
    model = CalibratedClassifierCV(estimator, method="sigmoid", cv=3)
    model.fit(x_train, y_train)
    metrics = evaluate(y_valid, model.predict_proba(x_valid)[:, 1], threshold)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metadata = {"model_version": datetime.now(timezone.utc).strftime("creditvision-%Y%m%dT%H%M%SZ"),
                "trained_at": datetime.now(timezone.utc).isoformat(), "features": FEATURE_COLUMNS,
                "dataset_sha256": dataset_hash(train_path), "seed": seed, "metrics": metrics,
                "libraries": {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "pandas": pd.__version__, "polars": pl.__version__}}
    model_path.with_suffix(model_path.suffix + ".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path(__file__).with_name("train.csv"))
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("credit_score_model.pkl"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.30)
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error("--threshold debe estar entre 0 y 1")
    print(json.dumps(train(args.train, args.model, args.seed, args.threshold)["metrics"], indent=2))
