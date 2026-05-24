"""Lightweight sklearn baselines for sufficiency estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class SufficiencyModel:
    pipeline: Pipeline
    feature_names: list[str]

    def predict_proba(self, rows: Iterable[dict[str, float]]) -> np.ndarray:
        x = np.array([[row[name] for name in self.feature_names] for row in rows], dtype=float)
        return self.pipeline.predict_proba(x)[:, 1]


def train_logistic_regression(
    rows: list[dict[str, float]],
    labels: list[int],
    feature_names: list[str],
) -> SufficiencyModel:
    x = np.array([[row[name] for name in feature_names] for row in rows], dtype=float)
    y = np.array(labels, dtype=int)
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )
    pipeline.fit(x, y)
    return SufficiencyModel(pipeline=pipeline, feature_names=feature_names)


def train_logistic_regression_balanced(
    rows: list[dict[str, float]],
    labels: list[int],
    feature_names: list[str],
) -> SufficiencyModel:
    x = np.array([[row[name] for name in feature_names] for row in rows], dtype=float)
    y = np.array(labels, dtype=int)
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    pipeline.fit(x, y)
    return SufficiencyModel(pipeline=pipeline, feature_names=feature_names)


def train_random_forest(
    rows: list[dict[str, float]],
    labels: list[int],
    feature_names: list[str],
) -> SufficiencyModel:
    x = np.array([[row[name] for name in feature_names] for row in rows], dtype=float)
    y = np.array(labels, dtype=int)
    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x, y)
    return SufficiencyModel(pipeline=model, feature_names=feature_names)


def train_gradient_boosting(
    rows: list[dict[str, float]],
    labels: list[int],
    feature_names: list[str],
) -> SufficiencyModel:
    x = np.array([[row[name] for name in feature_names] for row in rows], dtype=float)
    y = np.array(labels, dtype=int)
    model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=2,
        random_state=42,
    )
    model.fit(x, y)
    return SufficiencyModel(pipeline=model, feature_names=feature_names)


def train_estimator(
    estimator_name: str,
    rows: list[dict[str, float]],
    labels: list[int],
    feature_names: list[str],
) -> SufficiencyModel:
    if estimator_name == "logistic_regression":
        return train_logistic_regression(rows, labels, feature_names)
    if estimator_name == "logistic_regression_balanced":
        return train_logistic_regression_balanced(rows, labels, feature_names)
    if estimator_name == "random_forest":
        return train_random_forest(rows, labels, feature_names)
    if estimator_name == "gradient_boosting":
        return train_gradient_boosting(rows, labels, feature_names)
    raise ValueError(f"Unknown estimator: {estimator_name}")
