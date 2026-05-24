"""Risk calibration methods for CSR-RAG."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class IdentityCalibrator:
    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "IdentityCalibrator":
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return np.asarray(scores, dtype=float)


@dataclass
class PlattCalibrator:
    model: LogisticRegression | None = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        self.model = LogisticRegression(max_iter=2000, random_state=42)
        self.model.fit(np.asarray(scores, dtype=float).reshape(-1, 1), np.asarray(labels, dtype=int))
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("PlattCalibrator must be fitted before predict().")
        return self.model.predict_proba(np.asarray(scores, dtype=float).reshape(-1, 1))[:, 1]


@dataclass
class IsotonicCalibrator:
    model: IsotonicRegression | None = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        self.model = IsotonicRegression(out_of_bounds="clip")
        self.model.fit(np.asarray(scores, dtype=float), np.asarray(labels, dtype=int))
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("IsotonicCalibrator must be fitted before predict().")
        return self.model.predict(np.asarray(scores, dtype=float))


def make_calibrator(method: str):
    method = method.lower()
    if method == "identity":
        return IdentityCalibrator()
    if method == "platt":
        return PlattCalibrator()
    if method == "isotonic":
        return IsotonicCalibrator()
    raise ValueError(f"Unknown calibration method: {method}")
