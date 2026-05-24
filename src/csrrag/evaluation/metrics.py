"""Evaluation metrics for CSR-RAG experiments."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def accuracy(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    yt = np.asarray(list(y_true))
    yp = np.asarray(list(y_pred))
    return float((yt == yp).mean()) if len(yt) else 0.0


def brier_score(y_true: Iterable[int], y_prob: Iterable[float]) -> float:
    yt = np.asarray(list(y_true), dtype=float)
    yp = np.asarray(list(y_prob), dtype=float)
    return float(np.mean((yt - yp) ** 2)) if len(yt) else 0.0


def ece(y_true: Iterable[int], y_prob: Iterable[float], n_bins: int = 10) -> float:
    yt = np.asarray(list(y_true), dtype=float)
    yp = np.asarray(list(y_prob), dtype=float)
    if len(yt) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(yt)
    score = 0.0
    for i in range(n_bins):
        left, right = bins[i], bins[i + 1]
        mask = (yp >= left) & (yp < right) if i < n_bins - 1 else (yp >= left) & (yp <= right)
        if not mask.any():
            continue
        conf = yp[mask].mean()
        acc = yt[mask].mean()
        score += abs(acc - conf) * (mask.sum() / total)
    return float(score)


def selective_accuracy(y_true: Iterable[int], keep_mask: Iterable[int]) -> float:
    yt = np.asarray(list(y_true))
    km = np.asarray(list(keep_mask), dtype=bool)
    if not km.any():
        return 0.0
    return float((yt[km] == 0).mean())


def coverage(keep_mask: Iterable[int]) -> float:
    km = np.asarray(list(keep_mask), dtype=bool)
    return float(km.mean()) if len(km) else 0.0


def decision_metrics_from_risk(y_true: Iterable[int], risk_scores: Iterable[float], tau_answer: float) -> dict[str, float]:
    yt = np.asarray(list(y_true), dtype=int)
    risk = np.asarray(list(risk_scores), dtype=float)
    abstain = (risk > tau_answer).astype(int)
    keep = (abstain == 0).astype(int)
    return {
        "tau_answer": float(tau_answer),
        "decision_accuracy": accuracy(yt, abstain),
        "coverage": coverage(keep),
        "selective_accuracy": selective_accuracy(yt, keep),
    }
