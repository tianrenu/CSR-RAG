from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from csrrag.calibration.methods import make_calibrator
from csrrag.evaluation.metrics import brier_score, ece
from csrrag.experiments.feature_sets import EMBEDDING_FEATURES, FORBIDDEN_FEATURE_FIELDS
from csrrag.features.basic import extract_basic_features
from csrrag.models.baseline import train_estimator
from csrrag.utils.io import read_jsonl


ESTIMATORS = [
    "logistic_regression",
    "logistic_regression_balanced",
    "random_forest",
    "gradient_boosting",
]
POLICIES = [
    "balanced",
    "reliable@cov85",
    "risk_control@cov85",
    "high_precision@cov50",
]
MAIN_ESTIMATOR = "logistic_regression"
TARGET_COVERAGE = 0.85
TARGET_INSUFFICIENT_ANSWER_RATE = 0.50
BASELINE_GLOBAL_INSUFFICIENT_ANSWER_RATE = 0.6604


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hard-negative CSR-RAG estimator and policy sweep.")
    parser.add_argument("--split-dir", default="data/processed/hotpotqa_global_hardneg_splits_full_dev")
    parser.add_argument("--output-dir", default="results/tables/hotpotqa_global_hardneg_rag_full_dev")
    parser.add_argument("--calibration", default="isotonic")
    args = parser.parse_args()

    split_records = {split: read_jsonl(Path(args.split_dir) / f"{split}.jsonl") for split in ("train", "valid", "test")}
    _validate_split_records(split_records)
    feature_records = {split: [_feature_record(record) for record in records] for split, records in split_records.items()}
    _validate_feature_records(feature_records["train"] + feature_records["valid"] + feature_records["test"])

    runs = [_fit_and_score(estimator, args.calibration, split_records, feature_records) for estimator in ESTIMATORS]
    policy_rows = []
    curve_rows = []
    prediction_rows = []
    case_rows = []
    selected_by_estimator = {}
    for run in runs:
        run_policy_rows, selected = _select_policies(run)
        selected_by_estimator[run["estimator"]] = selected
        policy_rows.extend(run_policy_rows)
        curve_rows.extend(_threshold_curve_rows(run))
        prediction_rows.append(_prediction_metric_row(run))
        case_rows.extend(_case_study_rows(split_records["test"], run, selected))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    main_rows = _main_comparison_rows(split_records["test"], policy_rows)
    _write_csv(output_dir / "hardneg_main_comparison.csv", main_rows, list(main_rows[0].keys()))
    _write_csv(output_dir / "hardneg_policy_comparison.csv", policy_rows, list(policy_rows[0].keys()))
    _write_csv(output_dir / "hardneg_policy_curve_test.csv", curve_rows, list(curve_rows[0].keys()))
    _write_csv(output_dir / "hardneg_sufficiency_prediction_metrics.csv", prediction_rows, list(prediction_rows[0].keys()))
    _write_csv(output_dir / "hardneg_case_studies.csv", case_rows, list(case_rows[0].keys()))
    _write_summary(output_dir / "hardneg_policy_sweep_summary.md", policy_rows, prediction_rows)
    _write_validation(output_dir / "validation_summary.json", split_records, runs, policy_rows, curve_rows, case_rows)

    main_risk_control = next(
        row
        for row in policy_rows
        if row["estimator"] == MAIN_ESTIMATOR and row["policy"] == "risk_control@cov85"
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "estimators": ESTIMATORS,
                "policy_rows": len(policy_rows),
                "curve_rows": len(curve_rows),
                "main_risk_control_coverage": main_risk_control["test_coverage"],
                "main_risk_control_insufficient_answer_rate": main_risk_control["test_insufficient_answer_rate"],
                "target_met": bool(
                    float(main_risk_control["test_coverage"]) >= TARGET_COVERAGE
                    and float(main_risk_control["test_insufficient_answer_rate"]) < TARGET_INSUFFICIENT_ANSWER_RATE
                ),
            },
            ensure_ascii=False,
        )
    )


def _fit_and_score(
    estimator: str,
    calibration: str,
    split_records: dict[str, list[dict[str, Any]]],
    feature_records: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    train_rows = [_feature_row(record) for record in feature_records["train"]]
    valid_rows = [_feature_row(record) for record in feature_records["valid"]]
    test_rows = [_feature_row(record) for record in feature_records["test"]]
    model = train_estimator(estimator, train_rows, _sufficiency_labels(feature_records["train"]), EMBEDDING_FEATURES)

    valid_raw_risk = 1.0 - np.asarray(model.predict_proba(valid_rows), dtype=float)
    test_raw_risk = 1.0 - np.asarray(model.predict_proba(test_rows), dtype=float)
    valid_labels = _risk_labels(feature_records["valid"])
    test_labels = _risk_labels(feature_records["test"])
    calibrator = make_calibrator(calibration)
    calibrator.fit(valid_raw_risk, valid_labels)
    valid_risk = np.asarray(calibrator.predict(valid_raw_risk), dtype=float)
    test_risk = np.asarray(calibrator.predict(test_raw_risk), dtype=float)
    return {
        "estimator": estimator,
        "calibration": calibration,
        "valid_labels": valid_labels,
        "test_labels": test_labels,
        "valid_risk": valid_risk,
        "test_risk": test_risk,
        "valid_raw_risk": valid_raw_risk,
        "test_raw_risk": test_raw_risk,
        "test_records": split_records["test"],
        "raw_test_brier": brier_score(test_labels, test_raw_risk),
        "raw_test_ece": ece(test_labels, test_raw_risk),
        "calibrated_test_brier": brier_score(test_labels, test_risk),
        "calibrated_test_ece": ece(test_labels, test_risk),
    }


def _select_policies(run: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    candidates = []
    for tau in _threshold_candidates(run["valid_risk"]):
        candidates.append(
            {
                "tau": tau,
                "valid": _extended_decision_metrics(run["valid_labels"], run["valid_risk"], tau),
                "test": _extended_decision_metrics(run["test_labels"], run["test_risk"], tau),
            }
        )

    selected = {
        "balanced": _choose(candidates, _balanced_key),
        "reliable@cov85": _choose([item for item in candidates if item["valid"]["coverage"] >= 0.85], _reliable_key),
        "risk_control@cov85": _choose([item for item in candidates if item["valid"]["coverage"] >= 0.85], _risk_control_cov85_key),
        "high_precision@cov50": _choose([item for item in candidates if item["valid"]["coverage"] >= 0.50], _risk_control_cov85_key),
    }

    score_metrics = _score_metrics(run)
    rows = []
    for policy in POLICIES:
        item = selected[policy]
        row = {
            "estimator": run["estimator"],
            "calibration": run["calibration"],
            "policy": policy,
            "tau_answer": item["tau"],
            **score_metrics,
        }
        row.update(_prefix_metrics("valid", item["valid"]))
        row.update(_prefix_metrics("test", item["test"]))
        row["test_meets_coverage85"] = item["test"]["coverage"] >= TARGET_COVERAGE
        row["test_beats_global_iar_baseline"] = item["test"]["insufficient_answer_rate"] < BASELINE_GLOBAL_INSUFFICIENT_ANSWER_RATE
        row["test_meets_target"] = (
            item["test"]["coverage"] >= TARGET_COVERAGE
            and item["test"]["insufficient_answer_rate"] < TARGET_INSUFFICIENT_ANSWER_RATE
        )
        rows.append(row)
    return rows, selected


def _threshold_curve_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for tau in _threshold_candidates(run["valid_risk"]):
        valid_metrics = _extended_decision_metrics(run["valid_labels"], run["valid_risk"], tau)
        test_metrics = _extended_decision_metrics(run["test_labels"], run["test_risk"], tau)
        row = {"estimator": run["estimator"], "calibration": run["calibration"], "tau_answer": tau}
        row.update(_prefix_metrics("valid", valid_metrics))
        row.update(_prefix_metrics("test", test_metrics))
        rows.append(row)
    return rows


def _prediction_metric_row(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "estimator": run["estimator"],
        "calibration": run["calibration"],
        "raw_test_auroc": _safe_auc(run["test_labels"], run["test_raw_risk"]),
        "raw_test_auprc": _safe_average_precision(run["test_labels"], run["test_raw_risk"]),
        "calibrated_test_auroc": _safe_auc(run["test_labels"], run["test_risk"]),
        "calibrated_test_auprc": _safe_average_precision(run["test_labels"], run["test_risk"]),
        "raw_test_brier": run["raw_test_brier"],
        "raw_test_ece": run["raw_test_ece"],
        "calibrated_test_brier": run["calibrated_test_brier"],
        "calibrated_test_ece": run["calibrated_test_ece"],
    }


def _main_comparison_rows(test_records: list[dict[str, Any]], policy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    test_labels = _risk_labels(test_records)
    sufficient_count = int((test_labels == 0).sum())
    insufficient_count = int((test_labels == 1).sum())
    n_test = len(test_records)
    rows = [
        {
            "method": "Naive RAG",
            "estimator": "none",
            "calibration": "none",
            "policy": "always_answer",
            "tau_answer": "",
            "decision_accuracy": sufficient_count / n_test if n_test else 0.0,
            "coverage": 1.0,
            "selective_accuracy": sufficient_count / n_test if n_test else 0.0,
            "insufficient_answer_rate": 1.0 if insufficient_count else 0.0,
            "sufficient_abstain_rate": 0.0,
            "false_answer_count": insufficient_count,
            "over_abstain_count": 0,
            "n_test": n_test,
        }
    ]
    for policy in ("balanced", "risk_control@cov85"):
        policy_row = next(row for row in policy_rows if row["estimator"] == MAIN_ESTIMATOR and row["policy"] == policy)
        rows.append(
            {
                "method": "CSR-RAG",
                "estimator": policy_row["estimator"],
                "calibration": policy_row["calibration"],
                "policy": policy,
                "tau_answer": policy_row["tau_answer"],
                "decision_accuracy": policy_row["test_decision_accuracy"],
                "coverage": policy_row["test_coverage"],
                "selective_accuracy": policy_row["test_selective_accuracy"],
                "insufficient_answer_rate": policy_row["test_insufficient_answer_rate"],
                "sufficient_abstain_rate": policy_row["test_sufficient_abstain_rate"],
                "false_answer_count": policy_row["test_false_answer_count"],
                "over_abstain_count": policy_row["test_over_abstain_count"],
                "n_test": n_test,
            }
        )
    return rows


def _case_study_rows(
    test_records: list[dict[str, Any]],
    run: dict[str, Any],
    selected: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    risks = [float(risk) for risk in run["test_risk"]]
    enriched = list(zip(test_records, risks))
    for policy in POLICIES:
        tau = float(selected[policy]["tau"])
        buckets = {
            "successful_intercept": [
                (record, risk) for record, risk in enriched if record["sufficiency_label"] == "insufficient" and risk > tau
            ],
            "false_answer": [
                (record, risk) for record, risk in enriched if record["sufficiency_label"] == "insufficient" and risk <= tau
            ],
            "over_abstain": [
                (record, risk) for record, risk in enriched if record["sufficiency_label"] == "sufficient" and risk > tau
            ],
            "safe_answer": [
                (record, risk) for record, risk in enriched if record["sufficiency_label"] == "sufficient" and risk <= tau
            ],
        }
        for case_type, bucket in buckets.items():
            reverse = case_type in {"successful_intercept", "over_abstain"}
            for record, risk in sorted(bucket, key=lambda item: item[1], reverse=reverse)[:5]:
                rows.append(
                    {
                        "estimator": run["estimator"],
                        "calibration": run["calibration"],
                        "policy": policy,
                        "tau_answer": tau,
                        "case_type": case_type,
                        "id": record["id"],
                        "original_id": record["metadata"]["original_id"],
                        "record_kind": record["metadata"].get("record_kind", ""),
                        "question_type": record["metadata"].get("question_type", ""),
                        "sufficiency_label": record["sufficiency_label"],
                        "risk_score": risk,
                        "decision": "answer" if risk <= tau else "abstain",
                        "gold_answer": record["gold_answer"],
                        "question": record["query"],
                        "missing_support_titles": " || ".join(record["metadata"].get("missing_support_titles", [])),
                        "top5_titles": " || ".join(doc["title"] for doc in record["retrieved_docs"]),
                    }
                )
    return rows


def _extended_decision_metrics(labels: np.ndarray, risk_scores: np.ndarray, tau: float) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=int)
    risk_scores = np.asarray(risk_scores, dtype=float)
    answer = risk_scores <= tau
    abstain = ~answer
    sufficient = labels == 0
    insufficient = labels == 1
    correct_decision = ((answer & sufficient) | (abstain & insufficient))
    answered_sufficient = answer & sufficient
    answered_insufficient = answer & insufficient
    abstained_sufficient = abstain & sufficient
    abstained_insufficient = abstain & insufficient
    predicted_insufficient = abstain
    true_positive = int((predicted_insufficient & insufficient).sum())
    false_positive = int((predicted_insufficient & sufficient).sum())
    false_negative = int((answer & insufficient).sum())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / int(insufficient.sum()) if insufficient.any() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "decision_accuracy": float(correct_decision.mean()) if len(labels) else 0.0,
        "coverage": float(answer.mean()) if len(labels) else 0.0,
        "selective_accuracy": float(answered_sufficient.sum() / answer.sum()) if answer.any() else 0.0,
        "insufficient_answer_rate": float(answered_insufficient.sum() / insufficient.sum()) if insufficient.any() else 0.0,
        "sufficient_abstain_rate": float(abstained_sufficient.sum() / sufficient.sum()) if sufficient.any() else 0.0,
        "abstained_insufficient_rate": float(abstained_insufficient.sum() / insufficient.sum()) if insufficient.any() else 0.0,
        "answered_sufficient_rate": float(answered_sufficient.sum() / sufficient.sum()) if sufficient.any() else 0.0,
        "false_answer_count": int(answered_insufficient.sum()),
        "over_abstain_count": int(abstained_sufficient.sum()),
        "abstained_insufficient_count": int(abstained_insufficient.sum()),
        "answered_count": int(answer.sum()),
        "insufficient_precision": precision,
        "insufficient_recall": recall,
        "insufficient_f1": f1,
    }


def _threshold_candidates(valid_risk: np.ndarray) -> list[float]:
    candidates = {0.0, 1.0}
    candidates.update(float(score) for score in valid_risk if np.isfinite(score))
    return sorted(score for score in candidates if 0.0 <= score <= 1.0)


def _score_metrics(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw_test_brier": run["raw_test_brier"],
        "raw_test_ece": run["raw_test_ece"],
        "raw_test_auroc": _safe_auc(run["test_labels"], run["test_raw_risk"]),
        "raw_test_auprc": _safe_average_precision(run["test_labels"], run["test_raw_risk"]),
        "calibrated_test_brier": run["calibrated_test_brier"],
        "calibrated_test_ece": run["calibrated_test_ece"],
        "calibrated_test_auroc": _safe_auc(run["test_labels"], run["test_risk"]),
        "calibrated_test_auprc": _safe_average_precision(run["test_labels"], run["test_risk"]),
    }


def _balanced_key(item: dict[str, Any]) -> tuple[float, float, float]:
    metrics = item["valid"]
    return (metrics["decision_accuracy"], metrics["coverage"], metrics["selective_accuracy"])


def _reliable_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = item["valid"]
    return (metrics["selective_accuracy"], -metrics["insufficient_answer_rate"], metrics["decision_accuracy"], metrics["coverage"])


def _risk_control_cov85_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = item["valid"]
    return (-metrics["insufficient_answer_rate"], metrics["selective_accuracy"], metrics["coverage"], metrics["decision_accuracy"])


def _choose(candidates: list[dict[str, Any]], key_fn) -> dict[str, Any]:
    if not candidates:
        raise ValueError("No threshold candidates satisfy the policy constraints.")
    return max(candidates, key=key_fn)


def _prefix_metrics(prefix: str, metrics: dict[str, float | int]) -> dict[str, float | int]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _feature_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        **extract_basic_features(record),
        "sufficiency_label": record["sufficiency_label"],
    }


def _feature_row(record: dict[str, Any]) -> dict[str, float]:
    return {name: float(record[name]) for name in EMBEDDING_FEATURES}


def _sufficiency_labels(records: list[dict[str, Any]]) -> list[int]:
    return [1 if record["sufficiency_label"] == "sufficient" else 0 for record in records]


def _risk_labels(records: list[dict[str, Any]]) -> np.ndarray:
    return np.array([0 if record["sufficiency_label"] == "sufficient" else 1 for record in records], dtype=int)


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    if len(set(labels.tolist())) < 2:
        return 0.0
    return float(roc_auc_score(labels, scores))


def _safe_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    if len(set(labels.tolist())) < 2:
        return 0.0
    return float(average_precision_score(labels, scores))


def _validate_split_records(split_records: dict[str, list[dict[str, Any]]]) -> None:
    split_ids = {split: {record["metadata"]["original_id"] for record in records} for split, records in split_records.items()}
    _require(split_ids["train"].isdisjoint(split_ids["valid"]), "train and valid original_id overlap.")
    _require(split_ids["train"].isdisjoint(split_ids["test"]), "train and test original_id overlap.")
    _require(split_ids["valid"].isdisjoint(split_ids["test"]), "valid and test original_id overlap.")
    for split, records in split_records.items():
        _require(records, f"{split} split is empty.")
        labels = Counter(record["sufficiency_label"] for record in records)
        kinds = Counter(record["metadata"].get("record_kind", "unknown") for record in records)
        _require(labels["sufficient"] > 0 and labels["insufficient"] > 0, f"{split} must contain both labels.")
        _require(kinds["natural_global_top5"] > 0, f"{split} has no natural records.")
        _require(kinds["hardneg_missing_support_top5"] > 0, f"{split} has no hard-negative records.")
        for record in records:
            _require(len(record.get("retrieved_docs", [])) == 5, f"{record['id']} does not have top_k=5.")
            _require(all("is_support" not in doc for doc in record["retrieved_docs"]), f"{record['id']} exposes is_support.")
            _require(
                all(np.isfinite(float(doc.get("embedding_score", 0.0))) for doc in record["retrieved_docs"]),
                f"{record['id']} has non-finite embedding scores.",
            )
            if record["metadata"].get("record_kind") == "hardneg_missing_support_top5":
                _require(record["metadata"].get("missing_support_titles"), f"{record['id']} does not miss support.")


def _validate_feature_records(records: list[dict[str, Any]]) -> None:
    required = set(EMBEDDING_FEATURES)
    for record in records:
        missing = required - set(record)
        if missing:
            raise ValueError(f"Missing feature fields in {record.get('id')}: {sorted(missing)}")
        forbidden = FORBIDDEN_FEATURE_FIELDS & set(record)
        if forbidden:
            raise ValueError(f"Forbidden support-only feature fields in {record.get('id')}: {sorted(forbidden)}")


def _write_summary(
    path: Path,
    policy_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> None:
    lr_risk = next(row for row in policy_rows if row["estimator"] == MAIN_ESTIMATOR and row["policy"] == "risk_control@cov85")
    best_cov85 = min(
        [row for row in policy_rows if row["policy"] != "high_precision@cov50" and float(row["test_coverage"]) >= TARGET_COVERAGE],
        key=lambda row: (float(row["test_insufficient_answer_rate"]), -float(row["test_selective_accuracy"])),
    )
    best_auprc = max(prediction_rows, key=lambda row: float(row["calibrated_test_auprc"]))
    target_met = float(lr_risk["test_coverage"]) >= TARGET_COVERAGE and float(lr_risk["test_insufficient_answer_rate"]) < TARGET_INSUFFICIENT_ANSWER_RATE
    lower_bound = min(
        float(row["test_insufficient_answer_rate"])
        for row in policy_rows
        if row["policy"] != "high_precision@cov50" and float(row["test_coverage"]) >= TARGET_COVERAGE
    )
    text = f"""# Hard-Negative CSR-RAG Policy Sweep Summary

## Purpose

This stress setting adds hard-negative retrieval records that are embedding-relevant but missing at least one supporting title. It is designed to test whether CSR-RAG can reduce insufficient-answer risk under a harder retrieval distribution.

## Main Findings

- LR risk_control@cov85 coverage: {float(lr_risk["test_coverage"]):.4f}
- LR risk_control@cov85 insufficient answer rate: {float(lr_risk["test_insufficient_answer_rate"]):.4f}
- Target coverage>=0.85 and insufficient answer rate<0.50 met: {target_met}
- Best observed insufficient answer rate under coverage>=0.85: {lower_bound:.4f}
- Best coverage>=0.85 non-extreme policy: `{best_cov85["estimator"]}` / `{best_cov85["policy"]}` with coverage {float(best_cov85["test_coverage"]):.4f}, selective accuracy {float(best_cov85["test_selective_accuracy"]):.4f}, insufficient answer rate {float(best_cov85["test_insufficient_answer_rate"]):.4f}
- Best calibrated insufficient-retrieval AUPRC: `{best_auprc["estimator"]}` with AUPRC {float(best_auprc["calibrated_test_auprc"]):.4f}

## Paper Use

Use this as a stress-setting result, not as a replacement for the natural global retrieval main result. If the target is not met, report it as a failure-mode analysis of the current lightweight risk estimator and the coverage target under an insufficient-heavy stress distribution.
"""
    path.write_text(text, encoding="utf-8")


def _write_validation(
    path: Path,
    split_records: dict[str, list[dict[str, Any]]],
    runs: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
    curve_rows: list[dict[str, Any]],
    case_rows: list[dict[str, Any]],
) -> None:
    label_counts = {
        split: dict(Counter(record["sufficiency_label"] for record in records))
        for split, records in split_records.items()
    }
    hardneg_counts = {
        split: dict(Counter(record["metadata"].get("record_kind", "unknown") for record in records))
        for split, records in split_records.items()
    }
    summary = {
        "split_counts": {split: len(records) for split, records in split_records.items()},
        "split_question_counts": {
            split: len({record["metadata"]["original_id"] for record in records}) for split, records in split_records.items()
        },
        "label_counts": label_counts,
        "hardneg_counts": hardneg_counts,
        "estimators": [run["estimator"] for run in runs],
        "policies": POLICIES,
        "policy_rows": len(policy_rows),
        "expected_policy_rows": len(ESTIMATORS) * len(POLICIES),
        "curve_rows": len(curve_rows),
        "case_rows": len(case_rows),
        "risk_unique_counts": {
            run["estimator"]: len({round(float(score), 10) for score in run["test_risk"]}) for run in runs
        },
        "test_risk_ranges": {
            run["estimator"]: [float(np.min(run["test_risk"])), float(np.max(run["test_risk"]))] for run in runs
        },
        "target_coverage": TARGET_COVERAGE,
        "target_insufficient_answer_rate": TARGET_INSUFFICIENT_ANSWER_RATE,
        "test_coverage85_insufficient_answer_rate_lower_bound": _insufficient_answer_rate_lower_bound(
            split_records["test"],
            TARGET_COVERAGE,
        ),
        "has_coverage85_under_target_iar": any(
            float(row["test_coverage"]) >= TARGET_COVERAGE
            and float(row["test_insufficient_answer_rate"]) < TARGET_INSUFFICIENT_ANSWER_RATE
            for row in policy_rows
        ),
        "has_coverage85_lower_than_previous_global_iar": any(
            float(row["test_coverage"]) >= TARGET_COVERAGE
            and float(row["test_insufficient_answer_rate"]) < BASELINE_GLOBAL_INSUFFICIENT_ANSWER_RATE
            for row in policy_rows
        ),
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _insufficient_answer_rate_lower_bound(records: list[dict[str, Any]], target_coverage: float) -> float:
    labels = _risk_labels(records)
    sufficient = int((labels == 0).sum())
    insufficient = int((labels == 1).sum())
    required_answered = int(np.ceil(target_coverage * len(records)))
    required_insufficient_answered = max(0, required_answered - sufficient)
    return required_insufficient_answered / insufficient if insufficient else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
