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


TAU_GRID = [round(i / 100, 2) for i in range(0, 101, 5)]
ESTIMATORS = ["logistic_regression", "random_forest", "gradient_boosting"]
POLICIES = [
    "balanced",
    "reliable@cov85",
    "risk_control@suff_abstain15",
    "high_precision@cov50",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep CSR-RAG estimators under risk-sensitive threshold policies without API calls."
    )
    parser.add_argument("--split-dir", default="data/processed/hotpotqa_global_embedding_splits_1800")
    parser.add_argument(
        "--qa-details",
        default="results/tables/hotpotqa_global_real_rag_qa_eval_strict_100/real_rag_qa_details.csv",
    )
    parser.add_argument("--output-dir", default="results/tables/hotpotqa_global_embedding_rag_policy_sweep")
    parser.add_argument("--calibration", default="isotonic")
    args = parser.parse_args()

    split_records = {split: read_jsonl(Path(args.split_dir) / f"{split}.jsonl") for split in ("train", "valid", "test")}
    _validate_split_records(split_records)
    feature_records = {split: [_feature_record(record) for record in records] for split, records in split_records.items()}
    _validate_feature_records(feature_records["train"] + feature_records["valid"] + feature_records["test"])

    runs = [_fit_and_score(estimator, args.calibration, split_records, feature_records) for estimator in ESTIMATORS]
    selected_by_estimator = {}
    policy_rows = []
    curve_rows = []
    prediction_rows = []
    qa_rows = [_qa_baseline_row(_read_csv(Path(args.qa_details)))]
    case_rows = []

    qa_records = _read_csv(Path(args.qa_details))
    _validate_qa_records(qa_records, split_records["test"])
    for run in runs:
        run_policy_rows, selected = _select_policies(run)
        selected_by_estimator[run["estimator"]] = selected
        policy_rows.extend(run_policy_rows)
        curve_rows.extend(_threshold_curve_rows(run))
        prediction_rows.append(_prediction_metric_row(run))
        qa_rows.extend(_qa_policy_rows(qa_records, run, selected))
        case_rows.extend(_case_study_rows(qa_records, run, selected))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "estimator_policy_comparison.csv", policy_rows, list(policy_rows[0].keys()))
    _write_csv(output_dir / "estimator_policy_curve_test.csv", curve_rows, list(curve_rows[0].keys()))
    _write_csv(output_dir / "qa_estimator_policy_comparison.csv", qa_rows, list(qa_rows[0].keys()))
    _write_csv(output_dir / "sufficiency_prediction_metrics.csv", prediction_rows, list(prediction_rows[0].keys()))
    _write_csv(output_dir / "estimator_policy_case_studies.csv", case_rows, list(case_rows[0].keys()))
    _write_summary(output_dir / "estimator_policy_sweep_summary.md", policy_rows, qa_rows, prediction_rows)
    _write_validation(output_dir / "validation_summary.json", split_records, runs, policy_rows, curve_rows, qa_rows)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "estimators": ESTIMATORS,
                "policy_rows": len(policy_rows),
                "curve_rows": len(curve_rows),
                "qa_policy_rows": len(qa_rows),
                "qa_rescore_calls_llm": False,
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
        "test_original_ids": [record["metadata"]["original_id"] for record in split_records["test"]],
        "raw_test_brier": brier_score(test_labels, test_raw_risk),
        "raw_test_ece": ece(test_labels, test_raw_risk),
        "calibrated_test_brier": brier_score(test_labels, test_risk),
        "calibrated_test_ece": ece(test_labels, test_risk),
    }


def _select_policies(run: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    candidates = []
    for tau in TAU_GRID:
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
        "risk_control@suff_abstain15": _choose(
            [item for item in candidates if item["valid"]["sufficient_abstain_rate"] <= 0.15],
            _risk_control_key,
        ),
        "high_precision@cov50": _choose([item for item in candidates if item["valid"]["coverage"] >= 0.50], _risk_control_key),
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
        row["test_meets_coverage85"] = item["test"]["coverage"] >= 0.85
        rows.append(row)
    return rows, selected


def _threshold_curve_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for tau in TAU_GRID:
        valid_metrics = _extended_decision_metrics(run["valid_labels"], run["valid_risk"], tau)
        test_metrics = _extended_decision_metrics(run["test_labels"], run["test_risk"], tau)
        row = {"estimator": run["estimator"], "calibration": run["calibration"], "tau_answer": tau}
        row.update(_prefix_metrics("valid", valid_metrics))
        row.update(_prefix_metrics("test", test_metrics))
        rows.append(row)
    return rows


def _prediction_metric_row(run: dict[str, Any]) -> dict[str, Any]:
    valid_best = _choose(
        [
            {
                "tau": tau,
                "valid": _extended_decision_metrics(run["valid_labels"], run["valid_risk"], tau),
                "test": _extended_decision_metrics(run["test_labels"], run["test_risk"], tau),
            }
            for tau in TAU_GRID
        ],
        _balanced_key,
    )
    test_metrics = valid_best["test"]
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
        "balanced_tau_answer": valid_best["tau"],
        "balanced_test_insufficient_precision": test_metrics["insufficient_precision"],
        "balanced_test_insufficient_recall": test_metrics["insufficient_recall"],
        "balanced_test_insufficient_f1": test_metrics["insufficient_f1"],
    }


def _qa_policy_rows(
    qa_records: list[dict[str, str]],
    run: dict[str, Any],
    selected: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    risk_by_original_id = _risk_by_original_id(run)
    rows = []
    for policy in POLICIES:
        tau = float(selected[policy]["tau"])
        rows.append(_qa_row(qa_records, run["estimator"], run["calibration"], policy, tau, risk_by_original_id))
    return rows


def _qa_baseline_row(records: list[dict[str, str]]) -> dict[str, Any]:
    insufficient = [record for record in records if record["sufficiency_label"] == "insufficient"]
    return {
        "estimator": "none",
        "calibration": "none",
        "policy": "naive_always_answer",
        "tau_answer": "",
        "n": len(records),
        "coverage": 1.0,
        "answered_count": len(records),
        "answered_em": _mean(float(record["naive_em"]) for record in records),
        "answered_f1": _mean(float(record["naive_f1"]) for record in records),
        "answered_sufficient_rate": _mean(float(record["sufficiency_label"] == "sufficient") for record in records),
        "insufficient_answer_rate": 1.0 if insufficient else 0.0,
        "sufficient_abstain_rate": 0.0,
        "abstained_insufficient_rate": 0.0,
        "false_answer_count": len(insufficient),
        "over_abstain_count": 0,
        "json_parse_failure_rate": _mean(float(record["llm_json_parse_ok"] != "True") for record in records),
        "format_failure_rate": _mean(float(record["llm_format_ok"] != "True") for record in records),
        "qa_rescore_calls_llm": False,
    }


def _qa_row(
    records: list[dict[str, str]],
    estimator: str,
    calibration: str,
    policy: str,
    tau: float,
    risk_by_original_id: dict[str, float],
) -> dict[str, Any]:
    with_risk = [(record, risk_by_original_id[record["original_id"]]) for record in records]
    answered = [(record, risk) for record, risk in with_risk if risk <= tau]
    insufficient = [record for record in records if record["sufficiency_label"] == "insufficient"]
    sufficient = [record for record in records if record["sufficiency_label"] == "sufficient"]
    answered_insufficient = [record for record, _risk in answered if record["sufficiency_label"] == "insufficient"]
    abstained_insufficient = [record for record, risk in with_risk if record["sufficiency_label"] == "insufficient" and risk > tau]
    over_abstained = [record for record, risk in with_risk if record["sufficiency_label"] == "sufficient" and risk > tau]
    return {
        "estimator": estimator,
        "calibration": calibration,
        "policy": policy,
        "tau_answer": tau,
        "n": len(records),
        "coverage": len(answered) / len(records) if records else 0.0,
        "answered_count": len(answered),
        "answered_em": _mean(float(record["naive_em"]) for record, _risk in answered),
        "answered_f1": _mean(float(record["naive_f1"]) for record, _risk in answered),
        "answered_sufficient_rate": _mean(float(record["sufficiency_label"] == "sufficient") for record, _risk in answered),
        "insufficient_answer_rate": len(answered_insufficient) / len(insufficient) if insufficient else 0.0,
        "sufficient_abstain_rate": len(over_abstained) / len(sufficient) if sufficient else 0.0,
        "abstained_insufficient_rate": len(abstained_insufficient) / len(insufficient) if insufficient else 0.0,
        "false_answer_count": len(answered_insufficient),
        "over_abstain_count": len(over_abstained),
        "json_parse_failure_rate": _mean(float(record["llm_json_parse_ok"] != "True") for record in records),
        "format_failure_rate": _mean(float(record["llm_format_ok"] != "True") for record in records),
        "qa_rescore_calls_llm": False,
    }


def _case_study_rows(
    qa_records: list[dict[str, str]],
    run: dict[str, Any],
    selected: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    risk_by_original_id = _risk_by_original_id(run)
    rows = []
    for policy in POLICIES:
        tau = float(selected[policy]["tau"])
        enriched = [(record, risk_by_original_id[record["original_id"]]) for record in qa_records]
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
                        "sample_index": record["sample_index"],
                        "original_id": record["original_id"],
                        "sufficiency_label": record["sufficiency_label"],
                        "risk_score": risk,
                        "decision": "answer" if risk <= tau else "abstain",
                        "gold_answer": record["gold_answer"],
                        "naive_answer": record["naive_answer"],
                        "naive_em": record["naive_em"],
                        "naive_f1": record["naive_f1"],
                        "top5_titles": record["top5_titles"],
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


def _risk_control_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = item["valid"]
    return (-metrics["insufficient_answer_rate"], metrics["selective_accuracy"], metrics["decision_accuracy"], metrics["coverage"])


def _choose(candidates: list[dict[str, Any]], key_fn) -> dict[str, Any]:
    if not candidates:
        raise ValueError("No threshold candidates satisfy the policy constraints.")
    return max(candidates, key=key_fn)


def _prefix_metrics(prefix: str, metrics: dict[str, float | int]) -> dict[str, float | int]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _feature_record(record: dict[str, Any]) -> dict[str, Any]:
    return {"id": record["id"], **extract_basic_features(record), "sufficiency_label": record["sufficiency_label"]}


def _feature_row(record: dict[str, Any]) -> dict[str, float]:
    return {name: float(record[name]) for name in EMBEDDING_FEATURES}


def _sufficiency_labels(records: list[dict[str, Any]]) -> list[int]:
    return [1 if record["sufficiency_label"] == "sufficient" else 0 for record in records]


def _risk_labels(records: list[dict[str, Any]]) -> np.ndarray:
    return np.array([0 if record["sufficiency_label"] == "sufficient" else 1 for record in records], dtype=int)


def _risk_by_original_id(run: dict[str, Any]) -> dict[str, float]:
    return {original_id: float(risk) for original_id, risk in zip(run["test_original_ids"], run["test_risk"])}


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
        _require(labels["sufficient"] > 0 and labels["insufficient"] > 0, f"{split} must contain both labels.")
        for record in records:
            _require(len(record.get("retrieved_docs", [])) == 5, f"{record['id']} does not have top_k=5.")
            _require(all("is_support" not in doc for doc in record["retrieved_docs"]), f"{record['id']} exposes is_support.")


def _validate_feature_records(records: list[dict[str, Any]]) -> None:
    required = set(EMBEDDING_FEATURES)
    for record in records:
        missing = required - set(record)
        if missing:
            raise ValueError(f"Missing feature fields in {record.get('id')}: {sorted(missing)}")
        forbidden = FORBIDDEN_FEATURE_FIELDS & set(record)
        if forbidden:
            raise ValueError(f"Forbidden support-only feature fields in {record.get('id')}: {sorted(forbidden)}")


def _validate_qa_records(qa_records: list[dict[str, str]], test_records: list[dict[str, Any]]) -> None:
    test_ids = {record["metadata"]["original_id"] for record in test_records}
    qa_ids = {record["original_id"] for record in qa_records}
    missing = qa_ids - test_ids
    _require(not missing, f"QA records include original_id outside test split: {sorted(missing)[:5]}")


def _write_summary(
    path: Path,
    policy_rows: list[dict[str, Any]],
    qa_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> None:
    balanced_rows = [row for row in policy_rows if row["policy"] == "balanced"]
    coverage85_rows = [
        row
        for row in policy_rows
        if row["policy"] != "high_precision@cov50" and float(row["test_coverage"]) >= 0.85
    ]
    best_balanced = max(balanced_rows, key=lambda row: float(row["test_decision_accuracy"]))
    best_cov85 = min(
        coverage85_rows,
        key=lambda row: (float(row["test_insufficient_answer_rate"]), -float(row["test_selective_accuracy"])),
    )
    best_qa_reliability = min(
        [row for row in qa_rows if row["policy"] != "naive_always_answer"],
        key=lambda row: (float(row["insufficient_answer_rate"]), -float(row["coverage"])),
    )
    best_auprc = max(prediction_rows, key=lambda row: float(row["calibrated_test_auprc"]))
    text = f"""# Estimator-Policy Sweep Summary

## Purpose

This sweep checks whether a stronger sufficiency estimator can improve the coverage-risk trade-off under the global embedding retrieval setting. It does not call the embedding API or MiniMax; QA rows are recomputed from the existing 100-sample details.

## Main Findings

- Best balanced test decision accuracy: `{best_balanced["estimator"]}` with decision accuracy {float(best_balanced["test_decision_accuracy"]):.4f}, coverage {float(best_balanced["test_coverage"]):.4f}, insufficient answer rate {float(best_balanced["test_insufficient_answer_rate"]):.4f}.
- Best policy with test coverage >= 0.85 among non-extreme policies: `{best_cov85["estimator"]}` / `{best_cov85["policy"]}` with coverage {float(best_cov85["test_coverage"]):.4f}, selective accuracy {float(best_cov85["test_selective_accuracy"]):.4f}, insufficient answer rate {float(best_cov85["test_insufficient_answer_rate"]):.4f}.
- Best calibrated insufficient-retrieval AUPRC: `{best_auprc["estimator"]}` with AUPRC {float(best_auprc["calibrated_test_auprc"]):.4f} and AUROC {float(best_auprc["calibrated_test_auroc"]):.4f}.
- Most conservative QA100 reliability row: `{best_qa_reliability["estimator"]}` / `{best_qa_reliability["policy"]}` with coverage {float(best_qa_reliability["coverage"]):.4f}, answered F1 {float(best_qa_reliability["answered_f1"]):.4f}, insufficient answer rate {float(best_qa_reliability["insufficient_answer_rate"]):.4f}.

## Paper Use

Keep LogisticRegression as the lightweight CSR-RAG main method. Use RF/GB only as stronger estimator variants if they improve the trade-off under valid-selected policies. The paper claim should stay focused on controllable reliability/coverage trade-offs, not universal calibration improvement.
"""
    path.write_text(text, encoding="utf-8")


def _write_validation(
    path: Path,
    split_records: dict[str, list[dict[str, Any]]],
    runs: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
    curve_rows: list[dict[str, Any]],
    qa_rows: list[dict[str, Any]],
) -> None:
    summary = {
        "split_counts": {split: len(records) for split, records in split_records.items()},
        "estimators": [run["estimator"] for run in runs],
        "policies": POLICIES,
        "policy_rows": len(policy_rows),
        "expected_policy_rows": len(ESTIMATORS) * len(POLICIES),
        "curve_rows": len(curve_rows),
        "expected_curve_rows": len(ESTIMATORS) * len(TAU_GRID),
        "qa_policy_rows": len(qa_rows),
        "qa_rescore_calls_llm": False,
        "risk_unique_counts": {
            run["estimator"]: len({round(float(score), 8) for score in run["test_risk"]}) for run in runs
        },
        "test_risk_ranges": {
            run["estimator"]: [float(np.min(run["test_risk"])), float(np.max(run["test_risk"]))] for run in runs
        },
        "has_coverage85_lower_than_lr_balanced": _has_coverage85_lower_than_lr_balanced(policy_rows),
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _has_coverage85_lower_than_lr_balanced(policy_rows: list[dict[str, Any]]) -> bool:
    baseline = next(
        row
        for row in policy_rows
        if row["estimator"] == "logistic_regression" and row["policy"] == "balanced"
    )
    baseline_iar = float(baseline["test_insufficient_answer_rate"])
    return any(
        float(row["test_coverage"]) >= 0.85 and float(row["test_insufficient_answer_rate"]) < baseline_iar
        for row in policy_rows
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean(values) -> float:
    value_list = list(values)
    return float(sum(value_list) / len(value_list)) if value_list else 0.0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
