from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from csrrag.calibration.methods import make_calibrator
from csrrag.evaluation.metrics import brier_score, ece
from csrrag.experiments.feature_sets import EMBEDDING_FEATURES, FORBIDDEN_FEATURE_FIELDS
from csrrag.features.basic import extract_basic_features
from csrrag.models.baseline import train_estimator
from csrrag.utils.io import read_jsonl


TAU_GRID = [round(i / 100, 2) for i in range(0, 101, 5)]
POLICIES = [
    "balanced",
    "reliable@cov85",
    "risk_control@suff_abstain15",
    "high_precision@cov50",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run risk-sensitive threshold policies for global CSR-RAG.")
    parser.add_argument("--split-dir", default="data/processed/hotpotqa_global_embedding_splits_1800")
    parser.add_argument("--qa-details", default="results/tables/hotpotqa_global_real_rag_qa_eval_strict_100/real_rag_qa_details.csv")
    parser.add_argument("--output-dir", default="results/tables/hotpotqa_global_embedding_rag_thresholds")
    parser.add_argument("--estimator", default="logistic_regression")
    parser.add_argument("--calibration", default="isotonic")
    args = parser.parse_args()

    split_records = {split: read_jsonl(Path(args.split_dir) / f"{split}.jsonl") for split in ("train", "valid", "test")}
    _validate_split_records(split_records)
    feature_records = {split: [_feature_record(record) for record in records] for split, records in split_records.items()}
    _validate_feature_records(feature_records["train"] + feature_records["valid"] + feature_records["test"])

    run = _fit_and_score(args.estimator, args.calibration, feature_records)
    policy_rows, selected = _select_policies(run)
    curve_rows = _threshold_curve_rows(run)
    qa_rows = _qa_policy_rows(Path(args.qa_details), selected)
    case_rows = _case_study_rows(Path(args.qa_details), selected)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "threshold_policy_comparison.csv", policy_rows, list(policy_rows[0].keys()))
    _write_csv(output_dir / "threshold_policy_curve_test.csv", curve_rows, list(curve_rows[0].keys()))
    _write_csv(output_dir / "qa_threshold_policy_comparison.csv", qa_rows, list(qa_rows[0].keys()))
    _write_csv(output_dir / "threshold_case_studies.csv", case_rows, list(case_rows[0].keys()))
    _write_summary(output_dir / "risk_sensitive_threshold_summary.md", policy_rows, qa_rows)
    _write_validation(output_dir / "validation_summary.json", split_records, run, policy_rows, qa_rows)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "policies": len(policy_rows),
                "qa_policies": len(qa_rows),
                "balanced_tau": selected["balanced"]["tau"],
                "risk_control_tau": selected["risk_control@suff_abstain15"]["tau"],
            },
            ensure_ascii=False,
        )
    )


def _fit_and_score(estimator: str, calibration: str, feature_records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
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
        "test_brier": brier_score(test_labels, test_risk),
        "test_ece": ece(test_labels, test_risk),
        "raw_test_brier": brier_score(test_labels, test_raw_risk),
        "raw_test_ece": ece(test_labels, test_raw_risk),
    }


def _select_policies(run: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    candidates = []
    for tau in TAU_GRID:
        valid_metrics = _extended_decision_metrics(run["valid_labels"], run["valid_risk"], tau)
        test_metrics = _extended_decision_metrics(run["test_labels"], run["test_risk"], tau)
        candidates.append({"tau": tau, "valid": valid_metrics, "test": test_metrics})

    selected = {
        "balanced": _choose(candidates, _balanced_key),
        "reliable@cov85": _choose(
            [item for item in candidates if item["valid"]["coverage"] >= 0.85],
            _reliable_key,
        ),
        "risk_control@suff_abstain15": _choose(
            [item for item in candidates if item["valid"]["sufficient_abstain_rate"] <= 0.15],
            _risk_control_key,
        ),
        "high_precision@cov50": _choose(
            [item for item in candidates if item["valid"]["coverage"] >= 0.50],
            _risk_control_key,
        ),
    }

    rows = []
    for policy in POLICIES:
        item = selected[policy]
        row = {
            "policy": policy,
            "estimator": run["estimator"],
            "calibration": run["calibration"],
            "tau_answer": item["tau"],
            "raw_test_brier": run["raw_test_brier"],
            "raw_test_ece": run["raw_test_ece"],
            "calibrated_test_brier": run["test_brier"],
            "calibrated_test_ece": run["test_ece"],
        }
        row.update(_prefix_metrics("valid", item["valid"]))
        row.update(_prefix_metrics("test", item["test"]))
        rows.append(row)
    return rows, selected


def _threshold_curve_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for tau in TAU_GRID:
        valid_metrics = _extended_decision_metrics(run["valid_labels"], run["valid_risk"], tau)
        test_metrics = _extended_decision_metrics(run["test_labels"], run["test_risk"], tau)
        row = {"tau_answer": tau}
        row.update(_prefix_metrics("valid", valid_metrics))
        row.update(_prefix_metrics("test", test_metrics))
        rows.append(row)
    return rows


def _qa_policy_rows(qa_path: Path, selected: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    qa_records = _read_csv(qa_path)
    rows = [_qa_baseline_row(qa_records)]
    for policy in POLICIES:
        tau = float(selected[policy]["tau"])
        rows.append(_qa_row(policy, tau, qa_records))
    return rows


def _qa_baseline_row(records: list[dict[str, str]]) -> dict[str, Any]:
    insufficient = [record for record in records if record["sufficiency_label"] == "insufficient"]
    return {
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
    }


def _qa_row(policy: str, tau: float, records: list[dict[str, str]]) -> dict[str, Any]:
    answered = [record for record in records if float(record["risk_score"]) <= tau]
    insufficient = [record for record in records if record["sufficiency_label"] == "insufficient"]
    sufficient = [record for record in records if record["sufficiency_label"] == "sufficient"]
    answered_insufficient = [record for record in answered if record["sufficiency_label"] == "insufficient"]
    abstained_insufficient = [record for record in insufficient if float(record["risk_score"]) > tau]
    over_abstained = [record for record in sufficient if float(record["risk_score"]) > tau]
    return {
        "policy": policy,
        "tau_answer": tau,
        "n": len(records),
        "coverage": len(answered) / len(records) if records else 0.0,
        "answered_count": len(answered),
        "answered_em": _mean(float(record["naive_em"]) for record in answered),
        "answered_f1": _mean(float(record["naive_f1"]) for record in answered),
        "answered_sufficient_rate": _mean(float(record["sufficiency_label"] == "sufficient") for record in answered),
        "insufficient_answer_rate": len(answered_insufficient) / len(insufficient) if insufficient else 0.0,
        "sufficient_abstain_rate": len(over_abstained) / len(sufficient) if sufficient else 0.0,
        "abstained_insufficient_rate": len(abstained_insufficient) / len(insufficient) if insufficient else 0.0,
        "false_answer_count": len(answered_insufficient),
        "over_abstain_count": len(over_abstained),
        "json_parse_failure_rate": _mean(float(record["llm_json_parse_ok"] != "True") for record in records),
        "format_failure_rate": _mean(float(record["llm_format_ok"] != "True") for record in records),
    }


def _case_study_rows(qa_path: Path, selected: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    records = _read_csv(qa_path)
    rows = []
    for policy in POLICIES:
        tau = float(selected[policy]["tau"])
        buckets = {
            "successful_intercept": [record for record in records if record["sufficiency_label"] == "insufficient" and float(record["risk_score"]) > tau],
            "false_answer": [record for record in records if record["sufficiency_label"] == "insufficient" and float(record["risk_score"]) <= tau],
            "over_abstain": [record for record in records if record["sufficiency_label"] == "sufficient" and float(record["risk_score"]) > tau],
            "safe_answer": [record for record in records if record["sufficiency_label"] == "sufficient" and float(record["risk_score"]) <= tau],
        }
        sort_desc = {"successful_intercept", "over_abstain"}
        for case_type, bucket in buckets.items():
            bucket = sorted(bucket, key=lambda record: float(record["risk_score"]), reverse=case_type in sort_desc)
            for record in bucket[:5]:
                rows.append(
                    {
                        "policy": policy,
                        "tau_answer": tau,
                        "case_type": case_type,
                        "sample_index": record["sample_index"],
                        "original_id": record["original_id"],
                        "sufficiency_label": record["sufficiency_label"],
                        "risk_score": record["risk_score"],
                        "decision": "answer" if float(record["risk_score"]) <= tau else "abstain",
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


def _validate_split_records(split_records: dict[str, list[dict[str, Any]]]) -> None:
    split_ids = {split: {record["metadata"]["original_id"] for record in records} for split, records in split_records.items()}
    _require(split_ids["train"].isdisjoint(split_ids["valid"]), "train and valid original_id overlap.")
    _require(split_ids["train"].isdisjoint(split_ids["test"]), "train and test original_id overlap.")
    _require(split_ids["valid"].isdisjoint(split_ids["test"]), "valid and test original_id overlap.")
    for split, records in split_records.items():
        _require(records, f"{split} split is empty.")
        labels = {record["sufficiency_label"] for record in records}
        _require(labels == {"sufficient", "insufficient"}, f"{split} must contain both labels.")
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


def _write_summary(path: Path, policy_rows: list[dict[str, Any]], qa_rows: list[dict[str, Any]]) -> None:
    balanced = _find_row(policy_rows, "policy", "balanced")
    risk_control = _find_row(policy_rows, "policy", "risk_control@suff_abstain15")
    high_precision = _find_row(policy_rows, "policy", "high_precision@cov50")
    qa_balanced = _find_row(qa_rows, "policy", "balanced")
    qa_risk_control = _find_row(qa_rows, "policy", "risk_control@suff_abstain15")
    qa_high_precision = _find_row(qa_rows, "policy", "high_precision@cov50")
    text = f"""# Risk-Sensitive Threshold Summary

## Main Finding

Risk-sensitive thresholding trades coverage for lower insufficient-answer risk. The default balanced policy optimizes valid decision accuracy, while risk-control policies select stricter thresholds using only valid split metrics.

## Test Split

- Balanced tau: {balanced["tau_answer"]}, coverage: {balanced["test_coverage"]:.4f}, selective accuracy: {balanced["test_selective_accuracy"]:.4f}, insufficient answer rate: {balanced["test_insufficient_answer_rate"]:.4f}
- Risk-control tau: {risk_control["tau_answer"]}, coverage: {risk_control["test_coverage"]:.4f}, selective accuracy: {risk_control["test_selective_accuracy"]:.4f}, insufficient answer rate: {risk_control["test_insufficient_answer_rate"]:.4f}
- High-precision tau: {high_precision["tau_answer"]}, coverage: {high_precision["test_coverage"]:.4f}, selective accuracy: {high_precision["test_selective_accuracy"]:.4f}, insufficient answer rate: {high_precision["test_insufficient_answer_rate"]:.4f}

## QA 100 Rescoring

- Balanced coverage: {qa_balanced["coverage"]:.4f}, answered F1: {qa_balanced["answered_f1"]:.4f}, insufficient answer rate: {qa_balanced["insufficient_answer_rate"]:.4f}
- Risk-control coverage: {qa_risk_control["coverage"]:.4f}, answered F1: {qa_risk_control["answered_f1"]:.4f}, insufficient answer rate: {qa_risk_control["insufficient_answer_rate"]:.4f}
- High-precision coverage: {qa_high_precision["coverage"]:.4f}, answered F1: {qa_high_precision["answered_f1"]:.4f}, insufficient answer rate: {qa_high_precision["insufficient_answer_rate"]:.4f}

## Paper Interpretation

Use balanced CSR-RAG as the default method and risk-sensitive CSR-RAG as the reliability-oriented variant. The reliability variant should be presented as a coverage-risk control mechanism, not as a free improvement.
"""
    path.write_text(text, encoding="utf-8")


def _write_validation(
    path: Path,
    split_records: dict[str, list[dict[str, Any]]],
    run: dict[str, Any],
    policy_rows: list[dict[str, Any]],
    qa_rows: list[dict[str, Any]],
) -> None:
    summary = {
        "split_counts": {split: len(records) for split, records in split_records.items()},
        "policies": [row["policy"] for row in policy_rows],
        "qa_policies": [row["policy"] for row in qa_rows],
        "risk_unique_count": len({round(float(score), 8) for score in run["test_risk"]}),
        "test_risk_min": float(np.min(run["test_risk"])),
        "test_risk_max": float(np.max(run["test_risk"])),
        "threshold_curve_rows": len(TAU_GRID),
        "qa_rescore_calls_llm": False,
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _find_row(rows: list[dict[str, Any]], key: str, value: str) -> dict[str, Any]:
    for row in rows:
        if row[key] == value:
            return row
    raise ValueError(f"Row not found: {key}={value}")


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
