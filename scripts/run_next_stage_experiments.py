from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from csrrag.calibration.methods import make_calibrator
from csrrag.experiments.feature_sets import (
    EMBEDDING_FEATURES,
    EMBEDDING_SCORE,
    FEATURE_GROUPS_V2,
)
from csrrag.features.enhanced import (
    AUDIT_FEATURES,
    V3_EVIDENCE_FEATURES,
    V3_FEATURES,
    V3_RETRIEVAL_INTERACTION_FEATURES,
    extract_audit_features,
    extract_enhanced_features,
)
from csrrag.models.baseline import train_estimator
from csrrag.utils.io import read_jsonl


POLICIES = [
    "balanced",
    "reliable@cov85",
    "risk_control@suff_abstain15",
    "high_precision@cov50",
]
CALIBRATION_METHODS = ["identity", "platt", "isotonic"]
TAU_GRID = [round(i / 100, 2) for i in range(0, 101, 5)]
BOOTSTRAP_METRICS = [
    "coverage",
    "selective_accuracy",
    "insufficient_answer_rate",
    "sufficient_abstain_rate",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run next-stage no-API CSR-RAG experiments with v3 features, baselines, diagnostics, and CIs."
    )
    parser.add_argument("--split-dir", default="data/processed/hotpotqa_global_embedding_splits_1800")
    parser.add_argument(
        "--qa-details",
        default="results/tables/hotpotqa_global_real_rag_qa_eval_strict_100/real_rag_qa_details.csv",
    )
    parser.add_argument("--output-dir", default="results/tables/hotpotqa_global_embedding_rag_next")
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--record-kind-filter",
        default="",
        help="Optional metadata.record_kind value to keep, e.g. natural_global_top5.",
    )
    args = parser.parse_args()

    split_records = {split: read_jsonl(Path(args.split_dir) / f"{split}.jsonl") for split in ("train", "valid", "test")}
    if args.record_kind_filter:
        split_records = {
            split: [record for record in records if record.get("metadata", {}).get("record_kind") == args.record_kind_filter]
            for split, records in split_records.items()
        }
    _validate_split_records(split_records)
    valid_calib_records, valid_policy_records = _split_valid(split_records["valid"])
    feature_records = {split: [_feature_record(record) for record in records] for split, records in split_records.items()}
    valid_calib_features = [_feature_record(record) for record in valid_calib_records]
    valid_policy_features = [_feature_record(record) for record in valid_policy_records]
    feature_sets = _feature_sets()
    _validate_features(feature_records["train"] + valid_calib_features + valid_policy_features + feature_records["test"], feature_sets)

    model_runs = _model_runs(
        train_records=feature_records["train"],
        valid_calib_records=valid_calib_features,
        valid_policy_records=valid_policy_features,
        test_records=feature_records["test"],
        split_records=split_records,
        feature_sets=feature_sets,
    )
    baseline_runs = _score_baseline_runs(split_records, valid_policy_records)
    all_runs = model_runs + baseline_runs
    policy_rows, selected = _policy_rows(all_runs)
    main_rows = _main_rows(policy_rows)
    qa_rows = _qa_rows(Path(args.qa_details), selected) if Path(args.qa_details).exists() else []
    feature_ablation_rows = _feature_ablation_rows(policy_rows)
    calibration_rows = _calibration_rows(policy_rows)
    prediction_rows = _prediction_rows(all_runs)
    case_rows, taxonomy_rows, audit_rows = _diagnostic_rows(split_records["test"], feature_records["test"], selected)
    bootstrap_rows = _bootstrap_rows(selected, args.bootstrap_iters, args.seed)
    if qa_rows:
        bootstrap_rows.extend(_qa_bootstrap_rows(Path(args.qa_details), selected, args.bootstrap_iters, args.seed))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "main_comparison.csv", main_rows, list(main_rows[0].keys()))
    _write_csv(output_dir / "policy_comparison.csv", policy_rows, list(policy_rows[0].keys()))
    _write_csv(output_dir / "feature_ablation.csv", feature_ablation_rows, list(feature_ablation_rows[0].keys()))
    _write_csv(output_dir / "calibration_comparison.csv", calibration_rows, list(calibration_rows[0].keys()))
    _write_csv(output_dir / "prediction_metrics.csv", prediction_rows, list(prediction_rows[0].keys()))
    if qa_rows:
        _write_csv(output_dir / "qa_rescore_comparison.csv", qa_rows, list(qa_rows[0].keys()))
    _write_csv(output_dir / "case_studies.csv", case_rows, list(case_rows[0].keys()))
    _write_csv(output_dir / "failure_taxonomy.csv", taxonomy_rows, list(taxonomy_rows[0].keys()))
    _write_csv(output_dir / "label_audit_summary.csv", audit_rows, list(audit_rows[0].keys()))
    _write_csv(output_dir / "bootstrap_ci.csv", bootstrap_rows, list(bootstrap_rows[0].keys()))
    _write_summary(output_dir / "next_stage_summary.md", main_rows, policy_rows, qa_rows, taxonomy_rows)
    _write_validation(
        output_dir / "validation_summary.json",
        split_records,
        valid_calib_records,
        valid_policy_records,
        all_runs,
        selected,
        qa_rows,
        args,
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "model_runs": len(model_runs),
                "baseline_runs": len(baseline_runs),
                "policy_rows": len(policy_rows),
                "qa_rows": len(qa_rows),
                "no_api_calls": True,
            },
            ensure_ascii=False,
        )
    )


def _feature_record(record: dict[str, Any]) -> dict[str, Any]:
    features = extract_enhanced_features(record)
    audit = extract_audit_features(record)
    return {
        "id": record["id"],
        "original_id": record["metadata"]["original_id"],
        "query": record["query"],
        "gold_answer": record["gold_answer"],
        "sufficiency_label": record["sufficiency_label"],
        **features,
        **audit,
    }


def _feature_sets() -> dict[str, list[str]]:
    retrieval_quality = (
        FEATURE_GROUPS_V2["retrieval"]
        + FEATURE_GROUPS_V2["diversity"]
        + EMBEDDING_SCORE
        + V3_RETRIEVAL_INTERACTION_FEATURES
    )
    return {
        "v2_all": EMBEDDING_FEATURES,
        "v3_all": EMBEDDING_FEATURES + V3_FEATURES,
        "v3_no_query": FEATURE_GROUPS_V2["retrieval"]
        + FEATURE_GROUPS_V2["lexical"]
        + FEATURE_GROUPS_V2["coverage"]
        + FEATURE_GROUPS_V2["diversity"]
        + EMBEDDING_SCORE
        + V3_FEATURES,
        "v3_evidence_only": FEATURE_GROUPS_V2["coverage"] + V3_EVIDENCE_FEATURES,
        "retrieval_quality_only": retrieval_quality,
        "embedding_score_only": EMBEDDING_SCORE,
    }


def _model_runs(
    train_records: list[dict[str, Any]],
    valid_calib_records: list[dict[str, Any]],
    valid_policy_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    split_records: dict[str, list[dict[str, Any]]],
    feature_sets: dict[str, list[str]],
) -> list[dict[str, Any]]:
    run_specs = []
    for feature_set in feature_sets:
        run_specs.append(("logistic_regression", feature_set))
    for estimator in ("random_forest", "gradient_boosting"):
        run_specs.append((estimator, "v3_all"))
        run_specs.append((estimator, "retrieval_quality_only"))

    runs = []
    for estimator, feature_set in run_specs:
        feature_names = feature_sets[feature_set]
        model = train_estimator(
            estimator,
            [_feature_row(record, feature_names) for record in train_records],
            _sufficiency_labels(train_records),
            feature_names,
        )
        raw_by_split = {
            "valid_calib": 1.0 - np.asarray(model.predict_proba([_feature_row(record, feature_names) for record in valid_calib_records]), dtype=float),
            "valid_policy": 1.0 - np.asarray(model.predict_proba([_feature_row(record, feature_names) for record in valid_policy_records]), dtype=float),
            "test": 1.0 - np.asarray(model.predict_proba([_feature_row(record, feature_names) for record in test_records]), dtype=float),
        }
        labels_by_split = {
            "valid_calib": _risk_labels(valid_calib_records),
            "valid_policy": _risk_labels(valid_policy_records),
            "test": _risk_labels(test_records),
        }
        for calibration in CALIBRATION_METHODS:
            calibrator = make_calibrator(calibration)
            calibrator.fit(raw_by_split["valid_calib"], labels_by_split["valid_calib"])
            risk_by_split = {
                split: np.asarray(calibrator.predict(raw), dtype=float)
                for split, raw in raw_by_split.items()
            }
            runs.append(
                {
                    "method": "model",
                    "method_name": f"{estimator}/{feature_set}/{calibration}",
                    "estimator": estimator,
                    "feature_set": feature_set,
                    "calibration": calibration,
                    "n_features": len(feature_names),
                    "valid_policy_labels": labels_by_split["valid_policy"],
                    "test_labels": labels_by_split["test"],
                    "valid_policy_risk": risk_by_split["valid_policy"],
                    "test_risk": risk_by_split["test"],
                    "valid_policy_raw_risk": raw_by_split["valid_policy"],
                    "test_raw_risk": raw_by_split["test"],
                    "test_original_ids": [record["metadata"]["original_id"] for record in split_records["test"]],
                    "test_record_ids": [record["id"] for record in split_records["test"]],
                }
            )
    return runs


def _score_baseline_runs(
    split_records: dict[str, list[dict[str, Any]]],
    valid_policy_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    train_scores = np.asarray([_top1_embedding(record) for record in split_records["train"]], dtype=float)
    score_min = float(np.min(train_scores))
    score_max = float(np.max(train_scores))

    def risk(records: list[dict[str, Any]]) -> np.ndarray:
        scores = np.asarray([_top1_embedding(record) for record in records], dtype=float)
        if score_max <= score_min:
            return np.full(len(records), 0.5, dtype=float)
        normalized = np.clip((scores - score_min) / (score_max - score_min), 0.0, 1.0)
        return 1.0 - normalized

    return [
        {
            "method": "baseline",
            "method_name": "top1_embedding_threshold",
            "estimator": "none",
            "feature_set": "top1_embedding_score",
            "calibration": "identity",
            "n_features": 1,
            "valid_policy_labels": _risk_labels_raw(valid_policy_records),
            "test_labels": _risk_labels_raw(split_records["test"]),
            "valid_policy_risk": risk(valid_policy_records),
            "test_risk": risk(split_records["test"]),
            "valid_policy_raw_risk": risk(valid_policy_records),
            "test_raw_risk": risk(split_records["test"]),
            "test_original_ids": [record["metadata"]["original_id"] for record in split_records["test"]],
            "test_record_ids": [record["id"] for record in split_records["test"]],
        }
    ]


def _policy_rows(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = []
    selected: dict[str, dict[str, Any]] = {}
    for run in runs:
        candidates = []
        for tau in TAU_GRID:
            candidates.append(
                {
                    "tau": tau,
                    "valid": _extended_decision_metrics(run["valid_policy_labels"], run["valid_policy_risk"], tau),
                    "test": _extended_decision_metrics(run["test_labels"], run["test_risk"], tau),
                }
            )
        by_policy = {
            "balanced": _choose(candidates, _balanced_key),
            "reliable@cov85": _choose([item for item in candidates if item["valid"]["coverage"] >= 0.85], _reliable_key),
            "risk_control@suff_abstain15": _choose(
                [item for item in candidates if item["valid"]["sufficient_abstain_rate"] <= 0.15],
                _risk_control_key,
            ),
            "high_precision@cov50": _choose([item for item in candidates if item["valid"]["coverage"] >= 0.50], _risk_control_key),
        }
        for policy in POLICIES:
            item = by_policy[policy]
            row = {
                "method_name": run["method_name"],
                "method_type": run["method"],
                "estimator": run["estimator"],
                "feature_set": run["feature_set"],
                "calibration": run["calibration"],
                "policy": policy,
                "tau_answer": item["tau"],
                "n_features": run["n_features"],
                "raw_test_auroc": _safe_auc(run["test_labels"], run["test_raw_risk"]),
                "raw_test_auprc": _safe_average_precision(run["test_labels"], run["test_raw_risk"]),
                "test_auroc": _safe_auc(run["test_labels"], run["test_risk"]),
                "test_auprc": _safe_average_precision(run["test_labels"], run["test_risk"]),
            }
            row.update(_prefix("valid", item["valid"]))
            row.update(_prefix("test", item["test"]))
            rows.append(row)
            selected[f"{run['method_name']}::{policy}"] = {**run, "policy": policy, "tau": item["tau"], "valid": item["valid"], "test": item["test"]}
    return rows, selected


def _main_rows(policy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        {
            "method": "naive_always_answer",
            "estimator": "none",
            "feature_set": "none",
            "calibration": "none",
            "policy": "always_answer",
            "tau_answer": "",
            "test_decision_accuracy": _naive_sufficient_rate(policy_rows),
            "test_coverage": 1.0,
            "test_selective_accuracy": _naive_sufficient_rate(policy_rows),
            "test_insufficient_answer_rate": 1.0,
            "test_sufficient_abstain_rate": 0.0,
            "test_false_answer_count": _naive_false_answer_count(policy_rows),
            "test_over_abstain_count": 0,
        }
    ]
    targets = [
        ("top1_embedding_threshold", "balanced"),
        ("logistic_regression/embedding_score_only", "balanced"),
        ("logistic_regression/retrieval_quality_only", "balanced"),
        ("logistic_regression/v2_all", "balanced"),
        ("logistic_regression/v3_all", "balanced"),
        ("logistic_regression/v3_all", "risk_control@suff_abstain15"),
        ("gradient_boosting/v3_all", "reliable@cov85"),
    ]
    for method_prefix, policy in targets:
        candidates = [
            row for row in policy_rows if row["method_name"].startswith(method_prefix) and row["policy"] == policy
        ]
        if not candidates:
            continue
        row = max(candidates, key=lambda item: (float(item["valid_decision_accuracy"]), float(item["valid_selective_accuracy"])))
        rows.append(
            {
                "method": row["method_name"],
                "estimator": row["estimator"],
                "feature_set": row["feature_set"],
                "calibration": row["calibration"],
                "policy": row["policy"],
                "tau_answer": row["tau_answer"],
                "test_decision_accuracy": row["test_decision_accuracy"],
                "test_coverage": row["test_coverage"],
                "test_selective_accuracy": row["test_selective_accuracy"],
                "test_insufficient_answer_rate": row["test_insufficient_answer_rate"],
                "test_sufficient_abstain_rate": row["test_sufficient_abstain_rate"],
                "test_false_answer_count": row["test_false_answer_count"],
                "test_over_abstain_count": row["test_over_abstain_count"],
            }
        )
    return rows


def _feature_ablation_rows(policy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in policy_rows:
        if row["estimator"] == "logistic_regression" and row["calibration"] == "isotonic" and row["policy"] in {"balanced", "risk_control@suff_abstain15"}:
            rows.append(row)
    return rows


def _calibration_rows(policy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in policy_rows
        if row["estimator"] == "logistic_regression" and row["feature_set"] in {"v2_all", "v3_all"} and row["policy"] == "balanced"
    ]


def _prediction_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        rows.append(
            {
                "method_name": run["method_name"],
                "method_type": run["method"],
                "estimator": run["estimator"],
                "feature_set": run["feature_set"],
                "calibration": run["calibration"],
                "raw_test_auroc": _safe_auc(run["test_labels"], run["test_raw_risk"]),
                "raw_test_auprc": _safe_average_precision(run["test_labels"], run["test_raw_risk"]),
                "test_auroc": _safe_auc(run["test_labels"], run["test_risk"]),
                "test_auprc": _safe_average_precision(run["test_labels"], run["test_risk"]),
            }
        )
    return rows


def _qa_rows(qa_path: Path, selected: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    qa_records = _read_csv(qa_path)
    rows = [_qa_baseline_row(qa_records)]
    for selected_key in _selected_keys_for_qa(selected):
        run = selected[selected_key]
        risk_by_original_id = _risk_by_original_id(run)
        rows.append(_qa_row(qa_records, run, risk_by_original_id))
    return rows


def _selected_keys_for_qa(selected: dict[str, dict[str, Any]]) -> list[str]:
    prefixes = [
        "top1_embedding_threshold::balanced",
        "logistic_regression/embedding_score_only/isotonic::balanced",
        "logistic_regression/retrieval_quality_only/isotonic::balanced",
        "logistic_regression/v3_all/isotonic::balanced",
        "logistic_regression/v3_all/isotonic::risk_control@suff_abstain15",
        "gradient_boosting/v3_all/isotonic::reliable@cov85",
    ]
    return [key for key in prefixes if key in selected]


def _qa_baseline_row(records: list[dict[str, str]]) -> dict[str, Any]:
    insufficient = [record for record in records if record["sufficiency_label"] == "insufficient"]
    return {
        "method_name": "naive_always_answer",
        "policy": "always_answer",
        "tau_answer": "",
        "n": len(records),
        "coverage": 1.0,
        "answered_count": len(records),
        "answered_f1": _mean(float(record["naive_f1"]) for record in records),
        "insufficient_answer_rate": 1.0 if insufficient else 0.0,
        "false_answer_count": len(insufficient),
        "over_abstain_count": 0,
        "qa_rescore_calls_llm": False,
    }


def _qa_row(records: list[dict[str, str]], run: dict[str, Any], risk_by_original_id: dict[str, float]) -> dict[str, Any]:
    enriched = [(record, risk_by_original_id[record["original_id"]]) for record in records if record["original_id"] in risk_by_original_id]
    answered = [(record, risk) for record, risk in enriched if risk <= run["tau"]]
    insufficient = [record for record, _risk in enriched if record["sufficiency_label"] == "insufficient"]
    sufficient = [record for record, _risk in enriched if record["sufficiency_label"] == "sufficient"]
    answered_insufficient = [record for record, _risk in answered if record["sufficiency_label"] == "insufficient"]
    over_abstained = [
        record for record, risk in enriched if record["sufficiency_label"] == "sufficient" and risk > run["tau"]
    ]
    return {
        "method_name": run["method_name"],
        "policy": run["policy"],
        "tau_answer": run["tau"],
        "n": len(enriched),
        "coverage": len(answered) / len(enriched) if enriched else 0.0,
        "answered_count": len(answered),
        "answered_f1": _mean(float(record["naive_f1"]) for record, _risk in answered),
        "insufficient_answer_rate": len(answered_insufficient) / len(insufficient) if insufficient else 0.0,
        "false_answer_count": len(answered_insufficient),
        "over_abstain_count": len(over_abstained),
        "sufficient_abstain_rate": len(over_abstained) / len(sufficient) if sufficient else 0.0,
        "qa_rescore_calls_llm": False,
    }


def _diagnostic_rows(
    test_records: list[dict[str, Any]],
    test_feature_records: list[dict[str, Any]],
    selected: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    preferred_key = "logistic_regression/v3_all/isotonic::risk_control@suff_abstain15"
    fallback_key = "logistic_regression/v3_all/isotonic::balanced"
    run = selected.get(preferred_key) or selected[fallback_key]
    risks_by_id = {record_id: float(risk) for record_id, risk in zip(run["test_record_ids"], run["test_risk"])}
    features_by_id = {record["id"]: record for record in test_feature_records}
    case_rows = []
    taxonomy_counter: Counter[tuple[str, str]] = Counter()
    audit_counter: Counter[tuple[str, str]] = Counter()
    for record in test_records:
        features = features_by_id[record["id"]]
        risk = risks_by_id[record["id"]]
        decision = "answer" if risk <= run["tau"] else "abstain"
        label = record["sufficiency_label"]
        case_type = _case_type(label, decision)
        taxonomy = _taxonomy(record, features, risk, run["tau"], case_type)
        taxonomy_counter[(case_type, taxonomy)] += 1
        audit_counter[(label, _audit_bucket(features))] += 1
        if case_type in {"false_answer", "over_abstain", "successful_intercept", "safe_answer"}:
            case_rows.append(
                {
                    "method_name": run["method_name"],
                    "policy": run["policy"],
                    "tau_answer": run["tau"],
                    "case_type": case_type,
                    "taxonomy": taxonomy,
                    "id": record["id"],
                    "original_id": record["metadata"]["original_id"],
                    "question": record["query"],
                    "gold_answer": record["gold_answer"],
                    "sufficiency_label": label,
                    "risk_score": risk,
                    "decision": decision,
                    "audit_support_title_coverage": features["audit_support_title_coverage"],
                    "audit_gold_answer_in_top5": features["audit_gold_answer_in_top5"],
                    "query_token_coverage_union": features["query_token_coverage_union"],
                    "title_token_coverage_union": features["title_token_coverage_union"],
                    "embedding_top1_share": features["embedding_top1_share"],
                    "top5_titles": " || ".join(doc.get("title", "") for doc in record["retrieved_docs"]),
                }
            )
    case_rows = _limit_cases(case_rows, limit_per_type=20)
    taxonomy_rows = [
        {"case_type": case_type, "taxonomy": taxonomy, "count": count}
        for (case_type, taxonomy), count in sorted(taxonomy_counter.items())
    ]
    audit_rows = [
        {"sufficiency_label": label, "audit_bucket": bucket, "count": count}
        for (label, bucket), count in sorted(audit_counter.items())
    ]
    return case_rows, taxonomy_rows, audit_rows


def _limit_cases(rows: list[dict[str, Any]], limit_per_type: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["case_type"], []).append(row)
    limited = []
    for case_type, group in grouped.items():
        reverse = case_type in {"successful_intercept", "over_abstain"}
        limited.extend(sorted(group, key=lambda row: float(row["risk_score"]), reverse=reverse)[:limit_per_type])
    return limited


def _bootstrap_rows(selected: dict[str, dict[str, Any]], n_iters: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows = []
    for key in _selected_keys_for_bootstrap(selected):
        run = selected[key]
        labels = np.asarray(run["test_labels"], dtype=int)
        risk = np.asarray(run["test_risk"], dtype=float)
        tau = float(run["tau"])
        metrics_by_name = {metric: [] for metric in BOOTSTRAP_METRICS}
        n = len(labels)
        for _ in range(n_iters):
            indices = rng.integers(0, n, size=n)
            metrics = _extended_decision_metrics(labels[indices], risk[indices], tau)
            for metric in BOOTSTRAP_METRICS:
                metrics_by_name[metric].append(float(metrics[metric]))
        point = _extended_decision_metrics(labels, risk, tau)
        for metric, values in metrics_by_name.items():
            rows.append(_ci_row(run["method_name"], run["policy"], "retrieval_test", metric, float(point[metric]), values))
    return rows


def _qa_bootstrap_rows(qa_path: Path, selected: dict[str, dict[str, Any]], n_iters: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed + 17)
    records = _read_csv(qa_path)
    rows = []
    for key in _selected_keys_for_qa(selected):
        run = selected[key]
        risk_by_original_id = _risk_by_original_id(run)
        enriched = [(record, risk_by_original_id[record["original_id"]]) for record in records if record["original_id"] in risk_by_original_id]
        n = len(enriched)
        metric_values = {metric: [] for metric in ("coverage", "answered_f1", "insufficient_answer_rate")}
        for _ in range(n_iters):
            indices = rng.integers(0, n, size=n)
            sample = [enriched[index] for index in indices]
            metrics = _qa_metrics_from_enriched(sample, run["tau"])
            for metric in metric_values:
                metric_values[metric].append(float(metrics[metric]))
        point = _qa_metrics_from_enriched(enriched, run["tau"])
        for metric, values in metric_values.items():
            rows.append(_ci_row(run["method_name"], run["policy"], "qa100_rescore", metric, float(point[metric]), values))
    return rows


def _selected_keys_for_bootstrap(selected: dict[str, dict[str, Any]]) -> list[str]:
    keys = [
        "top1_embedding_threshold::balanced",
        "logistic_regression/retrieval_quality_only/isotonic::balanced",
        "logistic_regression/v2_all/isotonic::balanced",
        "logistic_regression/v3_all/isotonic::balanced",
        "logistic_regression/v3_all/isotonic::risk_control@suff_abstain15",
        "gradient_boosting/v3_all/isotonic::reliable@cov85",
    ]
    return [key for key in keys if key in selected]


def _ci_row(method_name: str, policy: str, split: str, metric: str, point: float, values: list[float]) -> dict[str, Any]:
    return {
        "method_name": method_name,
        "policy": policy,
        "split": split,
        "metric": metric,
        "point": point,
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "bootstrap_iters": len(values),
    }


def _qa_metrics_from_enriched(enriched: list[tuple[dict[str, str], float]], tau: float) -> dict[str, float]:
    answered = [(record, risk) for record, risk in enriched if risk <= tau]
    insufficient = [record for record, _risk in enriched if record["sufficiency_label"] == "insufficient"]
    answered_insufficient = [record for record, _risk in answered if record["sufficiency_label"] == "insufficient"]
    return {
        "coverage": len(answered) / len(enriched) if enriched else 0.0,
        "answered_f1": _mean(float(record["naive_f1"]) for record, _risk in answered),
        "insufficient_answer_rate": len(answered_insufficient) / len(insufficient) if insufficient else 0.0,
    }


def _write_summary(
    path: Path,
    main_rows: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
    qa_rows: list[dict[str, Any]],
    taxonomy_rows: list[dict[str, Any]],
) -> None:
    best_cov85 = min(
        [row for row in policy_rows if row["policy"] != "high_precision@cov50" and float(row["test_coverage"]) >= 0.85],
        key=lambda row: (float(row["test_insufficient_answer_rate"]), -float(row["test_selective_accuracy"])),
    )
    best_balanced = max(
        [row for row in policy_rows if row["policy"] == "balanced"],
        key=lambda row: float(row["test_decision_accuracy"]),
    )
    text = f"""# CSR-RAG Next-Stage No-API Experiment Summary

## Purpose

This run pauses paper writing and focuses on framework/experiment improvement. It adds deployable v3 features, no-API baselines, strict valid-only policy selection, failure taxonomy, QA100 rescoring, and bootstrap confidence intervals.

## Main Findings

- Best balanced decision accuracy: `{best_balanced["method_name"]}` with decision accuracy {float(best_balanced["test_decision_accuracy"]):.4f}, coverage {float(best_balanced["test_coverage"]):.4f}, insufficient answer rate {float(best_balanced["test_insufficient_answer_rate"]):.4f}.
- Best non-extreme policy with test coverage >= 0.85: `{best_cov85["method_name"]}` / `{best_cov85["policy"]}` with coverage {float(best_cov85["test_coverage"]):.4f}, selective accuracy {float(best_cov85["test_selective_accuracy"]):.4f}, insufficient answer rate {float(best_cov85["test_insufficient_answer_rate"]):.4f}.
- QA rescore rows: {len(qa_rows)}.
- Failure taxonomy rows: {len(taxonomy_rows)}.

## Interpretation

Use these results to decide which framework optimization is promising. Do not treat v3 as final until it improves insufficient-answer risk under reasonable coverage and survives stronger baselines.
"""
    path.write_text(text, encoding="utf-8")


def _write_validation(
    path: Path,
    split_records: dict[str, list[dict[str, Any]]],
    valid_calib_records: list[dict[str, Any]],
    valid_policy_records: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    selected: dict[str, dict[str, Any]],
    qa_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    summary = {
        "split_counts": {split: len(records) for split, records in split_records.items()},
        "valid_calib_count": len(valid_calib_records),
        "valid_policy_count": len(valid_policy_records),
        "valid_calib_label_counts": dict(Counter(record["sufficiency_label"] for record in valid_calib_records)),
        "valid_policy_label_counts": dict(Counter(record["sufficiency_label"] for record in valid_policy_records)),
        "runs": len(runs),
        "selected_policy_runs": len(selected),
        "qa_rows": len(qa_rows),
        "bootstrap_iters": args.bootstrap_iters,
        "seed": args.seed,
        "uses_embedding_api": False,
        "uses_llm_api": False,
        "record_kind_filter": args.record_kind_filter,
        "selection_protocol": "train estimator on train; fit calibration on valid_calib; select policy thresholds on valid_policy; report test only",
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _extended_decision_metrics(labels: np.ndarray, risk_scores: np.ndarray, tau: float) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=int)
    risk_scores = np.asarray(risk_scores, dtype=float)
    answer = risk_scores <= tau
    abstain = ~answer
    sufficient = labels == 0
    insufficient = labels == 1
    correct = (answer & sufficient) | (abstain & insufficient)
    answered_sufficient = answer & sufficient
    answered_insufficient = answer & insufficient
    abstained_sufficient = abstain & sufficient
    abstained_insufficient = abstain & insufficient
    return {
        "decision_accuracy": float(correct.mean()) if len(labels) else 0.0,
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


def _split_valid(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_label: dict[str, list[dict[str, Any]]] = {}
    for record in sorted(records, key=lambda item: item["metadata"]["original_id"]):
        by_label.setdefault(record["sufficiency_label"], []).append(record)
    calib = []
    policy = []
    for label_records in by_label.values():
        for idx, record in enumerate(label_records):
            if idx % 2 == 0:
                calib.append(record)
            else:
                policy.append(record)
    _require({record["sufficiency_label"] for record in calib} == {"sufficient", "insufficient"}, "valid_calib lacks both labels.")
    _require({record["sufficiency_label"] for record in policy} == {"sufficient", "insufficient"}, "valid_policy lacks both labels.")
    return calib, policy


def _validate_split_records(split_records: dict[str, list[dict[str, Any]]]) -> None:
    split_ids = {split: {record["metadata"]["original_id"] for record in records} for split, records in split_records.items()}
    _require(split_ids["train"].isdisjoint(split_ids["valid"]), "train and valid original_id overlap.")
    _require(split_ids["train"].isdisjoint(split_ids["test"]), "train and test original_id overlap.")
    _require(split_ids["valid"].isdisjoint(split_ids["test"]), "valid and test original_id overlap.")
    for split, records in split_records.items():
        _require(records, f"{split} split is empty.")
        labels = {record["sufficiency_label"] for record in records}
        _require(labels == {"sufficient", "insufficient"}, f"{split} must contain both labels.")


def _validate_features(records: list[dict[str, Any]], feature_sets: dict[str, list[str]]) -> None:
    required = set().union(*[set(features) for features in feature_sets.values()]) | set(AUDIT_FEATURES)
    for record in records:
        missing = required - set(record)
        if missing:
            raise ValueError(f"Missing features for {record.get('id')}: {sorted(missing)[:10]}")


def _feature_row(record: dict[str, Any], feature_names: list[str]) -> dict[str, float]:
    return {name: float(record[name]) for name in feature_names}


def _sufficiency_labels(records: list[dict[str, Any]]) -> list[int]:
    return [1 if record["sufficiency_label"] == "sufficient" else 0 for record in records]


def _risk_labels(records: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([0 if record["sufficiency_label"] == "sufficient" else 1 for record in records], dtype=int)


def _risk_labels_raw(records: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([0 if record["sufficiency_label"] == "sufficient" else 1 for record in records], dtype=int)


def _prefix(prefix: str, metrics: dict[str, float | int]) -> dict[str, float | int]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _top1_embedding(record: dict[str, Any]) -> float:
    docs = record.get("retrieved_docs", [])
    return float(docs[0].get("embedding_score", 0.0)) if docs else 0.0


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


def _case_type(label: str, decision: str) -> str:
    if label == "insufficient" and decision == "answer":
        return "false_answer"
    if label == "insufficient" and decision == "abstain":
        return "successful_intercept"
    if label == "sufficient" and decision == "abstain":
        return "over_abstain"
    return "safe_answer"


def _taxonomy(record: dict[str, Any], features: dict[str, Any], risk: float, tau: float, case_type: str) -> str:
    missing_support = float(features["audit_missing_support_title_count"]) > 0.0
    answer_present = float(features["audit_gold_answer_in_top5"]) > 0.0
    high_query_coverage = float(features["query_token_coverage_union"]) >= 0.80
    weak_title_bridge = float(features["title_token_coverage_union"]) < 0.25
    top1_embedding = _top1_embedding(record)
    high_embedding = top1_embedding >= 0.65
    if case_type == "false_answer":
        if missing_support and answer_present:
            return "answer_leakage_with_missing_support"
        if missing_support and high_query_coverage:
            return "missing_support_high_query_coverage"
        if high_embedding:
            return "high_embedding_false_safe"
        return "false_answer_other"
    if case_type == "over_abstain":
        if answer_present and not missing_support:
            return "conservative_despite_answer_and_support"
        if weak_title_bridge:
            return "weak_title_bridge_over_abstain"
        return "over_abstain_other"
    if case_type == "successful_intercept":
        if missing_support and not answer_present:
            return "missing_support_no_answer_leakage"
        if missing_support:
            return "missing_support_intercepted"
        return "intercepted_other"
    return "safe_answer"


def _audit_bucket(features: dict[str, Any]) -> str:
    support_coverage = float(features["audit_support_title_coverage"])
    answer_present = float(features["audit_gold_answer_in_top5"]) > 0.0
    if support_coverage >= 1.0 and answer_present:
        return "support_and_answer_present"
    if support_coverage < 1.0 and answer_present:
        return "missing_support_but_answer_present"
    if support_coverage >= 1.0 and not answer_present:
        return "support_present_answer_absent"
    return "missing_support_answer_absent"


def _naive_sufficient_rate(policy_rows: list[dict[str, Any]]) -> float:
    row = policy_rows[0]
    total = int(row["test_answered_count"]) + int(row["test_abstained_insufficient_count"]) + int(row["test_over_abstain_count"])
    sufficient = int(row["test_answered_count"]) - int(row["test_false_answer_count"]) + int(row["test_over_abstain_count"])
    return float(sufficient / total) if total else 0.0


def _naive_false_answer_count(policy_rows: list[dict[str, Any]]) -> int:
    row = policy_rows[0]
    return int(row["test_false_answer_count"]) + int(row["test_abstained_insufficient_count"])


def _mean(values) -> float:
    value_list = list(values)
    return float(sum(value_list) / len(value_list)) if value_list else 0.0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    merged_fieldnames = list(fieldnames)
    for row in rows:
        for key in row:
            if key not in merged_fieldnames:
                merged_fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=merged_fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
