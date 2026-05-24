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
from csrrag.experiments.feature_sets import EMBEDDING_FEATURES, EMBEDDING_SCORE, FEATURE_GROUPS_V2
from csrrag.features.enhanced import (
    V3_FEATURES,
    V3_RETRIEVAL_INTERACTION_FEATURES,
    extract_audit_features,
    extract_enhanced_features,
)
from csrrag.models.baseline import train_estimator
from csrrag.utils.io import read_jsonl


TOP_KS = [5, 10, 20]
FEATURE_SETS = ["v3_no_query", "v3_all", "retrieval_quality_only", "embedding_score_only"]
CALIBRATIONS = ["identity", "platt", "isotonic"]
ANSWER_POLICIES = ["balanced", "reliable@cov85", "risk_control@suff_abstain15", "high_precision@cov50"]
RETRIEVE_MORE_POLICIES = [
    "retrieve_more_balanced",
    "retrieve_more@cov85",
    "retrieve_more_risk_control@suff_abstain15",
    "retrieve_more_high_precision@cov50",
]
TAU_GRID = [round(i / 100, 2) for i in range(0, 101, 5)]
BOOTSTRAP_METRICS = ["coverage", "selective_accuracy", "insufficient_answer_rate", "sufficient_abstain_rate", "retrieval_rate"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run no-API retrieve-more policy experiments over link-bridge top-5/top-10/top-20 records. "
            "This script selects answer/retrieve-more/abstain thresholds on valid only."
        )
    )
    parser.add_argument("--split-dir", default="data/processed/hotpotqa_official_intro_link_bridge_splits_top20_full_dev")
    parser.add_argument("--record-prefix", default="official_intro_link_bridge_a0p85_p0p00_top")
    parser.add_argument("--output-dir", default="results/tables/hotpotqa_official_intro_link_bridge_retrieve_more")
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = _load_topk_records(Path(args.split_dir), args.record_prefix)
    _validate_records(records)
    valid_calib_ids, valid_policy_ids = _split_valid_ids(records[5]["valid"])
    features = {
        top_k: {
            split: [_feature_record(record) for record in split_records]
            for split, split_records in topk_records.items()
        }
        for top_k, topk_records in records.items()
    }
    feature_sets = _feature_sets()
    _validate_features(features, feature_sets)

    topk_rows = _topk_rows(records)
    prediction_rows, risks = _risk_runs(features, valid_calib_ids, feature_sets)
    policy_rows, selections = _policy_rows(records, risks, valid_policy_ids)
    main_rows, main_keys = _main_rows(policy_rows, selections)
    case_rows = _case_rows(records, selections, main_keys)
    bootstrap_rows = _bootstrap_rows(selections, main_keys, args.bootstrap_iters, args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "topk_sufficiency_curve.csv", topk_rows)
    _write_csv(output_dir / "prediction_metrics.csv", prediction_rows)
    _write_csv(output_dir / "policy_comparison.csv", policy_rows)
    _write_csv(output_dir / "main_comparison.csv", main_rows)
    _write_csv(output_dir / "case_studies.csv", case_rows)
    _write_csv(output_dir / "bootstrap_ci.csv", bootstrap_rows)
    _write_summary(output_dir / "link_bridge_retrieve_more_summary.md", main_rows, topk_rows)
    _write_validation(output_dir / "validation_summary.json", args, records, valid_calib_ids, valid_policy_ids, risks, selections)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "top_ks": sorted(records),
                "risk_runs": len(risks),
                "policy_rows": len(policy_rows),
                "main_rows": len(main_rows),
                "uses_embedding_api": False,
                "uses_llm_api": False,
            },
            ensure_ascii=False,
        )
    )


def _load_topk_records(split_dir: Path, record_prefix: str) -> dict[int, dict[str, list[dict[str, Any]]]]:
    records = {top_k: {"train": [], "valid": [], "test": []} for top_k in TOP_KS}
    for split in ("train", "valid", "test"):
        for record in read_jsonl(split_dir / f"{split}.jsonl"):
            kind = record.get("metadata", {}).get("record_kind", "")
            if not kind.startswith(record_prefix):
                continue
            top_k = int(record["metadata"]["top_k"])
            if top_k in records:
                records[top_k][split].append(record)
    return records


def _feature_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "original_id": record["metadata"]["original_id"],
        "query": record["query"],
        "gold_answer": record["gold_answer"],
        "sufficiency_label": record["sufficiency_label"],
        **extract_enhanced_features(record),
        **extract_audit_features(record),
    }


def _feature_sets() -> dict[str, list[str]]:
    retrieval_quality = (
        FEATURE_GROUPS_V2["retrieval"]
        + FEATURE_GROUPS_V2["diversity"]
        + EMBEDDING_SCORE
        + V3_RETRIEVAL_INTERACTION_FEATURES
    )
    return {
        "v3_no_query": FEATURE_GROUPS_V2["retrieval"]
        + FEATURE_GROUPS_V2["lexical"]
        + FEATURE_GROUPS_V2["coverage"]
        + FEATURE_GROUPS_V2["diversity"]
        + EMBEDDING_SCORE
        + V3_FEATURES,
        "v3_all": EMBEDDING_FEATURES + V3_FEATURES,
        "retrieval_quality_only": retrieval_quality,
        "embedding_score_only": EMBEDDING_SCORE,
    }


def _risk_runs(
    features: dict[int, dict[str, list[dict[str, Any]]]],
    valid_calib_ids: set[str],
    feature_sets: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], dict[tuple[int, str, str], dict[str, Any]]]:
    prediction_rows = []
    risks = {}
    for top_k, split_features in features.items():
        valid_calib = [record for record in split_features["valid"] if record["original_id"] in valid_calib_ids]
        valid_policy = [record for record in split_features["valid"] if record["original_id"] not in valid_calib_ids]
        for feature_set, names in feature_sets.items():
            model = train_estimator(
                "logistic_regression",
                [_feature_row(record, names) for record in split_features["train"]],
                _sufficiency_labels(split_features["train"]),
                names,
            )
            raw = {
                "valid_calib": 1.0 - np.asarray(model.predict_proba([_feature_row(record, names) for record in valid_calib]), dtype=float),
                "valid_policy": 1.0 - np.asarray(model.predict_proba([_feature_row(record, names) for record in valid_policy]), dtype=float),
                "test": 1.0 - np.asarray(model.predict_proba([_feature_row(record, names) for record in split_features["test"]]), dtype=float),
            }
            labels = {
                "valid_calib": _risk_labels(valid_calib),
                "valid_policy": _risk_labels(valid_policy),
                "test": _risk_labels(split_features["test"]),
            }
            for calibration in CALIBRATIONS:
                calibrator = make_calibrator(calibration)
                calibrator.fit(raw["valid_calib"], labels["valid_calib"])
                calibrated = {split: np.asarray(calibrator.predict(values), dtype=float) for split, values in raw.items()}
                key = (top_k, feature_set, calibration)
                risks[key] = {
                    "top_k": top_k,
                    "feature_set": feature_set,
                    "calibration": calibration,
                    "valid_policy_ids": [record["original_id"] for record in valid_policy],
                    "test_ids": [record["original_id"] for record in split_features["test"]],
                    "valid_policy_labels": labels["valid_policy"],
                    "test_labels": labels["test"],
                    "valid_policy_risk": calibrated["valid_policy"],
                    "test_risk": calibrated["test"],
                    "valid_policy_raw_risk": raw["valid_policy"],
                    "test_raw_risk": raw["test"],
                    "n_features": len(names),
                }
                prediction_rows.append(
                    {
                        "top_k": top_k,
                        "method_name": _method_name(top_k, feature_set, calibration),
                        "feature_set": feature_set,
                        "calibration": calibration,
                        "n_features": len(names),
                        "raw_test_auroc": _safe_auc(labels["test"], raw["test"]),
                        "raw_test_auprc": _safe_average_precision(labels["test"], raw["test"]),
                        "test_auroc": _safe_auc(labels["test"], calibrated["test"]),
                        "test_auprc": _safe_average_precision(labels["test"], calibrated["test"]),
                    }
                )
    return prediction_rows, risks


def _policy_rows(
    records: dict[int, dict[str, list[dict[str, Any]]]],
    risks: dict[tuple[int, str, str], dict[str, Any]],
    valid_policy_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = []
    selections = {}
    labels = {top_k: _labels_by_split(split_records, valid_policy_ids) for top_k, split_records in records.items()}
    for top_k in sorted(records):
        selection = _always_answer_selection(top_k, labels[5], labels[top_k])
        rows.append(_selection_row(selection))
        selections[selection["selection_key"]] = selection

    for key, run in risks.items():
        top_k, feature_set, calibration = key
        if top_k != 5:
            continue
        for policy in ANSWER_POLICIES:
            selection = _select_answer_abstain(run, policy)
            rows.append(_selection_row(selection))
            selections[selection["selection_key"]] = selection

    for target_k in (10, 20):
        for feature_set in FEATURE_SETS:
            for calibration in CALIBRATIONS:
                run5 = risks[(5, feature_set, calibration)]
                runk = risks[(target_k, feature_set, calibration)]
                for policy in RETRIEVE_MORE_POLICIES:
                    selection = _select_retrieve_more(run5, runk, labels[5], labels[target_k], target_k, policy)
                    rows.append(_selection_row(selection))
                    selections[selection["selection_key"]] = selection
    return rows, selections


def _always_answer_selection(top_k: int, labels5: dict[str, np.ndarray], labelsk: dict[str, np.ndarray]) -> dict[str, Any]:
    valid = _always_answer_metrics(top_k, labels5["valid_policy"], labelsk["valid_policy"])
    test = _always_answer_metrics(top_k, labels5["test"], labelsk["test"])
    return {
        "selection_key": f"always_answer_top{top_k}",
        "method_name": f"always_answer_top{top_k}",
        "action_type": "always_answer",
        "feature_set": "none",
        "calibration": "none",
        "policy": "always_answer",
        "target_k": top_k,
        "tau_answer_top5": "",
        "tau_answer_after_more": "",
        "n_features": 0,
        "valid": valid,
        "test": test,
        "test_arrays": {
            "labels5": labels5["test"],
            "labelsk": labelsk["test"],
            "risk5": np.zeros_like(labels5["test"], dtype=float),
            "riskk": np.zeros_like(labelsk["test"], dtype=float),
        },
    }


def _select_answer_abstain(run: dict[str, Any], policy: str) -> dict[str, Any]:
    candidates = []
    for tau in TAU_GRID:
        candidates.append(
            {
                "tau": tau,
                "valid": _answer_abstain_metrics(run["valid_policy_labels"], run["valid_policy_risk"], tau),
                "test": _answer_abstain_metrics(run["test_labels"], run["test_risk"], tau),
            }
        )
    item = _choose(candidates, policy)
    method = _method_name(5, run["feature_set"], run["calibration"])
    return {
        "selection_key": f"{method}::{policy}",
        "method_name": method,
        "action_type": "answer_abstain_top5",
        "feature_set": run["feature_set"],
        "calibration": run["calibration"],
        "policy": policy,
        "target_k": 5,
        "tau_answer_top5": item["tau"],
        "tau_answer_after_more": "",
        "n_features": run["n_features"],
        "valid": item["valid"],
        "test": item["test"],
        "test_arrays": {
            "labels5": run["test_labels"],
            "labelsk": run["test_labels"],
            "risk5": run["test_risk"],
            "riskk": run["test_risk"],
        },
    }


def _select_retrieve_more(
    run5: dict[str, Any],
    runk: dict[str, Any],
    labels5: dict[str, np.ndarray],
    labelsk: dict[str, np.ndarray],
    target_k: int,
    policy: str,
) -> dict[str, Any]:
    candidates = []
    for tau5 in TAU_GRID:
        for tauk in TAU_GRID:
            candidates.append(
                {
                    "tau5": tau5,
                    "tauk": tauk,
                    "valid": _retrieve_more_metrics(
                        labels5["valid_policy"], labelsk["valid_policy"], run5["valid_policy_risk"], runk["valid_policy_risk"], tau5, tauk, target_k
                    ),
                    "test": _retrieve_more_metrics(labels5["test"], labelsk["test"], run5["test_risk"], runk["test_risk"], tau5, tauk, target_k),
                }
            )
    item = _choose(candidates, policy)
    method = _method_name(target_k, runk["feature_set"], runk["calibration"])
    return {
        "selection_key": f"{method}::{policy}",
        "method_name": method,
        "action_type": "answer_retrieve_more_abstain",
        "feature_set": runk["feature_set"],
        "calibration": runk["calibration"],
        "policy": policy,
        "target_k": target_k,
        "tau_answer_top5": item["tau5"],
        "tau_answer_after_more": item["tauk"],
        "n_features": runk["n_features"],
        "valid": item["valid"],
        "test": item["test"],
        "test_arrays": {
            "labels5": labels5["test"],
            "labelsk": labelsk["test"],
            "risk5": run5["test_risk"],
            "riskk": runk["test_risk"],
        },
    }


def _answer_abstain_metrics(labels: np.ndarray, risk: np.ndarray, tau: float) -> dict[str, float | int]:
    answer = risk <= tau
    return _decision_metrics(labels, answer, target_k=5, retrieved=np.zeros_like(answer, dtype=bool))


def _retrieve_more_metrics(
    labels5: np.ndarray,
    labelsk: np.ndarray,
    risk5: np.ndarray,
    riskk: np.ndarray,
    tau5: float,
    tauk: float,
    target_k: int,
) -> dict[str, float | int]:
    answer_at5 = risk5 <= tau5
    retrieve = ~answer_at5
    answer_after_more = retrieve & (riskk <= tauk)
    final_answer = answer_at5 | answer_after_more
    final_labels = np.where(answer_at5, labels5, labelsk)
    return _decision_metrics(final_labels, final_answer, target_k=target_k, retrieved=retrieve)


def _always_answer_metrics(top_k: int, labels5: np.ndarray, labelsk: np.ndarray) -> dict[str, float | int]:
    labels = labels5 if top_k == 5 else labelsk
    answer = np.ones_like(labels, dtype=bool)
    retrieved = np.ones_like(labels, dtype=bool) if top_k > 5 else np.zeros_like(labels, dtype=bool)
    return _decision_metrics(labels, answer, target_k=top_k, retrieved=retrieved)


def _decision_metrics(labels: np.ndarray, answer: np.ndarray, target_k: int, retrieved: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=int)
    answer = np.asarray(answer, dtype=bool)
    retrieved = np.asarray(retrieved, dtype=bool)
    sufficient = labels == 0
    insufficient = labels == 1
    answered_sufficient = answer & sufficient
    answered_insufficient = answer & insufficient
    abstained_sufficient = (~answer) & sufficient
    abstained_insufficient = (~answer) & insufficient
    correct = answered_sufficient | abstained_insufficient
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
        "retrieval_rate": float(retrieved.mean()) if len(labels) else 0.0,
        "mean_added_docs": float(retrieved.mean() * max(0, target_k - 5)),
    }


def _choose(candidates: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    if "cov85" in policy:
        constrained = [item for item in candidates if item["valid"]["coverage"] >= 0.85]
        return max(constrained or candidates, key=lambda item: _reliable_key(item["valid"]))
    if "suff_abstain15" in policy:
        constrained = [item for item in candidates if item["valid"]["sufficient_abstain_rate"] <= 0.15]
        return max(constrained or candidates, key=lambda item: _risk_control_key(item["valid"]))
    if "cov50" in policy:
        constrained = [item for item in candidates if item["valid"]["coverage"] >= 0.50]
        return max(constrained or candidates, key=lambda item: _risk_control_key(item["valid"]))
    return max(candidates, key=lambda item: _balanced_key(item["valid"]))


def _balanced_key(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(metrics["decision_accuracy"]),
        float(metrics["coverage"]),
        float(metrics["selective_accuracy"]),
        -float(metrics["mean_added_docs"]),
    )


def _reliable_key(metrics: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        float(metrics["selective_accuracy"]),
        -float(metrics["insufficient_answer_rate"]),
        float(metrics["decision_accuracy"]),
        float(metrics["coverage"]),
        -float(metrics["mean_added_docs"]),
    )


def _risk_control_key(metrics: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        -float(metrics["insufficient_answer_rate"]),
        float(metrics["selective_accuracy"]),
        float(metrics["decision_accuracy"]),
        float(metrics["coverage"]),
        -float(metrics["mean_added_docs"]),
    )


def _main_rows(policy_rows: list[dict[str, Any]], selections: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    keys = []
    selected_specs = [
        ("always_answer_top5", None, None),
        ("always_answer_top10", None, None),
        ("always_answer_top20", None, None),
        (None, "answer_abstain_top5", "balanced"),
        (None, "answer_retrieve_more_abstain", "retrieve_more_balanced"),
        (None, "answer_retrieve_more_abstain", "retrieve_more_risk_control@suff_abstain15"),
        (None, "answer_retrieve_more_abstain", "retrieve_more@cov85"),
    ]
    for exact_key, action_type, policy in selected_specs:
        candidates = []
        if exact_key:
            if exact_key in selections:
                candidates = [selections[exact_key]]
        else:
            candidates = [
                selection
                for selection in selections.values()
                if selection["action_type"] == action_type and selection["policy"] == policy
            ]
        if not candidates:
            continue
        selection = max(candidates, key=lambda item: _valid_key_for_policy(item, item["policy"]))
        rows.append(_main_row(selection))
        keys.append(selection["selection_key"])
    return rows, keys


def _valid_key_for_policy(selection: dict[str, Any], policy: str) -> tuple[float, ...]:
    metrics = selection["valid"]
    if "cov85" in policy:
        return _reliable_key(metrics)
    if "risk_control" in policy or "high_precision" in policy:
        return _risk_control_key(metrics)
    return _balanced_key(metrics)


def _main_row(selection: dict[str, Any]) -> dict[str, Any]:
    return {
        "method_name": selection["method_name"],
        "action_type": selection["action_type"],
        "feature_set": selection["feature_set"],
        "calibration": selection["calibration"],
        "policy": selection["policy"],
        "target_k": selection["target_k"],
        "tau_answer_top5": selection["tau_answer_top5"],
        "tau_answer_after_more": selection["tau_answer_after_more"],
        **_prefix("test", selection["test"]),
    }


def _selection_row(selection: dict[str, Any]) -> dict[str, Any]:
    return {
        "selection_key": selection["selection_key"],
        "method_name": selection["method_name"],
        "action_type": selection["action_type"],
        "feature_set": selection["feature_set"],
        "calibration": selection["calibration"],
        "policy": selection["policy"],
        "target_k": selection["target_k"],
        "tau_answer_top5": selection["tau_answer_top5"],
        "tau_answer_after_more": selection["tau_answer_after_more"],
        "n_features": selection["n_features"],
        **_prefix("valid", selection["valid"]),
        **_prefix("test", selection["test"]),
    }


def _case_rows(
    records: dict[int, dict[str, list[dict[str, Any]]]],
    selections: dict[str, dict[str, Any]],
    selection_keys: list[str],
) -> list[dict[str, Any]]:
    rows = []
    records_by_k = {
        top_k: {record["metadata"]["original_id"]: record for record in records[top_k]["test"]}
        for top_k in sorted(records)
    }
    for key in selection_keys:
        selection = selections[key]
        if selection["action_type"] == "always_answer":
            continue
        arrays = selection["test_arrays"]
        labels5 = arrays["labels5"]
        labelsk = arrays["labelsk"]
        risk5 = arrays["risk5"]
        riskk = arrays["riskk"]
        target_k = int(selection["target_k"])
        ids = [record["metadata"]["original_id"] for record in records[target_k]["test"]]
        if selection["action_type"] == "answer_abstain_top5":
            answer = risk5 <= float(selection["tau_answer_top5"])
            retrieved = np.zeros_like(answer, dtype=bool)
            final_labels = labels5
        else:
            answer_at5 = risk5 <= float(selection["tau_answer_top5"])
            retrieved = ~answer_at5
            answer_after_more = retrieved & (riskk <= float(selection["tau_answer_after_more"]))
            answer = answer_at5 | answer_after_more
            final_labels = np.where(answer_at5, labels5, labelsk)
        for idx, original_id in enumerate(ids):
            if len(rows) >= 240:
                break
            case_type = _case_type(final_labels[idx], answer[idx])
            if case_type not in {"false_answer", "over_abstain", "successful_intercept"}:
                continue
            record5 = records_by_k[5][original_id]
            recordk = records_by_k[target_k][original_id]
            rows.append(
                {
                    "selection_key": key,
                    "case_type": case_type,
                    "original_id": original_id,
                    "question": record5["query"],
                    "gold_answer": record5["gold_answer"],
                    "label_top5": record5["sufficiency_label"],
                    "label_topk": recordk["sufficiency_label"],
                    "risk_top5": float(risk5[idx]),
                    "risk_topk": float(riskk[idx]),
                    "retrieved_more": int(bool(retrieved[idx])),
                    "answered": int(bool(answer[idx])),
                    "top5_titles": " || ".join(doc["title"] for doc in record5["retrieved_docs"]),
                    "topk_new_titles": " || ".join(doc["title"] for doc in recordk["retrieved_docs"][5:]),
                    "missing_support_top5": " || ".join(record5["metadata"].get("missing_support_titles", [])),
                    "missing_support_topk": " || ".join(recordk["metadata"].get("missing_support_titles", [])),
                }
            )
    return rows


def _case_type(label: int, answer: bool) -> str:
    if label == 1 and answer:
        return "false_answer"
    if label == 0 and not answer:
        return "over_abstain"
    if label == 1 and not answer:
        return "successful_intercept"
    return "safe_answer"


def _bootstrap_rows(selections: dict[str, dict[str, Any]], selection_keys: list[str], iters: int, seed: int) -> list[dict[str, Any]]:
    if iters <= 0:
        return []
    rng = np.random.default_rng(seed)
    rows = []
    seen = set()
    for key in selection_keys:
        if key in seen:
            continue
        seen.add(key)
        selection = selections[key]
        arrays = selection["test_arrays"]
        n = len(arrays["labels5"])
        values_by_metric = {metric: [] for metric in BOOTSTRAP_METRICS}
        for _ in range(iters):
            idx = rng.integers(0, n, size=n)
            metrics = _selection_metrics_on_indices(selection, idx)
            for metric in BOOTSTRAP_METRICS:
                values_by_metric[metric].append(float(metrics[metric]))
        for metric, values in values_by_metric.items():
            rows.append(
                {
                    "selection_key": key,
                    "method_name": selection["method_name"],
                    "policy": selection["policy"],
                    "target_k": selection["target_k"],
                    "metric": metric,
                    "point": float(selection["test"][metric]),
                    "ci_low": float(np.quantile(values, 0.025)),
                    "ci_high": float(np.quantile(values, 0.975)),
                    "bootstrap_iters": iters,
                }
            )
    return rows


def _selection_metrics_on_indices(selection: dict[str, Any], idx: np.ndarray) -> dict[str, float | int]:
    arrays = selection["test_arrays"]
    labels5 = arrays["labels5"][idx]
    labelsk = arrays["labelsk"][idx]
    risk5 = arrays["risk5"][idx]
    riskk = arrays["riskk"][idx]
    if selection["action_type"] == "always_answer":
        return _always_answer_metrics(int(selection["target_k"]), labels5, labelsk)
    if selection["action_type"] == "answer_abstain_top5":
        return _answer_abstain_metrics(labels5, risk5, float(selection["tau_answer_top5"]))
    return _retrieve_more_metrics(labels5, labelsk, risk5, riskk, float(selection["tau_answer_top5"]), float(selection["tau_answer_after_more"]), int(selection["target_k"]))


def _topk_rows(records: dict[int, dict[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    rows = []
    for top_k, split_records in records.items():
        for split, split_rows in split_records.items():
            labels = _risk_labels(split_rows)
            audits = [extract_audit_features(record) for record in split_rows]
            sufficient = labels == 0
            rows.append(
                {
                    "split": split,
                    "top_k": top_k,
                    "n": len(split_rows),
                    "sufficient_count": int(sufficient.sum()),
                    "insufficient_count": int((labels == 1).sum()),
                    "sufficient_rate": float(sufficient.mean()) if len(labels) else 0.0,
                    "mean_support_title_coverage": _mean(row["audit_support_title_coverage"] for row in audits),
                    "answer_present_in_returned_docs_rate": _mean(row["audit_gold_answer_in_top5"] for row in audits),
                }
            )
    return rows


def _labels_by_split(split_records: dict[str, list[dict[str, Any]]], valid_policy_ids: set[str]) -> dict[str, np.ndarray]:
    return {
        "valid_policy": _risk_labels([record for record in split_records["valid"] if record["metadata"]["original_id"] in valid_policy_ids]),
        "test": _risk_labels(split_records["test"]),
    }


def _split_valid_ids(valid_records: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    by_label: dict[str, list[dict[str, Any]]] = {}
    for record in sorted(valid_records, key=lambda item: item["metadata"]["original_id"]):
        by_label.setdefault(record["sufficiency_label"], []).append(record)
    calib = set()
    policy = set()
    for label_records in by_label.values():
        for idx, record in enumerate(label_records):
            target = calib if idx % 2 == 0 else policy
            target.add(record["metadata"]["original_id"])
    return calib, policy


def _validate_records(records: dict[int, dict[str, list[dict[str, Any]]]]) -> None:
    for top_k, split_records in records.items():
        split_ids = {split: {record["metadata"]["original_id"] for record in rows} for split, rows in split_records.items()}
        _require(split_ids["train"].isdisjoint(split_ids["valid"]), f"top{top_k}: train/valid overlap.")
        _require(split_ids["train"].isdisjoint(split_ids["test"]), f"top{top_k}: train/test overlap.")
        _require(split_ids["valid"].isdisjoint(split_ids["test"]), f"top{top_k}: valid/test overlap.")
        for split, rows in split_records.items():
            _require(rows, f"top{top_k} {split} split is empty.")
            _require({record["sufficiency_label"] for record in rows} == {"sufficient", "insufficient"}, f"top{top_k} {split} lacks both labels.")
            for record in rows:
                _require(int(record["metadata"]["top_k"]) == top_k, f"{record['id']} has wrong top_k.")
                _require(len(record["retrieved_docs"]) == top_k, f"{record['id']} has wrong doc count.")
    base_ids = {split: [record["metadata"]["original_id"] for record in records[5][split]] for split in ("train", "valid", "test")}
    for top_k in records:
        for split in ("train", "valid", "test"):
            ids = [record["metadata"]["original_id"] for record in records[top_k][split]]
            _require(ids == base_ids[split], f"top{top_k} {split} ids differ from top5.")


def _validate_features(features: dict[int, dict[str, list[dict[str, Any]]]], feature_sets: dict[str, list[str]]) -> None:
    required = set().union(*[set(names) for names in feature_sets.values()])
    for top_features in features.values():
        for records in top_features.values():
            for record in records:
                missing = required - set(record)
                if missing:
                    raise ValueError(f"Missing features for {record['id']}: {sorted(missing)[:10]}")


def _write_summary(path: Path, main_rows: list[dict[str, Any]], topk_rows: list[dict[str, Any]]) -> None:
    topk_lines = "\n".join(
        f"- top-{row['top_k']} {row['split']}: sufficient_rate={float(row['sufficient_rate']):.4f}"
        for row in topk_rows
        if row["split"] == "test"
    )
    main_lines = "\n".join(
        f"- `{row['method_name']}` / `{row['policy']}`: target_k={row['target_k']}, "
        f"coverage={float(row['test_coverage']):.4f}, IAR={float(row['test_insufficient_answer_rate']):.4f}, "
        f"retrieval_rate={float(row['test_retrieval_rate']):.4f}"
        for row in main_rows
    )
    text = f"""# Link-Bridge Retrieve-More Experiment Summary

## Purpose

This no-API run turns link-bridge top-10/top-20 candidates into an answer/retrieve-more/abstain policy. Thresholds are selected on valid_policy only and test is reported once.

## Test Top-k Sufficiency

{topk_lines}

## Main Rows

{main_lines}

## Interpretation

Compare retrieve-more rows against always-answer top-5/top-10/top-20 and answer/abstain top-5. A useful policy should reduce insufficient answer rate without collapsing coverage or over-abstaining on sufficient cases.
"""
    path.write_text(text, encoding="utf-8")


def _write_validation(
    path: Path,
    args: argparse.Namespace,
    records: dict[int, dict[str, list[dict[str, Any]]]],
    valid_calib_ids: set[str],
    valid_policy_ids: set[str],
    risks: dict[tuple[int, str, str], dict[str, Any]],
    selections: dict[str, dict[str, Any]],
) -> None:
    validation = {
        "split_dir": args.split_dir,
        "record_prefix": args.record_prefix,
        "top_ks": sorted(records),
        "split_counts": {top_k: {split: len(rows) for split, rows in split_records.items()} for top_k, split_records in records.items()},
        "valid_calib_count": len(valid_calib_ids),
        "valid_policy_count": len(valid_policy_ids),
        "risk_runs": len(risks),
        "policy_selections": len(selections),
        "bootstrap_iters": args.bootstrap_iters,
        "seed": args.seed,
        "uses_embedding_api": False,
        "uses_llm_api": False,
        "selection_protocol": "train on train; calibrate on valid_calib; select top5 answer and retrieve-more thresholds on valid_policy; report test only",
    }
    path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _feature_row(record: dict[str, Any], names: list[str]) -> dict[str, float]:
    return {name: float(record[name]) for name in names}


def _sufficiency_labels(records: list[dict[str, Any]]) -> list[int]:
    return [1 if record["sufficiency_label"] == "sufficient" else 0 for record in records]


def _risk_labels(records: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([0 if record["sufficiency_label"] == "sufficient" else 1 for record in records], dtype=int)


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(set(np.asarray(labels, dtype=int).tolist())) < 2:
        return 0.0
    return float(roc_auc_score(labels, scores))


def _safe_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(set(np.asarray(labels, dtype=int).tolist())) < 2:
        return 0.0
    return float(average_precision_score(labels, scores))


def _method_name(top_k: int, feature_set: str, calibration: str) -> str:
    return f"top{top_k}/logistic_regression/{feature_set}/{calibration}"


def _prefix(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _mean(values) -> float:
    value_list = list(values)
    return float(sum(value_list) / len(value_list)) if value_list else 0.0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
