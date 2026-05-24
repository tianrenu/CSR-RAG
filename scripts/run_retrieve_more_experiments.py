from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from csrrag.calibration.methods import make_calibrator
from csrrag.experiments.feature_sets import EMBEDDING_FEATURES, EMBEDDING_SCORE, FEATURE_GROUPS_V2
from csrrag.features.enhanced import (
    V3_EVIDENCE_FEATURES,
    V3_FEATURES,
    V3_RETRIEVAL_INTERACTION_FEATURES,
    extract_audit_features,
    extract_enhanced_features,
)
from csrrag.models.baseline import train_estimator
from csrrag.utils.io import read_jsonl
from prepare_global_hardneg_retrieval_dataset import (
    _build_global_doc_pool,
    _doc_embedding_text,
    _embedding_cache_get,
    _load_embedding_caches,
    _load_raw_rows,
    _make_record,
    _make_retrieved_docs,
)


TOP_KS = [5, 8, 10, 15]
CALIBRATION_METHODS = ["identity", "platt", "isotonic"]
TAU_GRID = [round(i / 100, 2) for i in range(0, 101, 5)]
POLICIES = [
    "balanced",
    "reliable@cov85",
    "risk_control@suff_abstain15",
]
RETRIEVE_MORE_POLICIES = [
    "retrieve_more_balanced",
    "retrieve_more@cov85",
    "retrieve_more_risk_control@suff_abstain15",
]
BOOTSTRAP_METRICS = [
    "coverage",
    "selective_accuracy",
    "insufficient_answer_rate",
    "sufficient_abstain_rate",
    "retrieval_rate",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run no-API top-k expansion and retrieve-more experiments for CSR-RAG. "
            "The script only reads existing HotpotQA data and embedding caches."
        )
    )
    parser.add_argument("--raw-hotpot", default="data/raw/hotpotqa/hotpot_dev_fullwiki_v1.json")
    parser.add_argument("--split-dir", default="data/processed/hotpotqa_global_hardneg_splits_full_dev")
    parser.add_argument("--cache-path", default="data/cache/hotpotqa_text_embedding_v4_full_dev.jsonl")
    parser.add_argument(
        "--seed-cache-path",
        action="append",
        default=["data/cache/hotpotqa_text_embedding_v4_1800.jsonl"],
        help="Additional existing embedding cache to reuse. May be passed multiple times.",
    )
    parser.add_argument("--embedding-model", default="text-embedding-v4")
    parser.add_argument("--output-dir", default="results/tables/hotpotqa_full_dev_retrieve_more")
    parser.add_argument("--top-ks", default="5,8,10,15")
    parser.add_argument("--record-kind-filter", default="natural_global_top5")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-questions-per-split",
        type=int,
        default=0,
        help="Debug only. 0 means use all natural records in each split.",
    )
    args = parser.parse_args()

    started = time.time()
    top_ks = _parse_top_ks(args.top_ks)
    base_records = _load_base_natural_records(Path(args.split_dir), args.record_kind_filter, args.max_questions_per_split)
    _validate_split_records(base_records)
    split_ids = {
        split: [record["metadata"]["original_id"] for record in records]
        for split, records in base_records.items()
    }
    selected_ids = [original_id for split in ("train", "valid", "test") for original_id in split_ids[split]]

    raw_rows = _load_raw_rows(args.raw_hotpot)
    global_docs = _build_global_doc_pool(raw_rows, selected_ids)
    embedding_cache = _load_embedding_caches(_dedupe_paths([Path(args.cache_path), *(Path(path) for path in args.seed_cache_path or [])]))
    expanded_records = _build_expanded_records(
        raw_rows=raw_rows,
        split_ids=split_ids,
        global_docs=global_docs,
        embedding_cache=embedding_cache,
        embedding_model=args.embedding_model,
        top_ks=top_ks,
        batch_size=args.batch_size,
    )
    _validate_expanded_records(expanded_records)

    valid_calib_ids, valid_policy_ids = _split_valid_ids(expanded_records[5]["valid"])
    feature_records = {
        top_k: {
            split: [_feature_record(record) for record in records]
            for split, records in split_records.items()
        }
        for top_k, split_records in expanded_records.items()
    }
    feature_sets = _feature_sets()
    _validate_features(feature_records, feature_sets)

    topk_rows = _topk_curve_rows(expanded_records)
    prediction_rows, risks = _risk_runs(
        feature_records=feature_records,
        valid_calib_ids=valid_calib_ids,
        feature_sets=feature_sets,
    )
    policy_rows, selections = _policy_rows(
        expanded_records=expanded_records,
        feature_records=feature_records,
        risks=risks,
        valid_policy_ids=valid_policy_ids,
    )
    main_rows, main_selection_keys = _main_rows(topk_rows, policy_rows, selections, top_ks)
    feature_rows = _feature_ablation_rows(policy_rows, top_ks)
    calibration_rows = _calibration_rows(policy_rows, top_ks)
    case_rows = _case_study_rows(expanded_records, risks, selections, policy_rows, top_ks)
    bootstrap_rows = _bootstrap_rows(selections, main_selection_keys, args.bootstrap_iters, args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "topk_sufficiency_curve.csv", topk_rows)
    _write_csv(output_dir / "prediction_metrics.csv", prediction_rows)
    _write_csv(output_dir / "policy_comparison.csv", policy_rows)
    _write_csv(output_dir / "main_comparison.csv", main_rows)
    _write_csv(output_dir / "feature_ablation.csv", feature_rows)
    _write_csv(output_dir / "calibration_comparison.csv", calibration_rows)
    _write_csv(output_dir / "case_studies.csv", case_rows)
    _write_csv(output_dir / "bootstrap_ci.csv", bootstrap_rows)
    _write_summary(output_dir / "retrieve_more_summary.md", main_rows, topk_rows, case_rows)
    _write_validation(
        output_dir / "validation_summary.json",
        args=args,
        top_ks=top_ks,
        base_records=base_records,
        expanded_records=expanded_records,
        global_docs=global_docs,
        risks=risks,
        elapsed_seconds=time.time() - started,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "top_ks": top_ks,
                "policy_rows": len(policy_rows),
                "main_rows": len(main_rows),
                "bootstrap_rows": len(bootstrap_rows),
                "no_api_calls": True,
            },
            ensure_ascii=False,
        )
    )


def _parse_top_ks(value: str) -> list[int]:
    top_ks = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if 5 not in top_ks:
        top_ks.insert(0, 5)
    if any(top_k < 5 for top_k in top_ks):
        raise ValueError("top-ks must be >= 5.")
    return top_ks


def _load_base_natural_records(split_dir: Path, record_kind: str, max_questions_per_split: int) -> dict[str, list[dict[str, Any]]]:
    split_records = {}
    for split in ("train", "valid", "test"):
        records = [
            record
            for record in read_jsonl(split_dir / f"{split}.jsonl")
            if record.get("metadata", {}).get("record_kind") == record_kind
        ]
        if max_questions_per_split > 0:
            records = records[:max_questions_per_split]
        split_records[split] = records
    return split_records


def _build_expanded_records(
    raw_rows: dict[str, dict[str, Any]],
    split_ids: dict[str, list[str]],
    global_docs: list[dict[str, Any]],
    embedding_cache: dict[str, list[float]],
    embedding_model: str,
    top_ks: list[int],
    batch_size: int,
) -> dict[int, dict[str, list[dict[str, Any]]]]:
    max_k = max(top_ks)
    doc_matrix = np.asarray(
        [_embedding_cache_get(embedding_cache, embedding_model, _doc_embedding_text(doc)) for doc in global_docs],
        dtype=np.float32,
    )
    doc_norms = np.linalg.norm(doc_matrix, axis=1, keepdims=True)
    doc_matrix = doc_matrix / np.maximum(doc_norms, 1e-12)
    expanded = {top_k: {"train": [], "valid": [], "test": []} for top_k in top_ks}
    for split in ("train", "valid", "test"):
        ids = split_ids[split]
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start : start + batch_size]
            query_matrix = np.asarray(
                [_embedding_cache_get(embedding_cache, embedding_model, raw_rows[original_id]["question"]) for original_id in batch_ids],
                dtype=np.float32,
            )
            query_norms = np.linalg.norm(query_matrix, axis=1, keepdims=True)
            query_matrix = query_matrix / np.maximum(query_norms, 1e-12)
            score_matrix = query_matrix.dot(doc_matrix.T)
            top_indices = _top_indices(score_matrix, max_k)
            for row_idx, original_id in enumerate(batch_ids):
                raw = raw_rows[original_id]
                max_docs = _make_retrieved_docs(raw["question"], global_docs, score_matrix[row_idx], top_indices[row_idx])
                for top_k in top_ks:
                    record = _make_record(
                        raw=raw,
                        split=split,
                        retrieved_docs=max_docs[:top_k],
                        top_k=top_k,
                        record_kind=f"natural_global_top{top_k}",
                        forced_missing_support_titles=[],
                    )
                    expanded[top_k][split].append(record)
        print(
            json.dumps(
                {
                    "expanded_split": split,
                    "questions": len(ids),
                    "top_ks": top_ks,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return expanded


def _top_indices(score_matrix: np.ndarray, top_k: int) -> np.ndarray:
    candidate = np.argpartition(-score_matrix, kth=top_k - 1, axis=1)[:, :top_k]
    candidate_scores = np.take_along_axis(score_matrix, candidate, axis=1)
    order = np.argsort(-candidate_scores, axis=1)
    return np.take_along_axis(candidate, order, axis=1)


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


def _risk_runs(
    feature_records: dict[int, dict[str, list[dict[str, Any]]]],
    valid_calib_ids: set[str],
    feature_sets: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], dict[tuple[int, str, str], dict[str, Any]]]:
    prediction_rows = []
    risks: dict[tuple[int, str, str], dict[str, Any]] = {}
    for top_k, split_records in feature_records.items():
        valid_calib = [record for record in split_records["valid"] if record["original_id"] in valid_calib_ids]
        valid_policy = [record for record in split_records["valid"] if record["original_id"] not in valid_calib_ids]
        for feature_set, feature_names in feature_sets.items():
            model = train_estimator(
                "logistic_regression",
                [_feature_row(record, feature_names) for record in split_records["train"]],
                _sufficiency_labels(split_records["train"]),
                feature_names,
            )
            raw_risk = {
                "valid_calib": 1.0 - np.asarray(model.predict_proba([_feature_row(record, feature_names) for record in valid_calib]), dtype=float),
                "valid_policy": 1.0 - np.asarray(model.predict_proba([_feature_row(record, feature_names) for record in valid_policy]), dtype=float),
                "test": 1.0 - np.asarray(model.predict_proba([_feature_row(record, feature_names) for record in split_records["test"]]), dtype=float),
            }
            labels = {
                "valid_calib": _risk_labels(valid_calib),
                "valid_policy": _risk_labels(valid_policy),
                "test": _risk_labels(split_records["test"]),
            }
            for calibration in CALIBRATION_METHODS:
                calibrator = make_calibrator(calibration)
                calibrator.fit(raw_risk["valid_calib"], labels["valid_calib"])
                calibrated = {
                    split: np.asarray(calibrator.predict(values), dtype=float)
                    for split, values in raw_risk.items()
                }
                key = (top_k, feature_set, calibration)
                risks[key] = {
                    "top_k": top_k,
                    "feature_set": feature_set,
                    "calibration": calibration,
                    "valid_policy_ids": [record["original_id"] for record in valid_policy],
                    "test_ids": [record["original_id"] for record in split_records["test"]],
                    "valid_policy_labels": labels["valid_policy"],
                    "test_labels": labels["test"],
                    "valid_policy_risk": calibrated["valid_policy"],
                    "test_risk": calibrated["test"],
                    "valid_policy_raw_risk": raw_risk["valid_policy"],
                    "test_raw_risk": raw_risk["test"],
                    "n_features": len(feature_names),
                }
                prediction_rows.append(
                    {
                        "top_k": top_k,
                        "method_name": _method_name(top_k, feature_set, calibration),
                        "feature_set": feature_set,
                        "calibration": calibration,
                        "n_features": len(feature_names),
                        "raw_test_auroc": _safe_auc(labels["test"], raw_risk["test"]),
                        "raw_test_auprc": _safe_average_precision(labels["test"], raw_risk["test"]),
                        "test_auroc": _safe_auc(labels["test"], calibrated["test"]),
                        "test_auprc": _safe_average_precision(labels["test"], calibrated["test"]),
                    }
                )
    return prediction_rows, risks


def _policy_rows(
    expanded_records: dict[int, dict[str, list[dict[str, Any]]]],
    feature_records: dict[int, dict[str, list[dict[str, Any]]]],
    risks: dict[tuple[int, str, str], dict[str, Any]],
    valid_policy_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    selections: dict[str, dict[str, Any]] = {}
    labels5 = _labels_by_split(expanded_records[5], valid_policy_ids)
    for top_k, split_records in expanded_records.items():
        baseline = _always_answer_selection(top_k, labels5, _labels_by_split(split_records, valid_policy_ids))
        rows.append(_selection_row(baseline))
        selections[baseline["selection_key"]] = baseline

    for key, run in risks.items():
        top_k, feature_set, calibration = key
        if top_k != 5:
            continue
        for policy in POLICIES:
            selection = _select_answer_abstain(run, policy)
            rows.append(_selection_row(selection))
            selections[selection["selection_key"]] = selection

    for target_k in sorted(k for k in expanded_records if k > 5):
        target_labels = _labels_by_split(expanded_records[target_k], valid_policy_ids)
        for feature_set in _retrieve_more_feature_sets():
            for calibration in CALIBRATION_METHODS:
                run5 = risks[(5, feature_set, calibration)]
                runk = risks[(target_k, feature_set, calibration)]
                for policy in RETRIEVE_MORE_POLICIES:
                    selection = _select_retrieve_more(
                        run5=run5,
                        runk=runk,
                        labels5=labels5,
                        target_labels=target_labels,
                        target_k=target_k,
                        policy=policy,
                    )
                    rows.append(_selection_row(selection))
                    selections[selection["selection_key"]] = selection
    return rows, selections


def _retrieve_more_feature_sets() -> list[str]:
    return ["v3_no_query", "v3_all", "retrieval_quality_only", "embedding_score_only"]


def _always_answer_selection(
    top_k: int,
    labels5: dict[str, np.ndarray],
    labelsk: dict[str, np.ndarray],
) -> dict[str, Any]:
    valid_metrics = _always_answer_metrics(top_k, labels5["valid_policy"], labelsk["valid_policy"])
    test_metrics = _always_answer_metrics(top_k, labels5["test"], labelsk["test"])
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
        "valid": valid_metrics,
        "test": test_metrics,
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
        valid = _answer_abstain_metrics(run["valid_policy_labels"], run["valid_policy_risk"], tau)
        test = _answer_abstain_metrics(run["test_labels"], run["test_risk"], tau)
        candidates.append({"tau": tau, "valid": valid, "test": test})
    item = _choose_for_policy(candidates, policy)
    selection_key = f"{_method_name(5, run['feature_set'], run['calibration'])}::{policy}"
    return {
        "selection_key": selection_key,
        "method_name": _method_name(5, run["feature_set"], run["calibration"]),
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
    target_labels: dict[str, np.ndarray],
    target_k: int,
    policy: str,
) -> dict[str, Any]:
    candidates = []
    for tau5 in TAU_GRID:
        for tauk in TAU_GRID:
            valid = _retrieve_more_metrics(
                labels5=labels5["valid_policy"],
                labelsk=target_labels["valid_policy"],
                risk5=run5["valid_policy_risk"],
                riskk=runk["valid_policy_risk"],
                tau5=tau5,
                tauk=tauk,
                target_k=target_k,
            )
            test = _retrieve_more_metrics(
                labels5=labels5["test"],
                labelsk=target_labels["test"],
                risk5=run5["test_risk"],
                riskk=runk["test_risk"],
                tau5=tau5,
                tauk=tauk,
                target_k=target_k,
            )
            candidates.append({"tau5": tau5, "tauk": tauk, "valid": valid, "test": test})
    item = _choose_for_retrieve_policy(candidates, policy)
    method_name = f"retrieve_more_top{target_k}/logistic_regression/{run5['feature_set']}/{run5['calibration']}"
    selection_key = f"{method_name}::{policy}"
    return {
        "selection_key": selection_key,
        "method_name": method_name,
        "action_type": "answer_retrieve_more_abstain",
        "feature_set": run5["feature_set"],
        "calibration": run5["calibration"],
        "policy": policy,
        "target_k": target_k,
        "tau_answer_top5": item["tau5"],
        "tau_answer_after_more": item["tauk"],
        "n_features": run5["n_features"],
        "valid": item["valid"],
        "test": item["test"],
        "test_arrays": {
            "labels5": labels5["test"],
            "labelsk": target_labels["test"],
            "risk5": run5["test_risk"],
            "riskk": runk["test_risk"],
        },
    }


def _choose_for_policy(candidates: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    if policy == "balanced":
        return max(candidates, key=lambda item: _balanced_key(item["valid"]))
    if policy == "reliable@cov85":
        filtered = [item for item in candidates if item["valid"]["coverage"] >= 0.85]
        return max(filtered, key=lambda item: _reliable_key(item["valid"]))
    if policy == "risk_control@suff_abstain15":
        filtered = [item for item in candidates if item["valid"]["sufficient_abstain_rate"] <= 0.15]
        return max(filtered, key=lambda item: _risk_control_key(item["valid"]))
    raise ValueError(f"Unknown policy: {policy}")


def _choose_for_retrieve_policy(candidates: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    if policy == "retrieve_more_balanced":
        return max(candidates, key=lambda item: (*_balanced_key(item["valid"]), -item["valid"]["mean_added_docs"]))
    if policy == "retrieve_more@cov85":
        filtered = [item for item in candidates if item["valid"]["coverage"] >= 0.85]
        return max(filtered, key=lambda item: (*_reliable_key(item["valid"]), -item["valid"]["mean_added_docs"]))
    if policy == "retrieve_more_risk_control@suff_abstain15":
        filtered = [item for item in candidates if item["valid"]["sufficient_abstain_rate"] <= 0.15]
        return max(filtered, key=lambda item: (*_risk_control_key(item["valid"]), -item["valid"]["mean_added_docs"]))
    raise ValueError(f"Unknown retrieve-more policy: {policy}")


def _always_answer_metrics(top_k: int, labels5: np.ndarray, labelsk: np.ndarray) -> dict[str, float | int]:
    n = len(labelsk)
    answer = np.ones(n, dtype=bool)
    retrieve = np.full(n, top_k > 5, dtype=bool)
    docs_used = np.full(n, top_k, dtype=float)
    return _action_metrics(labels5, labelsk, labelsk, answer, retrieve, docs_used)


def _answer_abstain_metrics(labels: np.ndarray, risk: np.ndarray, tau: float) -> dict[str, float | int]:
    answer = risk <= tau
    retrieve = np.zeros(len(labels), dtype=bool)
    docs_used = np.full(len(labels), 5.0, dtype=float)
    return _action_metrics(labels, labels, labels, answer, retrieve, docs_used)


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
    answer = answer_at5 | answer_after_more
    final_labels = np.where(answer_at5, labels5, labelsk)
    docs_used = np.where(answer_at5, 5.0, float(target_k))
    metrics = _action_metrics(labels5, labelsk, final_labels, answer, retrieve, docs_used)
    metrics["answer_at5_count"] = int(answer_at5.sum())
    metrics["answer_after_more_count"] = int(answer_after_more.sum())
    metrics["abstain_after_more_count"] = int((retrieve & ~answer_after_more).sum())
    return metrics


def _action_metrics(
    labels5: np.ndarray,
    labelsk: np.ndarray,
    final_labels: np.ndarray,
    answer: np.ndarray,
    retrieve: np.ndarray,
    docs_used: np.ndarray,
) -> dict[str, float | int]:
    sufficient = final_labels == 0
    insufficient = final_labels == 1
    answered_sufficient = answer & sufficient
    answered_insufficient = answer & insufficient
    abstained_sufficient = ~answer & sufficient
    abstained_insufficient = ~answer & insufficient
    correct = answered_sufficient | abstained_insufficient
    top5_insufficient = labels5 == 1
    top5_answered_insufficient = answer & top5_insufficient
    resolved_by_more_answered = retrieve & answer & (labels5 == 1) & (labelsk == 0)
    unresolved_after_more_answered = retrieve & answer & (labelsk == 1)
    return {
        "n": int(len(final_labels)),
        "decision_accuracy": float(correct.mean()) if len(final_labels) else 0.0,
        "coverage": float(answer.mean()) if len(final_labels) else 0.0,
        "selective_accuracy": float(answered_sufficient.sum() / answer.sum()) if answer.any() else 0.0,
        "insufficient_answer_rate": float(answered_insufficient.sum() / insufficient.sum()) if insufficient.any() else 0.0,
        "sufficient_abstain_rate": float(abstained_sufficient.sum() / sufficient.sum()) if sufficient.any() else 0.0,
        "abstained_insufficient_rate": float(abstained_insufficient.sum() / insufficient.sum()) if insufficient.any() else 0.0,
        "answered_count": int(answer.sum()),
        "false_answer_count": int(answered_insufficient.sum()),
        "over_abstain_count": int(abstained_sufficient.sum()),
        "retrieval_rate": float(retrieve.mean()) if len(final_labels) else 0.0,
        "retrieved_count": int(retrieve.sum()),
        "mean_docs_used": float(docs_used.mean()) if len(docs_used) else 0.0,
        "mean_added_docs": float(np.maximum(docs_used - 5.0, 0.0).mean()) if len(docs_used) else 0.0,
        "top5_insufficient_answer_rate": (
            float(top5_answered_insufficient.sum() / top5_insufficient.sum()) if top5_insufficient.any() else 0.0
        ),
        "resolved_by_more_answered_count": int(resolved_by_more_answered.sum()),
        "unresolved_after_more_answered_count": int(unresolved_after_more_answered.sum()),
    }


def _selection_row(selection: dict[str, Any]) -> dict[str, Any]:
    row = {
        "method_name": selection["method_name"],
        "action_type": selection["action_type"],
        "feature_set": selection["feature_set"],
        "calibration": selection["calibration"],
        "policy": selection["policy"],
        "target_k": selection["target_k"],
        "tau_answer_top5": selection["tau_answer_top5"],
        "tau_answer_after_more": selection["tau_answer_after_more"],
        "n_features": selection["n_features"],
    }
    row.update(_prefix("valid", selection["valid"]))
    row.update(_prefix("test", selection["test"]))
    return row


def _main_rows(
    topk_rows: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
    selections: dict[str, dict[str, Any]],
    top_ks: list[int],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    selection_keys: list[str] = []
    for top_k in top_ks:
        key = f"always_answer_top{top_k}"
        rows.append(_main_row_from_selection(selections[key]))
        selection_keys.append(key)
    top5_targets = [
        ("v3_no_query", "balanced"),
        ("v3_no_query", "risk_control@suff_abstain15"),
        ("retrieval_quality_only", "balanced"),
    ]
    for feature_set, policy in top5_targets:
        candidates = [
            row
            for row in policy_rows
            if row["action_type"] == "answer_abstain_top5"
            and row["feature_set"] == feature_set
            and row["policy"] == policy
        ]
        if candidates:
            best = max(candidates, key=lambda row: (float(row["valid_decision_accuracy"]), -float(row["valid_insufficient_answer_rate"])))
            key = f"{best['method_name']}::{best['policy']}"
            rows.append(_main_row_from_selection(selections[key]))
            selection_keys.append(key)
    for target_k in (k for k in top_ks if k > 5):
        for policy in ("retrieve_more@cov85", "retrieve_more_risk_control@suff_abstain15"):
            candidates = [
                row
                for row in policy_rows
                if row["action_type"] == "answer_retrieve_more_abstain"
                and int(row["target_k"]) == target_k
                and row["feature_set"] == "v3_no_query"
                and row["policy"] == policy
            ]
            if candidates:
                best = max(candidates, key=lambda row: _valid_row_key_for_policy(row, policy))
                key = f"{best['method_name']}::{best['policy']}"
                rows.append(_main_row_from_selection(selections[key]))
                selection_keys.append(key)
    return rows, selection_keys


def _main_row_from_selection(selection: dict[str, Any]) -> dict[str, Any]:
    row = {
        "method_name": selection["method_name"],
        "action_type": selection["action_type"],
        "feature_set": selection["feature_set"],
        "calibration": selection["calibration"],
        "policy": selection["policy"],
        "target_k": selection["target_k"],
        "tau_answer_top5": selection["tau_answer_top5"],
        "tau_answer_after_more": selection["tau_answer_after_more"],
    }
    row.update(_prefix("test", selection["test"]))
    return row


def _feature_ablation_rows(policy_rows: list[dict[str, Any]], top_ks: list[int]) -> list[dict[str, Any]]:
    target_k = max(top_ks)
    return [
        row
        for row in policy_rows
        if row["action_type"] == "answer_retrieve_more_abstain"
        and int(row["target_k"]) == target_k
        and row["policy"] in {"retrieve_more_balanced", "retrieve_more_risk_control@suff_abstain15"}
        and row["calibration"] == "isotonic"
    ]


def _calibration_rows(policy_rows: list[dict[str, Any]], top_ks: list[int]) -> list[dict[str, Any]]:
    target_k = max(top_ks)
    return [
        row
        for row in policy_rows
        if row["action_type"] == "answer_retrieve_more_abstain"
        and int(row["target_k"]) == target_k
        and row["feature_set"] == "v3_no_query"
        and row["policy"] == "retrieve_more_balanced"
    ]


def _case_study_rows(
    expanded_records: dict[int, dict[str, list[dict[str, Any]]]],
    risks: dict[tuple[int, str, str], dict[str, Any]],
    selections: dict[str, dict[str, Any]],
    policy_rows: list[dict[str, Any]],
    top_ks: list[int],
) -> list[dict[str, Any]]:
    target_k = max(top_ks)
    candidates = [
        row
        for row in policy_rows
        if row["action_type"] == "answer_retrieve_more_abstain"
        and int(row["target_k"]) == target_k
        and row["feature_set"] == "v3_no_query"
        and row["policy"] == "retrieve_more_risk_control@suff_abstain15"
    ]
    if not candidates:
        return []
    best = max(candidates, key=lambda row: _valid_row_key_for_policy(row, "retrieve_more_risk_control@suff_abstain15"))
    selection = selections[f"{best['method_name']}::{best['policy']}"]
    calibration = selection["calibration"]
    risk5 = risks[(5, "v3_no_query", calibration)]["test_risk"]
    riskk = risks[(target_k, "v3_no_query", calibration)]["test_risk"]
    records5 = expanded_records[5]["test"]
    recordsk = expanded_records[target_k]["test"]
    labels5 = _risk_labels(records5)
    labelsk = _risk_labels(recordsk)
    tau5 = float(selection["tau_answer_top5"])
    tauk = float(selection["tau_answer_after_more"])
    answer_at5 = risk5 <= tau5
    retrieve = ~answer_at5
    answer_after_more = retrieve & (riskk <= tauk)
    answer = answer_at5 | answer_after_more
    final_labels = np.where(answer_at5, labels5, labelsk)
    rows = []
    for idx, (record5, recordk) in enumerate(zip(records5, recordsk)):
        if answer[idx] and final_labels[idx] == 1:
            case_type = "false_answer"
        elif (not answer[idx]) and final_labels[idx] == 0:
            case_type = "over_abstain"
        elif answer_after_more[idx] and labels5[idx] == 1 and labelsk[idx] == 0:
            case_type = "resolved_by_retrieve_more"
        elif (not answer[idx]) and final_labels[idx] == 1:
            case_type = "successful_intercept"
        else:
            continue
        rows.append(
            {
                "method_name": selection["method_name"],
                "policy": selection["policy"],
                "case_type": case_type,
                "original_id": record5["metadata"]["original_id"],
                "question": record5["query"],
                "gold_answer": record5["gold_answer"],
                "label_top5": record5["sufficiency_label"],
                "label_topk": recordk["sufficiency_label"],
                "risk_top5": float(risk5[idx]),
                "risk_topk": float(riskk[idx]),
                "action": _action_name(bool(answer_at5[idx]), bool(answer_after_more[idx]), bool(answer[idx])),
                "top5_titles": " || ".join(doc.get("title", "") for doc in record5["retrieved_docs"]),
                "topk_new_titles": " || ".join(doc.get("title", "") for doc in recordk["retrieved_docs"][5:]),
                "missing_support_top5": " || ".join(record5["metadata"].get("missing_support_titles", [])),
                "missing_support_topk": " || ".join(recordk["metadata"].get("missing_support_titles", [])),
            }
        )
    return _limit_cases(rows, 25)


def _action_name(answer_at5: bool, answer_after_more: bool, answer: bool) -> str:
    if answer_at5:
        return "answer@5"
    if answer_after_more:
        return "retrieve_more_then_answer"
    if not answer:
        return "retrieve_more_then_abstain"
    return "unknown"


def _valid_row_key_for_policy(row: dict[str, Any], policy: str) -> tuple[float, ...]:
    if policy in {"retrieve_more@cov85", "reliable@cov85"}:
        return (
            float(row["valid_selective_accuracy"]),
            -float(row["valid_insufficient_answer_rate"]),
            float(row["valid_decision_accuracy"]),
            float(row["valid_coverage"]),
            -float(row["valid_mean_added_docs"]),
        )
    if "risk_control" in policy:
        return (
            -float(row["valid_insufficient_answer_rate"]),
            float(row["valid_selective_accuracy"]),
            float(row["valid_decision_accuracy"]),
            float(row["valid_coverage"]),
            -float(row["valid_mean_added_docs"]),
        )
    return (
        float(row["valid_decision_accuracy"]),
        float(row["valid_coverage"]),
        float(row["valid_selective_accuracy"]),
        -float(row["valid_mean_added_docs"]),
    )


def _limit_cases(rows: list[dict[str, Any]], limit_per_type: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["case_type"], []).append(row)
    limited = []
    for case_type, group in sorted(grouped.items()):
        if case_type == "false_answer":
            limited.extend(sorted(group, key=lambda row: float(row["risk_topk"]))[:limit_per_type])
        else:
            limited.extend(group[:limit_per_type])
    return limited


def _topk_curve_rows(expanded_records: dict[int, dict[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    rows = []
    labels5_by_split = {
        split: _risk_labels(records)
        for split, records in expanded_records[5].items()
    }
    for top_k, split_records in expanded_records.items():
        for split, records in split_records.items():
            labels = _risk_labels(records)
            labels5 = labels5_by_split[split]
            audit_rows = [extract_audit_features(record) for record in records]
            sufficient = labels == 0
            top5_insufficient = labels5 == 1
            newly_sufficient = top5_insufficient & sufficient
            rows.append(
                {
                    "split": split,
                    "top_k": top_k,
                    "n": len(records),
                    "sufficient_count": int(sufficient.sum()),
                    "insufficient_count": int((labels == 1).sum()),
                    "sufficient_rate": float(sufficient.mean()) if len(labels) else 0.0,
                    "newly_sufficient_vs_top5_count": int(newly_sufficient.sum()),
                    "newly_sufficient_vs_top5_rate": (
                        float(newly_sufficient.sum() / top5_insufficient.sum()) if top5_insufficient.any() else 0.0
                    ),
                    "answer_present_in_returned_docs_rate": _mean(_answer_present(record) for record in records),
                    "mean_support_title_coverage": _mean(row["audit_support_title_coverage"] for row in audit_rows),
                }
            )
    return rows


def _answer_present(record: dict[str, Any]) -> float:
    answer = " ".join(str(record.get("gold_answer", "")).lower().split())
    if not answer:
        return 0.0
    context = " ".join(
        " ".join(f"{doc.get('title', '')} {doc.get('text', '')}".lower().split())
        for doc in record.get("retrieved_docs", [])
    )
    return float(answer in context)


def _bootstrap_rows(
    selections: dict[str, dict[str, Any]],
    selection_keys: list[str],
    n_iters: int,
    seed: int,
) -> list[dict[str, Any]]:
    if n_iters <= 0:
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
        for _ in range(n_iters):
            indices = rng.integers(0, n, size=n)
            metrics = _selection_metrics_on_indices(selection, indices)
            for metric in BOOTSTRAP_METRICS:
                values_by_metric[metric].append(float(metrics[metric]))
        point = selection["test"]
        for metric, values in values_by_metric.items():
            rows.append(
                {
                    "method_name": selection["method_name"],
                    "policy": selection["policy"],
                    "target_k": selection["target_k"],
                    "metric": metric,
                    "point": float(point[metric]),
                    "ci_low": float(np.quantile(values, 0.025)),
                    "ci_high": float(np.quantile(values, 0.975)),
                    "bootstrap_iters": n_iters,
                }
            )
    return rows


def _selection_metrics_on_indices(selection: dict[str, Any], indices: np.ndarray) -> dict[str, float | int]:
    arrays = selection["test_arrays"]
    labels5 = arrays["labels5"][indices]
    labelsk = arrays["labelsk"][indices]
    risk5 = arrays["risk5"][indices]
    riskk = arrays["riskk"][indices]
    if selection["action_type"] == "always_answer":
        return _always_answer_metrics(int(selection["target_k"]), labels5, labelsk)
    if selection["action_type"] == "answer_abstain_top5":
        return _answer_abstain_metrics(labels5, risk5, float(selection["tau_answer_top5"]))
    return _retrieve_more_metrics(
        labels5=labels5,
        labelsk=labelsk,
        risk5=risk5,
        riskk=riskk,
        tau5=float(selection["tau_answer_top5"]),
        tauk=float(selection["tau_answer_after_more"]),
        target_k=int(selection["target_k"]),
    )


def _labels_by_split(split_records: dict[str, list[dict[str, Any]]], valid_policy_ids: set[str]) -> dict[str, np.ndarray]:
    return {
        "valid_policy": _risk_labels([record for record in split_records["valid"] if record["metadata"]["original_id"] in valid_policy_ids]),
        "test": _risk_labels(split_records["test"]),
    }


def _split_valid_ids(valid_records: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    by_label: dict[str, list[dict[str, Any]]] = {}
    for record in sorted(valid_records, key=lambda item: item["metadata"]["original_id"]):
        by_label.setdefault(record["sufficiency_label"], []).append(record)
    valid_calib_ids = set()
    valid_policy_ids = set()
    for label_records in by_label.values():
        for idx, record in enumerate(label_records):
            target = valid_calib_ids if idx % 2 == 0 else valid_policy_ids
            target.add(record["metadata"]["original_id"])
    if not valid_calib_ids or not valid_policy_ids:
        raise AssertionError("valid split could not be divided into calibration and policy ids.")
    return valid_calib_ids, valid_policy_ids


def _validate_split_records(split_records: dict[str, list[dict[str, Any]]]) -> None:
    split_ids = {split: {record["metadata"]["original_id"] for record in records} for split, records in split_records.items()}
    _require(split_ids["train"].isdisjoint(split_ids["valid"]), "train and valid original_id overlap.")
    _require(split_ids["train"].isdisjoint(split_ids["test"]), "train and test original_id overlap.")
    _require(split_ids["valid"].isdisjoint(split_ids["test"]), "valid and test original_id overlap.")
    for split, records in split_records.items():
        _require(records, f"{split} split is empty.")
        labels = {record["sufficiency_label"] for record in records}
        _require(labels == {"sufficient", "insufficient"}, f"{split} must contain both labels.")


def _validate_expanded_records(expanded_records: dict[int, dict[str, list[dict[str, Any]]]]) -> None:
    for top_k, split_records in expanded_records.items():
        _validate_split_records(split_records)
        for split, records in split_records.items():
            for record in records:
                if len(record["retrieved_docs"]) != top_k:
                    raise AssertionError(f"{record['id']} has wrong retrieved doc count.")
                if record["metadata"].get("top_k") != top_k:
                    raise AssertionError(f"{record['id']} has wrong metadata top_k.")
                if record["metadata"].get("record_kind") != f"natural_global_top{top_k}":
                    raise AssertionError(f"{record['id']} has wrong record_kind.")


def _validate_features(
    feature_records: dict[int, dict[str, list[dict[str, Any]]]],
    feature_sets: dict[str, list[str]],
) -> None:
    required = set().union(*[set(features) for features in feature_sets.values()])
    for split_records in feature_records.values():
        for records in split_records.values():
            for record in records:
                missing = required - set(record)
                if missing:
                    raise ValueError(f"Missing features for {record.get('id')}: {sorted(missing)[:10]}")


def _write_validation(
    path: Path,
    args: argparse.Namespace,
    top_ks: list[int],
    base_records: dict[str, list[dict[str, Any]]],
    expanded_records: dict[int, dict[str, list[dict[str, Any]]]],
    global_docs: list[dict[str, Any]],
    risks: dict[tuple[int, str, str], dict[str, Any]],
    elapsed_seconds: float,
) -> None:
    top5_reconstruction = _top5_reconstruction_check(base_records, expanded_records[5])
    summary = {
        "top_ks": top_ks,
        "split_counts": {split: len(records) for split, records in base_records.items()},
        "global_doc_count": len(global_docs),
        "risk_runs": len(risks),
        "bootstrap_iters": args.bootstrap_iters,
        "seed": args.seed,
        "cache_path": args.cache_path,
        "seed_cache_paths": args.seed_cache_path,
        "embedding_model": args.embedding_model,
        "uses_embedding_api": False,
        "uses_llm_api": False,
        "selection_protocol": "train estimator on train; fit calibration on valid_calib; select thresholds on valid_policy; report test only",
        "top5_reconstruction_check": top5_reconstruction,
        "elapsed_seconds": round(float(elapsed_seconds), 2),
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _top5_reconstruction_check(
    base_records: dict[str, list[dict[str, Any]]],
    reconstructed_records: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    rows = {}
    for split in ("train", "valid", "test"):
        base_by_id = {record["metadata"]["original_id"]: record for record in base_records[split]}
        recon_by_id = {record["metadata"]["original_id"]: record for record in reconstructed_records[split]}
        title_matches = 0
        label_matches = 0
        for original_id, base in base_by_id.items():
            recon = recon_by_id[original_id]
            base_titles = [doc["title"] for doc in base["retrieved_docs"]]
            recon_titles = [doc["title"] for doc in recon["retrieved_docs"]]
            title_matches += int(base_titles == recon_titles)
            label_matches += int(base["sufficiency_label"] == recon["sufficiency_label"])
        rows[split] = {
            "n": len(base_by_id),
            "exact_title_match_count": title_matches,
            "exact_title_match_rate": title_matches / len(base_by_id) if base_by_id else 0.0,
            "label_match_count": label_matches,
            "label_match_rate": label_matches / len(base_by_id) if base_by_id else 0.0,
        }
    return rows


def _write_summary(
    path: Path,
    main_rows: list[dict[str, Any]],
    topk_rows: list[dict[str, Any]],
    case_rows: list[dict[str, Any]],
) -> None:
    test_curve = [row for row in topk_rows if row["split"] == "test"]
    best_retrieve = min(
        [row for row in main_rows if row["action_type"] == "answer_retrieve_more_abstain"],
        key=lambda row: (float(row["test_insufficient_answer_rate"]), -float(row["test_selective_accuracy"])),
        default=None,
    )
    topk_lines = "\n".join(
        f"- top-{row['top_k']}: sufficient_rate={float(row['sufficient_rate']):.4f}, "
        f"newly_sufficient_vs_top5={float(row['newly_sufficient_vs_top5_rate']):.4f}"
        for row in test_curve
    )
    best_line = (
        "- Best retrieve-more main row: "
        f"`{best_retrieve['method_name']}` / `{best_retrieve['policy']}`, "
        f"coverage={float(best_retrieve['test_coverage']):.4f}, "
        f"IAR={float(best_retrieve['test_insufficient_answer_rate']):.4f}, "
        f"retrieval_rate={float(best_retrieve['test_retrieval_rate']):.4f}."
        if best_retrieve
        else "- No retrieve-more row was selected."
    )
    text = f"""# CSR-RAG Retrieve-More No-API Experiment Summary

## Purpose

This run tests whether lightweight top-k expansion can improve retrieval sufficiency before answer/abstain. It reads existing HotpotQA data and local embedding caches only; it does not call embedding or LLM APIs.

## Test Top-k Sufficiency Curve

{topk_lines}

## Main Retrieve-More Finding

{best_line}

## Diagnostics

- Case study rows: {len(case_rows)}.
- Interpret retrieve-more gains as retrieval-level evidence sufficiency gains. QA expansion should wait until these gains are stable.
- If top-k expansion only yields a small sufficiency gain, treat it as a diagnostic baseline rather than the next main CSR-RAG method.
"""
    path.write_text(text, encoding="utf-8")


def _method_name(top_k: int, feature_set: str, calibration: str) -> str:
    return f"top{top_k}/logistic_regression/{feature_set}/{calibration}"


def _feature_row(record: dict[str, Any], feature_names: list[str]) -> dict[str, float]:
    return {name: float(record[name]) for name in feature_names}


def _sufficiency_labels(records: list[dict[str, Any]]) -> list[int]:
    return [1 if record["sufficiency_label"] == "sufficient" else 0 for record in records]


def _risk_labels(records: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([0 if record["sufficiency_label"] == "sufficient" else 1 for record in records], dtype=int)


def _balanced_key(metrics: dict[str, Any]) -> tuple[float, float, float]:
    return (float(metrics["decision_accuracy"]), float(metrics["coverage"]), float(metrics["selective_accuracy"]))


def _reliable_key(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(metrics["selective_accuracy"]),
        -float(metrics["insufficient_answer_rate"]),
        float(metrics["decision_accuracy"]),
        float(metrics["coverage"]),
    )


def _risk_control_key(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        -float(metrics["insufficient_answer_rate"]),
        float(metrics["selective_accuracy"]),
        float(metrics["decision_accuracy"]),
        float(metrics["coverage"]),
    )


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


def _prefix(prefix: str, metrics: dict[str, float | int]) -> dict[str, float | int]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _mean(values) -> float:
    value_list = list(values)
    return float(sum(value_list) / len(value_list)) if value_list else 0.0


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
        for row in rows:
            writer.writerow(row)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    result = []
    seen = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


if __name__ == "__main__":
    main()
