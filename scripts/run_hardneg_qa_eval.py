from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import string
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from csrrag.calibration.methods import make_calibrator
from csrrag.experiments.feature_sets import EMBEDDING_FEATURES
from csrrag.features.basic import extract_basic_features
from csrrag.models.baseline import train_estimator
from csrrag.rag.api_clients import OpenAICompatibleChatClient
from csrrag.utils.env import load_dotenv
from csrrag.utils.io import read_jsonl


POLICIES = [
    "balanced",
    "reliable@cov85",
    "risk_control@cov85",
    "high_precision@cov50",
]
DETAIL_FIELDNAMES = [
    "sample_index",
    "sample_bucket",
    "id",
    "original_id",
    "record_kind",
    "question",
    "gold_answer",
    "sufficiency_label",
    "support_present_in_topk",
    "risk_score",
    "naive_answer",
    "naive_em",
    "naive_f1",
    "naive_is_dont_know",
    "naive_is_substantive",
    "llm_json_parse_ok",
    "llm_format_ok",
    "llm_attempt",
    "llm_used_fallback",
    "llm_had_thinking",
    "llm_had_reasoning_details",
    "llm_finish_reason",
    "llm_error",
    "missing_support_titles",
    "top5_titles",
    "top5_embedding_scores",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QA300 stress evaluation on hard-negative CSR-RAG records.")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--split-dir", default="data/processed/hotpotqa_global_hardneg_splits_full_dev")
    parser.add_argument("--output-dir", default="results/tables/hotpotqa_global_hardneg_qa_eval_strict_300")
    parser.add_argument("--estimator", default="logistic_regression")
    parser.add_argument("--calibration", default="isotonic")
    parser.add_argument("--per-bucket", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--qa-workers", type=int, default=4)
    args = parser.parse_args()

    load_dotenv(args.env_file)
    chat_client = OpenAICompatibleChatClient(
        base_url=_required_env("LLM_BASE_URL"),
        api_key=_required_env("LLM_API_KEY"),
        model=_required_env("LLM_MODEL"),
    )

    split_records = {split: read_jsonl(Path(args.split_dir) / f"{split}.jsonl") for split in ("train", "valid", "test")}
    _validate_split_records(split_records)
    model_bundle = _fit_risk_model(split_records, args.estimator, args.calibration)
    selected_records = _select_stress_records(split_records["test"], model_bundle, args.per_bucket, args.sample_seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "hardneg_qa_details.csv"
    existing_rows = {
        row["id"]: row
        for row in _read_csv_if_exists(details_path)
        if _to_bool(row.get("llm_format_ok", False))
    }

    detail_by_id = dict(existing_rows)
    pending = [
        (idx, item)
        for idx, item in enumerate(selected_records, start=1)
        if item["record"]["id"] not in detail_by_id
    ]
    workers = max(1, int(args.qa_workers))
    if workers == 1:
        for idx, item in pending:
            row = _answer_detail_row(chat_client, item, idx, model_bundle["test_risk_by_id"], args.max_tokens)
            detail_by_id[row["id"]] = row
            _write_details(details_path, selected_records, detail_by_id)
            _print_progress(row, len(detail_by_id), len(selected_records))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_answer_detail_row, chat_client, item, idx, model_bundle["test_risk_by_id"], args.max_tokens)
                for idx, item in pending
            ]
            for future in as_completed(futures):
                row = future.result()
                detail_by_id[row["id"]] = row
                _write_details(details_path, selected_records, detail_by_id)
                _print_progress(row, len(detail_by_id), len(selected_records))

    detail_rows = _ordered_detail_rows(selected_records, detail_by_id)
    _write_details(details_path, selected_records, detail_by_id)
    policy_rows = _qa_policy_rows(detail_rows, model_bundle["selected_policies"], args.estimator, args.calibration)
    case_rows = _qa_case_study_rows(detail_rows, model_bundle["selected_policies"], args.estimator, args.calibration)
    _write_csv(output_dir / "hardneg_qa_policy_comparison.csv", policy_rows, list(policy_rows[0].keys()))
    _write_csv(output_dir / "hardneg_qa_case_studies.csv", case_rows, list(case_rows[0].keys()))
    _write_csv(
        output_dir / "abstention_cases.csv",
        [row for row in case_rows if row["decision"] == "abstain"],
        list(case_rows[0].keys()),
    )
    config = {
        "llm_model": os.environ["LLM_MODEL"],
        "estimator": args.estimator,
        "calibration": args.calibration,
        "per_bucket": args.per_bucket,
        "sample_seed": args.sample_seed,
        "selected_policy_taus": {policy: model_bundle["selected_policies"][policy]["tau"] for policy in POLICIES},
        "qa_rescore_calls_llm": False,
        "selected_count": len(selected_records),
        "completed_count": len(detail_rows),
    }
    (output_dir / "hardneg_qa_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "hardneg_qa_summary.md").write_text(_summary_markdown(policy_rows), encoding="utf-8")
    _write_validation(output_dir / "validation_summary.json", selected_records, detail_rows, policy_rows)

    risk_control = next(row for row in policy_rows if row["policy"] == "risk_control@cov85")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "completed": len(detail_rows),
                "risk_control_coverage": risk_control["coverage"],
                "risk_control_decision_insufficient_answer_rate": risk_control["decision_insufficient_answer_rate"],
                "risk_control_unsupported_substantive_answer_rate": risk_control["unsupported_substantive_answer_rate"],
            },
            ensure_ascii=False,
        )
    )


def _answer_detail_row(
    chat_client: OpenAICompatibleChatClient,
    item: dict[str, Any],
    sample_index: int,
    test_risk_by_id: dict[str, float],
    max_tokens: int,
) -> dict[str, Any]:
    record = item["record"]
    risk_score = float(test_risk_by_id[record["id"]])
    answer_result = _safe_answer(chat_client, record["query"], record["retrieved_docs"], max_tokens)
    naive_answer = answer_result["answer"]
    return {
        "sample_index": sample_index,
        "sample_bucket": item["sample_bucket"],
        "id": record["id"],
        "original_id": record["metadata"]["original_id"],
        "record_kind": record["metadata"].get("record_kind", ""),
        "question": record["query"],
        "gold_answer": record["gold_answer"],
        "sufficiency_label": record["sufficiency_label"],
        "support_present_in_topk": record["metadata"].get("support_present_in_topk", False),
        "risk_score": risk_score,
        "naive_answer": naive_answer,
        "naive_em": exact_match(naive_answer, record["gold_answer"]),
        "naive_f1": f1_score(naive_answer, record["gold_answer"]),
        "naive_is_dont_know": _is_dont_know(naive_answer),
        "naive_is_substantive": not _is_dont_know(naive_answer),
        "llm_json_parse_ok": answer_result["json_parse_ok"],
        "llm_format_ok": answer_result.get("format_ok", False),
        "llm_attempt": answer_result.get("attempt", 0),
        "llm_used_fallback": answer_result["used_fallback"],
        "llm_had_thinking": answer_result["had_thinking"],
        "llm_had_reasoning_details": answer_result.get("had_reasoning_details", False),
        "llm_finish_reason": answer_result.get("finish_reason", ""),
        "llm_error": answer_result["error"],
        "missing_support_titles": " || ".join(record["metadata"].get("missing_support_titles", [])),
        "top5_titles": " || ".join(doc["title"] for doc in record["retrieved_docs"]),
        "top5_embedding_scores": " || ".join(str(doc.get("embedding_score", "")) for doc in record["retrieved_docs"]),
    }


def _print_progress(row: dict[str, Any], processed: int, target: int) -> None:
    print(
        json.dumps(
            {
                "processed": processed,
                "target": target,
                "sample_bucket": row["sample_bucket"],
                "risk": round(float(row["risk_score"]), 4),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def _fit_risk_model(split_records: dict[str, list[dict[str, Any]]], estimator: str, calibration: str) -> dict[str, Any]:
    feature_records = {split: [_feature_record(record) for record in records] for split, records in split_records.items()}
    model = train_estimator(
        estimator,
        [_feature_row(record) for record in feature_records["train"]],
        _sufficiency_labels(feature_records["train"]),
        EMBEDDING_FEATURES,
    )
    valid_rows = [_feature_row(record) for record in feature_records["valid"]]
    test_rows = [_feature_row(record) for record in feature_records["test"]]
    valid_labels = _risk_labels(feature_records["valid"])
    valid_raw_risk = 1.0 - np.asarray(model.predict_proba(valid_rows), dtype=float)
    test_raw_risk = 1.0 - np.asarray(model.predict_proba(test_rows), dtype=float)
    calibrator = make_calibrator(calibration)
    calibrator.fit(valid_raw_risk, valid_labels)
    valid_risk = np.asarray(calibrator.predict(valid_raw_risk), dtype=float)
    test_risk = np.asarray(calibrator.predict(test_raw_risk), dtype=float)
    selected_policies = _select_policies(valid_labels, valid_risk)
    return {
        "model": model,
        "calibrator": calibrator,
        "selected_policies": selected_policies,
        "test_risk_by_id": {record["id"]: float(risk) for record, risk in zip(split_records["test"], test_risk)},
    }


def _select_stress_records(
    test_records: list[dict[str, Any]],
    model_bundle: dict[str, Any],
    per_bucket: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    natural_sufficient = [
        record
        for record in test_records
        if record["metadata"].get("record_kind") == "natural_global_top5" and record["sufficiency_label"] == "sufficient"
    ]
    natural_insufficient = [
        record
        for record in test_records
        if record["metadata"].get("record_kind") == "natural_global_top5" and record["sufficiency_label"] == "insufficient"
    ]
    hardneg_insufficient = [
        record
        for record in test_records
        if record["metadata"].get("record_kind") == "hardneg_missing_support_top5" and record["sufficiency_label"] == "insufficient"
    ]
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    for bucket_name, candidates in (
        ("natural_sufficient", natural_sufficient),
        ("natural_insufficient", natural_insufficient),
        ("hardneg_insufficient", hardneg_insufficient),
    ):
        sampled = _sample_records(candidates, per_bucket, rng)
        if bucket_name == "natural_insufficient" and len(sampled) < per_bucket:
            needed = per_bucket - len(sampled)
            fallback = [record for record in hardneg_insufficient if record["id"] not in selected_ids and record["id"] not in {row["id"] for row in sampled}]
            sampled.extend(_sample_records(fallback, needed, rng))
        for record in sampled:
            if record["id"] in selected_ids:
                continue
            selected_ids.add(record["id"])
            selected.append({"sample_bucket": bucket_name, "record": record, "risk": model_bundle["test_risk_by_id"][record["id"]]})
    return selected


def _sample_records(records: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    shuffled = list(records)
    rng.shuffle(shuffled)
    return shuffled[:count]


def _select_policies(valid_labels: np.ndarray, valid_risk: np.ndarray) -> dict[str, dict[str, Any]]:
    candidates = [
        {"tau": tau, "valid": _extended_decision_metrics(valid_labels, valid_risk, tau)}
        for tau in _threshold_candidates(valid_risk)
    ]
    return {
        "balanced": _choose(candidates, _balanced_key),
        "reliable@cov85": _choose([item for item in candidates if item["valid"]["coverage"] >= 0.85], _reliable_key),
        "risk_control@cov85": _choose([item for item in candidates if item["valid"]["coverage"] >= 0.85], _risk_control_key),
        "high_precision@cov50": _choose([item for item in candidates if item["valid"]["coverage"] >= 0.50], _risk_control_key),
    }


def _qa_policy_rows(
    detail_rows: list[dict[str, Any]],
    selected_policies: dict[str, dict[str, Any]],
    estimator: str,
    calibration: str,
) -> list[dict[str, Any]]:
    rows = [_qa_row(detail_rows, estimator, calibration, "naive_always_answer", None)]
    for policy in POLICIES:
        rows.append(_qa_row(detail_rows, estimator, calibration, policy, float(selected_policies[policy]["tau"])))
    lower_bound = _qa_insufficient_answer_rate_lower_bound(detail_rows, 0.85)
    for row in rows:
        row["coverage85_decision_insufficient_answer_rate_lower_bound"] = lower_bound
    return rows


def _qa_row(
    detail_rows: list[dict[str, Any]],
    estimator: str,
    calibration: str,
    policy: str,
    tau: float | None,
) -> dict[str, Any]:
    if tau is None:
        answered = list(detail_rows)
    else:
        answered = [row for row in detail_rows if float(row["risk_score"]) <= tau]
    insufficient = [row for row in detail_rows if row["sufficiency_label"] == "insufficient"]
    sufficient = [row for row in detail_rows if row["sufficiency_label"] == "sufficient"]
    answered_insufficient = [row for row in answered if row["sufficiency_label"] == "insufficient"]
    answered_unsupported = [row for row in answered_insufficient if _to_bool(row["naive_is_substantive"])]
    over_abstained = [
        row
        for row in sufficient
        if tau is not None and float(row["risk_score"]) > tau
    ]
    abstained_insufficient = [
        row
        for row in insufficient
        if tau is not None and float(row["risk_score"]) > tau
    ]
    parse_failures = [row for row in detail_rows if not _to_bool(row["llm_json_parse_ok"])]
    format_failures = [row for row in detail_rows if not _to_bool(row["llm_format_ok"])]
    return {
        "estimator": "none" if tau is None else estimator,
        "calibration": "none" if tau is None else calibration,
        "policy": policy,
        "tau_answer": "" if tau is None else tau,
        "n": len(detail_rows),
        "coverage": len(answered) / len(detail_rows) if detail_rows else 0.0,
        "answered_count": len(answered),
        "answered_em": _mean(float(row["naive_em"]) for row in answered),
        "answered_f1": _mean(float(row["naive_f1"]) for row in answered),
        "answered_sufficient_rate": _mean(float(row["sufficiency_label"] == "sufficient") for row in answered),
        "decision_insufficient_answer_rate": len(answered_insufficient) / len(insufficient) if insufficient else 0.0,
        "unsupported_substantive_answer_rate": len(answered_unsupported) / len(insufficient) if insufficient else 0.0,
        "unsupported_substantive_rate_among_answered_insufficient": (
            len(answered_unsupported) / len(answered_insufficient) if answered_insufficient else 0.0
        ),
        "sufficient_abstain_rate": len(over_abstained) / len(sufficient) if sufficient else 0.0,
        "abstained_insufficient_rate": len(abstained_insufficient) / len(insufficient) if insufficient else 0.0,
        "false_answer_count": len(answered_insufficient),
        "unsupported_substantive_count": len(answered_unsupported),
        "over_abstain_count": len(over_abstained),
        "json_parse_failure_rate": len(parse_failures) / len(detail_rows) if detail_rows else 0.0,
        "format_failure_rate": len(format_failures) / len(detail_rows) if detail_rows else 0.0,
        "qa_rescore_calls_llm": False,
    }


def _qa_case_study_rows(
    detail_rows: list[dict[str, Any]],
    selected_policies: dict[str, dict[str, Any]],
    estimator: str,
    calibration: str,
) -> list[dict[str, Any]]:
    rows = []
    for policy in POLICIES:
        tau = float(selected_policies[policy]["tau"])
        buckets = {
            "successful_intercept": [
                row for row in detail_rows if row["sufficiency_label"] == "insufficient" and float(row["risk_score"]) > tau
            ],
            "false_answer": [
                row for row in detail_rows if row["sufficiency_label"] == "insufficient" and float(row["risk_score"]) <= tau
            ],
            "over_abstain": [
                row for row in detail_rows if row["sufficiency_label"] == "sufficient" and float(row["risk_score"]) > tau
            ],
            "safe_answer": [
                row for row in detail_rows if row["sufficiency_label"] == "sufficient" and float(row["risk_score"]) <= tau
            ],
        }
        for case_type, bucket in buckets.items():
            reverse = case_type in {"successful_intercept", "over_abstain"}
            for row in sorted(bucket, key=lambda item: float(item["risk_score"]), reverse=reverse)[:5]:
                rows.append(
                    {
                        "estimator": estimator,
                        "calibration": calibration,
                        "policy": policy,
                        "tau_answer": tau,
                        "case_type": case_type,
                        "sample_index": row["sample_index"],
                        "sample_bucket": row["sample_bucket"],
                        "id": row["id"],
                        "original_id": row["original_id"],
                        "record_kind": row["record_kind"],
                        "sufficiency_label": row["sufficiency_label"],
                        "risk_score": row["risk_score"],
                        "decision": "answer" if float(row["risk_score"]) <= tau else "abstain",
                        "gold_answer": row["gold_answer"],
                        "naive_answer": row["naive_answer"],
                        "naive_is_substantive": row["naive_is_substantive"],
                        "naive_em": row["naive_em"],
                        "naive_f1": row["naive_f1"],
                        "missing_support_titles": row["missing_support_titles"],
                        "top5_titles": row["top5_titles"],
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
    }


def _threshold_candidates(valid_risk: np.ndarray) -> list[float]:
    candidates = {0.0, 1.0}
    candidates.update(float(score) for score in valid_risk if np.isfinite(score))
    return sorted(score for score in candidates if 0.0 <= score <= 1.0)


def _balanced_key(item: dict[str, Any]) -> tuple[float, float, float]:
    metrics = item["valid"]
    return (metrics["decision_accuracy"], metrics["coverage"], metrics["selective_accuracy"])


def _reliable_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = item["valid"]
    return (metrics["selective_accuracy"], -metrics["insufficient_answer_rate"], metrics["decision_accuracy"], metrics["coverage"])


def _risk_control_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = item["valid"]
    return (-metrics["insufficient_answer_rate"], metrics["selective_accuracy"], metrics["coverage"], metrics["decision_accuracy"])


def _choose(candidates: list[dict[str, Any]], key_fn) -> dict[str, Any]:
    if not candidates:
        raise ValueError("No threshold candidates satisfy the policy constraints.")
    return max(candidates, key=key_fn)


def _feature_record(record: dict[str, Any]) -> dict[str, Any]:
    return {"id": record["id"], **extract_basic_features(record), "sufficiency_label": record["sufficiency_label"]}


def _feature_row(record: dict[str, Any]) -> dict[str, float]:
    return {name: float(record[name]) for name in EMBEDDING_FEATURES}


def _sufficiency_labels(records: list[dict[str, Any]]) -> list[int]:
    return [1 if record["sufficiency_label"] == "sufficient" else 0 for record in records]


def _risk_labels(records: list[dict[str, Any]]) -> np.ndarray:
    return np.array([0 if record["sufficiency_label"] == "sufficient" else 1 for record in records], dtype=int)


def _safe_answer(
    chat_client: OpenAICompatibleChatClient,
    question: str,
    contexts: list[dict[str, Any]],
    max_tokens: int,
) -> dict[str, Any]:
    try:
        result = chat_client.answer_with_metadata(question, contexts, max_tokens=max_tokens)
        result["error"] = ""
        return result
    except Exception as exc:  # noqa: BLE001 - record the failure and continue the batch.
        return {
            "answer": "",
            "json_parse_ok": False,
            "json_mode": True,
            "raw_content_length": 0,
            "had_thinking": False,
            "had_reasoning_details": False,
            "finish_reason": "",
            "used_fallback": False,
            "format_ok": False,
            "attempt": 0,
            "error": type(exc).__name__,
        }


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        return "".join(ch for ch in value if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(str(text).lower())))


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = set(prediction_tokens) & set(ground_truth_tokens)
    num_same = sum(min(prediction_tokens.count(token), ground_truth_tokens.count(token)) for token in common)
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return float(prediction_tokens == ground_truth_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def _is_dont_know(answer: str) -> bool:
    normalized = normalize_answer(answer)
    return normalized in {"i dont know", "dont know", "unknown", "not enough information", "cannot answer"}


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def _mean(values: Iterable[float]) -> float:
    value_list = list(values)
    return float(sum(value_list) / len(value_list)) if value_list else 0.0


def _ordered_detail_rows(selected_records: list[dict[str, Any]], detail_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [detail_by_id[item["record"]["id"]] for item in selected_records if item["record"]["id"] in detail_by_id]


def _write_details(path: Path, selected_records: list[dict[str, Any]], detail_by_id: dict[str, dict[str, Any]]) -> None:
    _write_csv(path, _ordered_detail_rows(selected_records, detail_by_id), DETAIL_FIELDNAMES)


def _read_csv_if_exists(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summary_markdown(policy_rows: list[dict[str, Any]]) -> str:
    first_row = policy_rows[0] if policy_rows else {}
    lower_bound = first_row.get("coverage85_decision_insufficient_answer_rate_lower_bound", "")
    lines = ["# Hard-Negative QA300 Stress Evaluation", ""]
    lines.append("| Policy | Coverage | Answered F1 | Decision Insufficient Answer Rate | Unsupported Substantive Answer Rate |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in policy_rows:
        lines.append(
            f"| {row['policy']} | {float(row['coverage']):.4f} | {float(row['answered_f1']):.4f} | "
            f"{float(row['decision_insufficient_answer_rate']):.4f} | {float(row['unsupported_substantive_answer_rate']):.4f} |"
        )
    lines.append("")
    if lower_bound != "":
        lines.append(f"Coverage>=0.85 lower bound for decision insufficient answer rate on this QA sample: {float(lower_bound):.4f}.")
        lines.append("")
    lines.append("This table is a stress-setting complement. It should be interpreted as reliability/coverage control, not as a free QA-quality improvement.")
    return "\n".join(lines) + "\n"


def _write_validation(
    path: Path,
    selected_records: list[dict[str, Any]],
    detail_rows: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
) -> None:
    sample_counts = Counter(item["sample_bucket"] for item in selected_records)
    completed_counts = Counter(row["sample_bucket"] for row in detail_rows)
    summary = {
        "selected_count": len(selected_records),
        "completed_count": len(detail_rows),
        "sample_counts": dict(sample_counts),
        "completed_counts": dict(completed_counts),
        "policies": [row["policy"] for row in policy_rows],
        "json_parse_failure_rate": max(float(row["json_parse_failure_rate"]) for row in policy_rows),
        "format_failure_rate": max(float(row["format_failure_rate"]) for row in policy_rows),
        "coverage85_decision_insufficient_answer_rate_lower_bound": _qa_insufficient_answer_rate_lower_bound(
            detail_rows,
            0.85,
        ),
        "qa_rescore_calls_llm": False,
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _qa_insufficient_answer_rate_lower_bound(detail_rows: list[dict[str, Any]], target_coverage: float) -> float:
    sufficient = sum(row["sufficiency_label"] == "sufficient" for row in detail_rows)
    insufficient = sum(row["sufficiency_label"] == "insufficient" for row in detail_rows)
    required_answered = int(np.ceil(target_coverage * len(detail_rows)))
    required_insufficient_answered = max(0, required_answered - sufficient)
    return required_insufficient_answered / insufficient if insufficient else 0.0


def _validate_split_records(split_records: dict[str, list[dict[str, Any]]]) -> None:
    split_ids = {split: {record["metadata"]["original_id"] for record in records} for split, records in split_records.items()}
    if not split_ids["train"].isdisjoint(split_ids["valid"]):
        raise AssertionError("train and valid original_id overlap.")
    if not split_ids["train"].isdisjoint(split_ids["test"]):
        raise AssertionError("train and test original_id overlap.")
    if not split_ids["valid"].isdisjoint(split_ids["test"]):
        raise AssertionError("valid and test original_id overlap.")
    for split, records in split_records.items():
        if not records:
            raise AssertionError(f"{split} split is empty.")
        labels = Counter(record["sufficiency_label"] for record in records)
        if labels["sufficient"] <= 0 or labels["insufficient"] <= 0:
            raise AssertionError(f"{split} must contain both labels.")


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return value


if __name__ == "__main__":
    main()
