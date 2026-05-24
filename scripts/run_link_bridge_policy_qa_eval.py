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

from csrrag.rag.api_clients import OpenAICompatibleChatClient
from csrrag.utils.env import load_dotenv

from run_link_bridge_retrieve_more_experiments import (
    _feature_record,
    _feature_sets,
    _load_topk_records,
    _risk_runs,
    _split_valid_ids,
    _validate_features,
    _validate_records,
)


DETAIL_FIELDNAMES = [
    "sample_index",
    "sample_bucket",
    "original_id",
    "question",
    "gold_answer",
    "label_top5",
    "label_top10",
    "label_top20",
    "top5_answer",
    "top5_em",
    "top5_f1",
    "top5_is_substantive",
    "top5_json_parse_ok",
    "top5_format_ok",
    "top5_error",
    "top20_answer",
    "top20_em",
    "top20_f1",
    "top20_is_substantive",
    "top20_json_parse_ok",
    "top20_format_ok",
    "top20_error",
    "top5_titles",
    "top20_titles",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a stratified MiniMax QA evaluation for link-bridge top5/top20 policies. "
            "The script calls the LLM for top5 and top20 contexts once per sampled question, then reuses those answers across policies."
        )
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--split-dir", default="data/processed/hotpotqa_official_intro_link_bridge_support_reranker_splits")
    parser.add_argument("--record-prefix", default="official_intro_link_bridge_support_reranker_random_forest_balanced_all_blend1_00_top")
    parser.add_argument(
        "--policy-main",
        default="results/tables/hotpotqa_official_intro_link_bridge_support_reranker_retrieve_more/main_comparison.csv",
    )
    parser.add_argument("--output-dir", default="results/tables/hotpotqa_official_intro_link_bridge_support_reranker_qa_eval")
    parser.add_argument("--sample-strategy", choices=["stratified", "natural"], default="stratified")
    parser.add_argument("--per-bucket", type=int, default=30)
    parser.add_argument("--max-examples", type=int, default=120, help="Used only when --sample-strategy natural.")
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

    records = _load_topk_records(Path(args.split_dir), args.record_prefix)
    _validate_records(records)
    features = {
        top_k: {
            split: [_feature_record(record) for record in split_records]
            for split, split_records in topk_records.items()
        }
        for top_k, topk_records in records.items()
    }
    feature_sets = _feature_sets()
    _validate_features(features, feature_sets)
    valid_calib_ids, _ = _split_valid_ids(records[5]["valid"])
    _, risks = _risk_runs(features, valid_calib_ids, feature_sets)
    policy_specs = _policy_specs(Path(args.policy_main), risks)
    selected = _select_samples(records, args.sample_strategy, args.per_bucket, args.max_examples, args.sample_seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "link_bridge_qa_details.csv"
    existing = {
        row["original_id"]: row
        for row in _read_csv_if_exists(details_path)
        if _to_bool(row.get("top5_format_ok", False)) and _to_bool(row.get("top20_format_ok", False))
    }
    detail_by_id = dict(existing)
    pending = [(idx, item) for idx, item in enumerate(selected, start=1) if item["original_id"] not in detail_by_id]
    workers = max(1, int(args.qa_workers))
    if workers == 1:
        for idx, item in pending:
            row = _answer_row(chat_client, item, idx, args.max_tokens)
            detail_by_id[row["original_id"]] = row
            _write_details(details_path, selected, detail_by_id)
            _print_progress(row, len(detail_by_id), len(selected))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_answer_row, chat_client, item, idx, args.max_tokens) for idx, item in pending]
            for future in as_completed(futures):
                row = future.result()
                detail_by_id[row["original_id"]] = row
                _write_details(details_path, selected, detail_by_id)
                _print_progress(row, len(detail_by_id), len(selected))

    detail_rows = _ordered_rows(selected, detail_by_id)
    policy_rows = _policy_rows(detail_rows, selected, policy_specs)
    case_rows = _case_rows(detail_rows, selected, policy_specs)
    _write_details(details_path, selected, detail_by_id)
    _write_csv(output_dir / "link_bridge_qa_policy_comparison.csv", policy_rows, list(policy_rows[0].keys()))
    _write_csv(output_dir / "link_bridge_qa_case_studies.csv", case_rows, list(case_rows[0].keys()))
    _write_validation(output_dir / "validation_summary.json", args, selected, detail_rows, policy_specs, policy_rows)
    (output_dir / "link_bridge_qa_summary.md").write_text(_summary_markdown(policy_rows), encoding="utf-8")

    cov85 = next((row for row in policy_rows if row["policy"] == "retrieve_more@cov85"), policy_rows[0])
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "selected": len(selected),
                "completed": len(detail_rows),
                "cov85_coverage": cov85["coverage"],
                "cov85_insufficient_substantive_answer_rate": cov85["insufficient_substantive_answer_rate"],
                "cov85_answered_f1": cov85["answered_f1"],
                "uses_llm_api": True,
            },
            ensure_ascii=False,
        )
    )


def _policy_specs(policy_main: Path, risks: dict[tuple[int, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = _read_csv_if_exists(policy_main)
    specs = [
        {"policy": "naive_top5", "action_type": "always_answer", "target_k": 5},
        {"policy": "naive_top20", "action_type": "always_answer", "target_k": 20},
    ]
    wanted = {"balanced", "retrieve_more_balanced", "retrieve_more_risk_control@suff_abstain15", "retrieve_more@cov85"}
    for row in rows:
        if row.get("policy") not in wanted:
            continue
        spec = dict(row)
        spec["target_k"] = int(float(spec["target_k"]))
        spec["tau_answer_top5"] = _optional_float(spec.get("tau_answer_top5", ""))
        spec["tau_answer_after_more"] = _optional_float(spec.get("tau_answer_after_more", ""))
        spec["risk5_by_id"] = _risk_by_id(risks[(5, spec["feature_set"], spec["calibration"])])
        spec["riskk_by_id"] = _risk_by_id(risks[(spec["target_k"], spec["feature_set"], spec["calibration"])])
        specs.append(spec)
    return specs


def _select_samples(
    records: dict[int, dict[str, list[dict[str, Any]]]],
    sample_strategy: str,
    per_bucket: int,
    max_examples: int,
    seed: int,
) -> list[dict[str, Any]]:
    by_id = {
        top_k: {record["metadata"]["original_id"]: record for record in records[top_k]["test"]}
        for top_k in (5, 10, 20)
    }
    buckets = {
        "top5_sufficient": [],
        "rescued_by_top20": [],
        "unresolved_top20": [],
    }
    for original_id, record5 in by_id[5].items():
        record20 = by_id[20][original_id]
        if record5["sufficiency_label"] == "sufficient":
            bucket = "top5_sufficient"
        elif record20["sufficiency_label"] == "sufficient":
            bucket = "rescued_by_top20"
        else:
            bucket = "unresolved_top20"
        buckets[bucket].append(
            {
                "sample_bucket": bucket,
                "original_id": original_id,
                "record5": record5,
                "record10": by_id[10][original_id],
                "record20": record20,
            }
        )
    rng = random.Random(seed)
    if sample_strategy == "natural":
        selected = [item for bucket_rows in buckets.values() for item in bucket_rows]
        rng.shuffle(selected)
        return selected[:max_examples]
    if sample_strategy != "stratified":
        raise ValueError(f"Unknown sample strategy: {sample_strategy}")
    selected = []
    for bucket_name in ("top5_sufficient", "rescued_by_top20", "unresolved_top20"):
        rows = list(buckets[bucket_name])
        rng.shuffle(rows)
        selected.extend(rows[:per_bucket])
    return selected


def _answer_row(chat_client: OpenAICompatibleChatClient, item: dict[str, Any], sample_index: int, max_tokens: int) -> dict[str, Any]:
    record5 = item["record5"]
    record20 = item["record20"]
    top5 = _safe_answer(chat_client, record5["query"], record5["retrieved_docs"], max_tokens)
    top20 = _safe_answer(chat_client, record20["query"], record20["retrieved_docs"], max_tokens)
    return {
        "sample_index": sample_index,
        "sample_bucket": item["sample_bucket"],
        "original_id": item["original_id"],
        "question": record5["query"],
        "gold_answer": record5["gold_answer"],
        "label_top5": record5["sufficiency_label"],
        "label_top10": item["record10"]["sufficiency_label"],
        "label_top20": record20["sufficiency_label"],
        "top5_answer": top5["answer"],
        "top5_em": exact_match(top5["answer"], record5["gold_answer"]),
        "top5_f1": f1_score(top5["answer"], record5["gold_answer"]),
        "top5_is_substantive": _is_substantive(top5["answer"]),
        "top5_json_parse_ok": top5["json_parse_ok"],
        "top5_format_ok": top5.get("format_ok", False),
        "top5_error": top5["error"],
        "top20_answer": top20["answer"],
        "top20_em": exact_match(top20["answer"], record20["gold_answer"]),
        "top20_f1": f1_score(top20["answer"], record20["gold_answer"]),
        "top20_is_substantive": _is_substantive(top20["answer"]),
        "top20_json_parse_ok": top20["json_parse_ok"],
        "top20_format_ok": top20.get("format_ok", False),
        "top20_error": top20["error"],
        "top5_titles": " || ".join(doc["title"] for doc in record5["retrieved_docs"]),
        "top20_titles": " || ".join(doc["title"] for doc in record20["retrieved_docs"]),
    }


def _policy_rows(detail_rows: list[dict[str, Any]], selected: list[dict[str, Any]], specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    item_by_id = {item["original_id"]: item for item in selected}
    rows = []
    for spec in specs:
        decisions = [_decision(spec, item_by_id[row["original_id"]]) for row in detail_rows]
        answered = [(row, decision) for row, decision in zip(detail_rows, decisions) if decision["decision"] == "answer"]
        insufficient = [(row, decision) for row, decision in zip(detail_rows, decisions) if decision["sufficiency_label"] == "insufficient"]
        sufficient = [(row, decision) for row, decision in zip(detail_rows, decisions) if decision["sufficiency_label"] == "sufficient"]
        answered_insufficient = [(row, decision) for row, decision in answered if decision["sufficiency_label"] == "insufficient"]
        insufficient_substantive = [
            (row, decision)
            for row, decision in answered_insufficient
            if _to_bool(row[f"top{decision['answer_top_k']}_is_substantive"])
        ]
        wrong_substantive = [
            (row, decision)
            for row, decision in answered
            if _to_bool(row[f"top{decision['answer_top_k']}_is_substantive"]) and float(row[f"top{decision['answer_top_k']}_em"]) == 0.0
        ]
        over_abstained = [(row, decision) for row, decision in zip(detail_rows, decisions) if decision["decision"] == "abstain" and decision["sufficiency_label"] == "sufficient"]
        rows.append(
            {
                "policy": spec["policy"],
                "action_type": spec["action_type"],
                "target_k": spec["target_k"],
                "n": len(detail_rows),
                "coverage": len(answered) / len(detail_rows) if detail_rows else 0.0,
                "answered_count": len(answered),
                "answered_em": _mean(float(row[f"top{decision['answer_top_k']}_em"]) for row, decision in answered),
                "answered_f1": _mean(float(row[f"top{decision['answer_top_k']}_f1"]) for row, decision in answered),
                "answered_sufficient_rate": _mean(float(decision["sufficiency_label"] == "sufficient") for _, decision in answered),
                "decision_insufficient_answer_rate": len(answered_insufficient) / len(insufficient) if insufficient else 0.0,
                "insufficient_substantive_answer_rate": len(insufficient_substantive) / len(insufficient) if insufficient else 0.0,
                "sufficient_abstain_rate": len(over_abstained) / len(sufficient) if sufficient else 0.0,
                "false_answer_count": len(answered_insufficient),
                "insufficient_substantive_count": len(insufficient_substantive),
                "wrong_substantive_count": len(wrong_substantive),
                "over_abstain_count": len(over_abstained),
                "tau_answer_top5": spec.get("tau_answer_top5", ""),
                "tau_answer_after_more": spec.get("tau_answer_after_more", ""),
            }
        )
    return rows


def _case_rows(detail_rows: list[dict[str, Any]], selected: list[dict[str, Any]], specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    item_by_id = {item["original_id"]: item for item in selected}
    rows = []
    for spec in specs:
        for row in detail_rows:
            decision = _decision(spec, item_by_id[row["original_id"]])
            if decision["decision"] == "abstain":
                case_type = "over_abstain" if decision["sufficiency_label"] == "sufficient" else "successful_intercept"
                answer = ""
                f1 = 0.0
            else:
                answer = row[f"top{decision['answer_top_k']}_answer"]
                f1 = float(row[f"top{decision['answer_top_k']}_f1"])
                if decision["sufficiency_label"] == "insufficient":
                    case_type = "false_answer"
                else:
                    case_type = "safe_answer"
            if case_type in {"safe_answer", "successful_intercept"} and len([r for r in rows if r["policy"] == spec["policy"] and r["case_type"] == case_type]) >= 5:
                continue
            rows.append(
                {
                    "policy": spec["policy"],
                    "case_type": case_type,
                    "original_id": row["original_id"],
                    "sample_bucket": row["sample_bucket"],
                    "decision": decision["decision"],
                    "answer_top_k": decision["answer_top_k"],
                    "sufficiency_label": decision["sufficiency_label"],
                    "risk_top5": decision.get("risk_top5", ""),
                    "risk_topk": decision.get("risk_topk", ""),
                    "question": row["question"],
                    "gold_answer": row["gold_answer"],
                    "answer": answer,
                    "f1": f1,
                    "top5_titles": row["top5_titles"],
                    "top20_titles": row["top20_titles"],
                }
            )
    return rows


def _decision(spec: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    original_id = item["original_id"]
    if spec["action_type"] == "always_answer":
        top_k = int(spec["target_k"])
        return {
            "decision": "answer",
            "answer_top_k": top_k,
            "sufficiency_label": item[f"record{top_k}"]["sufficiency_label"],
        }
    if spec["action_type"] == "answer_abstain_top5":
        risk5 = float(spec["risk5_by_id"][original_id])
        answer = risk5 <= float(spec["tau_answer_top5"])
        return {
            "decision": "answer" if answer else "abstain",
            "answer_top_k": 5,
            "sufficiency_label": item["record5"]["sufficiency_label"],
            "risk_top5": risk5,
            "risk_topk": risk5,
        }
    risk5 = float(spec["risk5_by_id"][original_id])
    if risk5 <= float(spec["tau_answer_top5"]):
        return {
            "decision": "answer",
            "answer_top_k": 5,
            "sufficiency_label": item["record5"]["sufficiency_label"],
            "risk_top5": risk5,
            "risk_topk": risk5,
        }
    target_k = int(spec["target_k"])
    riskk = float(spec["riskk_by_id"][original_id])
    answer = riskk <= float(spec["tau_answer_after_more"])
    return {
        "decision": "answer" if answer else "abstain",
        "answer_top_k": target_k if answer else target_k,
        "sufficiency_label": item[f"record{target_k}"]["sufficiency_label"],
        "risk_top5": risk5,
        "risk_topk": riskk,
    }


def _risk_by_id(run: dict[str, Any]) -> dict[str, float]:
    return {original_id: float(risk) for original_id, risk in zip(run["test_ids"], run["test_risk"])}


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
    except Exception as exc:  # noqa: BLE001 - record API failures as experimental data.
        return {
            "answer": "",
            "json_parse_ok": False,
            "format_ok": False,
            "used_fallback": False,
            "had_thinking": False,
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


def _is_substantive(answer: str) -> bool:
    return bool(normalize_answer(answer)) and not _is_dont_know(answer)


def _optional_float(value: Any) -> float | str:
    if value in {"", None}:
        return ""
    return float(value)


def _required_env(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return value


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def _mean(values: Iterable[float]) -> float:
    value_list = list(values)
    return float(sum(value_list) / len(value_list)) if value_list else 0.0


def _read_csv_if_exists(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_details(path: Path, selected: list[dict[str, Any]], detail_by_id: dict[str, dict[str, Any]]) -> None:
    _write_csv(path, _ordered_rows(selected, detail_by_id), DETAIL_FIELDNAMES)


def _ordered_rows(selected: list[dict[str, Any]], detail_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [detail_by_id[item["original_id"]] for item in selected if item["original_id"] in detail_by_id]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_validation(
    path: Path,
    args: argparse.Namespace,
    selected: list[dict[str, Any]],
    detail_rows: list[dict[str, Any]],
    policy_specs: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
) -> None:
    validation = {
        "split_dir": args.split_dir,
        "record_prefix": args.record_prefix,
        "policy_main": args.policy_main,
        "sample_strategy": args.sample_strategy,
        "per_bucket": args.per_bucket,
        "max_examples": args.max_examples,
        "sample_seed": args.sample_seed,
        "sample_counts": dict(Counter(item["sample_bucket"] for item in selected)),
        "selected_count": len(selected),
        "completed_count": len(detail_rows),
        "policy_count": len(policy_specs),
        "policies": [spec["policy"] for spec in policy_specs],
        "policy_rows": policy_rows,
        "llm_model": os.environ.get("LLM_MODEL", ""),
        "uses_llm_api": True,
        "uses_embedding_api": False,
        "qa_answer_cache": "top5 and top20 answers are cached in link_bridge_qa_details.csv",
    }
    path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")


def _summary_markdown(policy_rows: list[dict[str, Any]]) -> str:
    lines = ["# Link-Bridge Reranker QA Evaluation", ""]
    lines.append("| Policy | Coverage | Answered F1 | Insufficient Substantive Answer Rate | Wrong Substantive Count |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in policy_rows:
        lines.append(
            f"| {row['policy']} | {float(row['coverage']):.4f} | {float(row['answered_f1']):.4f} | "
            f"{float(row['insufficient_substantive_answer_rate']):.4f} | {int(row['wrong_substantive_count'])} |"
        )
    lines.append("")
    lines.append("This is a stratified QA sample over reranked link-bridge records. It is meant to validate whether retrieval-level reliability gains reduce unsupported substantive answers.")
    return "\n".join(lines) + "\n"


def _print_progress(row: dict[str, Any], processed: int, target: int) -> None:
    print(
        json.dumps(
            {
                "processed": processed,
                "target": target,
                "bucket": row["sample_bucket"],
                "top5_format_ok": row["top5_format_ok"],
                "top20_format_ok": row["top20_format_ok"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
