from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from csrrag.calibration.methods import make_calibrator
from csrrag.evaluation.metrics import decision_metrics_from_risk
from csrrag.experiments.feature_sets import EMBEDDING_FEATURES
from csrrag.features.basic import extract_basic_features
from csrrag.models.baseline import train_estimator
from csrrag.rag.api_clients import OpenAICompatibleChatClient
from csrrag.utils.env import load_dotenv
from csrrag.utils.io import read_jsonl


TAU_GRID = [round(i / 100, 2) for i in range(5, 100, 5)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small MiniMax QA evaluation on embedding-retrieval CSR-RAG outputs.")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--split-dir", default="data/processed/hotpotqa_embedding_splits_1800")
    parser.add_argument("--output-dir", default="results/tables/hotpotqa_real_rag_qa_eval")
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    load_dotenv(args.env_file)
    chat_client = OpenAICompatibleChatClient(
        base_url=_required_env("LLM_BASE_URL"),
        api_key=_required_env("LLM_API_KEY"),
        model=_required_env("LLM_MODEL"),
    )

    split_records = {split: read_jsonl(Path(args.split_dir) / f"{split}.jsonl") for split in ("train", "valid", "test")}
    model_bundle = _fit_risk_model(split_records)
    selected_records = _select_test_records(split_records["test"], args.max_examples, args.sample_seed)

    detail_rows = []
    for idx, record in enumerate(selected_records, start=1):
        risk_score = _predict_risk(model_bundle, record)
        decision = "answer" if risk_score <= model_bundle["tau_answer"] else "abstain"
        answer_result = _safe_answer(chat_client, record["query"], record["retrieved_docs"], args.max_tokens)
        naive_answer = answer_result["answer"]
        naive_em = exact_match(naive_answer, record["gold_answer"])
        naive_f1 = f1_score(naive_answer, record["gold_answer"])
        csr_answer = naive_answer if decision == "answer" else ""
        csr_em = exact_match(csr_answer, record["gold_answer"]) if decision == "answer" else 0.0
        csr_f1 = f1_score(csr_answer, record["gold_answer"]) if decision == "answer" else 0.0
        support_present = record["sufficiency_label"] == "sufficient"

        detail_rows.append(
            {
                "sample_index": idx,
                "original_id": record["metadata"]["original_id"],
                "question": record["query"],
                "gold_answer": record["gold_answer"],
                "sufficiency_label": record["sufficiency_label"],
                "support_present_in_topk": support_present,
                "risk_score": risk_score,
                "tau_answer": model_bundle["tau_answer"],
                "csr_decision": decision,
                "naive_answer": naive_answer,
                "naive_em": naive_em,
                "naive_f1": naive_f1,
                "csr_answer": csr_answer,
                "csr_em": csr_em,
                "csr_f1": csr_f1,
                "llm_json_parse_ok": answer_result["json_parse_ok"],
                "llm_format_ok": answer_result.get("format_ok", False),
                "llm_attempt": answer_result.get("attempt", 0),
                "llm_used_fallback": answer_result["used_fallback"],
                "llm_had_thinking": answer_result["had_thinking"],
                "llm_had_reasoning_details": answer_result.get("had_reasoning_details", False),
                "llm_finish_reason": answer_result.get("finish_reason", ""),
                "llm_error": answer_result["error"],
                "top5_titles": " || ".join(doc["title"] for doc in record["retrieved_docs"]),
                "top5_embedding_scores": " || ".join(str(doc.get("embedding_score", "")) for doc in record["retrieved_docs"]),
            }
        )
        print(json.dumps({"processed": idx, "total": len(selected_records), "decision": decision, "risk": round(risk_score, 4)}, ensure_ascii=False))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _summary_row(detail_rows)
    _write_csv(output_dir / "real_rag_qa_details.csv", detail_rows, list(detail_rows[0].keys()))
    _write_csv(output_dir / "real_rag_qa_summary.csv", [summary], list(summary.keys()))
    _write_csv(output_dir / "abstention_cases.csv", [row for row in detail_rows if row["csr_decision"] == "abstain"], list(detail_rows[0].keys()))
    (output_dir / "real_rag_qa_config.json").write_text(
        json.dumps(
            {
                "llm_model": os.environ["LLM_MODEL"],
                "max_examples": args.max_examples,
                "sample_seed": args.sample_seed,
                "tau_answer": model_bundle["tau_answer"],
                "feature_set": "all_embedding",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "real_rag_qa_summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), **summary}, ensure_ascii=False))


def _fit_risk_model(split_records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    train_features = [_feature_record(record) for record in split_records["train"]]
    valid_features = [_feature_record(record) for record in split_records["valid"]]
    model = train_estimator(
        "logistic_regression",
        [_feature_row(record, EMBEDDING_FEATURES) for record in train_features],
        _sufficiency_labels(train_features),
        EMBEDDING_FEATURES,
    )
    valid_rows = [_feature_row(record, EMBEDDING_FEATURES) for record in valid_features]
    valid_labels = _risk_labels(valid_features)
    valid_raw_risk = 1.0 - np.asarray(model.predict_proba(valid_rows), dtype=float)
    calibrator = make_calibrator("isotonic")
    calibrator.fit(valid_raw_risk, valid_labels)
    valid_risk = np.asarray(calibrator.predict(valid_raw_risk), dtype=float)
    tau_answer = _select_tau(valid_labels, valid_risk)
    return {"model": model, "calibrator": calibrator, "tau_answer": tau_answer}


def _feature_record(record: dict[str, Any]) -> dict[str, Any]:
    return {"id": record["id"], **extract_basic_features(record), "sufficiency_label": record["sufficiency_label"]}


def _predict_risk(model_bundle: dict[str, Any], retrieval_record: dict[str, Any]) -> float:
    features = _feature_record(retrieval_record)
    sufficiency_score = float(model_bundle["model"].predict_proba([_feature_row(features, EMBEDDING_FEATURES)])[0])
    raw_risk = np.array([1.0 - sufficiency_score], dtype=float)
    return float(model_bundle["calibrator"].predict(raw_risk)[0])


def _feature_row(record: dict[str, Any], feature_names: list[str]) -> dict[str, float]:
    return {name: float(record[name]) for name in feature_names}


def _sufficiency_labels(records: list[dict[str, Any]]) -> list[int]:
    return [1 if record["sufficiency_label"] == "sufficient" else 0 for record in records]


def _risk_labels(records: list[dict[str, Any]]) -> np.ndarray:
    return np.array([0 if record["sufficiency_label"] == "sufficient" else 1 for record in records], dtype=int)


def _select_tau(labels: Iterable[int], risk_scores: Iterable[float]) -> float:
    best_tau = TAU_GRID[0]
    best_metrics = decision_metrics_from_risk(labels, risk_scores, best_tau)
    for tau in TAU_GRID[1:]:
        metrics = decision_metrics_from_risk(labels, risk_scores, tau)
        if metrics["decision_accuracy"] > best_metrics["decision_accuracy"]:
            best_tau = tau
            best_metrics = metrics
        elif metrics["decision_accuracy"] == best_metrics["decision_accuracy"] and metrics["coverage"] > best_metrics["coverage"]:
            best_tau = tau
            best_metrics = metrics
    return best_tau


def _select_test_records(records: list[dict[str, Any]], max_examples: int, seed: int) -> list[dict[str, Any]]:
    shuffled = list(records)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    return shuffled[:max_examples]


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


def _summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(row["sufficiency_label"] for row in rows)
    answered_rows = [row for row in rows if row["csr_decision"] == "answer"]
    insufficient_rows = [row for row in rows if row["sufficiency_label"] == "insufficient"]
    insufficient_answered = [row for row in insufficient_rows if row["csr_decision"] == "answer"]
    parse_failures = [row for row in rows if not row["llm_json_parse_ok"]]
    format_failures = [row for row in rows if not row["llm_format_ok"]]
    api_errors = [row for row in rows if row["llm_error"]]
    return {
        "n": len(rows),
        "sufficient": labels["sufficient"],
        "insufficient": labels["insufficient"],
        "naive_em": _mean(row["naive_em"] for row in rows),
        "naive_f1": _mean(row["naive_f1"] for row in rows),
        "csr_coverage": len(answered_rows) / len(rows) if rows else 0.0,
        "csr_answered_em": _mean(row["csr_em"] for row in answered_rows),
        "csr_answered_f1": _mean(row["csr_f1"] for row in answered_rows),
        "csr_answered_sufficiency_rate": _mean(float(row["sufficiency_label"] == "sufficient") for row in answered_rows),
        "support_present_rate": _mean(float(row["support_present_in_topk"]) for row in rows),
        "insufficient_answer_rate": len(insufficient_answered) / len(insufficient_rows) if insufficient_rows else 0.0,
        "llm_json_parse_failure_rate": len(parse_failures) / len(rows) if rows else 0.0,
        "llm_format_failure_rate": len(format_failures) / len(rows) if rows else 0.0,
        "llm_api_error_count": len(api_errors),
    }


def _summary_markdown(summary: dict[str, Any]) -> str:
    return f"""# 真实 RAG QA 小规模评测摘要

本评测使用 embedding top-5 检索和 MiniMax 回答，只作为真实 RAG 主实验前的小规模 QA 验证，不作为全量最终结论。

- 样本数：{summary["n"]}
- sufficient / insufficient：{summary["sufficient"]} / {summary["insufficient"]}
- Naive RAG EM / F1：{summary["naive_em"]:.4f} / {summary["naive_f1"]:.4f}
- CSR-RAG coverage：{summary["csr_coverage"]:.4f}
- CSR-RAG answered EM / F1：{summary["csr_answered_em"]:.4f} / {summary["csr_answered_f1"]:.4f}
- Insufficient answer rate：{summary["insufficient_answer_rate"]:.4f}
- LLM JSON parse failure rate：{summary["llm_json_parse_failure_rate"]:.4f}
- LLM strict format failure rate：{summary["llm_format_failure_rate"]:.4f}

解释时需要谨慎：如果 CSR-RAG 的 answered EM/F1 高于 Naive RAG，但 coverage 较低，说明选择性回答提高了回答子集可靠性；如果没有提升，应把问题归因拆开看，包括检索召回、风险阈值和 LLM 短答案稳定性。
"""


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


def _mean(values: Iterable[float]) -> float:
    value_list = list(values)
    return float(sum(value_list) / len(value_list)) if value_list else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return value


if __name__ == "__main__":
    main()
