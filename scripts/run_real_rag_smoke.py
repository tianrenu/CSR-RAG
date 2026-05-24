from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import string
from pathlib import Path
from typing import Any

import numpy as np

from csrrag.calibration.methods import make_calibrator
from csrrag.evaluation.metrics import decision_metrics_from_risk
from csrrag.features.basic import extract_basic_features
from csrrag.models.baseline import train_estimator
from csrrag.rag.api_clients import OpenAICompatibleChatClient, OpenAICompatibleEmbeddingClient
from csrrag.utils.env import load_dotenv
from csrrag.utils.io import read_jsonl, write_jsonl
from csrrag.utils.text import lexical_score


FEATURE_COLUMNS = [
    "query_length",
    "has_time_constraint",
    "has_constraint_term",
    "topk_score_mean",
    "topk_score_std",
    "top1_top2_gap",
    "doc_count",
    "doc_redundancy",
    "title_overlap_max",
    "title_overlap_mean",
    "text_overlap_max",
    "text_overlap_mean",
    "top1_score",
    "top3_score_mean",
    "top5_score_min",
    "top1_top5_gap",
    "score_entropy",
    "query_token_coverage_union",
    "query_token_coverage_top1",
    "query_token_coverage_top3",
    "uncovered_query_token_ratio",
    "pairwise_doc_overlap_mean",
    "pairwise_doc_overlap_max",
    "unique_title_token_ratio",
    "doc_text_length_mean",
    "doc_text_length_std",
    "is_comparison_question",
    "is_bridge_like_question",
    "wh_who",
    "wh_what",
    "wh_when",
    "wh_where",
    "wh_which",
    "wh_how",
    "wh_other",
]
TAU_GRID = [round(i / 100, 2) for i in range(5, 100, 5)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small real RAG smoke test with embedding retrieval and LLM answers.")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--raw-hotpot", default="data/raw/hotpotqa/hotpot_dev_fullwiki_v1.json")
    parser.add_argument("--test-split", default="data/processed/hotpotqa_dev_splits_1800/test.jsonl")
    parser.add_argument("--train-features", default="data/features/hotpotqa_dev_train_1800_features.jsonl")
    parser.add_argument("--valid-features", default="data/features/hotpotqa_dev_valid_1800_features.jsonl")
    parser.add_argument("--output-dir", default="results/tables/hotpotqa_real_rag_smoke")
    parser.add_argument("--artifact-dir", default="data/outputs/hotpotqa_real_rag_smoke")
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedding-batch-size", type=int, default=10)
    args = parser.parse_args()

    load_dotenv(args.env_file)
    embedding_client = OpenAICompatibleEmbeddingClient(
        base_url=_required_env("EMBEDDING_BASE_URL"),
        api_key=_required_env("EMBEDDING_API_KEY"),
        model=_required_env("EMBEDDING_MODEL"),
    )
    chat_client = OpenAICompatibleChatClient(
        base_url=_required_env("LLM_BASE_URL"),
        api_key=_required_env("LLM_API_KEY"),
        model=_required_env("LLM_MODEL"),
    )

    output_dir = Path(args.output_dir)
    artifact_dir = Path(args.artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_bundle = _fit_risk_model(args.train_features, args.valid_features)
    raw_rows = _load_raw_rows(args.raw_hotpot)
    selected_ids = _select_test_original_ids(args.test_split, args.max_examples)

    detail_rows = []
    retrieval_records = []
    for original_id in selected_ids:
        raw = raw_rows[original_id]
        retrieved_docs = _embedding_retrieve(raw, embedding_client, args.top_k, args.embedding_batch_size)
        support_titles = _support_titles(raw)
        support_present = support_titles.issubset({_normalize_title(doc["title"]) for doc in retrieved_docs})
        retrieval_record = {
            "id": f"{original_id}__real_rag",
            "query": raw["question"],
            "gold_answer": raw["answer"],
            "sufficiency_label": "sufficient" if support_present else "insufficient",
            "retrieved_docs": retrieved_docs,
            "metadata": {
                "dataset": "hotpotqa",
                "original_id": original_id,
                "support_present_in_topk": support_present,
            },
        }
        retrieval_records.append(retrieval_record)

        risk_score = _predict_risk(model_bundle, retrieval_record)
        decision = "answer" if risk_score <= model_bundle["tau_answer"] else "abstain"
        naive_answer = chat_client.answer(raw["question"], retrieved_docs)

        naive_em = exact_match(naive_answer, raw["answer"])
        naive_f1 = f1_score(naive_answer, raw["answer"])
        csr_answer = naive_answer if decision == "answer" else ""
        csr_em = exact_match(csr_answer, raw["answer"]) if decision == "answer" else 0.0
        csr_f1 = f1_score(csr_answer, raw["answer"]) if decision == "answer" else 0.0

        detail_rows.append(
            {
                "original_id": original_id,
                "question": raw["question"],
                "gold_answer": raw["answer"],
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
                "top5_titles": " || ".join(doc["title"] for doc in retrieved_docs),
                "top5_scores": " || ".join(str(doc["score"]) for doc in retrieved_docs),
                "top5_embedding_scores": " || ".join(str(doc["embedding_score"]) for doc in retrieved_docs),
            }
        )

    write_jsonl(artifact_dir / "real_rag_retrieval_records.jsonl", retrieval_records)
    _write_csv(output_dir / "real_rag_smoke_details.csv", detail_rows, list(detail_rows[0].keys()))
    _write_csv(output_dir / "real_rag_smoke_summary.csv", [_summary_row(detail_rows)], list(_summary_row(detail_rows).keys()))
    (output_dir / "real_rag_smoke_config.json").write_text(
        json.dumps(
            {
                "embedding_model": os.environ["EMBEDDING_MODEL"],
                "llm_model": os.environ["LLM_MODEL"],
                "max_examples": args.max_examples,
                "top_k": args.top_k,
                "tau_answer": model_bundle["tau_answer"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print({"output_dir": str(output_dir), "artifact_dir": str(artifact_dir), **_summary_row(detail_rows)})


def _fit_risk_model(train_features_path: str, valid_features_path: str) -> dict[str, Any]:
    train_records = read_jsonl(train_features_path)
    valid_records = read_jsonl(valid_features_path)
    model = train_estimator(
        "logistic_regression",
        [_feature_row(record) for record in train_records],
        [1 if record["sufficiency_label"] == "sufficient" else 0 for record in train_records],
        FEATURE_COLUMNS,
    )
    valid_rows = [_feature_row(record) for record in valid_records]
    valid_labels = np.array([0 if record["sufficiency_label"] == "sufficient" else 1 for record in valid_records])
    valid_raw_risk = 1.0 - np.asarray(model.predict_proba(valid_rows), dtype=float)
    calibrator = make_calibrator("isotonic")
    calibrator.fit(valid_raw_risk, valid_labels)
    valid_risk = np.asarray(calibrator.predict(valid_raw_risk), dtype=float)
    tau_answer = _select_tau(valid_labels, valid_risk)
    return {"model": model, "calibrator": calibrator, "tau_answer": tau_answer}


def _feature_row(record: dict[str, Any]) -> dict[str, float]:
    return {name: float(record[name]) for name in FEATURE_COLUMNS}


def _predict_risk(model_bundle: dict[str, Any], retrieval_record: dict[str, Any]) -> float:
    features = extract_basic_features(retrieval_record)
    sufficiency_score = float(model_bundle["model"].predict_proba([_feature_row(features)])[0])
    raw_risk = np.array([1.0 - sufficiency_score], dtype=float)
    return float(model_bundle["calibrator"].predict(raw_risk)[0])


def _select_tau(labels: np.ndarray, risk_scores: np.ndarray) -> float:
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


def _embedding_retrieve(
    raw: dict[str, Any],
    embedding_client: OpenAICompatibleEmbeddingClient,
    top_k: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    docs = []
    for title, sentences in raw["context"]:
        text = " ".join(sentences)
        docs.append({"title": title, "text": text})
    texts = [raw["question"]] + [f"{doc['title']}\n{doc['text']}" for doc in docs]
    embeddings = []
    for start in range(0, len(texts), batch_size):
        embeddings.extend(embedding_client.embed(texts[start:start + batch_size]))
    query_embedding = np.asarray(embeddings[0], dtype=float)
    doc_embeddings = np.asarray(embeddings[1:], dtype=float)
    scores = _cosine_scores(query_embedding, doc_embeddings)
    ranked_indices = np.argsort(-scores)[:top_k]
    retrieved_docs = []
    for rank, doc_index in enumerate(ranked_indices, start=1):
        doc = docs[int(doc_index)]
        retrieved_docs.append(
            {
                "doc_id": f"{raw['_id']}::{doc['title']}",
                "rank": rank,
                "score": round(float(lexical_score(raw["question"], doc["title"], doc["text"])), 6),
                "embedding_score": round(float(scores[doc_index]), 6),
                "title": doc["title"],
                "text": doc["text"],
                "source": "hotpotqa_context_embedding",
            }
        )
    return retrieved_docs


def _cosine_scores(query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_embedding)
    doc_norms = np.linalg.norm(doc_embeddings, axis=1)
    denom = np.maximum(query_norm * doc_norms, 1e-12)
    return doc_embeddings.dot(query_embedding) / denom


def _load_raw_rows(path: str) -> dict[str, dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        rows = json.load(f)
    return {row["_id"]: row for row in rows}


def _select_test_original_ids(path: str, max_examples: int) -> list[str]:
    records = read_jsonl(path)
    selected = []
    seen = set()
    for record in records:
        original_id = record["metadata"]["original_id"]
        if original_id in seen:
            continue
        selected.append(original_id)
        seen.add(original_id)
        if len(selected) >= max_examples:
            break
    return selected


def _support_titles(raw: dict[str, Any]) -> set[str]:
    return {_normalize_title(title) for title, _sent_idx in raw["supporting_facts"]}


def _normalize_title(title: str) -> str:
    return html.unescape(str(title)).strip().lower()


def _summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answered_rows = [row for row in rows if row["csr_decision"] == "answer"]
    insufficient_rows = [row for row in rows if not row["support_present_in_topk"]]
    insufficient_answered = [row for row in insufficient_rows if row["csr_decision"] == "answer"]
    return {
        "n": len(rows),
        "naive_em": _mean(row["naive_em"] for row in rows),
        "naive_f1": _mean(row["naive_f1"] for row in rows),
        "csr_coverage": len(answered_rows) / len(rows) if rows else 0.0,
        "csr_answered_em": _mean(row["csr_em"] for row in answered_rows),
        "csr_answered_f1": _mean(row["csr_f1"] for row in answered_rows),
        "support_present_rate": _mean(float(row["support_present_in_topk"]) for row in rows),
        "insufficient_answer_rate": len(insufficient_answered) / len(insufficient_rows) if insufficient_rows else 0.0,
    }


def _mean(values: Any) -> float:
    value_list = list(values)
    return float(sum(value_list) / len(value_list)) if value_list else 0.0


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        return "".join(ch for ch in value if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(text.lower())))


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


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
