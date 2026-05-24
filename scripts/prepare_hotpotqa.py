from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from csrrag.utils.io import write_jsonl
from csrrag.utils.text import lexical_score


TOP_K = 5


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare HotpotQA data for CSR-RAG experiments.")
    parser.add_argument("--input", required=True, help="Path to raw HotpotQA JSON.")
    parser.add_argument("--output-raw", required=True, help="Path to output normalized raw samples JSONL.")
    parser.add_argument(
        "--output-retrieval",
        required=True,
        help="Path to output retrieval-style sufficiency dataset JSONL.",
    )
    parser.add_argument("--max-samples", type=int, default=1800, help="Maximum number of HotpotQA questions to convert.")
    args = parser.parse_args()

    with Path(args.input).open("r", encoding="utf-8") as f:
        data = json.load(f)

    raw_records: list[dict[str, Any]] = []
    retrieval_records: list[dict[str, Any]] = []

    skipped_missing_support = 0
    skipped_not_enough_context = 0
    skipped_too_many_support_docs = 0

    for row in data:
        if len(raw_records) >= args.max_samples:
            break

        original_id = row["_id"]
        question = row["question"]
        answer = row["answer"]
        sample_type = row.get("type", "unknown")
        sample_level = row.get("level", "unknown")

        support_title_set = set()
        for title, _sent_idx in row["supporting_facts"]:
            normalized_title = _normalize_title(title)
            if normalized_title not in support_title_set:
                support_title_set.add(normalized_title)

        context_docs = []
        for context_item in row["context"]:
            title, sentences = context_item
            text = " ".join(sentences)
            normalized_title = _normalize_title(title)
            context_docs.append(
                {
                    "doc_id": f"{original_id}::{title}",
                    "title": title,
                    "text": text,
                    "source": "hotpotqa_context",
                    "score": lexical_score(question, title, text),
                    "_normalized_title": normalized_title,
                    "_is_support": normalized_title in support_title_set,
                }
            )

        present_support_titles = {doc["_normalized_title"] for doc in context_docs if doc["_is_support"]}
        if not support_title_set.issubset(present_support_titles):
            skipped_missing_support += 1
            continue

        support_docs = [doc for doc in context_docs if doc["_is_support"]]
        distractor_docs = [doc for doc in context_docs if not doc["_is_support"]]

        if len(support_docs) > TOP_K:
            skipped_too_many_support_docs += 1
            continue

        needed_sufficient_distractors = TOP_K - len(support_docs)
        needed_insufficient_distractors = TOP_K - (len(support_docs) - 1)
        if (
            len(context_docs) < TOP_K
            or len(distractor_docs) < needed_sufficient_distractors
            or len(distractor_docs) < needed_insufficient_distractors
        ):
            skipped_not_enough_context += 1
            continue

        sufficient_docs = _build_sufficient_docs(support_docs, distractor_docs)
        insufficient_docs, dropped_support_doc = _build_insufficient_docs(support_docs, distractor_docs)
        if len(sufficient_docs) != TOP_K or len(insufficient_docs) != TOP_K:
            skipped_not_enough_context += 1
            continue

        raw_records.append(
            {
                "id": original_id,
                "query": question,
                "gold_answer": answer,
                "dataset": "hotpotqa",
                "metadata": {
                    "split": "validation",
                    "question_type": sample_type,
                    "difficulty": sample_level,
                    "support_title_count": len(support_docs),
                },
            }
        )

        retrieval_records.append(
            _make_retrieval_record(
                record_id=f"{original_id}__sufficient",
                original_id=original_id,
                query=question,
                answer=answer,
                docs=sufficient_docs,
                label="sufficient",
                sample_type=sample_type,
                sample_level=sample_level,
                variant="all_support_docs_present",
                support_doc_ids=[doc["doc_id"] for doc in support_docs],
                dropped_support_doc_id=None,
            )
        )
        retrieval_records.append(
            _make_retrieval_record(
                record_id=f"{original_id}__insufficient",
                original_id=original_id,
                query=question,
                answer=answer,
                docs=insufficient_docs,
                label="insufficient",
                sample_type=sample_type,
                sample_level=sample_level,
                variant="one_support_doc_dropped",
                support_doc_ids=[doc["doc_id"] for doc in support_docs],
                dropped_support_doc_id=dropped_support_doc["doc_id"],
            )
        )

    write_jsonl(args.output_raw, raw_records)
    write_jsonl(args.output_retrieval, retrieval_records)

    print(
        {
            "input_rows": len(data),
            "raw_samples": len(raw_records),
            "retrieval_records": len(retrieval_records),
            "skipped_missing_support": skipped_missing_support,
            "skipped_not_enough_context": skipped_not_enough_context,
            "skipped_too_many_support_docs": skipped_too_many_support_docs,
            "top_k": TOP_K,
            "output_raw": args.output_raw,
            "output_retrieval": args.output_retrieval,
        }
    )


def _build_sufficient_docs(
    support_docs: list[dict[str, Any]],
    distractor_docs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    needed_distractors = TOP_K - len(support_docs)
    chosen_distractors = sorted(distractor_docs, key=lambda doc: (-float(doc["score"]), doc["title"]))[:needed_distractors]
    return _finalize_docs(support_docs + chosen_distractors)


def _build_insufficient_docs(
    support_docs: list[dict[str, Any]],
    distractor_docs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dropped_support_doc = min(support_docs, key=lambda doc: (float(doc["score"]), doc["title"]))
    remaining_support_docs = [doc for doc in support_docs if doc["doc_id"] != dropped_support_doc["doc_id"]]
    needed_distractors = TOP_K - len(remaining_support_docs)
    chosen_distractors = sorted(distractor_docs, key=lambda doc: (-float(doc["score"]), doc["title"]))[:needed_distractors]
    return _finalize_docs(remaining_support_docs + chosen_distractors), dropped_support_doc


def _finalize_docs(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_docs = sorted(docs, key=lambda doc: (-float(doc["score"]), doc["title"]))[:TOP_K]
    finalized = []
    for rank, doc in enumerate(sorted_docs, start=1):
        finalized.append(
            {
                "doc_id": doc["doc_id"],
                "rank": rank,
                "score": round(float(doc["score"]), 6),
                "title": doc["title"],
                "text": doc["text"],
                "source": doc["source"],
            }
        )
    return finalized


def _make_retrieval_record(
    record_id: str,
    original_id: str,
    query: str,
    answer: str,
    docs: list[dict[str, Any]],
    label: str,
    sample_type: str,
    sample_level: str,
    variant: str,
    support_doc_ids: list[str],
    dropped_support_doc_id: str | None,
) -> dict[str, Any]:
    return {
        "id": record_id,
        "query": query,
        "gold_answer": answer,
        "sufficiency_label": label,
        "retrieved_docs": docs,
        "metadata": {
            "dataset": "hotpotqa",
            "split": "validation",
            "original_id": original_id,
            "question_type": sample_type,
            "difficulty": sample_level,
            "variant": variant,
            "support_doc_ids": support_doc_ids,
            "dropped_support_doc_id": dropped_support_doc_id,
        },
    }


def _normalize_title(title: str) -> str:
    return html.unescape(str(title)).strip().lower()


if __name__ == "__main__":
    main()
