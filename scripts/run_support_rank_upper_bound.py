from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from csrrag.utils.io import read_jsonl
from prepare_global_hardneg_retrieval_dataset import (
    _build_global_doc_pool,
    _doc_embedding_text,
    _embedding_cache_get,
    _load_embedding_caches,
    _load_raw_rows,
    _normalize_title,
    _support_titles,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run no-API support-title rank upper-bound analysis for CSR-RAG. "
            "This estimates whether deeper global embedding retrieval contains the full HotpotQA support chain."
        )
    )
    parser.add_argument("--raw-hotpot", default="data/raw/hotpotqa/hotpot_dev_fullwiki_v1.json")
    parser.add_argument("--split-dir", default="data/processed/hotpotqa_global_hardneg_splits_full_dev")
    parser.add_argument("--record-kind-filter", default="natural_global_top5")
    parser.add_argument("--cache-path", default="data/cache/hotpotqa_text_embedding_v4_full_dev.jsonl")
    parser.add_argument(
        "--seed-cache-path",
        action="append",
        default=["data/cache/hotpotqa_text_embedding_v4_1800.jsonl"],
        help="Additional existing embedding cache to reuse. May be passed multiple times.",
    )
    parser.add_argument("--embedding-model", default="text-embedding-v4")
    parser.add_argument("--top-ks", default="5,8,10,15,20,30,50,100")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", default="results/tables/hotpotqa_full_dev_support_rank_upper_bound")
    parser.add_argument("--max-questions-per-split", type=int, default=0, help="Debug only. 0 means full split.")
    args = parser.parse_args()

    started = time.time()
    top_ks = _parse_top_ks(args.top_ks)
    base_records = _load_base_records(Path(args.split_dir), args.record_kind_filter, args.max_questions_per_split)
    _validate_split_records(base_records)
    split_ids = {split: [record["metadata"]["original_id"] for record in records] for split, records in base_records.items()}
    selected_ids = [original_id for split in ("train", "valid", "test") for original_id in split_ids[split]]

    raw_rows = _load_raw_rows(args.raw_hotpot)
    global_docs = _build_global_doc_pool(raw_rows, selected_ids)
    title_to_indices = _title_to_indices(global_docs)
    embedding_cache = _load_embedding_caches(_dedupe_paths([Path(args.cache_path), *(Path(path) for path in args.seed_cache_path or [])]))
    doc_matrix = _normalized_doc_matrix(global_docs, embedding_cache, args.embedding_model)

    question_rows = []
    for split in ("train", "valid", "test"):
        rows = _analyze_split(
            split=split,
            original_ids=split_ids[split],
            raw_rows=raw_rows,
            base_records=base_records[split],
            title_to_indices=title_to_indices,
            doc_matrix=doc_matrix,
            embedding_cache=embedding_cache,
            embedding_model=args.embedding_model,
            top_ks=top_ks,
            batch_size=args.batch_size,
        )
        question_rows.extend(rows)
        print(json.dumps({"split": split, "questions": len(rows), "top_ks": top_ks}, ensure_ascii=False), flush=True)

    summary_rows = _summary_rows(question_rows, top_ks)
    bucket_rows = _rank_bucket_rows(question_rows)
    validation = _validation_summary(
        args=args,
        top_ks=top_ks,
        base_records=base_records,
        question_rows=question_rows,
        global_docs=global_docs,
        title_to_indices=title_to_indices,
        elapsed_seconds=time.time() - started,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "support_rank_question_details.csv", question_rows)
    _write_csv(output_dir / "support_rank_upper_bound_summary.csv", summary_rows)
    _write_csv(output_dir / "support_rank_buckets.csv", bucket_rows)
    (output_dir / "validation_summary.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown_summary(output_dir / "support_rank_upper_bound_summary.md", summary_rows, bucket_rows, validation)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "question_rows": len(question_rows),
                "summary_rows": len(summary_rows),
                "no_api_calls": True,
            },
            ensure_ascii=False,
        )
    )


def _load_base_records(split_dir: Path, record_kind: str, max_questions_per_split: int) -> dict[str, list[dict[str, Any]]]:
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


def _analyze_split(
    split: str,
    original_ids: list[str],
    raw_rows: dict[str, dict[str, Any]],
    base_records: list[dict[str, Any]],
    title_to_indices: dict[str, np.ndarray],
    doc_matrix: np.ndarray,
    embedding_cache: dict[str, list[float]],
    embedding_model: str,
    top_ks: list[int],
    batch_size: int,
) -> list[dict[str, Any]]:
    base_by_id = {record["metadata"]["original_id"]: record for record in base_records}
    rows = []
    for start in range(0, len(original_ids), batch_size):
        batch_ids = original_ids[start : start + batch_size]
        query_matrix = np.asarray(
            [_embedding_cache_get(embedding_cache, embedding_model, raw_rows[original_id]["question"]) for original_id in batch_ids],
            dtype=np.float32,
        )
        query_matrix = query_matrix / np.maximum(np.linalg.norm(query_matrix, axis=1, keepdims=True), 1e-12)
        score_matrix = query_matrix.dot(doc_matrix.T)
        for row_idx, original_id in enumerate(batch_ids):
            raw = raw_rows[original_id]
            scores = score_matrix[row_idx]
            support_titles = sorted(_support_titles(raw))
            support_ranks = []
            missing_titles = []
            for title in support_titles:
                indices = title_to_indices.get(title)
                if indices is None or len(indices) == 0:
                    support_ranks.append(-1)
                    missing_titles.append(title)
                    continue
                best_score = float(np.max(scores[indices]))
                support_ranks.append(int(np.count_nonzero(scores > best_score) + 1))
            max_support_rank = max(support_ranks) if support_ranks and all(rank > 0 for rank in support_ranks) else -1
            min_support_rank = min(support_ranks) if support_ranks and all(rank > 0 for rank in support_ranks) else -1
            base_label = base_by_id[original_id]["sufficiency_label"]
            row = {
                "split": split,
                "original_id": original_id,
                "question_type": raw.get("type", "unknown"),
                "difficulty": raw.get("level", "unknown"),
                "question": raw["question"],
                "gold_answer": raw["answer"],
                "support_title_count": len(support_titles),
                "support_titles": " || ".join(support_titles),
                "support_title_min_ranks": " || ".join(str(rank) for rank in support_ranks),
                "min_support_rank": min_support_rank,
                "max_support_rank": max_support_rank,
                "base_top5_label": base_label,
                "missing_support_title_count": len(missing_titles),
                "missing_support_titles": " || ".join(missing_titles),
            }
            for top_k in top_ks:
                covered_count = sum(1 for rank in support_ranks if 0 < rank <= top_k)
                row[f"covered_all_at_{top_k}"] = int(max_support_rank > 0 and max_support_rank <= top_k)
                row[f"covered_any_at_{top_k}"] = int(covered_count > 0)
                row[f"support_title_coverage_at_{top_k}"] = covered_count / len(support_titles) if support_titles else 0.0
            rows.append(row)
    return rows


def _summary_rows(question_rows: list[dict[str, Any]], top_ks: list[int]) -> list[dict[str, Any]]:
    rows = []
    for split in ("train", "valid", "test", "all"):
        split_rows = question_rows if split == "all" else [row for row in question_rows if row["split"] == split]
        if not split_rows:
            continue
        top5_insufficient = [row for row in split_rows if int(row["covered_all_at_5"]) == 0]
        ranked_support_rows = [row for row in split_rows if int(row["max_support_rank"]) > 0]
        missing_support_rows = [row for row in split_rows if int(row["max_support_rank"]) <= 0]
        for top_k in top_ks:
            covered = [row for row in split_rows if int(row[f"covered_all_at_{top_k}"]) == 1]
            newly_covered = [
                row
                for row in split_rows
                if int(row["covered_all_at_5"]) == 0 and int(row[f"covered_all_at_{top_k}"]) == 1
            ]
            rows.append(
                {
                    "split": split,
                    "top_k": top_k,
                    "n": len(split_rows),
                    "all_support_covered_count": len(covered),
                    "all_support_covered_rate": len(covered) / len(split_rows),
                    "all_support_covered_rate_among_ranked_support": (
                        len(covered) / len(ranked_support_rows) if ranked_support_rows else 0.0
                    ),
                    "newly_all_support_covered_vs_top5_count": len(newly_covered),
                    "newly_all_support_covered_vs_top5_rate": (
                        len(newly_covered) / len(top5_insufficient) if top5_insufficient else 0.0
                    ),
                    "mean_support_title_coverage": _mean(float(row[f"support_title_coverage_at_{top_k}"]) for row in split_rows),
                    "any_support_covered_rate": _mean(float(row[f"covered_any_at_{top_k}"]) for row in split_rows),
                    "all_support_ranked_count": len(ranked_support_rows),
                    "all_support_ranked_rate": len(ranked_support_rows) / len(split_rows),
                    "support_missing_from_pool_question_count": len(missing_support_rows),
                    "support_missing_from_pool_question_rate": len(missing_support_rows) / len(split_rows),
                    "missing_support_title_count": sum(int(row["missing_support_title_count"]) for row in split_rows),
                }
            )
    return rows


def _rank_bucket_rows(question_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets = [
        ("<=5", lambda rank: 0 < rank <= 5),
        ("6-15", lambda rank: 5 < rank <= 15),
        ("16-30", lambda rank: 15 < rank <= 30),
        ("31-50", lambda rank: 30 < rank <= 50),
        ("51-100", lambda rank: 50 < rank <= 100),
        (">100", lambda rank: rank > 100),
        ("missing", lambda rank: rank <= 0),
    ]
    rows = []
    for split in ("train", "valid", "test", "all"):
        split_rows = question_rows if split == "all" else [row for row in question_rows if row["split"] == split]
        if not split_rows:
            continue
        ranks = [int(row["max_support_rank"]) for row in split_rows]
        for bucket, predicate in buckets:
            count = sum(1 for rank in ranks if predicate(rank))
            rows.append(
                {
                    "split": split,
                    "max_support_rank_bucket": bucket,
                    "count": count,
                    "rate": count / len(ranks),
                    "n": len(ranks),
                }
            )
    return rows


def _validation_summary(
    args: argparse.Namespace,
    top_ks: list[int],
    base_records: dict[str, list[dict[str, Any]]],
    question_rows: list[dict[str, Any]],
    global_docs: list[dict[str, Any]],
    title_to_indices: dict[str, np.ndarray],
    elapsed_seconds: float,
) -> dict[str, Any]:
    rows_by_id = {row["original_id"]: row for row in question_rows}
    top5_checks = {}
    for split, records in base_records.items():
        label_matches = 0
        for record in records:
            row = rows_by_id[record["metadata"]["original_id"]]
            inferred_label = "sufficient" if int(row["covered_all_at_5"]) == 1 else "insufficient"
            label_matches += int(inferred_label == record["sufficiency_label"])
        top5_checks[split] = {
            "n": len(records),
            "label_match_count": label_matches,
            "label_match_rate": label_matches / len(records) if records else 0.0,
        }
    return {
        "top_ks": top_ks,
        "split_counts": {split: len(records) for split, records in base_records.items()},
        "global_doc_count": len(global_docs),
        "unique_title_count": len(title_to_indices),
        "cache_path": args.cache_path,
        "seed_cache_paths": args.seed_cache_path,
        "embedding_model": args.embedding_model,
        "uses_embedding_api": False,
        "uses_llm_api": False,
        "top5_label_reconstruction": top5_checks,
        "selection_protocol": "no training; upper-bound analysis only; all statistics computed from fixed train/valid/test splits",
        "elapsed_seconds": round(float(elapsed_seconds), 2),
    }


def _write_markdown_summary(
    path: Path,
    summary_rows: list[dict[str, Any]],
    bucket_rows: list[dict[str, Any]],
    validation: dict[str, Any],
) -> None:
    test_rows = [row for row in summary_rows if row["split"] == "test"]
    test_lines = "\n".join(
        f"- top-{row['top_k']}: all_support_covered_rate={float(row['all_support_covered_rate']):.4f}, "
        f"among_ranked={float(row['all_support_covered_rate_among_ranked_support']):.4f}, "
        f"newly_vs_top5={float(row['newly_all_support_covered_vs_top5_rate']):.4f}, "
        f"mean_support_title_coverage={float(row['mean_support_title_coverage']):.4f}"
        for row in test_rows
    )
    bucket_lines = "\n".join(
        f"- {row['max_support_rank_bucket']}: {float(row['rate']):.4f}"
        for row in bucket_rows
        if row["split"] == "test"
    )
    text = f"""# CSR-RAG Support-Rank Upper-Bound Summary

## Purpose

This no-API run estimates how deep the global embedding ranking must go before all HotpotQA supporting titles are present. It is an upper-bound diagnostic for retrieval refinement.

## Test Coverage by Top-k

{test_lines}

## Test Max Support-Rank Buckets

{bucket_lines}

## Validation

- Global docs: {validation["global_doc_count"]}
- Unique titles: {validation["unique_title_count"]}
- Embedding API: no
- LLM API: no
- Top-5 label reconstruction: {json.dumps(validation["top5_label_reconstruction"], ensure_ascii=False)}

## Interpretation

The overall top-k coverage is capped by corpus completeness. If many support titles are missing from the current pool, the next research step is to build or obtain a support-complete retrieval corpus before treating retrieval failures as model failures. Among questions whose full support chain exists in the pool, high top-50/top-100 coverage means reranking and bridge-aware candidate selection are promising.
"""
    path.write_text(text, encoding="utf-8")


def _normalized_doc_matrix(
    global_docs: list[dict[str, Any]],
    embedding_cache: dict[str, list[float]],
    embedding_model: str,
) -> np.ndarray:
    doc_matrix = np.asarray(
        [_embedding_cache_get(embedding_cache, embedding_model, _doc_embedding_text(doc)) for doc in global_docs],
        dtype=np.float32,
    )
    return doc_matrix / np.maximum(np.linalg.norm(doc_matrix, axis=1, keepdims=True), 1e-12)


def _title_to_indices(global_docs: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, doc in enumerate(global_docs):
        grouped[_normalize_title(doc["title"])].append(idx)
    return {title: np.asarray(indices, dtype=np.int64) for title, indices in grouped.items()}


def _parse_top_ks(value: str) -> list[int]:
    top_ks = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if 5 not in top_ks:
        top_ks.insert(0, 5)
    if any(top_k <= 0 for top_k in top_ks):
        raise ValueError("top-ks must be positive.")
    return top_ks


def _validate_split_records(split_records: dict[str, list[dict[str, Any]]]) -> None:
    split_ids = {split: {record["metadata"]["original_id"] for record in records} for split, records in split_records.items()}
    _require(split_ids["train"].isdisjoint(split_ids["valid"]), "train and valid original_id overlap.")
    _require(split_ids["train"].isdisjoint(split_ids["test"]), "train and test original_id overlap.")
    _require(split_ids["valid"].isdisjoint(split_ids["test"]), "valid and test original_id overlap.")
    for split, records in split_records.items():
        _require(records, f"{split} split is empty.")
        labels = {record["sufficiency_label"] for record in records}
        _require(labels == {"sufficient", "insufficient"}, f"{split} must contain both labels.")


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


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    result = []
    seen = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def _mean(values) -> float:
    value_list = list(values)
    return float(sum(value_list) / len(value_list)) if value_list else 0.0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
