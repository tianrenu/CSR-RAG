from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from csrrag.rag.api_clients import OpenAICompatibleEmbeddingClient
from csrrag.utils.env import load_dotenv
from csrrag.utils.io import write_jsonl
from csrrag.utils.text import lexical_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full-dev global hard-negative retrieval records for CSR-RAG.")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--raw-hotpot", default="data/raw/hotpotqa/hotpot_dev_fullwiki_v1.json")
    parser.add_argument(
        "--output-retrieval",
        default="data/retrieval/hotpotqa_global_hardneg_retrieval_full_dev.jsonl",
    )
    parser.add_argument(
        "--output-split-dir",
        default="data/processed/hotpotqa_global_hardneg_splits_full_dev",
    )
    parser.add_argument("--cache-path", default="data/cache/hotpotqa_text_embedding_v4_full_dev.jsonl")
    parser.add_argument(
        "--seed-cache-path",
        action="append",
        default=["data/cache/hotpotqa_text_embedding_v4_1800.jsonl"],
        help="Existing embedding cache to reuse. May be passed multiple times.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--embedding-workers", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-questions", type=int, default=0, help="Debug only. 0 means full dev set.")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    if args.batch_size > 10:
        raise ValueError("text-embedding-v4 accepts at most 10 inputs per request; use --batch-size <= 10.")

    load_dotenv(args.env_file)
    client = OpenAICompatibleEmbeddingClient(
        base_url=_required_env("EMBEDDING_BASE_URL"),
        api_key=_required_env("EMBEDDING_API_KEY"),
        model=_required_env("EMBEDDING_MODEL"),
    )

    raw_rows = _load_raw_rows(args.raw_hotpot)
    selected_ids = list(raw_rows)
    if args.max_questions > 0:
        selected_ids = selected_ids[: args.max_questions]
    split_ids = _split_original_ids(selected_ids, args.split_seed)
    global_docs = _build_global_doc_pool(raw_rows, selected_ids)
    cache_paths = _dedupe_paths([Path(args.cache_path), *(Path(path) for path in args.seed_cache_path or [])])
    embedding_cache = _load_embedding_caches(cache_paths)

    _prefetch_missing_embeddings(
        raw_rows=raw_rows,
        selected_ids=selected_ids,
        global_docs=global_docs,
        embedding_client=client,
        embedding_cache=embedding_cache,
        cache_path=Path(args.cache_path),
        model=client.model,
        batch_size=args.batch_size,
        workers=args.embedding_workers,
    )

    doc_matrix = np.asarray(
        [_embedding_cache_get(embedding_cache, client.model, _doc_embedding_text(doc)) for doc in global_docs],
        dtype=np.float32,
    )
    doc_norms = np.linalg.norm(doc_matrix, axis=1)

    all_records: list[dict[str, Any]] = []
    split_records: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    total = sum(len(ids) for ids in split_ids.values())
    processed = 0
    started = time.time()
    for split in ("train", "valid", "test"):
        for original_id in split_ids[split]:
            raw = raw_rows[original_id]
            query_embedding = np.asarray(_embedding_cache_get(embedding_cache, client.model, raw["question"]), dtype=np.float32)
            scores = _cosine_scores(query_embedding, doc_matrix, doc_norms)
            ranked_indices = np.argsort(-scores)

            natural_docs = _make_retrieved_docs(raw["question"], global_docs, scores, ranked_indices[: args.top_k])
            hardneg_docs, missing_support_titles = _make_hardneg_docs(
                raw["question"],
                global_docs,
                scores,
                ranked_indices,
                _support_titles(raw),
                args.top_k,
            )

            natural_record = _make_record(
                raw=raw,
                split=split,
                retrieved_docs=natural_docs,
                top_k=args.top_k,
                record_kind="natural_global_top5",
                forced_missing_support_titles=[],
            )
            hardneg_record = _make_record(
                raw=raw,
                split=split,
                retrieved_docs=hardneg_docs,
                top_k=args.top_k,
                record_kind="hardneg_missing_support_top5",
                forced_missing_support_titles=missing_support_titles,
            )

            _validate_hardneg_record(hardneg_record)
            all_records.extend([natural_record, hardneg_record])
            split_records[split].extend([natural_record, hardneg_record])
            processed += 1
            if processed == 1 or processed % args.progress_every == 0 or processed == total:
                elapsed = time.time() - started
                print(
                    json.dumps(
                        {
                            "processed_questions": processed,
                            "total_questions": total,
                            "records": len(all_records),
                            "split": split,
                            "elapsed_seconds": round(elapsed, 1),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    _validate_split_records(split_records)
    write_jsonl(args.output_retrieval, all_records)
    output_split_dir = Path(args.output_split_dir)
    for split, records in split_records.items():
        write_jsonl(output_split_dir / f"{split}.jsonl", records)
    summary = _summary(split_records, global_docs, args.top_k, selected_ids)
    (output_split_dir / "global_hardneg_retrieval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output_retrieval": args.output_retrieval, "output_split_dir": args.output_split_dir, **summary}, ensure_ascii=False))


def _make_record(
    raw: dict[str, Any],
    split: str,
    retrieved_docs: list[dict[str, Any]],
    top_k: int,
    record_kind: str,
    forced_missing_support_titles: list[str],
) -> dict[str, Any]:
    support_titles = _support_titles(raw)
    retrieved_titles = {_normalize_title(doc["title"]) for doc in retrieved_docs}
    support_present = support_titles.issubset(retrieved_titles)
    label = "sufficient" if support_present else "insufficient"
    if record_kind == "hardneg_missing_support_top5":
        label = "insufficient"
    missing_support_titles = sorted(support_titles - retrieved_titles)
    return {
        "id": f"{raw['_id']}__{record_kind}",
        "query": raw["question"],
        "gold_answer": raw["answer"],
        "sufficiency_label": label,
        "retrieved_docs": retrieved_docs,
        "metadata": {
            "dataset": "hotpotqa",
            "split": split,
            "original_id": raw["_id"],
            "question_type": raw.get("type", "unknown"),
            "difficulty": raw.get("level", "unknown"),
            "retriever": "text-embedding-v4-global-hardneg-context-pool",
            "record_kind": record_kind,
            "top_k": top_k,
            "support_present_in_topk": support_present,
            "support_titles": sorted(support_titles),
            "missing_support_titles": missing_support_titles,
            "forced_missing_support_titles": forced_missing_support_titles,
        },
    }


def _make_hardneg_docs(
    query: str,
    global_docs: list[dict[str, Any]],
    scores: np.ndarray,
    ranked_indices: np.ndarray,
    support_titles: set[str],
    top_k: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    missing_title = _choose_forced_missing_support(global_docs, ranked_indices, support_titles)
    selected = [idx for idx in ranked_indices if global_docs[int(idx)]["normalized_title"] != missing_title][:top_k]
    if len(selected) < top_k:
        raise RuntimeError("Could not build a top-k hard-negative record.")
    retrieved_docs = _make_retrieved_docs(query, global_docs, scores, selected)
    retrieved_titles = {_normalize_title(doc["title"]) for doc in retrieved_docs}
    missing_titles = sorted(support_titles - retrieved_titles)
    if not missing_titles:
        raise AssertionError("Hard-negative retrieval unexpectedly contains all supporting titles.")
    return retrieved_docs, missing_titles


def _choose_forced_missing_support(global_docs: list[dict[str, Any]], ranked_indices: np.ndarray, support_titles: set[str]) -> str:
    if not support_titles:
        raise ValueError("Record has no supporting titles.")
    best_rank = {title: float("inf") for title in support_titles}
    for rank, doc_index in enumerate(ranked_indices):
        title = global_docs[int(doc_index)]["normalized_title"]
        if title in best_rank and best_rank[title] == float("inf"):
            best_rank[title] = rank
        if all(rank_value < float("inf") for rank_value in best_rank.values()):
            break
    return min(support_titles, key=lambda title: (best_rank[title], title))


def _make_retrieved_docs(
    query: str,
    global_docs: list[dict[str, Any]],
    scores: np.ndarray,
    indices: list[int] | np.ndarray,
) -> list[dict[str, Any]]:
    retrieved_docs = []
    for rank, doc_index in enumerate(indices, start=1):
        doc = global_docs[int(doc_index)]
        embedding_score = float(scores[int(doc_index)])
        if not np.isfinite(embedding_score):
            raise ValueError("Non-finite embedding score.")
        retrieved_docs.append(
            {
                "doc_id": doc["doc_id"],
                "rank": rank,
                "score": round(float(lexical_score(query, doc["title"], doc["text"])), 6),
                "embedding_score": round(embedding_score, 6),
                "title": doc["title"],
                "text": doc["text"],
                "source": "hotpotqa_global_hardneg_context_embedding",
            }
        )
    return retrieved_docs


def _build_global_doc_pool(raw_rows: dict[str, dict[str, Any]], selected_ids: list[str]) -> list[dict[str, Any]]:
    docs_by_key: dict[str, dict[str, Any]] = {}
    source_ids: dict[str, set[str]] = defaultdict(set)
    for original_id in selected_ids:
        raw = raw_rows[original_id]
        for title, sentences in raw["context"]:
            text = " ".join(sentences)
            key = _doc_key(title, text)
            if key not in docs_by_key:
                docs_by_key[key] = {
                    "doc_id": f"global::{key[:16]}",
                    "title": title,
                    "normalized_title": _normalize_title(title),
                    "text": text,
                }
            source_ids[key].add(original_id)
    docs = []
    for key in sorted(docs_by_key):
        doc = dict(docs_by_key[key])
        doc["source_original_count"] = len(source_ids[key])
        docs.append(doc)
    return docs


def _prefetch_missing_embeddings(
    raw_rows: dict[str, dict[str, Any]],
    selected_ids: list[str],
    global_docs: list[dict[str, Any]],
    embedding_client: OpenAICompatibleEmbeddingClient,
    embedding_cache: dict[str, list[float]],
    cache_path: Path,
    model: str,
    batch_size: int,
    workers: int,
) -> None:
    texts = [raw_rows[original_id]["question"] for original_id in selected_ids]
    texts.extend(_doc_embedding_text(doc) for doc in global_docs)
    unique_missing: dict[str, str] = {}
    for text in texts:
        key = _cache_key(model, text)
        if key not in embedding_cache:
            unique_missing[key] = text
    if not unique_missing:
        print(json.dumps({"prefetch_missing_embeddings": 0, "cache_items": len(embedding_cache)}, ensure_ascii=False), flush=True)
        return

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    missing_items = list(unique_missing.items())
    batches = [missing_items[start : start + batch_size] for start in range(0, len(missing_items), batch_size)]
    workers = max(1, int(workers))
    done = 0
    next_progress = 500
    with cache_path.open("a", encoding="utf-8") as f:
        if workers == 1:
            for batch in batches:
                keys, embeddings = _embed_missing_batch(embedding_client, batch)
                _append_embeddings(f, embedding_cache, keys, embeddings, model)
                done += len(keys)
                next_progress = _maybe_print_prefetch_progress(done, len(missing_items), next_progress)
            return

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_embed_missing_batch, embedding_client, batch) for batch in batches]
            for future in as_completed(futures):
                keys, embeddings = future.result()
                _append_embeddings(f, embedding_cache, keys, embeddings, model)
                done += len(keys)
                next_progress = _maybe_print_prefetch_progress(done, len(missing_items), next_progress)


def _embed_missing_batch(
    embedding_client: OpenAICompatibleEmbeddingClient,
    batch: list[tuple[str, str]],
) -> tuple[list[str], list[list[float]]]:
    keys = [key for key, _text in batch]
    batch_texts = [text for _key, text in batch]
    embeddings = embedding_client.embed(batch_texts)
    if len(embeddings) != len(batch_texts):
        raise RuntimeError("Embedding API returned an unexpected number of embeddings.")
    return keys, embeddings


def _append_embeddings(
    file_obj,
    embedding_cache: dict[str, list[float]],
    keys: list[str],
    embeddings: list[list[float]],
    model: str,
) -> None:
    for key, embedding in zip(keys, embeddings):
        embedding_cache[key] = embedding
        file_obj.write(json.dumps({"key": key, "model": model, "embedding": embedding}, ensure_ascii=False) + "\n")
    file_obj.flush()


def _maybe_print_prefetch_progress(done: int, total: int, next_progress: int) -> int:
    if done >= total or done >= next_progress:
        print(json.dumps({"prefetch_embeddings": done, "prefetch_total": total}, ensure_ascii=False), flush=True)
        while next_progress <= done:
            next_progress += 500
    return next_progress


def _split_original_ids(original_ids: list[str], seed: int) -> dict[str, list[str]]:
    shuffled = list(original_ids)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_count = int(total * 0.70)
    valid_count = int(total * 0.15)
    return {
        "train": sorted(shuffled[:train_count]),
        "valid": sorted(shuffled[train_count : train_count + valid_count]),
        "test": sorted(shuffled[train_count + valid_count :]),
    }


def _load_raw_rows(path: str) -> dict[str, dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return {row["_id"]: row for row in json.load(f)}


def _load_embedding_caches(paths: list[Path]) -> dict[str, list[float]]:
    cache: dict[str, list[float]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                cache[row["key"]] = row["embedding"]
    return cache


def _embedding_cache_get(cache: dict[str, list[float]], model: str, text: str) -> list[float]:
    key = _cache_key(model, text)
    if key not in cache:
        raise KeyError(f"Missing embedding cache item: {key}")
    return cache[key]


def _cosine_scores(query_embedding: np.ndarray, doc_matrix: np.ndarray, doc_norms: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_embedding)
    denom = np.maximum(query_norm * doc_norms, 1e-12)
    return doc_matrix.dot(query_embedding) / denom


def _doc_embedding_text(doc: dict[str, Any]) -> str:
    return f"{doc['title']}\n{doc['text']}"


def _support_titles(raw: dict[str, Any]) -> set[str]:
    return {_normalize_title(title) for title, _sent_idx in raw["supporting_facts"]}


def _normalize_title(title: str) -> str:
    return html.unescape(str(title)).strip().lower()


def _doc_key(title: str, text: str) -> str:
    payload = f"{_normalize_title(title)}\0{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _cache_key(model: str, text: str) -> str:
    payload = f"{model}\0{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    result = []
    seen = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def _summary(
    split_records: dict[str, list[dict[str, Any]]],
    global_docs: list[dict[str, Any]],
    top_k: int,
    selected_ids: list[str],
) -> dict[str, Any]:
    split_rows = []
    for split, records in split_records.items():
        labels = Counter(record["sufficiency_label"] for record in records)
        kinds = Counter(record["metadata"].get("record_kind", "unknown") for record in records)
        qtypes = Counter(record["metadata"].get("question_type", "unknown") for record in records)
        natural = [record for record in records if record["metadata"].get("record_kind") == "natural_global_top5"]
        split_rows.append(
            {
                "split": split,
                "records": len(records),
                "questions": len({record["metadata"]["original_id"] for record in records}),
                "sufficient": labels["sufficient"],
                "insufficient": labels["insufficient"],
                "sufficient_rate": labels["sufficient"] / len(records) if records else 0.0,
                "natural_sufficient_rate": (
                    sum(record["sufficiency_label"] == "sufficient" for record in natural) / len(natural)
                    if natural
                    else 0.0
                ),
                "record_kinds": dict(kinds),
                "question_types": dict(qtypes),
            }
        )
    all_records = [record for records in split_records.values() for record in records]
    labels = Counter(record["sufficiency_label"] for record in all_records)
    return {
        "top_k": top_k,
        "selected_questions": len(selected_ids),
        "global_doc_count": len(global_docs),
        "total_records": len(all_records),
        "splits": split_rows,
        "all_sufficient": labels["sufficient"],
        "all_insufficient": labels["insufficient"],
        "non_finite_embedding_scores": 0,
    }


def _validate_hardneg_record(record: dict[str, Any]) -> None:
    if record["metadata"].get("record_kind") != "hardneg_missing_support_top5":
        return
    if record["sufficiency_label"] != "insufficient":
        raise AssertionError(f"{record['id']} is not labeled insufficient.")
    if not record["metadata"].get("missing_support_titles"):
        raise AssertionError(f"{record['id']} does not miss a supporting title.")


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
        for record in records:
            if len(record.get("retrieved_docs", [])) != 5:
                raise AssertionError(f"{record['id']} does not have top_k=5.")
            if any("is_support" in doc for doc in record["retrieved_docs"]):
                raise AssertionError(f"{record['id']} exposes is_support.")
            if record["metadata"].get("record_kind") == "hardneg_missing_support_top5":
                _validate_hardneg_record(record)


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return value


if __name__ == "__main__":
    main()
