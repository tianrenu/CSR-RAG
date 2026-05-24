from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from csrrag.rag.api_clients import OpenAICompatibleEmbeddingClient
from csrrag.utils.env import load_dotenv
from csrrag.utils.io import read_jsonl, write_jsonl
from csrrag.utils.text import lexical_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a global-pool embedding retrieval dataset for CSR-RAG.")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--raw-hotpot", default="data/raw/hotpotqa/hotpot_dev_fullwiki_v1.json")
    parser.add_argument("--controlled-split-dir", default="data/processed/hotpotqa_dev_splits_1800")
    parser.add_argument("--output-retrieval", default="data/retrieval/hotpotqa_global_embedding_retrieval_1800.jsonl")
    parser.add_argument("--output-split-dir", default="data/processed/hotpotqa_global_embedding_splits_1800")
    parser.add_argument("--cache-path", default="data/cache/hotpotqa_text_embedding_v4_1800.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=10)
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
    split_ids = _load_split_original_ids(args.controlled_split_dir)
    selected_ids = [original_id for split in ("train", "valid", "test") for original_id in split_ids[split]]
    global_docs = _build_global_doc_pool(raw_rows, selected_ids)
    embedding_cache = _load_embedding_cache(Path(args.cache_path))

    _prefetch_missing_embeddings(
        raw_rows=raw_rows,
        selected_ids=selected_ids,
        global_docs=global_docs,
        embedding_client=client,
        embedding_cache=embedding_cache,
        cache_path=Path(args.cache_path),
        model=client.model,
        batch_size=args.batch_size,
    )

    doc_matrix = np.asarray(
        [_embedding_cache_get(embedding_cache, client.model, f"{doc['title']}\n{doc['text']}") for doc in global_docs],
        dtype=float,
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
            query_embedding = np.asarray(_embedding_cache_get(embedding_cache, client.model, raw["question"]), dtype=float)
            scores = _cosine_scores(query_embedding, doc_matrix, doc_norms)
            ranked_indices = np.argsort(-scores)[: args.top_k]
            retrieved_docs = []
            for rank, doc_index in enumerate(ranked_indices, start=1):
                doc = global_docs[int(doc_index)]
                retrieved_docs.append(
                    {
                        "doc_id": doc["doc_id"],
                        "rank": rank,
                        "score": round(float(lexical_score(raw["question"], doc["title"], doc["text"])), 6),
                        "embedding_score": round(float(scores[doc_index]), 6),
                        "title": doc["title"],
                        "text": doc["text"],
                        "source": "hotpotqa_global_context_embedding",
                    }
                )
            record = _make_record(raw, split, retrieved_docs, args.top_k)
            all_records.append(record)
            split_records[split].append(record)
            processed += 1
            if processed == 1 or processed % args.progress_every == 0 or processed == total:
                elapsed = time.time() - started
                print(
                    json.dumps(
                        {
                            "processed": processed,
                            "total": total,
                            "split": split,
                            "elapsed_seconds": round(elapsed, 1),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    write_jsonl(args.output_retrieval, all_records)
    output_split_dir = Path(args.output_split_dir)
    for split, records in split_records.items():
        write_jsonl(output_split_dir / f"{split}.jsonl", records)
    summary = _summary(split_records, global_docs, args.top_k)
    (output_split_dir / "global_embedding_retrieval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output_retrieval": args.output_retrieval, "output_split_dir": args.output_split_dir, **summary}, ensure_ascii=False))


def _make_record(raw: dict[str, Any], split: str, retrieved_docs: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    support_titles = _support_titles(raw)
    retrieved_titles = {_normalize_title(doc["title"]) for doc in retrieved_docs}
    support_present = support_titles.issubset(retrieved_titles)
    return {
        "id": f"{raw['_id']}__global_embedding_top{top_k}",
        "query": raw["question"],
        "gold_answer": raw["answer"],
        "sufficiency_label": "sufficient" if support_present else "insufficient",
        "retrieved_docs": retrieved_docs,
        "metadata": {
            "dataset": "hotpotqa",
            "split": split,
            "original_id": raw["_id"],
            "question_type": raw.get("type", "unknown"),
            "difficulty": raw.get("level", "unknown"),
            "retriever": "text-embedding-v4-global-context-pool",
            "top_k": top_k,
            "support_present_in_topk": support_present,
            "support_titles": sorted(support_titles),
        },
    }


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
) -> None:
    texts = [raw_rows[original_id]["question"] for original_id in selected_ids]
    texts.extend(f"{doc['title']}\n{doc['text']}" for doc in global_docs)
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
    for start in range(0, len(missing_items), batch_size):
        batch = missing_items[start : start + batch_size]
        keys = [key for key, _text in batch]
        batch_texts = [text for _key, text in batch]
        embeddings = embedding_client.embed(batch_texts)
        if len(embeddings) != len(batch_texts):
            raise RuntimeError("Embedding API returned an unexpected number of embeddings.")
        with cache_path.open("a", encoding="utf-8") as f:
            for key, embedding in zip(keys, embeddings):
                embedding_cache[key] = embedding
                f.write(json.dumps({"key": key, "model": model, "embedding": embedding}, ensure_ascii=False) + "\n")
        done = min(start + batch_size, len(missing_items))
        if done == len(missing_items) or done % 500 == 0:
            print(json.dumps({"prefetch_embeddings": done, "prefetch_total": len(missing_items)}, ensure_ascii=False), flush=True)


def _embedding_cache_get(cache: dict[str, list[float]], model: str, text: str) -> list[float]:
    key = _cache_key(model, text)
    if key not in cache:
        raise KeyError(f"Missing embedding cache item: {key}")
    return cache[key]


def _load_embedding_cache(path: Path) -> dict[str, list[float]]:
    cache: dict[str, list[float]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cache[row["key"]] = row["embedding"]
    return cache


def _load_raw_rows(path: str) -> dict[str, dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return {row["_id"]: row for row in json.load(f)}


def _load_split_original_ids(split_dir: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for split in ("train", "valid", "test"):
        ids = []
        seen = set()
        for record in read_jsonl(Path(split_dir) / f"{split}.jsonl"):
            original_id = record["metadata"]["original_id"]
            if original_id not in seen:
                seen.add(original_id)
                ids.append(original_id)
        result[split] = ids
    return result


def _cosine_scores(query_embedding: np.ndarray, doc_matrix: np.ndarray, doc_norms: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_embedding)
    denom = np.maximum(query_norm * doc_norms, 1e-12)
    return doc_matrix.dot(query_embedding) / denom


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


def _summary(split_records: dict[str, list[dict[str, Any]]], global_docs: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    split_rows = []
    for split, records in split_records.items():
        labels = Counter(record["sufficiency_label"] for record in records)
        qtypes = Counter(record["metadata"].get("question_type", "unknown") for record in records)
        split_rows.append(
            {
                "split": split,
                "records": len(records),
                "sufficient": labels["sufficient"],
                "insufficient": labels["insufficient"],
                "sufficient_rate": labels["sufficient"] / len(records) if records else 0.0,
                "question_types": dict(qtypes),
            }
        )
    all_records = [record for records in split_records.values() for record in records]
    return {
        "top_k": top_k,
        "global_doc_count": len(global_docs),
        "total_records": len(all_records),
        "splits": split_rows,
        "all_sufficient": sum(record["sufficiency_label"] == "sufficient" for record in all_records),
        "all_insufficient": sum(record["sufficiency_label"] == "insufficient" for record in all_records),
        "non_finite_embedding_scores": _non_finite_embedding_scores(all_records),
    }


def _non_finite_embedding_scores(records: list[dict[str, Any]]) -> int:
    count = 0
    for record in records:
        for doc in record["retrieved_docs"]:
            score = float(doc.get("embedding_score", 0.0))
            if not math.isfinite(score):
                count += 1
    return count


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return value


if __name__ == "__main__":
    main()
