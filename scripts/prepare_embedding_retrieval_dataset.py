from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from csrrag.rag.api_clients import OpenAICompatibleEmbeddingClient
from csrrag.utils.env import load_dotenv
from csrrag.utils.io import read_jsonl, write_jsonl
from csrrag.utils.text import lexical_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a HotpotQA embedding-retrieval dataset for real RAG experiments.")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--raw-hotpot", default="data/raw/hotpotqa/hotpot_dev_fullwiki_v1.json")
    parser.add_argument("--controlled-split-dir", default="data/processed/hotpotqa_dev_splits_1800")
    parser.add_argument("--output-retrieval", default="data/retrieval/hotpotqa_embedding_retrieval_1800.jsonl")
    parser.add_argument("--output-split-dir", default="data/processed/hotpotqa_embedding_splits_1800")
    parser.add_argument("--cache-path", default="data/cache/hotpotqa_text_embedding_v4_1800.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None, help="Optional original-question limit for debugging.")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
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
    if args.limit is not None:
        split_ids = _limit_split_ids(split_ids, args.limit)

    cache_path = Path(args.cache_path)
    embedding_cache = _load_embedding_cache(cache_path)
    _prefetch_embeddings(
        raw_rows=raw_rows,
        split_ids=split_ids,
        embedding_client=client,
        embedding_cache=embedding_cache,
        cache_path=cache_path,
        model=client.model,
        batch_size=args.batch_size,
        progress_every=max(1, args.progress_every * 20),
    )
    all_records: list[dict[str, Any]] = []
    split_records: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}

    total = sum(len(ids) for ids in split_ids.values())
    processed = 0
    started = time.time()
    for split, original_ids in split_ids.items():
        for original_id in original_ids:
            raw = raw_rows[original_id]
            record = _build_record(
                raw=raw,
                split=split,
                embedding_client=client,
                embedding_cache=embedding_cache,
                cache_path=cache_path,
                model=client.model,
                top_k=args.top_k,
                batch_size=args.batch_size,
            )
            all_records.append(record)
            split_records[split].append(record)
            processed += 1
            if args.sleep_seconds:
                time.sleep(args.sleep_seconds)
            if processed == 1 or processed % args.progress_every == 0 or processed == total:
                elapsed = time.time() - started
                print(
                    json.dumps(
                        {
                            "processed": processed,
                            "total": total,
                            "split": split,
                            "cache_items": len(embedding_cache),
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

    summary = _summary(split_records, top_k=args.top_k)
    summary_path = Path(args.output_split_dir) / "embedding_retrieval_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_retrieval": args.output_retrieval, "output_split_dir": args.output_split_dir, **summary}, ensure_ascii=False))


def _build_record(
    raw: dict[str, Any],
    split: str,
    embedding_client: OpenAICompatibleEmbeddingClient,
    embedding_cache: dict[str, list[float]],
    cache_path: Path,
    model: str,
    top_k: int,
    batch_size: int,
) -> dict[str, Any]:
    docs = _context_docs(raw)
    texts = [raw["question"]] + [f"{doc['title']}\n{doc['text']}" for doc in docs]
    embeddings = _embed_texts_cached(
        texts=texts,
        embedding_client=embedding_client,
        embedding_cache=embedding_cache,
        cache_path=cache_path,
        model=model,
        batch_size=batch_size,
    )
    query_embedding = np.asarray(embeddings[0], dtype=float)
    doc_embeddings = np.asarray(embeddings[1:], dtype=float)
    embedding_scores = _cosine_scores(query_embedding, doc_embeddings)
    ranked_indices = np.argsort(-embedding_scores)[:top_k]

    retrieved_docs = []
    for rank, doc_index in enumerate(ranked_indices, start=1):
        doc = docs[int(doc_index)]
        retrieved_docs.append(
            {
                "doc_id": f"{raw['_id']}::{doc['title']}",
                "rank": rank,
                "score": round(float(lexical_score(raw["question"], doc["title"], doc["text"])), 6),
                "embedding_score": round(float(embedding_scores[doc_index]), 6),
                "title": doc["title"],
                "text": doc["text"],
                "source": "hotpotqa_context_embedding",
            }
        )

    support_titles = _support_titles(raw)
    retrieved_titles = {_normalize_title(doc["title"]) for doc in retrieved_docs}
    support_present = support_titles.issubset(retrieved_titles)
    return {
        "id": f"{raw['_id']}__embedding_top{top_k}",
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
            "retriever": "text-embedding-v4",
            "top_k": top_k,
            "support_present_in_topk": support_present,
            "support_doc_ids": [f"{raw['_id']}::{title}" for title in sorted(support_titles)],
        },
    }


def _embed_texts_cached(
    texts: list[str],
    embedding_client: OpenAICompatibleEmbeddingClient,
    embedding_cache: dict[str, list[float]],
    cache_path: Path,
    model: str,
    batch_size: int,
) -> list[list[float]]:
    keys = [_cache_key(model, text) for text in texts]
    missing_texts: list[str] = []
    missing_keys: list[str] = []
    for key, text in zip(keys, texts):
        if key not in embedding_cache:
            missing_keys.append(key)
            missing_texts.append(text)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(missing_texts), batch_size):
        batch_texts = missing_texts[start : start + batch_size]
        batch_keys = missing_keys[start : start + batch_size]
        batch_embeddings = embedding_client.embed(batch_texts)
        if len(batch_embeddings) != len(batch_texts):
            raise RuntimeError("Embedding API returned an unexpected number of embeddings.")
        with cache_path.open("a", encoding="utf-8") as f:
            for key, embedding in zip(batch_keys, batch_embeddings):
                embedding_cache[key] = embedding
                f.write(json.dumps({"key": key, "model": model, "embedding": embedding}, ensure_ascii=False) + "\n")

    return [embedding_cache[key] for key in keys]


def _prefetch_embeddings(
    raw_rows: dict[str, dict[str, Any]],
    split_ids: dict[str, list[str]],
    embedding_client: OpenAICompatibleEmbeddingClient,
    embedding_cache: dict[str, list[float]],
    cache_path: Path,
    model: str,
    batch_size: int,
    progress_every: int,
) -> None:
    unique_texts: dict[str, str] = {}
    for original_ids in split_ids.values():
        for original_id in original_ids:
            raw = raw_rows[original_id]
            docs = _context_docs(raw)
            texts = [raw["question"]] + [f"{doc['title']}\n{doc['text']}" for doc in docs]
            for text in texts:
                key = _cache_key(model, text)
                if key not in embedding_cache:
                    unique_texts[key] = text

    missing_items = list(unique_texts.items())
    if not missing_items:
        print(json.dumps({"prefetch_missing_embeddings": 0, "cache_items": len(embedding_cache)}, ensure_ascii=False), flush=True)
        return

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(missing_items)
    for start in range(0, total, batch_size):
        batch = missing_items[start : start + batch_size]
        batch_keys = [key for key, _text in batch]
        batch_texts = [text for _key, text in batch]
        batch_embeddings = embedding_client.embed(batch_texts)
        if len(batch_embeddings) != len(batch_texts):
            raise RuntimeError("Embedding API returned an unexpected number of embeddings.")
        with cache_path.open("a", encoding="utf-8") as f:
            for key, embedding in zip(batch_keys, batch_embeddings):
                embedding_cache[key] = embedding
                f.write(json.dumps({"key": key, "model": model, "embedding": embedding}, ensure_ascii=False) + "\n")
        done = min(start + batch_size, total)
        if done == total or done % progress_every == 0:
            print(
                json.dumps(
                    {
                        "prefetch_embeddings": done,
                        "prefetch_total": total,
                        "cache_items": len(embedding_cache),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )


def _load_embedding_cache(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    cache: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cache[row["key"]] = row["embedding"]
    return cache


def _cache_key(model: str, text: str) -> str:
    payload = f"{model}\0{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_raw_rows(path: str) -> dict[str, dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return {row["_id"]: row for row in json.load(f)}


def _load_split_original_ids(split_dir: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for split in ("train", "valid", "test"):
        seen = set()
        ids = []
        for record in read_jsonl(Path(split_dir) / f"{split}.jsonl"):
            original_id = record["metadata"]["original_id"]
            if original_id not in seen:
                seen.add(original_id)
                ids.append(original_id)
        result[split] = ids
    return result


def _limit_split_ids(split_ids: dict[str, list[str]], limit: int) -> dict[str, list[str]]:
    remaining = limit
    limited: dict[str, list[str]] = {}
    for split in ("train", "valid", "test"):
        ids = split_ids[split][: max(0, remaining)]
        limited[split] = ids
        remaining -= len(ids)
    return limited


def _context_docs(raw: dict[str, Any]) -> list[dict[str, str]]:
    return [{"title": title, "text": " ".join(sentences)} for title, sentences in raw["context"]]


def _support_titles(raw: dict[str, Any]) -> set[str]:
    return {_normalize_title(title) for title, _sent_idx in raw["supporting_facts"]}


def _normalize_title(title: str) -> str:
    return html.unescape(str(title)).strip().lower()


def _cosine_scores(query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_embedding)
    doc_norms = np.linalg.norm(doc_embeddings, axis=1)
    denom = np.maximum(query_norm * doc_norms, 1e-12)
    return doc_embeddings.dot(query_embedding) / denom


def _summary(split_records: dict[str, list[dict[str, Any]]], top_k: int) -> dict[str, Any]:
    rows = []
    for split, records in split_records.items():
        labels = Counter(record["sufficiency_label"] for record in records)
        qtypes = Counter(record["metadata"].get("question_type", "unknown") for record in records)
        rows.append(
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
        "total_records": len(all_records),
        "splits": rows,
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
