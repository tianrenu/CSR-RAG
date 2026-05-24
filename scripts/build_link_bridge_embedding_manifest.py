from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from csrrag.utils.io import read_jsonl
from csrrag.utils.text import tokenize


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a no-API embedding manifest for candidate-level reranking over link-bridge records. "
            "The script estimates unique query/doc texts, cache coverage, and missing-token budget."
        )
    )
    parser.add_argument("--split-dir", default="data/processed/hotpotqa_official_intro_link_bridge_splits_top20_full_dev")
    parser.add_argument("--record-kind-filter", default="official_intro_link_bridge_a0p85_p0p00_top20")
    parser.add_argument("--output-dir", default="results/tables/hotpotqa_official_intro_link_bridge_embedding_manifest")
    parser.add_argument("--embedding-model", default="text-embedding-v4")
    parser.add_argument(
        "--cache-path",
        action="append",
        default=[
            "data/cache/hotpotqa_text_embedding_v4_full_dev.jsonl",
            "data/cache/hotpotqa_text_embedding_v4_1800.jsonl",
        ],
        help="Existing embedding cache JSONL paths. May be passed multiple times.",
    )
    parser.add_argument("--token-multiplier", type=float, default=1.3)
    parser.add_argument("--sample-missing", type=int, default=200)
    args = parser.parse_args()

    records = _load_records(Path(args.split_dir), args.record_kind_filter)
    cache_keys = _load_cache_keys([Path(path) for path in args.cache_path or []])
    manifest = _build_manifest(records, args.embedding_model, cache_keys, args.token_multiplier)
    summary_rows = _summary_rows(manifest)
    sample_rows = _sample_missing_rows(manifest, args.sample_missing)
    split_rows = _split_rows(records, manifest)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "embedding_manifest_summary.csv", summary_rows)
    _write_csv(output_dir / "embedding_manifest_split_summary.csv", split_rows)
    _write_csv(output_dir / "embedding_manifest_missing_samples.csv", sample_rows)
    (output_dir / "validation_summary.json").write_text(
        json.dumps(
            {
                "split_dir": args.split_dir,
                "record_kind_filter": args.record_kind_filter,
                "embedding_model": args.embedding_model,
                "cache_paths": args.cache_path,
                "cache_key_count": len(cache_keys),
                "token_multiplier": args.token_multiplier,
                "records": {split: len(rows) for split, rows in records.items()},
                "uses_embedding_api": False,
                "uses_llm_api": False,
                "purpose": "budget and cache-coverage manifest for candidate-level embedding reranking",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "embedding_manifest_summary.md").write_text(_summary_markdown(summary_rows), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "records": sum(len(rows) for rows in records.values()),
                "unique_texts": sum(int(row["unique_text_count"]) for row in summary_rows if row["text_type"] == "all"),
                "missing_texts": sum(int(row["missing_cache_count"]) for row in summary_rows if row["text_type"] == "all"),
                "estimated_missing_tokens": sum(int(row["estimated_missing_tokens"]) for row in summary_rows if row["text_type"] == "all"),
                "uses_embedding_api": False,
            },
            ensure_ascii=False,
        )
    )


def _load_records(split_dir: Path, record_kind: str) -> dict[str, list[dict[str, Any]]]:
    records = {}
    for split in ("train", "valid", "test"):
        rows = [
            record
            for record in read_jsonl(split_dir / f"{split}.jsonl")
            if record.get("metadata", {}).get("record_kind") == record_kind
        ]
        if not rows:
            raise ValueError(f"No {record_kind} records found for split {split}.")
        records[split] = rows
    return records


def _load_cache_keys(paths: list[Path]) -> set[str]:
    keys: set[str] = set()
    for path in _dedupe_paths(paths):
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = row.get("key")
                if key:
                    keys.add(str(key))
    return keys


def _build_manifest(
    records: dict[str, list[dict[str, Any]]],
    embedding_model: str,
    cache_keys: set[str],
    token_multiplier: float,
) -> dict[str, dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    for split, split_records in records.items():
        for record in split_records:
            _add_text(
                items,
                text_type="query",
                text=record["query"],
                embedding_model=embedding_model,
                cache_keys=cache_keys,
                token_multiplier=token_multiplier,
                split=split,
                original_id=record["metadata"]["original_id"],
                title="",
            )
            for doc in record["retrieved_docs"]:
                _add_text(
                    items,
                    text_type="doc",
                    text=_doc_embedding_text(doc),
                    embedding_model=embedding_model,
                    cache_keys=cache_keys,
                    token_multiplier=token_multiplier,
                    split=split,
                    original_id=record["metadata"]["original_id"],
                    title=str(doc.get("title", "")),
                )
    return items


def _add_text(
    items: dict[str, dict[str, Any]],
    text_type: str,
    text: str,
    embedding_model: str,
    cache_keys: set[str],
    token_multiplier: float,
    split: str,
    original_id: str,
    title: str,
) -> None:
    cache_key = _cache_key(embedding_model, text)
    text_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    estimated_tokens = int(math.ceil(len(tokenize(text)) * token_multiplier))
    if cache_key not in items:
        items[cache_key] = {
            "cache_key": cache_key,
            "text_key": text_key,
            "text_type": text_type,
            "cached": cache_key in cache_keys,
            "char_count": len(text),
            "word_token_count": len(tokenize(text)),
            "estimated_embedding_tokens": estimated_tokens,
            "occurrence_count": 0,
            "splits": Counter(),
            "example_original_ids": [],
            "example_title": title,
        }
    item = items[cache_key]
    item["occurrence_count"] += 1
    item["splits"][split] += 1
    if len(item["example_original_ids"]) < 3:
        item["example_original_ids"].append(original_id)


def _summary_rows(manifest: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for text_type in ("query", "doc", "all"):
        items = list(manifest.values()) if text_type == "all" else [item for item in manifest.values() if item["text_type"] == text_type]
        cached = [item for item in items if item["cached"]]
        missing = [item for item in items if not item["cached"]]
        rows.append(
            {
                "text_type": text_type,
                "unique_text_count": len(items),
                "cached_count": len(cached),
                "missing_cache_count": len(missing),
                "cache_coverage": len(cached) / len(items) if items else 0.0,
                "occurrence_count": sum(int(item["occurrence_count"]) for item in items),
                "estimated_total_tokens": sum(int(item["estimated_embedding_tokens"]) for item in items),
                "estimated_cached_tokens": sum(int(item["estimated_embedding_tokens"]) for item in cached),
                "estimated_missing_tokens": sum(int(item["estimated_embedding_tokens"]) for item in missing),
                "mean_estimated_tokens_per_text": (
                    sum(int(item["estimated_embedding_tokens"]) for item in items) / len(items) if items else 0.0
                ),
            }
        )
    return rows


def _split_rows(records: dict[str, list[dict[str, Any]]], manifest: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for split in ("train", "valid", "test", "all"):
        split_items = []
        for item in manifest.values():
            if split == "all" or item["splits"][split] > 0:
                split_items.append(item)
        rows.append(
            {
                "split": split,
                "record_count": sum(len(rows) for rows in records.values()) if split == "all" else len(records[split]),
                "unique_text_count": len(split_items),
                "unique_query_count": sum(1 for item in split_items if item["text_type"] == "query"),
                "unique_doc_count": sum(1 for item in split_items if item["text_type"] == "doc"),
                "missing_cache_count": sum(1 for item in split_items if not item["cached"]),
                "estimated_missing_tokens": sum(int(item["estimated_embedding_tokens"]) for item in split_items if not item["cached"]),
            }
        )
    return rows


def _sample_missing_rows(manifest: dict[str, dict[str, Any]], sample_count: int) -> list[dict[str, Any]]:
    missing = [item for item in manifest.values() if not item["cached"]]
    missing.sort(key=lambda item: (-int(item["occurrence_count"]), item["text_type"], item["text_key"]))
    rows = []
    for item in missing[:sample_count]:
        rows.append(
            {
                "cache_key": item["cache_key"],
                "text_key": item["text_key"],
                "text_type": item["text_type"],
                "occurrence_count": item["occurrence_count"],
                "estimated_embedding_tokens": item["estimated_embedding_tokens"],
                "example_title": item["example_title"],
                "example_original_ids": " || ".join(item["example_original_ids"]),
                "splits": " || ".join(f"{split}:{count}" for split, count in sorted(item["splits"].items()) if count),
            }
        )
    return rows


def _summary_markdown(summary_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Link-Bridge Embedding Manifest Summary",
        "",
        "This is a no-API budget manifest for candidate-level embedding reranking. Token counts are approximate and use a word-token multiplier, so they should be treated as planning estimates rather than billing truth.",
        "",
        "| Text Type | Unique Texts | Cached | Missing | Cache Coverage | Estimated Missing Tokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['text_type']} | {int(row['unique_text_count'])} | {int(row['cached_count'])} | "
            f"{int(row['missing_cache_count'])} | {float(row['cache_coverage']):.4f} | "
            f"{int(row['estimated_missing_tokens'])} |"
        )
    lines.append("")
    lines.append("Recommended next step: if the missing-token estimate is within budget, run candidate-level query/doc embedding only for missing manifest texts and cache the result before reranking.")
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _doc_embedding_text(doc: dict[str, Any]) -> str:
    return f"{doc['title']}\n{doc['text']}"


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


if __name__ == "__main__":
    main()
