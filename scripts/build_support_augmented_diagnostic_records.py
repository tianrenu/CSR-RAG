from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

from csrrag.utils.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build paired natural/oracle diagnostic records by augmenting retrieved docs with "
            "HotpotQA gold support-title documents. This is not a natural retrieval setting."
        )
    )
    parser.add_argument("--split-dir", default="data/processed/hotpotqa_official_intro_bm25_splits_full_dev")
    parser.add_argument("--record-kind-filter", default="official_intro_bm25_top5")
    parser.add_argument("--support-corpus", default="data/processed/hotpotqa_support_title_intro_corpus.jsonl")
    parser.add_argument(
        "--output-split-dir",
        default="data/processed/hotpotqa_official_intro_bm25_support_augmented_diagnostic_splits",
    )
    parser.add_argument(
        "--output-dir",
        default="results/tables/hotpotqa_official_intro_bm25_support_augmented_diagnostic",
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    support_docs = _load_support_docs(Path(args.support_corpus))
    split_records = {}
    detail_rows = []
    summary_rows = []
    for split in ("train", "valid", "test"):
        base_records = [
            record
            for record in read_jsonl(Path(args.split_dir) / f"{split}.jsonl")
            if record.get("metadata", {}).get("record_kind") == args.record_kind_filter
        ]
        paired = []
        for record in base_records:
            natural = _copy_natural_pair_record(record)
            oracle, detail = _make_oracle_record(record, support_docs, args.top_k)
            paired.append(natural)
            paired.append(oracle)
            detail_rows.append(detail)
        split_records[split] = paired
        summary_rows.append(_summary_row(split, base_records, paired, detail_rows))

    all_base = []
    all_paired = []
    for split in ("train", "valid", "test"):
        base = [
            record
            for record in read_jsonl(Path(args.split_dir) / f"{split}.jsonl")
            if record.get("metadata", {}).get("record_kind") == args.record_kind_filter
        ]
        all_base.extend(base)
        all_paired.extend(split_records[split])
    summary_rows.append(_summary_row("all", all_base, all_paired, detail_rows))

    output_split_dir = Path(args.output_split_dir)
    output_split_dir.mkdir(parents=True, exist_ok=True)
    for split, records in split_records.items():
        write_jsonl(output_split_dir / f"{split}.jsonl", records)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "support_augmented_summary.csv", summary_rows)
    _write_csv(output_dir / "support_augmented_question_details.csv", detail_rows)
    _write_validation(output_dir / "validation_summary.json", args, split_records, support_docs)
    _write_summary(output_dir / "support_augmented_diagnostic_summary.md", summary_rows, args)

    print(
        json.dumps(
            {
                "output_split_dir": str(output_split_dir),
                "output_dir": str(output_dir),
                "split_counts": {split: len(records) for split, records in split_records.items()},
                "support_docs": len(support_docs),
                "uses_embedding_api": False,
                "uses_llm_api": False,
                "oracle_diagnostic": True,
            },
            ensure_ascii=False,
        )
    )


def _load_support_docs(path: Path) -> dict[str, dict[str, Any]]:
    docs = {}
    for row in read_jsonl(path):
        title = _normalize_title(row["title"])
        docs[title] = row
    return docs


def _copy_natural_pair_record(record: dict[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(record, ensure_ascii=False))
    copied["id"] = f"{record['metadata']['original_id']}__diagnostic_natural_top5"
    copied["metadata"]["record_kind"] = "diagnostic_natural_top5"
    copied["metadata"]["diagnostic_pair_kind"] = "natural"
    copied["metadata"]["oracle_support_augmented"] = False
    return copied


def _make_oracle_record(
    record: dict[str, Any],
    support_docs: dict[str, dict[str, Any]],
    top_k: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    support_titles = [_normalize_title(title) for title in record["metadata"].get("support_titles", [])]
    base_docs = record.get("retrieved_docs", [])
    base_by_title = {_normalize_title(doc.get("title", "")): doc for doc in base_docs}
    selected = []
    selected_titles = set()
    injected_titles = []

    for title in support_titles:
        if title in base_by_title:
            selected.append(_copy_doc(base_by_title[title], source_override="hotpotqa_official_intro_bm25"))
            selected_titles.add(title)

    for title in support_titles:
        if title in selected_titles:
            continue
        if title not in support_docs:
            raise KeyError(f"Missing support-title doc in support corpus: {title}")
        selected.append(_support_doc_to_retrieved(support_docs[title]))
        selected_titles.add(title)
        injected_titles.append(title)

    for doc in base_docs:
        title = _normalize_title(doc.get("title", ""))
        if title in selected_titles:
            continue
        selected.append(_copy_doc(doc, source_override="hotpotqa_official_intro_bm25"))
        selected_titles.add(title)
        if len(selected) >= top_k:
            break

    selected = selected[:top_k]
    for rank, doc in enumerate(selected, start=1):
        doc["rank"] = rank

    support_set = set(support_titles)
    retrieved_titles = {_normalize_title(doc.get("title", "")) for doc in selected}
    missing = sorted(support_set - retrieved_titles)
    if missing:
        raise AssertionError(f"Oracle record still misses support titles: {missing}")

    oracle = json.loads(json.dumps(record, ensure_ascii=False))
    original_id = record["metadata"]["original_id"]
    oracle["id"] = f"{original_id}__diagnostic_oracle_support_augmented_top{top_k}"
    oracle["sufficiency_label"] = "sufficient"
    oracle["retrieved_docs"] = selected
    oracle["metadata"]["record_kind"] = f"diagnostic_oracle_support_augmented_top{top_k}"
    oracle["metadata"]["diagnostic_pair_kind"] = "oracle_support_augmented"
    oracle["metadata"]["oracle_support_augmented"] = True
    oracle["metadata"]["retriever"] = "bm25-official-intro-corpus+oracle-support-title-docs"
    oracle["metadata"]["top_k"] = top_k
    oracle["metadata"]["support_present_in_topk"] = True
    oracle["metadata"]["missing_support_titles"] = []
    oracle["metadata"]["injected_support_titles"] = injected_titles
    oracle["metadata"]["forced_missing_support_titles"] = []

    detail = {
        "split": record["metadata"]["split"],
        "original_id": original_id,
        "question": record["query"],
        "gold_answer": record["gold_answer"],
        "base_label": record["sufficiency_label"],
        "oracle_label": oracle["sufficiency_label"],
        "base_missing_support_titles": " || ".join(record["metadata"].get("missing_support_titles", [])),
        "injected_support_title_count": len(injected_titles),
        "injected_support_titles": " || ".join(injected_titles),
        "base_top_titles": " || ".join(doc.get("title", "") for doc in base_docs),
        "oracle_top_titles": " || ".join(doc.get("title", "") for doc in selected),
    }
    return oracle, detail


def _support_doc_to_retrieved(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "doc_id": doc["doc_id"],
        "rank": 0,
        "score": 0.0,
        "embedding_score": 0.0,
        "sparse_score": 0.0,
        "title": doc["title"],
        "text": doc["text"],
        "source": "hotpotqa_oracle_support_title_injection",
    }


def _copy_doc(doc: dict[str, Any], source_override: str) -> dict[str, Any]:
    copied = dict(doc)
    copied["source"] = source_override
    return copied


def _summary_row(
    split: str,
    base_records: list[dict[str, Any]],
    paired_records: list[dict[str, Any]],
    detail_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    split_details = [row for row in detail_rows if split == "all" or row["split"] == split]
    base_labels = Counter(record["sufficiency_label"] for record in base_records)
    paired_labels = Counter(record["sufficiency_label"] for record in paired_records)
    return {
        "split": split,
        "base_questions": len(base_records),
        "paired_records": len(paired_records),
        "base_sufficient_count": base_labels.get("sufficient", 0),
        "base_insufficient_count": base_labels.get("insufficient", 0),
        "base_sufficient_rate": base_labels.get("sufficient", 0) / len(base_records) if base_records else 0.0,
        "paired_sufficient_count": paired_labels.get("sufficient", 0),
        "paired_insufficient_count": paired_labels.get("insufficient", 0),
        "paired_sufficient_rate": paired_labels.get("sufficient", 0) / len(paired_records) if paired_records else 0.0,
        "mean_injected_support_title_count": _mean(int(row["injected_support_title_count"]) for row in split_details),
        "oracle_records": sum(1 for record in paired_records if record["metadata"].get("oracle_support_augmented")),
    }


def _write_validation(
    path: Path,
    args: argparse.Namespace,
    split_records: dict[str, list[dict[str, Any]]],
    support_docs: dict[str, dict[str, Any]],
) -> None:
    validation = {
        "input_split_dir": args.split_dir,
        "input_record_kind_filter": args.record_kind_filter,
        "support_corpus": args.support_corpus,
        "output_split_dir": args.output_split_dir,
        "top_k": args.top_k,
        "split_counts": {split: len(records) for split, records in split_records.items()},
        "support_docs": len(support_docs),
        "uses_embedding_api": False,
        "uses_llm_api": False,
        "oracle_diagnostic": True,
        "retrieval_uses_gold_support": True,
        "suitable_as_main_result": False,
        "purpose": "diagnostic contrast between natural top5 records and oracle support-complete top5 records",
    }
    path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_summary(path: Path, summary_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# BM25 Support-Augmented Diagnostic Records",
        "",
        "## Purpose",
        "",
        "This no-API diagnostic dataset pairs natural BM25 top-5 records with oracle support-complete records built by injecting official support-title documents. It must not be used as a natural retrieval main result.",
        "",
        "## Summary",
        "",
        "| Split | Base questions | Paired records | Base sufficient | Paired sufficient | Mean injected support titles |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['split']} | {row['base_questions']} | {row['paired_records']} | "
            f"{float(row['base_sufficient_rate']):.4f} | {float(row['paired_sufficient_rate']):.4f} | "
            f"{float(row['mean_injected_support_title_count']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- Input split dir: `{args.split_dir}`",
            f"- Support corpus: `{args.support_corpus}`",
            "- Embedding API: no",
            "- LLM API: no",
            "- Oracle diagnostic: yes",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def _normalize_title(title: str) -> str:
    return html.unescape(str(title)).strip().lower()


def _mean(values) -> float:
    value_list = list(values)
    return float(sum(value_list) / len(value_list)) if value_list else 0.0


if __name__ == "__main__":
    main()
