from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

from csrrag.utils.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit local HotpotQA corpus completeness for CSR-RAG retrieval experiments."
    )
    parser.add_argument("--raw-hotpot", default="data/raw/hotpotqa/hotpot_dev_fullwiki_v1.json")
    parser.add_argument("--split-dir", default="data/processed/hotpotqa_global_hardneg_splits_full_dev")
    parser.add_argument("--record-kind-filter", default="natural_global_top5")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="results/tables/hotpotqa_corpus_completeness_audit")
    parser.add_argument("--example-limit-per-split", type=int, default=30)
    args = parser.parse_args()

    raw_rows = _load_raw_rows(Path(args.raw_hotpot))
    split_ids = _load_split_ids(Path(args.split_dir), args.record_kind_filter)
    selected_ids = [original_id for split in ("train", "valid", "test") for original_id in split_ids[split]]
    global_titles = _global_titles(raw_rows, selected_ids)

    detail_rows = []
    for split in ("train", "valid", "test"):
        for original_id in split_ids[split]:
            detail_rows.append(_audit_row(split, raw_rows[original_id], global_titles))

    summary_rows = _summary_rows(detail_rows)
    missing_examples = _missing_examples(detail_rows, args.example_limit_per_split)
    manifest_rows = _manifest_rows(Path(args.data_root))
    validation = _validation_summary(args, raw_rows, split_ids, global_titles, detail_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "corpus_completeness_summary.csv", summary_rows)
    _write_csv(output_dir / "corpus_completeness_details.csv", detail_rows)
    _write_csv(output_dir / "missing_support_examples.csv", missing_examples)
    _write_csv(output_dir / "local_data_manifest.csv", manifest_rows)
    (output_dir / "validation_summary.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(output_dir / "corpus_completeness_audit.md", summary_rows, missing_examples, validation)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "detail_rows": len(detail_rows),
                "summary_rows": len(summary_rows),
                "manifest_rows": len(manifest_rows),
                "no_api_calls": True,
            },
            ensure_ascii=False,
        )
    )


def _load_raw_rows(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return {row["_id"]: row for row in json.load(f)}


def _load_split_ids(split_dir: Path, record_kind: str) -> dict[str, list[str]]:
    split_ids = {}
    for split in ("train", "valid", "test"):
        records = [
            record
            for record in read_jsonl(split_dir / f"{split}.jsonl")
            if record.get("metadata", {}).get("record_kind") == record_kind
        ]
        split_ids[split] = [record["metadata"]["original_id"] for record in records]
    return split_ids


def _global_titles(raw_rows: dict[str, dict[str, Any]], selected_ids: list[str]) -> set[str]:
    titles = set()
    for original_id in selected_ids:
        for title, _sentences in raw_rows[original_id]["context"]:
            titles.add(_normalize_title(title))
    return titles


def _audit_row(split: str, raw: dict[str, Any], global_titles: set[str]) -> dict[str, Any]:
    support_titles = sorted({_normalize_title(title) for title, _sent_idx in raw["supporting_facts"]})
    context_titles = [_normalize_title(title) for title, _sentences in raw["context"]]
    context_title_set = set(context_titles)
    in_context = [title for title in support_titles if title in context_title_set]
    in_global = [title for title in support_titles if title in global_titles]
    missing_context = [title for title in support_titles if title not in context_title_set]
    missing_global = [title for title in support_titles if title not in global_titles]
    fuzzy_context = _fuzzy_title_matches(support_titles, context_titles)
    return {
        "split": split,
        "original_id": raw["_id"],
        "question_type": raw.get("type", "unknown"),
        "difficulty": raw.get("level", "unknown"),
        "question": raw["question"],
        "gold_answer": raw["answer"],
        "context_doc_count": len(context_titles),
        "support_title_count": len(support_titles),
        "support_titles": " || ".join(support_titles),
        "context_all_support_present": int(len(in_context) == len(support_titles)),
        "context_support_title_coverage": len(in_context) / len(support_titles) if support_titles else 0.0,
        "context_missing_support_count": len(missing_context),
        "context_missing_support_titles": " || ".join(missing_context),
        "global_pool_all_support_present": int(len(in_global) == len(support_titles)),
        "global_pool_support_title_coverage": len(in_global) / len(support_titles) if support_titles else 0.0,
        "global_pool_missing_support_count": len(missing_global),
        "global_pool_missing_support_titles": " || ".join(missing_global),
        "missing_context_but_present_global_count": sum(1 for title in missing_context if title in global_titles),
        "missing_context_with_fuzzy_title_count": sum(1 for title in missing_context if fuzzy_context.get(title)),
        "fuzzy_context_matches": " || ".join(f"{title} => {fuzzy_context[title]}" for title in missing_context if fuzzy_context.get(title)),
        "first_context_titles": " || ".join(title for title, _sentences in raw["context"][:10]),
    }


def _fuzzy_title_matches(support_titles: list[str], context_titles: list[str]) -> dict[str, str]:
    matches = {}
    for support_title in support_titles:
        candidates = []
        support_tokens = set(support_title.split())
        for context_title in context_titles:
            context_tokens = set(context_title.split())
            if not support_tokens or not context_tokens:
                continue
            overlap = len(support_tokens & context_tokens) / len(support_tokens | context_tokens)
            if support_title in context_title or context_title in support_title or overlap >= 0.5:
                candidates.append((overlap, context_title))
        if candidates:
            matches[support_title] = ", ".join(title for _score, title in sorted(candidates, reverse=True)[:3])
    return matches


def _summary_rows(detail_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for split in ("train", "valid", "test", "all"):
        split_rows = detail_rows if split == "all" else [row for row in detail_rows if row["split"] == split]
        if not split_rows:
            continue
        context_counts = [int(row["context_doc_count"]) for row in split_rows]
        support_counts = [int(row["support_title_count"]) for row in split_rows]
        context_missing = [row for row in split_rows if int(row["context_all_support_present"]) == 0]
        global_missing = [row for row in split_rows if int(row["global_pool_all_support_present"]) == 0]
        context_missing_but_global_present = [
            row
            for row in split_rows
            if int(row["context_all_support_present"]) == 0 and int(row["global_pool_all_support_present"]) == 1
        ]
        fuzzy_missing = [row for row in context_missing if int(row["missing_context_with_fuzzy_title_count"]) > 0]
        rows.append(
            {
                "split": split,
                "n": len(split_rows),
                "context_doc_count_mean": mean(context_counts),
                "context_doc_count_median": median(context_counts),
                "context_doc_count_min": min(context_counts),
                "context_doc_count_max": max(context_counts),
                "support_title_count_mean": mean(support_counts),
                "context_all_support_present_count": len(split_rows) - len(context_missing),
                "context_all_support_present_rate": (len(split_rows) - len(context_missing)) / len(split_rows),
                "context_missing_support_question_count": len(context_missing),
                "context_missing_support_question_rate": len(context_missing) / len(split_rows),
                "global_pool_all_support_present_count": len(split_rows) - len(global_missing),
                "global_pool_all_support_present_rate": (len(split_rows) - len(global_missing)) / len(split_rows),
                "global_pool_missing_support_question_count": len(global_missing),
                "global_pool_missing_support_question_rate": len(global_missing) / len(split_rows),
                "context_missing_but_global_present_count": len(context_missing_but_global_present),
                "context_missing_but_global_present_rate": len(context_missing_but_global_present) / len(split_rows),
                "context_missing_with_fuzzy_title_count": len(fuzzy_missing),
                "context_missing_with_fuzzy_title_rate": len(fuzzy_missing) / len(split_rows),
                "mean_context_support_title_coverage": mean(float(row["context_support_title_coverage"]) for row in split_rows),
                "mean_global_pool_support_title_coverage": mean(float(row["global_pool_support_title_coverage"]) for row in split_rows),
            }
        )
    return rows


def _missing_examples(detail_rows: list[dict[str, Any]], limit_per_split: int) -> list[dict[str, Any]]:
    examples = []
    counters: Counter[str] = Counter()
    for row in detail_rows:
        if int(row["context_all_support_present"]) == 1 and int(row["global_pool_all_support_present"]) == 1:
            continue
        split = row["split"]
        if counters[split] >= limit_per_split:
            continue
        counters[split] += 1
        examples.append(row)
    return examples


def _manifest_rows(data_root: Path) -> list[dict[str, Any]]:
    if not data_root.exists():
        return []
    rows = []
    for path in sorted(data_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.as_posix()
        size = path.stat().st_size
        rows.append(
            {
                "path": rel,
                "size_bytes": size,
                "size_mb": round(size / 1024 / 1024, 3),
                "role": _guess_role(path),
                "lightweight_baseline_recommendation": _recommendation(path, size),
            }
        )
    return sorted(rows, key=lambda row: int(row["size_bytes"]), reverse=True)


def _guess_role(path: Path) -> str:
    parts = set(path.parts)
    name = path.name.lower()
    if "cache" in parts:
        return "embedding_cache"
    if "raw" in parts:
        return "raw_dataset"
    if "retrieval" in parts:
        return "retrieval_records"
    if "processed" in parts:
        return "processed_splits"
    if "features" in parts:
        return "features"
    if "outputs" in parts:
        return "model_or_qa_outputs"
    if name.endswith(".json") or name.endswith(".jsonl") or name.endswith(".csv"):
        return "data_artifact"
    return "other"


def _recommendation(path: Path, size: int) -> str:
    role = _guess_role(path)
    if role in {"embedding_cache", "raw_dataset"} or size > 10 * 1024 * 1024:
        return "register_path_only"
    if role in {"features", "processed_splits", "retrieval_records"} and size > 2 * 1024 * 1024:
        return "register_path_or_manifest_only"
    return "eligible_small_artifact"


def _validation_summary(
    args: argparse.Namespace,
    raw_rows: dict[str, dict[str, Any]],
    split_ids: dict[str, list[str]],
    global_titles: set[str],
    detail_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    split_sets = {split: set(ids) for split, ids in split_ids.items()}
    return {
        "raw_hotpot": args.raw_hotpot,
        "raw_rows": len(raw_rows),
        "split_counts": {split: len(ids) for split, ids in split_ids.items()},
        "global_pool_unique_titles": len(global_titles),
        "detail_rows": len(detail_rows),
        "train_valid_overlap": len(split_sets["train"] & split_sets["valid"]),
        "train_test_overlap": len(split_sets["train"] & split_sets["test"]),
        "valid_test_overlap": len(split_sets["valid"] & split_sets["test"]),
        "uses_embedding_api": False,
        "uses_llm_api": False,
    }


def _write_markdown(
    path: Path,
    summary_rows: list[dict[str, Any]],
    missing_examples: list[dict[str, Any]],
    validation: dict[str, Any],
) -> None:
    lines = ["# HotpotQA Corpus Completeness Audit", ""]
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "This no-API audit separates per-question context completeness from global context-pool completeness for CSR-RAG retrieval experiments."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| Split | N | Context all-support present | Global-pool all-support present | Context docs median | Missing context but global present |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        lines.append(
            f"| {row['split']} | {row['n']} | {float(row['context_all_support_present_rate']):.4f} | "
            f"{float(row['global_pool_all_support_present_rate']):.4f} | {float(row['context_doc_count_median']):.1f} | "
            f"{float(row['context_missing_but_global_present_rate']):.4f} |"
        )
    lines.append("")
    lines.append("## Validation")
    lines.append("")
    lines.append(f"- Raw rows: {validation['raw_rows']}")
    lines.append(f"- Global pool unique titles: {validation['global_pool_unique_titles']}")
    lines.append(f"- Split counts: {json.dumps(validation['split_counts'], ensure_ascii=False)}")
    lines.append("- Embedding API: no")
    lines.append("- LLM API: no")
    lines.append("")
    lines.append("## Example Missing-Support Cases")
    lines.append("")
    for row in missing_examples[:10]:
        lines.append(
            f"- `{row['split']}` / `{row['original_id']}`: missing context support `{row['context_missing_support_titles']}`; "
            f"missing global support `{row['global_pool_missing_support_titles']}`; question: {row['question']}"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "If global-pool all-support presence is low, current retrieval experiments are corpus-limited. A support-complete corpus is required before treating low sufficiency as a pure retriever or estimator failure."
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
    return " ".join(str(title).strip().lower().split())


if __name__ == "__main__":
    main()
