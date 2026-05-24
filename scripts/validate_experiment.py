from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from csrrag.utils.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sanity checks for the leak-free CSR-RAG experiment.")
    parser.add_argument("--retrieval-input", required=True, help="Full retrieval record JSONL.")
    parser.add_argument("--split-dir", required=True, help="Directory containing train/valid/test JSONL files.")
    parser.add_argument("--test-risk-input", required=True, help="Test risk JSONL.")
    parser.add_argument("--decision-metrics", required=True, help="Decision metrics CSV.")
    parser.add_argument("--coverage-curve", required=True, help="Coverage-risk curve CSV.")
    parser.add_argument("--expected-original-count", type=int, default=1800)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default=None, help="Optional JSON path for validation summary.")
    args = parser.parse_args()

    retrieval_records = read_jsonl(args.retrieval_input)
    split_dir = Path(args.split_dir)
    split_records = {
        "train": read_jsonl(split_dir / "train.jsonl"),
        "valid": read_jsonl(split_dir / "valid.jsonl"),
        "test": read_jsonl(split_dir / "test.jsonl"),
    }
    test_risk_records = read_jsonl(args.test_risk_input)

    summary = {
        "retrieval": _check_retrieval_records(
            retrieval_records,
            expected_original_count=args.expected_original_count,
            top_k=args.top_k,
        ),
        "splits": _check_splits(split_records, retrieval_records),
        "risk": _check_risk_scores(test_risk_records),
        "tables": _check_tables(Path(args.decision_metrics), Path(args.coverage_curve)),
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _check_retrieval_records(
    records: list[dict[str, Any]],
    expected_original_count: int,
    top_k: int,
) -> dict[str, Any]:
    by_original_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        original_id = record["metadata"]["original_id"]
        by_original_id[original_id].append(record)

    _require(len(by_original_id) == expected_original_count, "Unexpected original question count.")
    _require(len(records) == expected_original_count * 2, "Unexpected retrieval record count.")

    for original_id, group in by_original_id.items():
        labels = {record["sufficiency_label"] for record in group}
        _require(len(group) == 2, f"{original_id} does not have exactly 2 records.")
        _require(labels == {"sufficient", "insufficient"}, f"{original_id} is missing a paired label.")

    for record in records:
        docs = record["retrieved_docs"]
        doc_ids = {doc["doc_id"] for doc in docs}
        support_doc_ids = set(record["metadata"].get("support_doc_ids", []))
        _require(len(docs) == top_k, f"{record['id']} does not have top_k={top_k}.")
        _require(all("is_support" not in doc for doc in docs), f"{record['id']} exposes is_support.")

        if record["sufficiency_label"] == "sufficient":
            _require(support_doc_ids.issubset(doc_ids), f"{record['id']} misses support docs.")
        else:
            missing = support_doc_ids - doc_ids
            _require(len(missing) >= 1, f"{record['id']} does not drop a support doc.")

    label_counts = Counter(record["sufficiency_label"] for record in records)
    return {
        "record_count": len(records),
        "original_id_count": len(by_original_id),
        "sufficient_count": label_counts["sufficient"],
        "insufficient_count": label_counts["insufficient"],
        "top_k": top_k,
    }


def _check_splits(
    split_records: dict[str, list[dict[str, Any]]],
    all_records: list[dict[str, Any]],
) -> dict[str, Any]:
    split_ids = {
        split: {record["metadata"]["original_id"] for record in records}
        for split, records in split_records.items()
    }
    _require(split_ids["train"].isdisjoint(split_ids["valid"]), "train and valid overlap.")
    _require(split_ids["train"].isdisjoint(split_ids["test"]), "train and test overlap.")
    _require(split_ids["valid"].isdisjoint(split_ids["test"]), "valid and test overlap.")

    all_original_ids = {record["metadata"]["original_id"] for record in all_records}
    combined_ids = set().union(*split_ids.values())
    _require(combined_ids == all_original_ids, "Split IDs do not cover the retrieval dataset.")

    rows = {}
    for split, records in split_records.items():
        label_counts = Counter(record["sufficiency_label"] for record in records)
        _require(label_counts["sufficient"] > 0, f"{split} has no sufficient records.")
        _require(label_counts["insufficient"] > 0, f"{split} has no insufficient records.")
        rows[split] = {
            "record_count": len(records),
            "original_id_count": len(split_ids[split]),
            "sufficient_count": label_counts["sufficient"],
            "insufficient_count": label_counts["insufficient"],
        }
    return rows


def _check_risk_scores(records: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(record["risk_score"]) for record in records]
    rounded_unique = {round(score, 8) for score in scores}
    _require(len(rounded_unique) > 2, "Risk scores are nearly constant.")
    _require(not all(score in {0.0, 1.0} for score in rounded_unique), "Risk scores are all binary.")
    return {
        "record_count": len(records),
        "unique_score_count": len(rounded_unique),
        "min_risk": min(scores),
        "max_risk": max(scores),
    }


def _check_tables(decision_metrics_path: Path, coverage_curve_path: Path) -> dict[str, Any]:
    decision_rows = _read_csv(decision_metrics_path)
    curve_rows = _read_csv(coverage_curve_path)
    _require(len(decision_rows) >= 1, "Decision metrics table is empty.")
    _require(len(curve_rows) >= 10, "Coverage-risk curve has too few threshold points.")

    for row in decision_rows:
        coverage = float(row["coverage"] if "coverage" in row else row["test_coverage"])
        accuracy_key = "decision_accuracy" if "decision_accuracy" in row else "test_decision_accuracy"
        decision_accuracy = float(row[accuracy_key])
        _require(
            not (coverage == 0.5 and decision_accuracy == 1.0),
            "Decision metrics look like the old leaked debug table.",
        )

    return {
        "decision_rows": len(decision_rows),
        "coverage_curve_rows": len(curve_rows),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
