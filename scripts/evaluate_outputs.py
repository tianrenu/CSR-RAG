from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from csrrag.evaluation.metrics import accuracy, coverage, decision_metrics_from_risk, selective_accuracy
from csrrag.utils.io import read_jsonl


TAU_GRID = [round(x, 2) for x in [i / 100 for i in range(5, 100, 5)]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Write formal experiment tables for CSR-RAG.")
    parser.add_argument("--train-input", required=True, help="Train retrieval JSONL.")
    parser.add_argument("--valid-input", required=True, help="Validation retrieval JSONL.")
    parser.add_argument("--test-input", required=True, help="Test retrieval JSONL.")
    parser.add_argument("--test-risk-input", required=True, help="Calibrated test risk JSONL.")
    parser.add_argument("--test-decisions-input", required=True, help="Test decision JSONL.")
    parser.add_argument("--calibration-metadata", required=True, help="Calibration metadata JSON.")
    parser.add_argument("--threshold-metadata", required=True, help="Threshold metadata JSON.")
    parser.add_argument("--output-dir", required=True, help="Directory for formal tables.")
    args = parser.parse_args()

    train_records = read_jsonl(args.train_input)
    valid_records = read_jsonl(args.valid_input)
    test_records = read_jsonl(args.test_input)
    test_risk_records = read_jsonl(args.test_risk_input)
    test_decisions = {record["id"]: record for record in read_jsonl(args.test_decisions_input)}

    with Path(args.calibration_metadata).open("r", encoding="utf-8") as f:
        calibration_meta = json.load(f)
    with Path(args.threshold_metadata).open("r", encoding="utf-8") as f:
        threshold_meta = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_score_metrics(output_dir / "score_metrics_test.csv", calibration_meta, len(test_risk_records))
    _write_decision_metrics(output_dir / "decision_metrics_test.csv", test_records, test_decisions, threshold_meta)
    _write_coverage_curve(output_dir / "coverage_risk_curve_test.csv", test_risk_records)
    _write_data_summary(output_dir / "data_summary.csv", train_records, valid_records, test_records)


def _write_score_metrics(path: Path, calibration_meta: dict[str, object], test_size: int) -> None:
    row = {
        "method": calibration_meta["method"],
        "n_test": test_size,
        "raw_brier": calibration_meta["raw_brier_test"],
        "raw_ece": calibration_meta["raw_ece_test"],
        "calibrated_brier": calibration_meta["calibrated_brier_test"],
        "calibrated_ece": calibration_meta["calibrated_ece_test"],
    }
    _write_csv(path, [row], list(row.keys()))


def _write_decision_metrics(
    path: Path,
    test_records: list[dict[str, object]],
    test_decisions: dict[str, dict[str, object]],
    threshold_meta: dict[str, object],
) -> None:
    y_true = [1 if record["sufficiency_label"] == "insufficient" else 0 for record in test_records]
    y_pred = [1 if test_decisions[record["id"]]["decision"] == "abstain" else 0 for record in test_records]
    keep = [1 if test_decisions[record["id"]]["decision"] == "answer" else 0 for record in test_records]

    row = {
        "tau_answer": threshold_meta["tau_answer"],
        "decision_accuracy": accuracy(y_true, y_pred),
        "coverage": coverage(keep),
        "selective_accuracy": selective_accuracy(y_true, keep),
        "n_test": len(test_records),
    }
    _write_csv(path, [row], list(row.keys()))


def _write_coverage_curve(path: Path, test_risk_records: list[dict[str, object]]) -> None:
    y_true = [1 if record["sufficiency_label"] == "insufficient" else 0 for record in test_risk_records]
    risk_scores = [float(record["risk_score"]) for record in test_risk_records]
    rows = [decision_metrics_from_risk(y_true, risk_scores, tau) for tau in TAU_GRID]
    _write_csv(path, rows, ["tau_answer", "decision_accuracy", "coverage", "selective_accuracy"])


def _write_data_summary(
    path: Path,
    train_records: list[dict[str, object]],
    valid_records: list[dict[str, object]],
    test_records: list[dict[str, object]],
) -> None:
    rows = [
        _summarize_split("train", train_records),
        _summarize_split("valid", valid_records),
        _summarize_split("test", test_records),
    ]
    fieldnames = [
        "split",
        "record_count",
        "original_id_count",
        "sufficient_count",
        "insufficient_count",
        "bridge_count",
        "comparison_count",
    ]
    _write_csv(path, rows, fieldnames)


def _summarize_split(split_name: str, records: list[dict[str, object]]) -> dict[str, object]:
    label_counter = Counter(record["sufficiency_label"] for record in records)
    type_counter = Counter(record["metadata"].get("question_type", "unknown") for record in records)
    original_ids = {record["metadata"]["original_id"] for record in records}
    return {
        "split": split_name,
        "record_count": len(records),
        "original_id_count": len(original_ids),
        "sufficient_count": label_counter.get("sufficient", 0),
        "insufficient_count": label_counter.get("insufficient", 0),
        "bridge_count": type_counter.get("bridge", 0),
        "comparison_count": type_counter.get("comparison", 0),
    }


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
