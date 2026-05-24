from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from csrrag.calibration.methods import make_calibrator
from csrrag.evaluation.metrics import brier_score, ece
from csrrag.utils.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a calibrator on valid and apply it to test.")
    parser.add_argument("--valid-input", required=True, help="Validation sufficiency score JSONL.")
    parser.add_argument("--test-input", required=True, help="Test sufficiency score JSONL.")
    parser.add_argument("--output-calibrator", required=True, help="Path to output pickled calibrator.")
    parser.add_argument("--output-metadata", required=True, help="Path to output calibration metrics JSON.")
    parser.add_argument("--output-valid-risk", required=True, help="Path to output validation risk JSONL.")
    parser.add_argument("--output-test-risk", required=True, help="Path to output test risk JSONL.")
    parser.add_argument("--method", default="isotonic", choices=["identity", "platt", "isotonic"])
    args = parser.parse_args()

    valid_records = read_jsonl(args.valid_input)
    test_records = read_jsonl(args.test_input)

    label_map = {"sufficient": 0, "insufficient": 1}
    valid_raw_risk = np.array([1.0 - float(record["sufficiency_score"]) for record in valid_records], dtype=float)
    valid_labels = np.array([label_map[record["sufficiency_label"]] for record in valid_records], dtype=int)
    test_raw_risk = np.array([1.0 - float(record["sufficiency_score"]) for record in test_records], dtype=float)
    test_labels = np.array([label_map[record["sufficiency_label"]] for record in test_records], dtype=int)

    calibrator = make_calibrator(args.method)
    calibrator.fit(valid_raw_risk, valid_labels)

    valid_calibrated_risk = calibrator.predict(valid_raw_risk)
    test_calibrated_risk = calibrator.predict(test_raw_risk)

    write_jsonl(args.output_valid_risk, _risk_records(valid_records, valid_raw_risk, valid_calibrated_risk, args.method))
    write_jsonl(args.output_test_risk, _risk_records(test_records, test_raw_risk, test_calibrated_risk, args.method))

    calibrator_path = Path(args.output_calibrator)
    calibrator_path.parent.mkdir(parents=True, exist_ok=True)
    with calibrator_path.open("wb") as f:
        pickle.dump(calibrator, f)

    metadata = {
        "method": args.method,
        "valid_size": len(valid_records),
        "test_size": len(test_records),
        "raw_brier_test": brier_score(test_labels, test_raw_risk),
        "raw_ece_test": ece(test_labels, test_raw_risk),
        "calibrated_brier_test": brier_score(test_labels, test_calibrated_risk),
        "calibrated_ece_test": ece(test_labels, test_calibrated_risk),
    }
    metadata_path = Path(args.output_metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(metadata)


def _risk_records(records, raw_risk, calibrated_risk, method: str):
    output_records = []
    for record, raw_value, calibrated_value in zip(records, raw_risk, calibrated_risk):
        output_records.append(
            {
                "id": record["id"],
                "sufficiency_label": record["sufficiency_label"],
                "sufficiency_score": float(record["sufficiency_score"]),
                "raw_risk_score": float(raw_value),
                "risk_score": float(calibrated_value),
                "calibration_method": method,
            }
        )
    return output_records


if __name__ == "__main__":
    main()
