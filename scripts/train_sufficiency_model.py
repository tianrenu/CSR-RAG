from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from csrrag.models.baseline import train_logistic_regression
from csrrag.utils.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a train-only sufficiency model and score valid/test.")
    parser.add_argument("--train-input", required=True, help="Train feature JSONL.")
    parser.add_argument("--valid-input", required=True, help="Validation feature JSONL.")
    parser.add_argument("--test-input", required=True, help="Test feature JSONL.")
    parser.add_argument("--output-model", required=True, help="Path to output pickled model artifact.")
    parser.add_argument("--output-model-meta", required=True, help="Path to output model metadata JSON.")
    parser.add_argument("--output-valid-scores", required=True, help="Path to output validation score JSONL.")
    parser.add_argument("--output-test-scores", required=True, help="Path to output test score JSONL.")
    parser.add_argument(
        "--label-column",
        default="sufficiency_label",
        help="Column name for binary labels: sufficient / insufficient.",
    )
    args = parser.parse_args()

    label_map = {"sufficient": 1, "insufficient": 0}
    train_records = _load_feature_records(args.train_input, args.label_column, label_map)
    valid_records = _load_feature_records(args.valid_input, args.label_column, label_map)
    test_records = _load_feature_records(args.test_input, args.label_column, label_map)

    feature_names = [
        key
        for key in train_records[0].keys()
        if key not in {"id", args.label_column}
    ]

    model = train_logistic_regression(
        rows=[_feature_row(record, feature_names) for record in train_records],
        labels=[label_map[record[args.label_column]] for record in train_records],
        feature_names=feature_names,
    )

    _write_score_records(valid_records, model, feature_names, args.output_valid_scores, args.label_column)
    _write_score_records(test_records, model, feature_names, args.output_test_scores, args.label_column)

    model_path = Path(args.output_model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as f:
        pickle.dump(model, f)

    meta = {
        "model_name": "sklearn_logistic_regression_v1",
        "feature_names": feature_names,
        "train_size": len(train_records),
        "valid_size": len(valid_records),
        "test_size": len(test_records),
    }
    meta_path = Path(args.output_model_meta)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(meta)


def _load_feature_records(path: str, label_column: str, label_map: dict[str, int]) -> list[dict[str, object]]:
    records = read_jsonl(path)
    filtered = [record for record in records if record.get(label_column) in label_map]
    if not filtered:
        raise ValueError(f"No labeled records found in {path}")
    return filtered


def _feature_row(record: dict[str, object], feature_names: list[str]) -> dict[str, float]:
    return {name: float(record[name]) for name in feature_names}


def _write_score_records(
    records: list[dict[str, object]],
    model,
    feature_names: list[str],
    output_path: str,
    label_column: str,
) -> None:
    feature_rows = [_feature_row(record, feature_names) for record in records]
    probs = model.predict_proba(feature_rows)
    score_records = []
    for record, prob in zip(records, probs):
        score_records.append(
            {
                "id": record["id"],
                "sufficiency_label": record[label_column],
                "sufficiency_score": float(prob),
                "predicted_label": "sufficient" if prob >= 0.5 else "insufficient",
            }
        )
    write_jsonl(output_path, score_records)


if __name__ == "__main__":
    main()
