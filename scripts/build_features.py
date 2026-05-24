from __future__ import annotations

import argparse
from pathlib import Path

from csrrag.features.basic import extract_basic_features
from csrrag.utils.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build basic features for CSR-RAG.")
    parser.add_argument("--input", required=True, help="Path to retrieval JSONL.")
    parser.add_argument("--output", required=True, help="Path to output JSONL.")
    args = parser.parse_args()

    records = read_jsonl(args.input)
    rows = []
    for record in records:
        features = extract_basic_features(record)
        row = {"id": record["id"], **features}
        if "sufficiency_label" in record:
            row["sufficiency_label"] = record["sufficiency_label"]
        rows.append(row)

    write_jsonl(Path(args.output), rows)


if __name__ == "__main__":
    main()
