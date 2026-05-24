from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from csrrag.utils.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Split retrieval records by original question id.")
    parser.add_argument("--input", required=True, help="Path to retrieval JSONL.")
    parser.add_argument("--output-dir", required=True, help="Directory for train/valid/test JSONL files.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for grouped split.")
    args = parser.parse_args()

    records = read_jsonl(args.input)
    grouped = defaultdict(list)
    for record in records:
        original_id = record["metadata"]["original_id"]
        grouped[original_id].append(record)

    original_ids = list(grouped.keys())
    rng = random.Random(args.seed)
    rng.shuffle(original_ids)

    total = len(original_ids)
    train_count = int(total * 0.70)
    valid_count = int(total * 0.15)
    test_count = total - train_count - valid_count

    split_ids = {
        "train": set(original_ids[:train_count]),
        "valid": set(original_ids[train_count:train_count + valid_count]),
        "test": set(original_ids[train_count + valid_count:train_count + valid_count + test_count]),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, ids in split_ids.items():
        split_records = []
        for original_id in ids:
            split_records.extend(grouped[original_id])
        split_records.sort(key=lambda record: record["id"])
        write_jsonl(output_dir / f"{split_name}.jsonl", split_records)

    print(
        {
            "input_records": len(records),
            "original_question_count": total,
            "train_original_ids": len(split_ids["train"]),
            "valid_original_ids": len(split_ids["valid"]),
            "test_original_ids": len(split_ids["test"]),
            "output_dir": str(output_dir),
        }
    )


if __name__ == "__main__":
    main()
