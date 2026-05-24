from __future__ import annotations

import argparse
from pathlib import Path

from csrrag.utils.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize raw CSR-RAG samples.")
    parser.add_argument("--input", required=True, help="Path to raw JSONL samples.")
    parser.add_argument("--output", required=True, help="Path to normalized JSONL samples.")
    args = parser.parse_args()

    records = read_jsonl(args.input)
    normalized = []
    for i, record in enumerate(records, start=1):
        normalized.append(
            {
                "id": record.get("id", f"sample_{i:06d}"),
                "query": record["query"],
                "gold_answer": record.get("gold_answer", ""),
                "dataset": record.get("dataset", "unknown"),
                "metadata": record.get("metadata", {}),
            }
        )

    write_jsonl(Path(args.output), normalized)


if __name__ == "__main__":
    main()
