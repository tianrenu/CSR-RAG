from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np

from run_link_bridge_support_reranker import (
    _candidate_rows,
    _load_records,
    _train_and_score,
    _validate_records,
)


TARGET_VARIANTS = {
    "rf_all": "random_forest_balanced/all/blend1.00",
    "gb_all": "gradient_boosting/all/blend1.00",
    "lr_all": "logistic_regression_balanced/all/blend1.00",
}
TOP_KS = [5, 10, 20]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run no-API multi-seed support reranker stability analysis for fixed RF/GB/LR all/blend1.00 variants."
        )
    )
    parser.add_argument("--split-dir", default="data/processed/hotpotqa_official_intro_link_bridge_splits_top20_full_dev")
    parser.add_argument("--record-kind-filter", default="official_intro_link_bridge_a0p85_p0p00_top20")
    parser.add_argument(
        "--output-dir",
        default="results/tables/hotpotqa_official_intro_link_bridge_support_reranker_multiseed",
    )
    parser.add_argument("--seeds", default="13,21,42,87,100")
    args = parser.parse_args()

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    split_records = _load_records(Path(args.split_dir), args.record_kind_filter)
    _validate_records(split_records)
    candidate_rows = {split: _candidate_rows(records) for split, records in split_records.items()}

    seed_rows = []
    for seed in seeds:
        variants, _ = _train_and_score(candidate_rows, [1.0], seed)
        by_id = {row["variant_id"]: row for row in variants}
        missing = [variant_id for variant_id in TARGET_VARIANTS.values() if variant_id not in by_id]
        if missing:
            raise ValueError(f"Missing expected variants for seed {seed}: {missing}")
        for label, variant_id in TARGET_VARIANTS.items():
            variant = by_id[variant_id]
            row: dict[str, Any] = {
                "seed": seed,
                "variant_label": label,
                "variant_id": variant_id,
                "train_test_gap_top5": float(variant["train_top5_sufficient_rate"])
                - float(variant["test_top5_sufficient_rate"]),
                "valid_test_gap_top5": float(variant["valid_top5_sufficient_rate"])
                - float(variant["test_top5_sufficient_rate"]),
            }
            for split in ("train", "valid", "test"):
                for top_k in TOP_KS:
                    row[f"{split}_top{top_k}"] = float(variant[f"{split}_top{top_k}_sufficient_rate"])
            seed_rows.append(row)
        del variants
        gc.collect()

    aggregate_rows = _aggregate_rows(seed_rows)
    delta_rows = _delta_rows(seed_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "multiseed_variant_metrics.csv", seed_rows)
    _write_csv(output_dir / "multiseed_aggregate_summary.csv", aggregate_rows)
    _write_csv(output_dir / "multiseed_pairwise_deltas.csv", delta_rows)
    (output_dir / "multiseed_reranker_summary.md").write_text(
        _summary_markdown(seeds, aggregate_rows, delta_rows, args),
        encoding="utf-8",
    )
    (output_dir / "validation_summary.json").write_text(
        json.dumps(
            {
                "split_dir": args.split_dir,
                "record_kind_filter": args.record_kind_filter,
                "output_dir": args.output_dir,
                "seeds": seeds,
                "target_variants": TARGET_VARIANTS,
                "blend_alphas": [1.0],
                "selection_note": "Fixed RF/GB/LR all/blend1.00 variants; no test-based variant selection.",
                "uses_llm_api": False,
                "uses_embedding_api": False,
                "outputs": [
                    "multiseed_variant_metrics.csv",
                    "multiseed_aggregate_summary.csv",
                    "multiseed_pairwise_deltas.csv",
                    "multiseed_reranker_summary.md",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "seeds": seeds,
                "uses_llm_api": False,
                "uses_embedding_api": False,
            },
            ensure_ascii=False,
        )
    )


def _aggregate_rows(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    labels = sorted({row["variant_label"] for row in seed_rows})
    metrics = [
        "test_top5",
        "test_top10",
        "test_top20",
        "valid_top5",
        "train_test_gap_top5",
        "valid_test_gap_top5",
    ]
    for label in labels:
        subset = [row for row in seed_rows if row["variant_label"] == label]
        out: dict[str, Any] = {"variant_label": label, "variant_id": subset[0]["variant_id"], "n_seeds": len(subset)}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in subset], dtype=float)
            out[f"{metric}_mean"] = float(values.mean())
            out[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            out[f"{metric}_min"] = float(values.min())
            out[f"{metric}_max"] = float(values.max())
        rows.append(out)
    return rows


def _delta_rows(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_seed = {}
    for row in seed_rows:
        by_seed.setdefault(int(row["seed"]), {})[row["variant_label"]] = row
    pairs = [
        ("rf_vs_gb", "rf_all", "gb_all"),
        ("rf_vs_lr", "rf_all", "lr_all"),
        ("gb_vs_lr", "gb_all", "lr_all"),
    ]
    rows = []
    for comparison, left, right in pairs:
        for top_k in TOP_KS:
            values = []
            for seed, seed_map in sorted(by_seed.items()):
                delta = float(seed_map[left][f"test_top{top_k}"]) - float(seed_map[right][f"test_top{top_k}"])
                values.append(delta)
                rows.append(
                    {
                        "row_type": "seed",
                        "comparison": comparison,
                        "top_k": top_k,
                        "seed": seed,
                        "delta_left_minus_right": delta,
                        "mean_delta": "",
                        "std_delta": "",
                        "min_delta": "",
                        "max_delta": "",
                        "positive_seed_count": "",
                        "n_seeds": "",
                    }
                )
            array = np.asarray(values, dtype=float)
            rows.append(
                {
                    "row_type": "aggregate",
                    "comparison": comparison,
                    "top_k": top_k,
                    "seed": "",
                    "delta_left_minus_right": "",
                    "mean_delta": float(array.mean()),
                    "std_delta": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
                    "min_delta": float(array.min()),
                    "max_delta": float(array.max()),
                    "positive_seed_count": int((array > 0).sum()),
                    "n_seeds": len(array),
                }
            )
    return rows


def _summary_markdown(
    seeds: list[int],
    aggregate_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    aggregate_deltas = [row for row in delta_rows if row["row_type"] == "aggregate"]
    lines = [
        "# Multi-seed Reranker Summary",
        "",
        "This report reruns fixed RF/GB/LR all/blend1.00 rerankers with multiple seeds. It does not call LLM or embedding APIs.",
        "",
        "## Settings",
        "",
        f"- split dir: `{args.split_dir}`",
        f"- record kind: `{args.record_kind_filter}`",
        f"- seeds: `{seeds}`",
        "",
        "## Variant Aggregate Metrics",
        "",
        "| Variant | Test top-5 mean | Test top-5 std | Test top-10 mean | Train-test gap mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['variant_label']} | {_fmt(row['test_top5_mean'])} | {_fmt(row['test_top5_std'])} | "
            f"{_fmt(row['test_top10_mean'])} | {_fmt(row['train_test_gap_top5_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Pairwise Test Deltas",
            "",
            "| Comparison | Top-k | Mean delta | Std | Min | Max | Positive seeds |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate_deltas:
        lines.append(
            f"| {row['comparison']} | {row['top_k']} | {_fmt(row['mean_delta'])} | {_fmt(row['std_delta'])} | "
            f"{_fmt(row['min_delta'])} | {_fmt(row['max_delta'])} | {row['positive_seed_count']}/{row['n_seeds']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "1. A robust reranker advantage should remain positive across seeds and have small seed variance.",
            "2. RF top-5 strength must be weighed against its train-test gap.",
            "3. If RF top-10 deltas are near zero, retrieve-more policies may be less sensitive to RF than top-5 answer-only policies.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
