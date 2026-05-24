from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from run_link_bridge_support_reranker import (
    _candidate_rows,
    _load_records,
    _normalize_title,
    _rank_docs,
    _train_and_score,
    _validate_records,
)


VARIANTS = {
    "original_rank": "original_rank",
    "rf_all": "random_forest_balanced/all/blend1.00",
    "gb_all": "gradient_boosting/all/blend1.00",
    "lr_all": "logistic_regression_balanced/all/blend1.00",
}
TOP_KS = [5, 10, 20]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build paired question-level RF/GB/LR reranker details and bootstrap CIs. "
            "This script retrains the no-API support reranker with existing records and does not call LLM or embedding APIs."
        )
    )
    parser.add_argument("--split-dir", default="data/processed/hotpotqa_official_intro_link_bridge_splits_top20_full_dev")
    parser.add_argument("--record-kind-filter", default="official_intro_link_bridge_a0p85_p0p00_top20")
    parser.add_argument(
        "--output-dir",
        default="results/tables/hotpotqa_official_intro_link_bridge_support_reranker_paired_bootstrap",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iters", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260524)
    args = parser.parse_args()

    split_records = _load_records(Path(args.split_dir), args.record_kind_filter)
    _validate_records(split_records)
    candidate_rows = {split: _candidate_rows(records) for split, records in split_records.items()}
    variants, predictions = _train_and_score(candidate_rows, [1.0], args.seed)
    variant_ids = _validate_variant_ids(variants)

    details = _paired_details(split_records["test"], predictions, variant_ids)
    variant_summary = _variant_summary(details)
    pairwise = _pairwise_rows(details)
    bootstrap = _bootstrap_rows(details, args.bootstrap_iters, args.bootstrap_seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "paired_question_details.csv", details)
    _write_csv(output_dir / "paired_variant_summary.csv", variant_summary)
    _write_csv(output_dir / "paired_pairwise_deltas.csv", pairwise)
    _write_csv(output_dir / "paired_bootstrap_ci.csv", bootstrap)
    (output_dir / "paired_reranker_bootstrap_summary.md").write_text(
        _summary_markdown(variant_summary, pairwise, bootstrap, args),
        encoding="utf-8",
    )
    (output_dir / "validation_summary.json").write_text(
        json.dumps(
            {
                "split_dir": args.split_dir,
                "record_kind_filter": args.record_kind_filter,
                "output_dir": args.output_dir,
                "seed": args.seed,
                "bootstrap_iters": args.bootstrap_iters,
                "bootstrap_seed": args.bootstrap_seed,
                "test_question_count": len(split_records["test"]),
                "variants": VARIANTS,
                "selection_note": "Uses fixed all/blend1.00 variants for paired stability analysis; no test tuning.",
                "uses_llm_api": False,
                "uses_embedding_api": False,
                "outputs": [
                    "paired_question_details.csv",
                    "paired_variant_summary.csv",
                    "paired_pairwise_deltas.csv",
                    "paired_bootstrap_ci.csv",
                    "paired_reranker_bootstrap_summary.md",
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
                "test_question_count": len(split_records["test"]),
                "bootstrap_iters": args.bootstrap_iters,
                "uses_llm_api": False,
                "uses_embedding_api": False,
            },
            ensure_ascii=False,
        )
    )


def _validate_variant_ids(variants: list[dict[str, Any]]) -> dict[str, str]:
    available = {row["variant_id"] for row in variants}
    missing = [variant_id for label, variant_id in VARIANTS.items() if label != "original_rank" and variant_id not in available]
    if missing:
        raise ValueError(f"Missing expected variants: {missing}")
    return dict(VARIANTS)


def _paired_details(
    test_records: list[dict[str, Any]],
    predictions: dict[str, dict[str, dict[str, list[float]]]],
    variant_ids: dict[str, str],
) -> list[dict[str, Any]]:
    rows = []
    for record in test_records:
        original_id = record["metadata"]["original_id"]
        support_titles = set(_normalize_title(title) for title in record["metadata"]["support_titles"])
        row: dict[str, Any] = {
            "original_id": original_id,
            "question": record["query"],
            "gold_answer": record["gold_answer"],
            "support_titles": " || ".join(record["metadata"]["support_titles"]),
            "support_title_count": len(support_titles),
        }
        for label, variant_id in variant_ids.items():
            if label == "original_rank":
                ranked_docs = list(record["retrieved_docs"])
            else:
                ranked_docs = _rank_docs(record["retrieved_docs"], predictions[variant_id]["test"][original_id])
            for top_k in TOP_KS:
                titles = {_normalize_title(doc.get("title", "")) for doc in ranked_docs[:top_k]}
                row[f"{label}_top{top_k}_sufficient"] = int(support_titles.issubset(titles))
                row[f"{label}_top{top_k}_coverage"] = _support_coverage(support_titles, titles)
            row[f"{label}_top5_titles"] = " || ".join(doc["title"] for doc in ranked_docs[:5])
        rows.append(row)
    return rows


def _variant_summary(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for label, variant_id in VARIANTS.items():
        row: dict[str, Any] = {"variant_label": label, "variant_id": variant_id, "n": len(details)}
        for top_k in TOP_KS:
            values = np.asarray([int(item[f"{label}_top{top_k}_sufficient"]) for item in details], dtype=float)
            row[f"top{top_k}_sufficient_count"] = int(values.sum())
            row[f"top{top_k}_sufficient_rate"] = float(values.mean()) if len(values) else 0.0
        rows.append(row)
    return rows


def _pairwise_rows(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = [
        ("rf_vs_original", "rf_all", "original_rank"),
        ("gb_vs_original", "gb_all", "original_rank"),
        ("lr_vs_original", "lr_all", "original_rank"),
        ("rf_vs_gb", "rf_all", "gb_all"),
        ("rf_vs_lr", "rf_all", "lr_all"),
        ("gb_vs_lr", "gb_all", "lr_all"),
    ]
    rows = []
    for comparison, left, right in pairs:
        for top_k in TOP_KS:
            left_values = np.asarray([int(row[f"{left}_top{top_k}_sufficient"]) for row in details], dtype=int)
            right_values = np.asarray([int(row[f"{right}_top{top_k}_sufficient"]) for row in details], dtype=int)
            left_only = int(((left_values == 1) & (right_values == 0)).sum())
            right_only = int(((left_values == 0) & (right_values == 1)).sum())
            rows.append(
                {
                    "comparison": comparison,
                    "top_k": top_k,
                    "left_variant": left,
                    "right_variant": right,
                    "left_sufficient_count": int(left_values.sum()),
                    "right_sufficient_count": int(right_values.sum()),
                    "left_sufficient_rate": float(left_values.mean()),
                    "right_sufficient_rate": float(right_values.mean()),
                    "delta_left_minus_right": float(left_values.mean() - right_values.mean()),
                    "left_only_count": left_only,
                    "right_only_count": right_only,
                    "both_sufficient_count": int(((left_values == 1) & (right_values == 1)).sum()),
                    "both_insufficient_count": int(((left_values == 0) & (right_values == 0)).sum()),
                    "net_left_gain_count": left_only - right_only,
                }
            )
    return rows


def _bootstrap_rows(details: list[dict[str, Any]], iters: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    n = len(details)
    pairs = [
        ("rf_vs_original", "rf_all", "original_rank"),
        ("gb_vs_original", "gb_all", "original_rank"),
        ("lr_vs_original", "lr_all", "original_rank"),
        ("rf_vs_gb", "rf_all", "gb_all"),
        ("rf_vs_lr", "rf_all", "lr_all"),
        ("gb_vs_lr", "gb_all", "lr_all"),
    ]
    rows = []
    for comparison, left, right in pairs:
        for top_k in TOP_KS:
            left_values = np.asarray([int(row[f"{left}_top{top_k}_sufficient"]) for row in details], dtype=float)
            right_values = np.asarray([int(row[f"{right}_top{top_k}_sufficient"]) for row in details], dtype=float)
            diffs = left_values - right_values
            estimate = float(diffs.mean())
            sampled = np.empty(iters, dtype=float)
            for idx in range(iters):
                sample_idx = rng.integers(0, n, size=n)
                sampled[idx] = float(diffs[sample_idx].mean())
            rows.append(
                {
                    "comparison": comparison,
                    "top_k": top_k,
                    "estimate_delta": estimate,
                    "ci95_low": float(np.quantile(sampled, 0.025)),
                    "ci95_high": float(np.quantile(sampled, 0.975)),
                    "p_delta_le_0": float(np.mean(sampled <= 0.0)),
                    "bootstrap_iters": iters,
                    "bootstrap_seed": seed,
                }
            )
    return rows


def _summary_markdown(
    variant_summary: list[dict[str, Any]],
    pairwise: list[dict[str, Any]],
    bootstrap: list[dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    lines = [
        "# Paired Reranker Bootstrap Summary",
        "",
        "This report uses existing link-bridge records and no-API reranker training. It does not call LLM or embedding APIs.",
        "",
        "## Settings",
        "",
        f"- split dir: `{args.split_dir}`",
        f"- record kind: `{args.record_kind_filter}`",
        f"- reranker seed: `{args.seed}`",
        f"- bootstrap iters: `{args.bootstrap_iters}`",
        "",
        "## Variant Summary",
        "",
        "| Variant | Top-5 | Top-10 | Top-20 |",
        "|---|---:|---:|---:|",
    ]
    for row in variant_summary:
        lines.append(
            f"| {row['variant_label']} | {_fmt(row['top5_sufficient_rate'])} | "
            f"{_fmt(row['top10_sufficient_rate'])} | {_fmt(row['top20_sufficient_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Key Pairwise Deltas",
            "",
            "| Comparison | Top-k | Delta | Left-only | Right-only | Net left gain |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in pairwise:
        if int(row["top_k"]) not in {5, 10}:
            continue
        lines.append(
            f"| {row['comparison']} | {row['top_k']} | {_fmt(row['delta_left_minus_right'])} | "
            f"{row['left_only_count']} | {row['right_only_count']} | {row['net_left_gain_count']} |"
        )
    lines.extend(
        [
            "",
            "## Bootstrap CI",
            "",
            "| Comparison | Top-k | Estimate | 95% low | 95% high | P(delta <= 0) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in bootstrap:
        if int(row["top_k"]) not in {5, 10}:
            continue
        lines.append(
            f"| {row['comparison']} | {row['top_k']} | {_fmt(row['estimate_delta'])} | "
            f"{_fmt(row['ci95_low'])} | {_fmt(row['ci95_high'])} | {_fmt(row['p_delta_le_0'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "1. Use top-5 deltas to judge whether a reranker moves complete support chains into the answer context.",
            "2. Use top-10 deltas to check whether differences remain after a moderate retrieve-more expansion.",
            "3. If a 95% CI crosses 0, treat the paired advantage as not yet stable enough for a strong paper claim.",
            "4. This analysis still needs multi-seed reruns before freezing the final reranker choice.",
            "",
        ]
    )
    return "\n".join(lines)


def _support_coverage(support_titles: set[str], retrieved_titles: set[str]) -> float:
    if not support_titles:
        return 0.0
    return len(support_titles & retrieved_titles) / len(support_titles)


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
