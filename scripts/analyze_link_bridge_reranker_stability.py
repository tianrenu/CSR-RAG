from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ESTIMATOR_VARIANTS = {
    "original_rank": "original_rank",
    "random_forest_balanced": "random_forest_balanced/all/blend1.00",
    "gradient_boosting": "gradient_boosting/all/blend1.00",
    "logistic_regression_balanced": "logistic_regression_balanced/all/blend1.00",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize link-bridge support reranker stability from existing no-API artifacts."
    )
    parser.add_argument(
        "--input-dir",
        default="results/tables/hotpotqa_official_intro_link_bridge_support_reranker",
    )
    parser.add_argument(
        "--output-dir",
        default="results/tables/hotpotqa_official_intro_link_bridge_support_reranker_stability",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    estimator_rows = [_coerce(row) for row in _read_csv(input_dir / "estimator_ablation.csv")]
    doc_rows = [_coerce(row) for row in _read_csv(input_dir / "doc_prediction_metrics.csv")]
    validation = json.loads((input_dir / "validation_summary.json").read_text(encoding="utf-8"))

    summary_rows = _summary_rows(estimator_rows, doc_rows, validation)
    comparison_rows = _comparison_rows(summary_rows)
    _write_csv(output_dir / "reranker_stability_summary.csv", summary_rows)
    _write_csv(output_dir / "reranker_pairwise_deltas.csv", comparison_rows)
    (output_dir / "reranker_stability_summary.md").write_text(
        _summary_markdown(summary_rows, comparison_rows, validation, args),
        encoding="utf-8",
    )
    (output_dir / "validation_summary.json").write_text(
        json.dumps(
            {
                "input_dir": args.input_dir,
                "output_dir": args.output_dir,
                "source_files": [
                    "estimator_ablation.csv",
                    "doc_prediction_metrics.csv",
                    "validation_summary.json",
                ],
                "split_counts": validation.get("split_counts"),
                "candidate_counts": validation.get("candidate_counts"),
                "uses_llm_api": False,
                "uses_embedding_api": False,
                "outputs": [
                    "reranker_stability_summary.csv",
                    "reranker_pairwise_deltas.csv",
                    "reranker_stability_summary.md",
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
                "variants": [row["variant"] for row in summary_rows],
                "uses_llm_api": False,
                "uses_embedding_api": False,
            },
            ensure_ascii=False,
        )
    )


def _summary_rows(
    estimator_rows: list[dict[str, Any]],
    doc_rows: list[dict[str, Any]],
    validation: dict[str, Any],
) -> list[dict[str, Any]]:
    by_variant = {row["selected_variant"]: row for row in estimator_rows}
    by_doc = {
        (row["variant_id"], row["split"]): row
        for row in doc_rows
    }
    original = by_variant[ESTIMATOR_VARIANTS["original_rank"]]
    test_n = int(validation.get("split_counts", {}).get("test", 0))
    rows = []
    for label, variant in ESTIMATOR_VARIANTS.items():
        row = by_variant[variant]
        train = row["train_top5_sufficient_rate"]
        valid = row["valid_top5_sufficient_rate"]
        test = row["test_top5_sufficient_rate"]
        doc_train = by_doc.get((variant, "train"), {})
        doc_valid = by_doc.get((variant, "valid"), {})
        doc_test = by_doc.get((variant, "test"), {})
        rows.append(
            {
                "variant": variant,
                "estimator_label": label,
                "train_top5": train,
                "valid_top5": valid,
                "test_top5": test,
                "test_top10": row["test_top10_sufficient_rate"],
                "test_top20": row["test_top20_sufficient_rate"],
                "test_gain_vs_original": test - original["test_top5_sufficient_rate"],
                "approx_test_question_gain_vs_original": round(
                    (test - original["test_top5_sufficient_rate"]) * test_n
                ),
                "train_test_gap": train - test,
                "train_valid_gap": train - valid,
                "doc_train_auprc": doc_train.get("auprc", ""),
                "doc_valid_auprc": doc_valid.get("auprc", ""),
                "doc_test_auprc": doc_test.get("auprc", ""),
                "doc_train_test_auprc_gap": _maybe_diff(doc_train.get("auprc", ""), doc_test.get("auprc", "")),
                "doc_test_auroc": doc_test.get("auroc", ""),
            }
        )
    return rows


def _comparison_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_label = {row["estimator_label"]: row for row in summary_rows}
    rf = by_label["random_forest_balanced"]
    pairs = [
        ("rf_vs_gradient_boosting", rf, by_label["gradient_boosting"]),
        ("rf_vs_logistic_regression", rf, by_label["logistic_regression_balanced"]),
        ("gb_vs_logistic_regression", by_label["gradient_boosting"], by_label["logistic_regression_balanced"]),
    ]
    rows = []
    for name, left, right in pairs:
        rows.append(
            {
                "comparison": name,
                "left_variant": left["variant"],
                "right_variant": right["variant"],
                "delta_test_top5": left["test_top5"] - right["test_top5"],
                "delta_test_top10": left["test_top10"] - right["test_top10"],
                "delta_train_test_gap": left["train_test_gap"] - right["train_test_gap"],
                "delta_doc_train_test_auprc_gap": _maybe_diff(
                    left["doc_train_test_auprc_gap"],
                    right["doc_train_test_auprc_gap"],
                ),
            }
        )
    return rows


def _summary_markdown(
    summary_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    validation: dict[str, Any],
    args: argparse.Namespace,
) -> str:
    lines = [
        "# Link-bridge Support Reranker Stability",
        "",
        "This report summarizes existing reranker artifacts only. It does not call LLM or embedding APIs.",
        "",
        "## Scope",
        "",
        f"- Input dir: `{args.input_dir}`",
        f"- Split counts: `{validation.get('split_counts')}`",
        f"- Candidate counts: `{validation.get('candidate_counts')}`",
        "",
        "## Estimator Summary",
        "",
        "| Variant | Train top-5 | Valid top-5 | Test top-5 | Test top-10 | Gain vs orig. | Train-test gap | Doc test AUPRC | Doc AUPRC gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['variant']} | {_fmt(row['train_top5'])} | {_fmt(row['valid_top5'])} | "
            f"{_fmt(row['test_top5'])} | {_fmt(row['test_top10'])} | {_fmt(row['test_gain_vs_original'])} | "
            f"{_fmt(row['train_test_gap'])} | {_fmt(row['doc_test_auprc'])} | {_fmt(row['doc_train_test_auprc_gap'])} |"
        )
    lines.extend(
        [
            "",
            "## Pairwise Deltas",
            "",
            "| Comparison | Delta test top-5 | Delta test top-10 | Delta train-test gap |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in comparison_rows:
        lines.append(
            f"| {row['comparison']} | {_fmt(row['delta_test_top5'])} | "
            f"{_fmt(row['delta_test_top10'])} | {_fmt(row['delta_train_test_gap'])} |"
        )
    lines.extend(
        [
            "",
            "## Research Judgment",
            "",
            "1. RF/all/blend1.00 remains the strongest current reranker by test top-5 sufficiency.",
            "2. RF has a clear train-test gap; it should be treated as the strongest working variant, not a frozen final method.",
            "3. GB/LR have slightly weaker top-5 sufficiency but much cleaner train-test behavior, so they should remain mandatory robustness baselines.",
            "4. Next no-API work should add paired question-level RF/GB/LR details, multi-seed reruns, and bootstrap confidence intervals.",
            "",
        ]
    )
    return "\n".join(lines)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _coerce(row: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if value == "":
            out[key] = value
            continue
        try:
            out[key] = float(value)
        except ValueError:
            out[key] = value
    return out


def _maybe_diff(left: Any, right: Any) -> float | str:
    if left == "" or right == "":
        return ""
    return float(left) - float(right)


def _fmt(value: Any) -> str:
    if value == "":
        return ""
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
