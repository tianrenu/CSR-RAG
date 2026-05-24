from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from run_link_bridge_retrieve_more_experiments import (
    _bootstrap_rows,
    _case_rows,
    _feature_record,
    _feature_sets,
    _main_rows,
    _policy_rows,
    _risk_runs,
    _split_valid_ids,
    _topk_rows,
    _validate_features,
    _validate_records as _validate_topk_records,
    _write_csv,
    _write_summary,
)
from run_link_bridge_support_reranker import (
    _candidate_rows,
    _load_records,
    _reranked_split_records,
    _train_and_score,
    _validate_records as _validate_reranker_records,
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
            "Run no-API downstream policy sensitivity analysis for fixed RF/GB/LR support rerankers. "
            "The script generates reranked top-k records in memory and selects policies on valid only."
        )
    )
    parser.add_argument("--split-dir", default="data/processed/hotpotqa_official_intro_link_bridge_splits_top20_full_dev")
    parser.add_argument("--record-kind-filter", default="official_intro_link_bridge_a0p85_p0p00_top20")
    parser.add_argument(
        "--output-dir",
        default="results/tables/hotpotqa_official_intro_link_bridge_support_reranker_policy_sensitivity",
    )
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_records = _load_records(Path(args.split_dir), args.record_kind_filter)
    _validate_reranker_records(split_records)
    candidate_rows = {split: _candidate_rows(records) for split, records in split_records.items()}
    variants, predictions = _train_and_score(candidate_rows, [1.0], args.seed)
    variants_by_id = {row["variant_id"]: row for row in variants}
    missing = [variant_id for variant_id in TARGET_VARIANTS.values() if variant_id not in variants_by_id]
    if missing:
        raise ValueError(f"Missing expected variants: {missing}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = output_dir / "variant_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)

    combined_main_rows = []
    combined_policy_rows = []
    combined_prediction_rows = []
    combined_topk_rows = []
    combined_bootstrap_rows = []
    combined_case_rows = []
    validation_by_variant = {}

    for variant_label, variant_id in TARGET_VARIANTS.items():
        variant = variants_by_id[variant_id]
        reranked_records = _reranked_split_records(split_records, predictions[variant_id], TOP_KS, variant)
        records = _records_by_topk(reranked_records)
        _validate_topk_records(records)

        valid_calib_ids, valid_policy_ids = _split_valid_ids(records[5]["valid"])
        features = {
            top_k: {
                split: [_feature_record(record) for record in split_rows]
                for split, split_rows in topk_records.items()
            }
            for top_k, topk_records in records.items()
        }
        feature_sets = _feature_sets()
        _validate_features(features, feature_sets)

        topk_rows = _with_variant(variant_label, variant_id, _topk_rows(records))
        prediction_rows, risks = _risk_runs(features, valid_calib_ids, feature_sets)
        policy_rows, selections = _policy_rows(records, risks, valid_policy_ids)
        main_rows, main_keys = _main_rows(policy_rows, selections)
        case_rows = _case_rows(records, selections, main_keys)
        bootstrap_rows = _bootstrap_rows(selections, main_keys, args.bootstrap_iters, args.seed)

        combined_topk_rows.extend(topk_rows)
        combined_prediction_rows.extend(_with_variant(variant_label, variant_id, prediction_rows))
        combined_policy_rows.extend(_with_variant(variant_label, variant_id, policy_rows))
        combined_main_rows.extend(_with_variant(variant_label, variant_id, main_rows))
        combined_case_rows.extend(_with_variant(variant_label, variant_id, case_rows))
        combined_bootstrap_rows.extend(_with_variant(variant_label, variant_id, bootstrap_rows))
        _write_summary(summary_dir / f"{variant_label}_retrieve_more_summary.md", main_rows, _topk_rows(records))

        validation_by_variant[variant_label] = {
            "variant_id": variant_id,
            "split_counts": {
                top_k: {split: len(rows) for split, rows in topk_records.items()}
                for top_k, topk_records in records.items()
            },
            "valid_calib_count": len(valid_calib_ids),
            "valid_policy_count": len(valid_policy_ids),
            "risk_runs": len(risks),
            "policy_selections": len(selections),
            "main_rows": len(main_rows),
        }

    _write_csv(output_dir / "policy_sensitivity_main.csv", combined_main_rows)
    _write_csv(output_dir / "policy_sensitivity_policy_comparison.csv", combined_policy_rows)
    _write_csv(output_dir / "policy_sensitivity_prediction_metrics.csv", combined_prediction_rows)
    _write_csv(output_dir / "policy_sensitivity_topk.csv", combined_topk_rows)
    _write_csv(output_dir / "policy_sensitivity_case_studies.csv", combined_case_rows)
    _write_csv(output_dir / "policy_sensitivity_bootstrap_ci.csv", combined_bootstrap_rows)
    _write_csv(output_dir / "policy_sensitivity_key_deltas.csv", _key_delta_rows(combined_main_rows))
    (output_dir / "policy_sensitivity_summary.md").write_text(
        _summary_markdown(combined_main_rows, combined_topk_rows, args),
        encoding="utf-8",
    )
    (output_dir / "validation_summary.json").write_text(
        json.dumps(
            {
                "split_dir": args.split_dir,
                "record_kind_filter": args.record_kind_filter,
                "target_variants": TARGET_VARIANTS,
                "top_ks": TOP_KS,
                "seed": args.seed,
                "bootstrap_iters": args.bootstrap_iters,
                "variant_validation": validation_by_variant,
                "selection_protocol": (
                    "Train fixed reranker variants on train; generate reranked top-k records in memory; "
                    "calibrate risk on valid_calib; select thresholds on valid_policy; report test only."
                ),
                "writes_large_data_files": False,
                "uses_embedding_api": False,
                "uses_llm_api": False,
                "outputs": [
                    "policy_sensitivity_main.csv",
                    "policy_sensitivity_policy_comparison.csv",
                    "policy_sensitivity_prediction_metrics.csv",
                    "policy_sensitivity_topk.csv",
                    "policy_sensitivity_case_studies.csv",
                    "policy_sensitivity_bootstrap_ci.csv",
                    "policy_sensitivity_key_deltas.csv",
                    "policy_sensitivity_summary.md",
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
                "variants": TARGET_VARIANTS,
                "main_rows": len(combined_main_rows),
                "uses_embedding_api": False,
                "uses_llm_api": False,
                "writes_large_data_files": False,
            },
            ensure_ascii=False,
        )
    )


def _records_by_topk(reranked_records: dict[str, list[dict[str, Any]]]) -> dict[int, dict[str, list[dict[str, Any]]]]:
    records = {top_k: {"train": [], "valid": [], "test": []} for top_k in TOP_KS}
    for split, rows in reranked_records.items():
        for record in rows:
            top_k = int(record["metadata"]["top_k"])
            if top_k in records:
                records[top_k][split].append(record)
    return records


def _with_variant(variant_label: str, variant_id: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        out = {"reranker_label": variant_label, "reranker_variant_id": variant_id}
        out.update(row)
        output.append(out)
    return output


def _key_delta_rows(main_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_variant_policy = {
        (row["reranker_label"], row["policy"], str(row["target_k"])): row
        for row in main_rows
    }
    rows = []
    for reranker_label in sorted({row["reranker_label"] for row in main_rows}):
        top5 = by_variant_policy.get((reranker_label, "always_answer", "5"))
        for policy in ("balanced", "retrieve_more_balanced", "retrieve_more_risk_control@suff_abstain15", "retrieve_more@cov85"):
            candidates = [
                row
                for row in main_rows
                if row["reranker_label"] == reranker_label
                and row["policy"] == policy
            ]
            for row in candidates:
                out = {
                    "reranker_label": reranker_label,
                    "reranker_variant_id": row["reranker_variant_id"],
                    "policy": policy,
                    "method_name": row["method_name"],
                    "target_k": row["target_k"],
                    "test_coverage": row["test_coverage"],
                    "test_selective_accuracy": row["test_selective_accuracy"],
                    "test_insufficient_answer_rate": row["test_insufficient_answer_rate"],
                    "test_sufficient_abstain_rate": row["test_sufficient_abstain_rate"],
                    "test_false_answer_count": row["test_false_answer_count"],
                    "test_over_abstain_count": row["test_over_abstain_count"],
                    "test_retrieval_rate": row["test_retrieval_rate"],
                }
                if top5 is not None:
                    out["delta_iar_vs_always_top5"] = float(row["test_insufficient_answer_rate"]) - float(top5["test_insufficient_answer_rate"])
                    out["delta_false_answers_vs_always_top5"] = int(row["test_false_answer_count"]) - int(top5["test_false_answer_count"])
                    out["delta_coverage_vs_always_top5"] = float(row["test_coverage"]) - float(top5["test_coverage"])
                rows.append(out)
    return rows


def _summary_markdown(
    main_rows: list[dict[str, Any]],
    topk_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    lines = [
        "# Reranker Policy Sensitivity Summary",
        "",
        "This no-API diagnostic compares RF/GB/LR support rerankers after the downstream answer/retrieve-more/abstain policy stage.",
        "",
        "## Settings",
        "",
        f"- split dir: `{args.split_dir}`",
        f"- record kind: `{args.record_kind_filter}`",
        f"- bootstrap iters: `{args.bootstrap_iters}`",
        "- writes large data files: `false`",
        "- uses LLM API: `false`",
        "- uses embedding API: `false`",
        "",
        "## Test Top-k Sufficiency",
        "",
        "| Reranker | Top-5 | Top-10 | Top-20 |",
        "|---|---:|---:|---:|",
    ]
    for reranker_label in sorted({row["reranker_label"] for row in topk_rows}):
        cells = []
        for top_k in TOP_KS:
            match = next(
                row
                for row in topk_rows
                if row["reranker_label"] == reranker_label
                and row["split"] == "test"
                and int(row["top_k"]) == top_k
            )
            cells.append(_fmt(match["sufficient_rate"]))
        lines.append(f"| {reranker_label} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines.extend(
        [
            "",
            "## Main Policy Rows",
            "",
            "| Reranker | Policy | Target k | Coverage | Selective acc. | IAR | SAR | Retrieval rate |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    priority = {
        "always_answer": 0,
        "balanced": 1,
        "retrieve_more_balanced": 2,
        "retrieve_more_risk_control@suff_abstain15": 3,
        "retrieve_more@cov85": 4,
    }
    for row in sorted(main_rows, key=lambda item: (item["reranker_label"], priority.get(item["policy"], 99), int(item["target_k"]))):
        if row["policy"] == "always_answer" and int(row["target_k"]) not in (5, 20):
            continue
        lines.append(
            f"| {row['reranker_label']} | `{row['policy']}` | {row['target_k']} | "
            f"{_fmt(row['test_coverage'])} | {_fmt(row['test_selective_accuracy'])} | "
            f"{_fmt(row['test_insufficient_answer_rate'])} | {_fmt(row['test_sufficient_abstain_rate'])} | "
            f"{_fmt(row['test_retrieval_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "1. If policy metrics are close across RF/GB/LR, the downstream policy is robust to reranker choice.",
            "2. RF can remain the strongest top-5 reranker while retrieve-more narrows downstream differences.",
            "3. A lower IAR is not enough by itself; coverage and sufficient abstain rate must be checked together.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
