from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FLOAT_FIELDS = {
    "coverage",
    "answered_em",
    "answered_f1",
    "answered_sufficient_rate",
    "decision_insufficient_answer_rate",
    "insufficient_substantive_answer_rate",
    "sufficient_abstain_rate",
    "tau_answer_top5",
    "tau_answer_after_more",
}

INT_FIELDS = {
    "target_k",
    "n",
    "answered_count",
    "false_answer_count",
    "insufficient_substantive_count",
    "wrong_substantive_count",
    "over_abstain_count",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare link-bridge QA policies across stratified and natural QA samples. "
            "This script only reads existing QA artifacts and does not call LLM or embedding APIs."
        )
    )
    parser.add_argument(
        "--stratified-dir",
        default="results/tables/hotpotqa_official_intro_link_bridge_support_reranker_qa_eval",
    )
    parser.add_argument(
        "--natural-dir",
        default="results/tables/hotpotqa_official_intro_link_bridge_support_reranker_qa_eval_natural120",
    )
    parser.add_argument(
        "--output-dir",
        default="results/tables/hotpotqa_official_intro_link_bridge_qa_aware_policy_comparison",
    )
    args = parser.parse_args()

    runs = [
        _load_run("stratified_qa90", Path(args.stratified_dir)),
        _load_run("natural_qa120", Path(args.natural_dir)),
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_rows = _combined_rows(runs)
    robustness_rows = _robustness_rows(combined_rows)
    recommendations = _recommendations(combined_rows, robustness_rows)

    _write_csv(output_dir / "combined_policy_metrics.csv", combined_rows)
    _write_csv(output_dir / "policy_robustness_summary.csv", robustness_rows)
    (output_dir / "qa_aware_policy_recommendations.json").write_text(
        json.dumps(recommendations, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "qa_aware_policy_summary.md").write_text(
        _summary_markdown(combined_rows, robustness_rows, recommendations, runs),
        encoding="utf-8",
    )
    (output_dir / "validation_summary.json").write_text(
        json.dumps(
            {
                "stratified_dir": args.stratified_dir,
                "natural_dir": args.natural_dir,
                "output_dir": args.output_dir,
                "sample_runs": [
                    {
                        "sample_name": run["sample_name"],
                        "qa_dir": str(run["qa_dir"]),
                        "selected_count": run["validation"].get("selected_count"),
                        "completed_count": run["validation"].get("completed_count"),
                        "sample_counts": run["validation"].get("sample_counts"),
                        "uses_llm_api_source": run["validation"].get("uses_llm_api"),
                        "uses_embedding_api_source": run["validation"].get("uses_embedding_api"),
                    }
                    for run in runs
                ],
                "uses_llm_api": False,
                "uses_embedding_api": False,
                "input_tables": [
                    str(run["qa_dir"] / "link_bridge_qa_policy_comparison.csv")
                    for run in runs
                ],
                "outputs": [
                    "combined_policy_metrics.csv",
                    "policy_robustness_summary.csv",
                    "qa_aware_policy_recommendations.json",
                    "qa_aware_policy_summary.md",
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
                "policies": sorted({row["policy"] for row in combined_rows}),
                "risk_first_policy": recommendations["risk_first"].get("policy"),
                "f1_first_policy": recommendations["f1_first"].get("policy"),
                "uses_llm_api": False,
                "uses_embedding_api": False,
            },
            ensure_ascii=False,
        )
    )


def _load_run(sample_name: str, qa_dir: Path) -> dict[str, Any]:
    table_path = qa_dir / "link_bridge_qa_policy_comparison.csv"
    validation_path = qa_dir / "validation_summary.json"
    if not table_path.exists():
        raise FileNotFoundError(table_path)
    if not validation_path.exists():
        raise FileNotFoundError(validation_path)
    return {
        "sample_name": sample_name,
        "qa_dir": qa_dir,
        "rows": [_coerce_row(row) for row in _read_csv(table_path)],
        "validation": json.loads(validation_path.read_text(encoding="utf-8")),
    }


def _coerce_row(row: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in row.items():
        if key in FLOAT_FIELDS:
            result[key] = _float_or_blank(value)
        elif key in INT_FIELDS:
            result[key] = int(float(value)) if value not in {"", None} else ""
        else:
            result[key] = value
    return result


def _combined_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        by_policy = {row["policy"]: row for row in run["rows"]}
        top5 = by_policy["naive_top5"]
        top20 = by_policy["naive_top20"]
        for row in run["rows"]:
            out = {
                "sample": run["sample_name"],
                "policy": row["policy"],
                "action_type": row["action_type"],
                "target_k": row["target_k"],
                "n": row["n"],
                "coverage": row["coverage"],
                "answered_f1": row["answered_f1"],
                "answered_em": row["answered_em"],
                "answered_sufficient_rate": row["answered_sufficient_rate"],
                "insufficient_substantive_answer_rate": row["insufficient_substantive_answer_rate"],
                "decision_insufficient_answer_rate": row["decision_insufficient_answer_rate"],
                "sufficient_abstain_rate": row["sufficient_abstain_rate"],
                "false_answer_count": row["false_answer_count"],
                "insufficient_substantive_count": row["insufficient_substantive_count"],
                "wrong_substantive_count": row["wrong_substantive_count"],
                "over_abstain_count": row["over_abstain_count"],
                "delta_f1_vs_naive_top5": row["answered_f1"] - top5["answered_f1"],
                "delta_isar_vs_naive_top5": row["insufficient_substantive_answer_rate"]
                - top5["insufficient_substantive_answer_rate"],
                "delta_false_answer_vs_naive_top5": row["false_answer_count"] - top5["false_answer_count"],
                "delta_wrong_substantive_vs_naive_top5": row["wrong_substantive_count"]
                - top5["wrong_substantive_count"],
                "delta_f1_vs_naive_top20": row["answered_f1"] - top20["answered_f1"],
                "delta_isar_vs_naive_top20": row["insufficient_substantive_answer_rate"]
                - top20["insufficient_substantive_answer_rate"],
                "delta_false_answer_vs_naive_top20": row["false_answer_count"] - top20["false_answer_count"],
                "delta_wrong_substantive_vs_naive_top20": row["wrong_substantive_count"]
                - top20["wrong_substantive_count"],
                "dominates_naive_top5_on_f1_and_isar": row["answered_f1"] >= top5["answered_f1"]
                and row["insufficient_substantive_answer_rate"] <= top5["insufficient_substantive_answer_rate"],
                "dominates_naive_top20_on_f1_and_isar": row["answered_f1"] >= top20["answered_f1"]
                and row["insufficient_substantive_answer_rate"] <= top20["insufficient_substantive_answer_rate"],
            }
            rows.append(out)
    return rows


def _robustness_rows(combined_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_policy: dict[str, list[dict[str, Any]]] = {}
    for row in combined_rows:
        by_policy.setdefault(row["policy"], []).append(row)
    rows = []
    for policy, policy_rows in sorted(by_policy.items()):
        natural = _find_sample(policy_rows, "natural_qa120")
        stratified = _find_sample(policy_rows, "stratified_qa90")
        rows.append(
            {
                "policy": policy,
                "natural_coverage": natural["coverage"],
                "natural_answered_f1": natural["answered_f1"],
                "natural_isar": natural["insufficient_substantive_answer_rate"],
                "natural_sufficient_abstain_rate": natural["sufficient_abstain_rate"],
                "stratified_coverage": stratified["coverage"],
                "stratified_answered_f1": stratified["answered_f1"],
                "stratified_isar": stratified["insufficient_substantive_answer_rate"],
                "stratified_sufficient_abstain_rate": stratified["sufficient_abstain_rate"],
                "min_coverage": min(row["coverage"] for row in policy_rows),
                "mean_answered_f1": sum(row["answered_f1"] for row in policy_rows) / len(policy_rows),
                "max_isar": max(row["insufficient_substantive_answer_rate"] for row in policy_rows),
                "mean_isar": sum(row["insufficient_substantive_answer_rate"] for row in policy_rows)
                / len(policy_rows),
                "dominates_naive_top5_all_samples": all(
                    row["dominates_naive_top5_on_f1_and_isar"] for row in policy_rows
                ),
                "dominates_naive_top20_all_samples": all(
                    row["dominates_naive_top20_on_f1_and_isar"] for row in policy_rows
                ),
            }
        )
    return rows


def _recommendations(
    combined_rows: list[dict[str, Any]], robustness_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    top5_by_sample = {
        row["sample"]: row
        for row in combined_rows
        if row["policy"] == "naive_top5"
    }
    non_naive = [row for row in robustness_rows if not row["policy"].startswith("naive_")]

    def acceptable(row: dict[str, Any]) -> bool:
        return (
            row["natural_coverage"] >= 0.65
            and row["natural_answered_f1"] >= top5_by_sample["natural_qa120"]["answered_f1"]
            and row["stratified_answered_f1"] >= top5_by_sample["stratified_qa90"]["answered_f1"]
            and row["natural_isar"] <= top5_by_sample["natural_qa120"]["insufficient_substantive_answer_rate"]
            and row["stratified_isar"] <= top5_by_sample["stratified_qa90"]["insufficient_substantive_answer_rate"]
        )

    accepted = [row for row in non_naive if acceptable(row)]
    risk_first = min(
        accepted,
        key=lambda row: (row["max_isar"], -row["natural_answered_f1"], -row["natural_coverage"]),
        default={},
    )
    f1_first = max(
        accepted,
        key=lambda row: (row["natural_answered_f1"], -row["max_isar"], row["natural_coverage"]),
        default={},
    )
    high_coverage = [
        row
        for row in accepted
        if row["natural_coverage"] >= 0.80
        and row["natural_isar"] <= top5_by_sample["natural_qa120"]["insufficient_substantive_answer_rate"]
    ]
    high_coverage_first = max(
        high_coverage,
        key=lambda row: (row["natural_coverage"], row["natural_answered_f1"]),
        default={},
    )
    return {
        "selection_protocol": {
            "input_samples": ["stratified_qa90", "natural_qa120"],
            "hard_filters": {
                "natural_coverage_min": 0.65,
                "f1_must_beat_naive_top5_on_both_samples": True,
                "isar_must_not_exceed_naive_top5_on_both_samples": True,
            },
            "risk_first_sort": ["max_isar asc", "natural_answered_f1 desc", "natural_coverage desc"],
            "f1_first_sort": ["natural_answered_f1 desc", "max_isar asc", "natural_coverage desc"],
        },
        "accepted_policies": [row["policy"] for row in accepted],
        "risk_first": risk_first,
        "f1_first": f1_first,
        "high_coverage_first": high_coverage_first,
        "high_coverage_note": (
            "No non-naive policy currently satisfies natural coverage >= 0.80 while also keeping natural "
            "insufficient-substantive answer rate no worse than naive_top5."
            if not high_coverage_first
            else ""
        ),
    }


def _summary_markdown(
    combined_rows: list[dict[str, Any]],
    robustness_rows: list[dict[str, Any]],
    recommendations: dict[str, Any],
    runs: list[dict[str, Any]],
) -> str:
    lines = [
        "# QA-aware Policy Comparison",
        "",
        "This report reuses existing MiniMax QA artifacts only. It does not call LLM or embedding APIs.",
        "",
        "## Inputs",
        "",
    ]
    for run in runs:
        validation = run["validation"]
        lines.append(
            f"- {run['sample_name']}: `{run['qa_dir']}`, selected `{validation.get('selected_count')}`, "
            f"completed `{validation.get('completed_count')}`, sample counts `{validation.get('sample_counts')}`"
        )
    lines.extend(
        [
            "",
            "## Combined Metrics",
            "",
            "| Sample | Policy | Coverage | Answered F1 | ISAR | False Ans. | Wrong Subst. |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in combined_rows:
        lines.append(
            f"| {row['sample']} | {row['policy']} | {_fmt(row['coverage'])} | "
            f"{_fmt(row['answered_f1'])} | {_fmt(row['insufficient_substantive_answer_rate'])} | "
            f"{row['false_answer_count']} | {row['wrong_substantive_count']} |"
        )
    lines.extend(
        [
            "",
            "## Robustness Summary",
            "",
            "| Policy | Natural cov. | Natural F1 | Natural ISAR | Strat. cov. | Strat. F1 | Strat. ISAR | Max ISAR |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in robustness_rows:
        lines.append(
            f"| {row['policy']} | {_fmt(row['natural_coverage'])} | {_fmt(row['natural_answered_f1'])} | "
            f"{_fmt(row['natural_isar'])} | {_fmt(row['stratified_coverage'])} | "
            f"{_fmt(row['stratified_answered_f1'])} | {_fmt(row['stratified_isar'])} | "
            f"{_fmt(row['max_isar'])} |"
        )
    lines.extend(
        [
            "",
            "## QA-aware Selection",
            "",
            f"- Accepted policies: `{', '.join(recommendations['accepted_policies'])}`",
            f"- Risk-first candidate: `{recommendations['risk_first'].get('policy', '')}`",
            f"- F1-first candidate: `{recommendations['f1_first'].get('policy', '')}`",
            f"- High-coverage candidate: `{recommendations['high_coverage_first'].get('policy', '')}`",
        ]
    )
    if recommendations["high_coverage_note"]:
        lines.append(f"- High-coverage note: {recommendations['high_coverage_note']}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "1. `balanced` is the current risk-first policy: it gives the lowest maximum ISAR across the two QA samples under the selection filters.",
            "2. `retrieve_more_risk_control@suff_abstain15` is the current F1-first policy among QA-safe candidates, but it has lower stratified coverage and more sufficient abstention.",
            "3. `retrieve_more@cov85` is not yet QA-safe: its natural coverage is attractive, but its natural ISAR is slightly worse than naive top-5.",
            "4. The next method work should improve high-coverage risk control, not merely reduce coverage.",
            "",
        ]
    )
    return "\n".join(lines)


def _find_sample(rows: list[dict[str, Any]], sample: str) -> dict[str, Any]:
    for row in rows:
        if row["sample"] == sample:
            return row
    raise KeyError(sample)


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


def _float_or_blank(value: str | None) -> float | str:
    if value in {"", None}:
        return ""
    return float(value)


def _fmt(value: float) -> str:
    return f"{value:.4f}"


if __name__ == "__main__":
    main()
