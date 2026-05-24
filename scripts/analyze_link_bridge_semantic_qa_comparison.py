from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FLOAT_FIELDS = {
    "coverage",
    "strict_answered_em",
    "answered_f1",
    "alias_corrected_answer_accuracy",
    "lenient_corrected_answer_accuracy",
    "judge_bad_answer_rate",
    "strict_insufficient_substantive_answer_rate",
    "alias_corrected_insufficient_substantive_answer_rate",
    "lenient_corrected_insufficient_substantive_answer_rate",
}

INT_FIELDS = {
    "target_k",
    "n",
    "answered_count",
    "strict_wrong_substantive_count",
    "alias_corrected_wrong_substantive_count",
    "lenient_corrected_wrong_substantive_count",
    "judge_bad_answer_count",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare semantic QA rescoring across natural QA120 and stratified QA90 without API calls."
    )
    parser.add_argument(
        "--natural-dir",
        default="results/tables/hotpotqa_official_intro_link_bridge_support_reranker_semantic_qa_rescore_natural120",
    )
    parser.add_argument(
        "--stratified-dir",
        default="results/tables/hotpotqa_official_intro_link_bridge_support_reranker_semantic_qa_rescore_stratified90",
    )
    parser.add_argument(
        "--output-dir",
        default="results/tables/hotpotqa_official_intro_link_bridge_support_reranker_semantic_qa_comparison",
    )
    args = parser.parse_args()

    runs = [
        _load_run("natural_qa120", Path(args.natural_dir)),
        _load_run("stratified_qa90", Path(args.stratified_dir)),
    ]
    combined = []
    for run in runs:
        for row in run["rows"]:
            out = {"sample": run["sample_name"], **row}
            combined.append(out)
    robustness = _robustness_rows(combined)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "combined_semantic_qa_metrics.csv", combined)
    _write_csv(output_dir / "semantic_qa_robustness_summary.csv", robustness)
    (output_dir / "semantic_qa_comparison_summary.md").write_text(
        _summary_markdown(combined, robustness, runs),
        encoding="utf-8",
    )
    (output_dir / "validation_summary.json").write_text(
        json.dumps(
            {
                "natural_dir": args.natural_dir,
                "stratified_dir": args.stratified_dir,
                "output_dir": args.output_dir,
                "input_tables": [
                    str(Path(args.natural_dir) / "semantic_qa_policy_rescore.csv"),
                    str(Path(args.stratified_dir) / "semantic_qa_policy_rescore.csv"),
                ],
                "uses_llm_api": False,
                "uses_embedding_api": False,
                "outputs": [
                    "combined_semantic_qa_metrics.csv",
                    "semantic_qa_robustness_summary.csv",
                    "semantic_qa_comparison_summary.md",
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
                "policies": sorted({row["policy"] for row in combined}),
                "uses_llm_api": False,
                "uses_embedding_api": False,
            },
            ensure_ascii=False,
        )
    )


def _load_run(sample_name: str, directory: Path) -> dict[str, Any]:
    rows = [_coerce(row) for row in _read_csv(directory / "semantic_qa_policy_rescore.csv")]
    validation_path = directory / "validation_summary.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else {}
    return {"sample_name": sample_name, "directory": directory, "rows": rows, "validation": validation}


def _coerce(row: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if key in FLOAT_FIELDS:
            out[key] = float(value)
        elif key in INT_FIELDS:
            out[key] = int(float(value))
        else:
            out[key] = value
    return out


def _robustness_rows(combined: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_policy: dict[str, list[dict[str, Any]]] = {}
    for row in combined:
        by_policy.setdefault(row["policy"], []).append(row)
    rows = []
    for policy, policy_rows in sorted(by_policy.items()):
        natural = _find(policy_rows, "natural_qa120")
        stratified = _find(policy_rows, "stratified_qa90")
        rows.append(
            {
                "policy": policy,
                "natural_coverage": natural["coverage"],
                "natural_alias_accuracy": natural["alias_corrected_answer_accuracy"],
                "natural_judge_bad_rate": natural["judge_bad_answer_rate"],
                "natural_alias_isar": natural["alias_corrected_insufficient_substantive_answer_rate"],
                "stratified_coverage": stratified["coverage"],
                "stratified_alias_accuracy": stratified["alias_corrected_answer_accuracy"],
                "stratified_judge_bad_rate": stratified["judge_bad_answer_rate"],
                "stratified_alias_isar": stratified["alias_corrected_insufficient_substantive_answer_rate"],
                "mean_alias_accuracy": _mean(row["alias_corrected_answer_accuracy"] for row in policy_rows),
                "max_judge_bad_rate": max(row["judge_bad_answer_rate"] for row in policy_rows),
                "max_alias_isar": max(row["alias_corrected_insufficient_substantive_answer_rate"] for row in policy_rows),
                "min_coverage": min(row["coverage"] for row in policy_rows),
            }
        )
    return rows


def _summary_markdown(combined: list[dict[str, Any]], robustness: list[dict[str, Any]], runs: list[dict[str, Any]]) -> str:
    lines = [
        "# Semantic QA Comparison",
        "",
        "This report combines existing semantic QA rescore artifacts. It does not call LLM or embedding APIs.",
        "",
        "## Inputs",
        "",
    ]
    for run in runs:
        lines.append(f"- {run['sample_name']}: `{run['directory']}`")
    lines.extend(
        [
            "",
            "## Combined Metrics",
            "",
            "| Sample | Policy | Coverage | Alias acc. | Judge-bad rate | Alias ISAR |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in combined:
        lines.append(
            f"| {row['sample']} | {row['policy']} | {_fmt(row['coverage'])} | "
            f"{_fmt(row['alias_corrected_answer_accuracy'])} | {_fmt(row['judge_bad_answer_rate'])} | "
            f"{_fmt(row['alias_corrected_insufficient_substantive_answer_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Robustness",
            "",
            "| Policy | Natural cov. | Natural alias acc. | Natural bad | Strat. cov. | Strat. alias acc. | Strat. bad | Max alias ISAR |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in robustness:
        lines.append(
            f"| {row['policy']} | {_fmt(row['natural_coverage'])} | {_fmt(row['natural_alias_accuracy'])} | "
            f"{_fmt(row['natural_judge_bad_rate'])} | {_fmt(row['stratified_coverage'])} | "
            f"{_fmt(row['stratified_alias_accuracy'])} | {_fmt(row['stratified_judge_bad_rate'])} | "
            f"{_fmt(row['max_alias_isar'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "1. Retrieve-more policies consistently improve alias-corrected answer accuracy over naive top-5 and balanced answer/abstain.",
            "2. High coverage still needs better risk control: `retrieve_more@cov85` has attractive coverage but worse max alias-corrected ISAR than stricter policies.",
            "3. Semantic rescore is an audit layer; strict EM/F1 should remain in reports for comparability.",
            "",
        ]
    )
    return "\n".join(lines)


def _find(rows: list[dict[str, Any]], sample: str) -> dict[str, Any]:
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


def _mean(values: Any) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _fmt(value: float) -> str:
    return f"{value:.4f}"


if __name__ == "__main__":
    main()
