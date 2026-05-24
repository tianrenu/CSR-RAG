from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from run_link_bridge_policy_qa_eval import _decision, _policy_specs
from run_link_bridge_retrieve_more_experiments import (
    _feature_record,
    _feature_sets,
    _load_topk_records,
    _risk_runs,
    _split_valid_ids,
    _validate_features,
    _validate_records,
)


BAD_TAXONOMIES = {"generation_failure", "retrieval_insufficient"}
ALIAS_TAXONOMIES = {"answer_alias_or_metric_mismatch"}
LENIENT_CORRECT_TAXONOMIES = {"answer_alias_or_metric_mismatch", "ambiguous_or_gold_issue"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rescore link-bridge QA policies with semantic failure judgments. "
            "This script only reads existing QA and judge artifacts."
        )
    )
    parser.add_argument(
        "--qa-dir",
        default="results/tables/hotpotqa_official_intro_link_bridge_support_reranker_qa_eval_natural120",
    )
    parser.add_argument(
        "--judge-dir",
        default="results/tables/hotpotqa_official_intro_link_bridge_support_reranker_qa_failure_judge_natural120_v2",
    )
    parser.add_argument(
        "--split-dir",
        default="data/processed/hotpotqa_official_intro_link_bridge_support_reranker_splits",
    )
    parser.add_argument(
        "--record-prefix",
        default="official_intro_link_bridge_support_reranker_random_forest_balanced_all_blend1_00_top",
    )
    parser.add_argument(
        "--policy-main",
        default="results/tables/hotpotqa_official_intro_link_bridge_support_reranker_retrieve_more/main_comparison.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="results/tables/hotpotqa_official_intro_link_bridge_support_reranker_semantic_qa_rescore_natural120",
    )
    parser.add_argument("--sample-name", default="natural_qa120")
    args = parser.parse_args()

    details = _read_csv(Path(args.qa_dir) / "link_bridge_qa_details.csv")
    judgments = {
        row["case_id"]: row
        for row in _read_csv(Path(args.judge_dir) / "qa_failure_judgments.csv")
    }
    records = _load_topk_records(Path(args.split_dir), args.record_prefix)
    _validate_records(records)
    features = {
        top_k: {
            split: [_feature_record(record) for record in split_records]
            for split, split_records in topk_records.items()
        }
        for top_k, topk_records in records.items()
    }
    feature_sets = _feature_sets()
    _validate_features(features, feature_sets)
    valid_calib_ids, _ = _split_valid_ids(records[5]["valid"])
    _, risks = _risk_runs(features, valid_calib_ids, feature_sets)
    specs = _policy_specs(Path(args.policy_main), risks)

    selected = _selected_from_details(details, records)
    rows = _policy_rows(args.sample_name, details, selected, specs, judgments)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "semantic_qa_policy_rescore.csv", rows)
    (output_dir / "semantic_qa_rescore_summary.md").write_text(_summary_markdown(rows, args), encoding="utf-8")
    (output_dir / "validation_summary.json").write_text(
        json.dumps(
            {
                "qa_dir": args.qa_dir,
                "judge_dir": args.judge_dir,
                "split_dir": args.split_dir,
                "record_prefix": args.record_prefix,
                "policy_main": args.policy_main,
                "sample_name": args.sample_name,
                "detail_rows": len(details),
                "judge_rows": len(judgments),
                "policy_count": len(specs),
                "uses_llm_api": False,
                "uses_embedding_api": False,
                "outputs": ["semantic_qa_policy_rescore.csv", "semantic_qa_rescore_summary.md"],
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
                "policies": [row["policy"] for row in rows],
                "uses_llm_api": False,
                "uses_embedding_api": False,
            },
            ensure_ascii=False,
        )
    )


def _selected_from_details(
    details: list[dict[str, str]],
    records: dict[int, dict[str, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    by_id = {
        top_k: {record["metadata"]["original_id"]: record for record in records[top_k]["test"]}
        for top_k in (5, 10, 20)
    }
    selected = []
    for row in details:
        original_id = row["original_id"]
        selected.append(
            {
                "sample_bucket": row["sample_bucket"],
                "original_id": original_id,
                "record5": by_id[5][original_id],
                "record10": by_id[10][original_id],
                "record20": by_id[20][original_id],
            }
        )
    return selected


def _policy_rows(
    sample_name: str,
    detail_rows: list[dict[str, str]],
    selected: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    judgments: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    item_by_id = {item["original_id"]: item for item in selected}
    rows = []
    for spec in specs:
        decisions = [_decision(spec, item_by_id[row["original_id"]]) for row in detail_rows]
        answered = [(row, decision) for row, decision in zip(detail_rows, decisions) if decision["decision"] == "answer"]
        insufficient = [(row, decision) for row, decision in zip(detail_rows, decisions) if decision["sufficiency_label"] == "insufficient"]
        answered_insufficient = [(row, decision) for row, decision in answered if decision["sufficiency_label"] == "insufficient"]
        strict_wrong_substantive = [
            (row, decision)
            for row, decision in answered
            if _substantive(row, decision) and _em(row, decision) == 0.0
        ]
        alias_wrong_substantive = [
            (row, decision)
            for row, decision in answered
            if _substantive(row, decision) and not _semantic_correct(sample_name, row, decision, judgments, "alias")
        ]
        lenient_wrong_substantive = [
            (row, decision)
            for row, decision in answered
            if _substantive(row, decision) and not _semantic_correct(sample_name, row, decision, judgments, "lenient")
        ]
        judge_bad = [
            (row, decision)
            for row, decision in answered
            if _taxonomy(sample_name, row, decision, judgments) in BAD_TAXONOMIES
        ]
        insuff_alias_bad = [
            (row, decision)
            for row, decision in answered_insufficient
            if _substantive(row, decision) and not _semantic_correct(sample_name, row, decision, judgments, "alias")
        ]
        insuff_lenient_bad = [
            (row, decision)
            for row, decision in answered_insufficient
            if _substantive(row, decision) and not _semantic_correct(sample_name, row, decision, judgments, "lenient")
        ]
        rows.append(
            {
                "policy": spec["policy"],
                "action_type": spec["action_type"],
                "target_k": spec["target_k"],
                "n": len(detail_rows),
                "coverage": len(answered) / len(detail_rows) if detail_rows else 0.0,
                "answered_count": len(answered),
                "strict_answered_em": _mean(_em(row, decision) for row, decision in answered),
                "answered_f1": _mean(_f1(row, decision) for row, decision in answered),
                "alias_corrected_answer_accuracy": _mean(
                    float(_semantic_correct(sample_name, row, decision, judgments, "alias"))
                    for row, decision in answered
                ),
                "lenient_corrected_answer_accuracy": _mean(
                    float(_semantic_correct(sample_name, row, decision, judgments, "lenient"))
                    for row, decision in answered
                ),
                "strict_wrong_substantive_count": len(strict_wrong_substantive),
                "alias_corrected_wrong_substantive_count": len(alias_wrong_substantive),
                "lenient_corrected_wrong_substantive_count": len(lenient_wrong_substantive),
                "judge_bad_answer_count": len(judge_bad),
                "judge_bad_answer_rate": len(judge_bad) / len(answered) if answered else 0.0,
                "strict_insufficient_substantive_answer_rate": _insuff_rate(
                    answered_insufficient, insufficient, sample_name, judgments, "strict"
                ),
                "alias_corrected_insufficient_substantive_answer_rate": len(insuff_alias_bad) / len(insufficient)
                if insufficient
                else 0.0,
                "lenient_corrected_insufficient_substantive_answer_rate": len(insuff_lenient_bad) / len(insufficient)
                if insufficient
                else 0.0,
            }
        )
    return rows


def _insuff_rate(
    answered_insufficient: list[tuple[dict[str, str], dict[str, Any]]],
    insufficient: list[tuple[dict[str, str], dict[str, Any]]],
    sample_name: str,
    judgments: dict[str, dict[str, str]],
    mode: str,
) -> float:
    if not insufficient:
        return 0.0
    count = 0
    for row, decision in answered_insufficient:
        if not _substantive(row, decision):
            continue
        if mode == "strict" or not _semantic_correct(sample_name, row, decision, judgments, mode):
            count += 1
    return count / len(insufficient)


def _semantic_correct(
    sample_name: str,
    row: dict[str, str],
    decision: dict[str, Any],
    judgments: dict[str, dict[str, str]],
    mode: str,
) -> bool:
    if _em(row, decision) == 1.0:
        return True
    taxonomy = _taxonomy(sample_name, row, decision, judgments)
    if mode == "alias":
        return taxonomy in ALIAS_TAXONOMIES
    if mode == "lenient":
        return taxonomy in LENIENT_CORRECT_TAXONOMIES
    return False


def _taxonomy(
    sample_name: str,
    row: dict[str, str],
    decision: dict[str, Any],
    judgments: dict[str, dict[str, str]],
) -> str:
    top_k = decision["answer_top_k"]
    case_id = f"{sample_name}::{row['original_id']}::top{top_k}"
    return judgments.get(case_id, {}).get("taxonomy", "")


def _em(row: dict[str, str], decision: dict[str, Any]) -> float:
    return float(row[f"top{decision['answer_top_k']}_em"])


def _f1(row: dict[str, str], decision: dict[str, Any]) -> float:
    return float(row[f"top{decision['answer_top_k']}_f1"])


def _substantive(row: dict[str, str], decision: dict[str, Any]) -> bool:
    return str(row[f"top{decision['answer_top_k']}_is_substantive"]).lower() == "true"


def _summary_markdown(rows: list[dict[str, Any]], args: argparse.Namespace) -> str:
    lines = [
        "# Semantic QA Rescore",
        "",
        f"This report combines {args.sample_name} QA details with MiniMax failure taxonomy. It does not call LLM or embedding APIs.",
        "",
        "## Inputs",
        "",
        f"- QA dir: `{args.qa_dir}`",
        f"- Judge dir: `{args.judge_dir}`",
        "",
        "## Policy Rescore",
        "",
        "| Policy | Coverage | Strict EM | F1 | Alias acc. | Lenient acc. | Judge-bad rate | Strict ISAR | Alias ISAR | Lenient ISAR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['policy']} | {_fmt(row['coverage'])} | {_fmt(row['strict_answered_em'])} | "
            f"{_fmt(row['answered_f1'])} | {_fmt(row['alias_corrected_answer_accuracy'])} | "
            f"{_fmt(row['lenient_corrected_answer_accuracy'])} | {_fmt(row['judge_bad_answer_rate'])} | "
            f"{_fmt(row['strict_insufficient_substantive_answer_rate'])} | "
            f"{_fmt(row['alias_corrected_insufficient_substantive_answer_rate'])} | "
            f"{_fmt(row['lenient_corrected_insufficient_substantive_answer_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "1. Alias-corrected accuracy only treats answer_alias_or_metric_mismatch as correct.",
            "2. Lenient accuracy also treats ambiguous_or_gold_issue as non-error, so it should be reported as an audit upper bound.",
            "3. Judge-bad rate counts generation_failure and retrieval_insufficient among answered cases.",
            "4. Strict EM/F1 remain necessary for comparability; semantic rescore is an audit layer, not a replacement.",
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


def _mean(values: Any) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _fmt(value: float) -> str:
    return f"{value:.4f}"


if __name__ == "__main__":
    main()
