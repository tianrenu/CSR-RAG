from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from csrrag.calibration.methods import make_calibrator
from csrrag.evaluation.metrics import brier_score, decision_metrics_from_risk, ece
from csrrag.models.baseline import train_estimator
from csrrag.utils.io import read_jsonl, write_jsonl


TAU_GRID = [round(i / 100, 2) for i in range(5, 100, 5)]
ESTIMATORS = ["logistic_regression", "random_forest", "gradient_boosting"]
CALIBRATION_METHODS = ["identity", "platt", "isotonic"]

V1_QUERY = ["query_length", "has_time_constraint", "has_constraint_term"]
V1_RETRIEVAL = ["topk_score_mean", "topk_score_std", "top1_top2_gap", "doc_count", "doc_redundancy"]
V1_LEXICAL = ["title_overlap_max", "title_overlap_mean", "text_overlap_max", "text_overlap_mean"]

RANK_AWARE = ["top1_score", "top3_score_mean", "top5_score_min", "top1_top5_gap", "score_entropy"]
COVERAGE = [
    "query_token_coverage_union",
    "query_token_coverage_top1",
    "query_token_coverage_top3",
    "uncovered_query_token_ratio",
]
DIVERSITY = [
    "pairwise_doc_overlap_mean",
    "pairwise_doc_overlap_max",
    "unique_title_token_ratio",
    "doc_text_length_mean",
    "doc_text_length_std",
]
QUESTION_FORM = [
    "is_comparison_question",
    "is_bridge_like_question",
    "wh_who",
    "wh_what",
    "wh_when",
    "wh_where",
    "wh_which",
    "wh_how",
    "wh_other",
]

FEATURE_GROUPS = {
    "query": V1_QUERY + QUESTION_FORM,
    "retrieval": V1_RETRIEVAL + RANK_AWARE,
    "lexical": V1_LEXICAL,
    "coverage": COVERAGE,
    "diversity": DIVERSITY,
}
FEATURE_V1 = V1_QUERY + V1_RETRIEVAL + V1_LEXICAL
FEATURE_V2 = FEATURE_GROUPS["query"] + FEATURE_GROUPS["retrieval"] + FEATURE_GROUPS["lexical"] + FEATURE_GROUPS["coverage"] + FEATURE_GROUPS["diversity"]
GROUP_ABLATIONS = {
    "all_v2": FEATURE_V2,
    "query_only": FEATURE_GROUPS["query"],
    "retrieval_only": FEATURE_GROUPS["retrieval"],
    "lexical_only": FEATURE_GROUPS["lexical"],
    "coverage_only": FEATURE_GROUPS["coverage"],
    "diversity_only": FEATURE_GROUPS["diversity"],
    "no_query": FEATURE_GROUPS["retrieval"] + FEATURE_GROUPS["lexical"] + FEATURE_GROUPS["coverage"] + FEATURE_GROUPS["diversity"],
    "no_retrieval": FEATURE_GROUPS["query"] + FEATURE_GROUPS["lexical"] + FEATURE_GROUPS["coverage"] + FEATURE_GROUPS["diversity"],
    "no_lexical": FEATURE_GROUPS["query"] + FEATURE_GROUPS["retrieval"] + FEATURE_GROUPS["coverage"] + FEATURE_GROUPS["diversity"],
    "no_coverage": FEATURE_GROUPS["query"] + FEATURE_GROUPS["retrieval"] + FEATURE_GROUPS["lexical"] + FEATURE_GROUPS["diversity"],
    "no_diversity": FEATURE_GROUPS["query"] + FEATURE_GROUPS["retrieval"] + FEATURE_GROUPS["lexical"] + FEATURE_GROUPS["coverage"],
}
FORBIDDEN_FEATURE_FIELDS = {"support_doc_ids", "dropped_support_doc_id", "is_support"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run third-round CSR-RAG experiments with feature v2 and case studies.")
    parser.add_argument("--train-features", default="data/features/hotpotqa_dev_train_1800_features.jsonl")
    parser.add_argument("--valid-features", default="data/features/hotpotqa_dev_valid_1800_features.jsonl")
    parser.add_argument("--test-features", default="data/features/hotpotqa_dev_test_1800_features.jsonl")
    parser.add_argument("--test-records", default="data/processed/hotpotqa_dev_splits_1800/test.jsonl")
    parser.add_argument("--output-dir", default="results/tables/hotpotqa_dev_1800_v3")
    parser.add_argument("--artifact-dir", default="data/outputs/hotpotqa_dev_1800_v3")
    args = parser.parse_args()

    train_records = read_jsonl(args.train_features)
    valid_records = read_jsonl(args.valid_features)
    test_records = read_jsonl(args.test_features)
    test_retrieval_records = read_jsonl(args.test_records)
    _validate_feature_inputs(train_records + valid_records + test_records)

    output_dir = Path(args.output_dir)
    artifact_dir = Path(args.artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for estimator in ESTIMATORS:
        runs.append(_run_experiment("model_comparison", estimator, "isotonic", "v2_all", FEATURE_V2, train_records, valid_records, test_records))
    for calibration_method in CALIBRATION_METHODS:
        runs.append(_run_experiment("calibration_comparison", "logistic_regression", calibration_method, "v2_all", FEATURE_V2, train_records, valid_records, test_records))
    for version_name, feature_names in {"v1": FEATURE_V1, "v2": FEATURE_V2}.items():
        runs.append(_run_experiment("feature_version", "logistic_regression", "isotonic", version_name, feature_names, train_records, valid_records, test_records))
    for feature_set_name, feature_names in GROUP_ABLATIONS.items():
        runs.append(_run_experiment("feature_group_ablation", "logistic_regression", "isotonic", feature_set_name, feature_names, train_records, valid_records, test_records))

    main_run = _find_run(runs, "feature_version", "logistic_regression", "v2", "isotonic")
    raw_v2_run = _find_run(runs, "calibration_comparison", "logistic_regression", "v2_all", "identity")

    _write_main_comparison(output_dir / "main_comparison.csv", test_records, raw_v2_run, main_run)
    _write_model_comparison(output_dir / "model_comparison.csv", runs)
    _write_feature_version_comparison(output_dir / "feature_version_comparison.csv", runs)
    _write_feature_group_ablation(output_dir / "feature_group_ablation.csv", runs)
    _write_calibration_comparison(output_dir / "calibration_comparison.csv", runs)
    _write_coverage_curves(output_dir / "coverage_risk_curve_test.csv", runs)
    _write_reliability_bins(output_dir / "reliability_bins_test.csv", main_run)
    _write_question_type_breakdown(output_dir / "question_type_breakdown.csv", main_run, test_retrieval_records)
    _write_case_studies(output_dir / "case_studies.csv", main_run, test_retrieval_records)
    _write_validation_summary(output_dir / "validation_summary.json", runs, main_run, train_records, valid_records, test_records)
    _write_summary(output_dir / "third_round_summary.md", runs, main_run, raw_v2_run)
    _write_main_artifacts(artifact_dir, main_run)

    print(
        {
            "output_dir": str(output_dir),
            "artifact_dir": str(artifact_dir),
            "feature_v1_count": len(FEATURE_V1),
            "feature_v2_count": len(FEATURE_V2),
            "total_runs": len(runs),
            "coverage_curve_rows": len(runs) * len(TAU_GRID),
        }
    )


def _run_experiment(
    experiment_type: str,
    estimator_name: str,
    calibration_method: str,
    feature_set_name: str,
    feature_names: list[str],
    train_records: list[dict[str, Any]],
    valid_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
) -> dict[str, Any]:
    train_rows = [_feature_row(record, feature_names) for record in train_records]
    valid_rows = [_feature_row(record, feature_names) for record in valid_records]
    test_rows = [_feature_row(record, feature_names) for record in test_records]

    model = train_estimator(estimator_name, train_rows, _sufficiency_labels(train_records), feature_names)
    valid_sufficiency_scores = np.asarray(model.predict_proba(valid_rows), dtype=float)
    test_sufficiency_scores = np.asarray(model.predict_proba(test_rows), dtype=float)
    valid_raw_risk = 1.0 - valid_sufficiency_scores
    test_raw_risk = 1.0 - test_sufficiency_scores
    valid_labels = _risk_labels(valid_records)
    test_labels = _risk_labels(test_records)

    calibrator = make_calibrator(calibration_method)
    calibrator.fit(valid_raw_risk, valid_labels)
    valid_risk = np.asarray(calibrator.predict(valid_raw_risk), dtype=float)
    test_risk = np.asarray(calibrator.predict(test_raw_risk), dtype=float)
    tau_answer, valid_decision_metrics = _select_tau(valid_labels, valid_risk)
    test_decision_metrics = decision_metrics_from_risk(test_labels, test_risk, tau_answer)

    return {
        "experiment_type": experiment_type,
        "estimator": estimator_name,
        "feature_set": feature_set_name,
        "feature_names": feature_names,
        "calibration_method": calibration_method,
        "n_features": len(feature_names),
        "n_train": len(train_records),
        "n_valid": len(valid_records),
        "n_test": len(test_records),
        "test_ids": [record["id"] for record in test_records],
        "test_labels": test_labels,
        "test_sufficiency_scores": test_sufficiency_scores,
        "test_raw_risk": test_raw_risk,
        "test_risk": test_risk,
        "tau_answer": tau_answer,
        "valid_decision_metrics": valid_decision_metrics,
        "test_decision_metrics": test_decision_metrics,
        "raw_brier": brier_score(test_labels, test_raw_risk),
        "raw_ece": ece(test_labels, test_raw_risk),
        "calibrated_brier": brier_score(test_labels, test_risk),
        "calibrated_ece": ece(test_labels, test_risk),
    }


def _validate_feature_inputs(records: list[dict[str, Any]]) -> None:
    required = set(FEATURE_V2)
    for record in records:
        missing = required - set(record.keys())
        if missing:
            raise ValueError(f"Missing feature fields in {record.get('id')}: {sorted(missing)}")
        forbidden = FORBIDDEN_FEATURE_FIELDS & set(record.keys())
        if forbidden:
            raise ValueError(f"Forbidden support-only feature fields in {record.get('id')}: {sorted(forbidden)}")


def _feature_row(record: dict[str, Any], feature_names: list[str]) -> dict[str, float]:
    return {name: float(record[name]) for name in feature_names}


def _sufficiency_labels(records: list[dict[str, Any]]) -> list[int]:
    return [1 if record["sufficiency_label"] == "sufficient" else 0 for record in records]


def _risk_labels(records: list[dict[str, Any]]) -> np.ndarray:
    return np.array([0 if record["sufficiency_label"] == "sufficient" else 1 for record in records], dtype=int)


def _select_tau(y_true: Iterable[int], risk_scores: Iterable[float]) -> tuple[float, dict[str, float]]:
    labels = list(y_true)
    scores = list(risk_scores)
    best_tau = TAU_GRID[0]
    best_metrics = decision_metrics_from_risk(labels, scores, best_tau)
    for tau in TAU_GRID[1:]:
        metrics = decision_metrics_from_risk(labels, scores, tau)
        if metrics["decision_accuracy"] > best_metrics["decision_accuracy"]:
            best_tau = tau
            best_metrics = metrics
        elif metrics["decision_accuracy"] == best_metrics["decision_accuracy"] and metrics["coverage"] > best_metrics["coverage"]:
            best_tau = tau
            best_metrics = metrics
    return best_tau, best_metrics


def _find_run(runs: list[dict[str, Any]], experiment_type: str, estimator: str, feature_set: str, calibration_method: str) -> dict[str, Any]:
    for run in runs:
        if (
            run["experiment_type"] == experiment_type
            and run["estimator"] == estimator
            and run["feature_set"] == feature_set
            and run["calibration_method"] == calibration_method
        ):
            return run
    raise ValueError(f"Run not found: {experiment_type}/{estimator}/{feature_set}/{calibration_method}")


def _metric_row(run: dict[str, Any]) -> dict[str, Any]:
    test_metrics = run["test_decision_metrics"]
    return {
        "experiment_type": run["experiment_type"],
        "estimator": run["estimator"],
        "feature_set": run["feature_set"],
        "calibration_method": run["calibration_method"],
        "n_features": run["n_features"],
        "n_test": run["n_test"],
        "raw_brier": run["raw_brier"],
        "raw_ece": run["raw_ece"],
        "calibrated_brier": run["calibrated_brier"],
        "calibrated_ece": run["calibrated_ece"],
        "brier_delta": run["calibrated_brier"] - run["raw_brier"],
        "ece_delta": run["calibrated_ece"] - run["raw_ece"],
        "tau_answer": run["tau_answer"],
        "decision_accuracy": test_metrics["decision_accuracy"],
        "coverage": test_metrics["coverage"],
        "selective_accuracy": test_metrics["selective_accuracy"],
        "abstention_rate": 1.0 - test_metrics["coverage"],
    }


def _write_main_comparison(path: Path, test_records: list[dict[str, Any]], raw_run: dict[str, Any], main_run: dict[str, Any]) -> None:
    labels = _risk_labels(test_records)
    naive_metrics = decision_metrics_from_risk(labels, np.zeros_like(labels, dtype=float), 0.95)
    rows = [
        {
            "method": "Naive RAG",
            "estimator": "none",
            "feature_set": "none",
            "calibration_method": "none",
            "tau_answer": "",
            "decision_accuracy": naive_metrics["decision_accuracy"],
            "coverage": naive_metrics["coverage"],
            "selective_accuracy": naive_metrics["selective_accuracy"],
            "abstention_rate": 1.0 - naive_metrics["coverage"],
            "risk_brier": "",
            "risk_ece": "",
            "n_test": len(test_records),
        },
        {
            "method": "Uncalibrated CSR",
            "estimator": raw_run["estimator"],
            "feature_set": raw_run["feature_set"],
            "calibration_method": raw_run["calibration_method"],
            "tau_answer": raw_run["tau_answer"],
            "decision_accuracy": raw_run["test_decision_metrics"]["decision_accuracy"],
            "coverage": raw_run["test_decision_metrics"]["coverage"],
            "selective_accuracy": raw_run["test_decision_metrics"]["selective_accuracy"],
            "abstention_rate": 1.0 - raw_run["test_decision_metrics"]["coverage"],
            "risk_brier": raw_run["calibrated_brier"],
            "risk_ece": raw_run["calibrated_ece"],
            "n_test": len(test_records),
        },
        {
            "method": "CSR-RAG",
            "estimator": main_run["estimator"],
            "feature_set": main_run["feature_set"],
            "calibration_method": main_run["calibration_method"],
            "tau_answer": main_run["tau_answer"],
            "decision_accuracy": main_run["test_decision_metrics"]["decision_accuracy"],
            "coverage": main_run["test_decision_metrics"]["coverage"],
            "selective_accuracy": main_run["test_decision_metrics"]["selective_accuracy"],
            "abstention_rate": 1.0 - main_run["test_decision_metrics"]["coverage"],
            "risk_brier": main_run["calibrated_brier"],
            "risk_ece": main_run["calibrated_ece"],
            "n_test": len(test_records),
        },
    ]
    _write_csv(path, rows, list(rows[0].keys()))


def _write_model_comparison(path: Path, runs: list[dict[str, Any]]) -> None:
    rows = [_metric_row(_find_run(runs, "model_comparison", estimator, "v2_all", "isotonic")) for estimator in ESTIMATORS]
    _write_csv(path, rows, list(rows[0].keys()))


def _write_feature_version_comparison(path: Path, runs: list[dict[str, Any]]) -> None:
    rows = [_metric_row(_find_run(runs, "feature_version", "logistic_regression", version, "isotonic")) for version in ["v1", "v2"]]
    _write_csv(path, rows, list(rows[0].keys()))


def _write_feature_group_ablation(path: Path, runs: list[dict[str, Any]]) -> None:
    rows = [_metric_row(_find_run(runs, "feature_group_ablation", "logistic_regression", feature_set, "isotonic")) for feature_set in GROUP_ABLATIONS]
    _write_csv(path, rows, list(rows[0].keys()))


def _write_calibration_comparison(path: Path, runs: list[dict[str, Any]]) -> None:
    rows = [_metric_row(_find_run(runs, "calibration_comparison", "logistic_regression", "v2_all", method)) for method in CALIBRATION_METHODS]
    _write_csv(path, rows, list(rows[0].keys()))


def _write_coverage_curves(path: Path, runs: list[dict[str, Any]]) -> None:
    rows = []
    for run in runs:
        for tau in TAU_GRID:
            metrics = decision_metrics_from_risk(run["test_labels"], run["test_risk"], tau)
            rows.append(
                {
                    "experiment_type": run["experiment_type"],
                    "estimator": run["estimator"],
                    "feature_set": run["feature_set"],
                    "calibration_method": run["calibration_method"],
                    **metrics,
                }
            )
    _write_csv(path, rows, list(rows[0].keys()))


def _write_reliability_bins(path: Path, run: dict[str, Any], n_bins: int = 10) -> None:
    labels = run["test_labels"]
    risks = run["test_risk"]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for bin_id in range(n_bins):
        left, right = bins[bin_id], bins[bin_id + 1]
        mask = (risks >= left) & (risks < right) if bin_id < n_bins - 1 else (risks >= left) & (risks <= right)
        n = int(mask.sum())
        rows.append(
            {
                "bin_id": bin_id,
                "bin_left": left,
                "bin_right": right,
                "n": n,
                "mean_predicted_risk": float(risks[mask].mean()) if n else "",
                "observed_risk_rate": float(labels[mask].mean()) if n else "",
                "abs_gap": abs(float(risks[mask].mean()) - float(labels[mask].mean())) if n else "",
            }
        )
    _write_csv(path, rows, list(rows[0].keys()))


def _write_question_type_breakdown(path: Path, run: dict[str, Any], retrieval_records: list[dict[str, Any]]) -> None:
    metadata_by_id = {record["id"]: record["metadata"] for record in retrieval_records}
    grouped: dict[str, list[int]] = {}
    for index, record_id in enumerate(run["test_ids"]):
        grouped.setdefault(metadata_by_id[record_id].get("question_type", "unknown"), []).append(index)
    rows = []
    for question_type, indices in sorted(grouped.items()):
        labels = run["test_labels"][indices]
        risks = run["test_risk"][indices]
        metrics = decision_metrics_from_risk(labels, risks, run["tau_answer"])
        rows.append(
            {
                "question_type": question_type,
                "n_test": len(indices),
                "sufficient_count": int((labels == 0).sum()),
                "insufficient_count": int((labels == 1).sum()),
                "brier": brier_score(labels, risks),
                "ece": ece(labels, risks),
                "decision_accuracy": metrics["decision_accuracy"],
                "coverage": metrics["coverage"],
                "selective_accuracy": metrics["selective_accuracy"],
            }
        )
    _write_csv(path, rows, list(rows[0].keys()))


def _write_case_studies(path: Path, run: dict[str, Any], retrieval_records: list[dict[str, Any]]) -> None:
    retrieval_by_id = {record["id"]: record for record in retrieval_records}
    rows = []
    for case_type, indices in _case_indices(run).items():
        for index in indices[:5]:
            record_id = run["test_ids"][index]
            record = retrieval_by_id[record_id]
            docs = record["retrieved_docs"][:5]
            risk = float(run["test_risk"][index])
            label = "insufficient" if int(run["test_labels"][index]) == 1 else "sufficient"
            decision = "answer" if risk <= float(run["tau_answer"]) else "abstain"
            rows.append(
                {
                    "case_type": case_type,
                    "id": record_id,
                    "query": record["query"],
                    "gold_answer": record.get("gold_answer", ""),
                    "label": label,
                    "risk_score": risk,
                    "decision": decision,
                    "tau_answer": run["tau_answer"],
                    "top5_titles": " || ".join(str(doc["title"]) for doc in docs),
                    "top5_scores": " || ".join(str(doc["score"]) for doc in docs),
                }
            )
    _write_csv(
        path,
        rows,
        ["case_type", "id", "query", "gold_answer", "label", "risk_score", "decision", "tau_answer", "top5_titles", "top5_scores"],
    )


def _case_indices(run: dict[str, Any]) -> dict[str, list[int]]:
    risks = run["test_risk"]
    labels = run["test_labels"]
    tau = float(run["tau_answer"])
    indices = list(range(len(labels)))
    return {
        "success_answer": sorted([i for i in indices if labels[i] == 0 and risks[i] <= tau], key=lambda i: risks[i]),
        "success_abstain": sorted([i for i in indices if labels[i] == 1 and risks[i] > tau], key=lambda i: risks[i], reverse=True),
        "false_answer": sorted([i for i in indices if labels[i] == 1 and risks[i] <= tau], key=lambda i: risks[i]),
        "over_abstain": sorted([i for i in indices if labels[i] == 0 and risks[i] > tau], key=lambda i: risks[i], reverse=True),
    }


def _write_validation_summary(
    path: Path,
    runs: list[dict[str, Any]],
    main_run: dict[str, Any],
    train_records: list[dict[str, Any]],
    valid_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
) -> None:
    unique_scores = {round(float(score), 8) for score in main_run["test_risk"]}
    summary = {
        "splits": {"train": len(train_records), "valid": len(valid_records), "test": len(test_records)},
        "feature_counts": {"v1": len(FEATURE_V1), "v2": len(FEATURE_V2)},
        "runs": {"total": len(runs), "coverage_curve_rows": len(runs) * len(TAU_GRID)},
        "main_risk": {
            "unique_score_count": len(unique_scores),
            "min_risk": float(main_run["test_risk"].min()),
            "max_risk": float(main_run["test_risk"].max()),
            "all_binary": all(score in {0.0, 1.0} for score in unique_scores),
        },
        "support_only_features_present": False,
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_summary(path: Path, runs: list[dict[str, Any]], main_run: dict[str, Any], raw_run: dict[str, Any]) -> None:
    v1_run = _find_run(runs, "feature_version", "logistic_regression", "v1", "isotonic")
    model_runs = [_find_run(runs, "model_comparison", estimator, "v2_all", "isotonic") for estimator in ESTIMATORS]
    best_model = max(model_runs, key=lambda run: run["test_decision_metrics"]["decision_accuracy"])
    ablation_runs = [_find_run(runs, "feature_group_ablation", "logistic_regression", name, "isotonic") for name in GROUP_ABLATIONS]
    best_ablation = max(ablation_runs, key=lambda run: run["test_decision_metrics"]["decision_accuracy"])
    lines = [
        "# CSR-RAG 第三轮实验结果摘要",
        "",
        "## 1. 当前最好结果",
        "",
        f"- 主方法：LogisticRegression + isotonic + feature v2",
        f"- decision accuracy：{main_run['test_decision_metrics']['decision_accuracy']:.4f}",
        f"- coverage：{main_run['test_decision_metrics']['coverage']:.4f}",
        f"- selective accuracy：{main_run['test_decision_metrics']['selective_accuracy']:.4f}",
        f"- calibrated Brier：{main_run['calibrated_brier']:.4f}",
        f"- calibrated ECE：{main_run['calibrated_ece']:.4f}",
        "",
        "## 2. feature v1 vs v2",
        "",
        f"- v1 decision accuracy：{v1_run['test_decision_metrics']['decision_accuracy']:.4f}",
        f"- v2 decision accuracy：{main_run['test_decision_metrics']['decision_accuracy']:.4f}",
        f"- v1 calibrated Brier：{v1_run['calibrated_brier']:.4f}",
        f"- v2 calibrated Brier：{main_run['calibrated_brier']:.4f}",
        "",
        "## 3. 模型、特征与校准",
        "",
        f"- 最好 estimator：{best_model['estimator']}，decision accuracy = {best_model['test_decision_metrics']['decision_accuracy']:.4f}",
        f"- 最好特征组设置：{best_ablation['feature_set']}，decision accuracy = {best_ablation['test_decision_metrics']['decision_accuracy']:.4f}",
        f"- 未校准 CSR decision accuracy：{raw_run['test_decision_metrics']['decision_accuracy']:.4f}",
        f"- 校准后 CSR-RAG decision accuracy：{main_run['test_decision_metrics']['decision_accuracy']:.4f}",
        "",
        "## 4. 论文表述建议",
        "",
        "- 继续把主张收敛在检索充分性风险预测和选择性回答。",
        "- 如果 feature v2 未明显提升，就不继续堆特征，应转向论文写作和局限性分析。",
        "- 如果 feature v2 明显提升，可以进入最小 QA generation 评测，但仍保持 answer / abstain 主线。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_main_artifacts(artifact_dir: Path, run: dict[str, Any]) -> None:
    risk_records = []
    decision_records = []
    for record_id, label, sufficiency_score, raw_risk, risk in zip(
        run["test_ids"], run["test_labels"], run["test_sufficiency_scores"], run["test_raw_risk"], run["test_risk"]
    ):
        label_name = "insufficient" if int(label) == 1 else "sufficient"
        decision = "answer" if float(risk) <= float(run["tau_answer"]) else "abstain"
        risk_records.append(
            {
                "id": record_id,
                "sufficiency_label": label_name,
                "sufficiency_score": float(sufficiency_score),
                "raw_risk_score": float(raw_risk),
                "risk_score": float(risk),
                "estimator": run["estimator"],
                "feature_set": run["feature_set"],
                "calibration_method": run["calibration_method"],
            }
        )
        decision_records.append({"id": record_id, "risk_score": float(risk), "decision": decision, "tau_answer": float(run["tau_answer"])})
    write_jsonl(artifact_dir / "main_test_risk.jsonl", risk_records)
    write_jsonl(artifact_dir / "main_test_decisions.jsonl", decision_records)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
