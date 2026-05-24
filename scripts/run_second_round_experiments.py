from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
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

FEATURE_GROUPS = {
    "query": ["query_length", "has_time_constraint", "has_constraint_term"],
    "retrieval": ["topk_score_mean", "topk_score_std", "top1_top2_gap", "doc_count", "doc_redundancy"],
    "lexical": ["title_overlap_max", "title_overlap_mean", "text_overlap_max", "text_overlap_mean"],
}
FEATURE_SETS = {
    "all_features": FEATURE_GROUPS["query"] + FEATURE_GROUPS["retrieval"] + FEATURE_GROUPS["lexical"],
    "query_only": FEATURE_GROUPS["query"],
    "retrieval_only": FEATURE_GROUPS["retrieval"],
    "lexical_only": FEATURE_GROUPS["lexical"],
    "no_query": FEATURE_GROUPS["retrieval"] + FEATURE_GROUPS["lexical"],
    "no_retrieval": FEATURE_GROUPS["query"] + FEATURE_GROUPS["lexical"],
    "no_lexical": FEATURE_GROUPS["query"] + FEATURE_GROUPS["retrieval"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper-ready second-round CSR-RAG experiments.")
    parser.add_argument("--train-features", default="data/features/hotpotqa_dev_train_1800_features.jsonl")
    parser.add_argument("--valid-features", default="data/features/hotpotqa_dev_valid_1800_features.jsonl")
    parser.add_argument("--test-features", default="data/features/hotpotqa_dev_test_1800_features.jsonl")
    parser.add_argument("--test-records", default="data/processed/hotpotqa_dev_splits_1800/test.jsonl")
    parser.add_argument("--output-dir", default="results/tables/hotpotqa_dev_1800_v2")
    parser.add_argument("--artifact-dir", default="data/outputs/hotpotqa_dev_1800_v2")
    args = parser.parse_args()

    train_records = read_jsonl(args.train_features)
    valid_records = read_jsonl(args.valid_features)
    test_records = read_jsonl(args.test_features)
    test_retrieval_records = read_jsonl(args.test_records)

    output_dir = Path(args.output_dir)
    artifact_dir = Path(args.artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for estimator in ESTIMATORS:
        for calibration_method in CALIBRATION_METHODS:
            runs.append(
                _run_experiment(
                    experiment_type="model_comparison",
                    estimator_name=estimator,
                    calibration_method=calibration_method,
                    feature_set_name="all_features",
                    feature_names=FEATURE_SETS["all_features"],
                    train_records=train_records,
                    valid_records=valid_records,
                    test_records=test_records,
                )
            )

    for feature_set_name, feature_names in FEATURE_SETS.items():
        runs.append(
            _run_experiment(
                experiment_type="feature_ablation",
                estimator_name="logistic_regression",
                calibration_method="isotonic",
                feature_set_name=feature_set_name,
                feature_names=feature_names,
                train_records=train_records,
                valid_records=valid_records,
                test_records=test_records,
            )
        )

    main_run = _find_run(runs, "model_comparison", "logistic_regression", "all_features", "isotonic")
    raw_lr_run = _find_run(runs, "model_comparison", "logistic_regression", "all_features", "identity")

    _write_score_metrics(output_dir / "score_metrics_test.csv", runs)
    _write_decision_metrics(output_dir / "decision_metrics_test.csv", runs)
    _write_coverage_curves(output_dir / "coverage_risk_curve_test.csv", runs)
    _write_main_comparison(output_dir / "main_comparison.csv", test_records, raw_lr_run, main_run)
    _write_model_comparison(output_dir / "model_comparison.csv", runs)
    _write_feature_ablation(output_dir / "feature_ablation.csv", runs)
    _write_question_type_breakdown(output_dir / "question_type_breakdown.csv", main_run, test_retrieval_records)
    _write_reliability_bins(output_dir / "reliability_bins_test.csv", main_run)
    _write_validation_summary(output_dir / "second_round_validation_summary.json", runs, train_records, valid_records, test_records)
    _write_summary(output_dir / "second_round_summary.md", runs, main_run, raw_lr_run)

    _write_main_artifacts(artifact_dir, main_run)
    print(
        {
            "output_dir": str(output_dir),
            "artifact_dir": str(artifact_dir),
            "model_comparison_rows": 3,
            "feature_ablation_rows": len(FEATURE_SETS),
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

    train_sufficiency_labels = _sufficiency_labels(train_records)
    valid_risk_labels = _risk_labels(valid_records)
    test_risk_labels = _risk_labels(test_records)

    model = train_estimator(estimator_name, train_rows, train_sufficiency_labels, feature_names)
    valid_sufficiency_scores = np.asarray(model.predict_proba(valid_rows), dtype=float)
    test_sufficiency_scores = np.asarray(model.predict_proba(test_rows), dtype=float)
    valid_raw_risk = 1.0 - valid_sufficiency_scores
    test_raw_risk = 1.0 - test_sufficiency_scores

    calibrator = make_calibrator(calibration_method)
    calibrator.fit(valid_raw_risk, valid_risk_labels)
    valid_risk = np.asarray(calibrator.predict(valid_raw_risk), dtype=float)
    test_risk = np.asarray(calibrator.predict(test_raw_risk), dtype=float)

    tau_answer, valid_decision_metrics = _select_tau(valid_risk_labels, valid_risk)
    test_decision_metrics = decision_metrics_from_risk(test_risk_labels, test_risk, tau_answer)

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
        "test_labels": test_risk_labels,
        "valid_raw_risk": valid_raw_risk,
        "test_raw_risk": test_raw_risk,
        "valid_risk": valid_risk,
        "test_risk": test_risk,
        "valid_sufficiency_scores": valid_sufficiency_scores,
        "test_sufficiency_scores": test_sufficiency_scores,
        "tau_answer": tau_answer,
        "valid_decision_metrics": valid_decision_metrics,
        "test_decision_metrics": test_decision_metrics,
        "raw_brier": brier_score(test_risk_labels, test_raw_risk),
        "raw_ece": ece(test_risk_labels, test_raw_risk),
        "calibrated_brier": brier_score(test_risk_labels, test_risk),
        "calibrated_ece": ece(test_risk_labels, test_risk),
    }


def _feature_row(record: dict[str, Any], feature_names: list[str]) -> dict[str, float]:
    return {name: float(record[name]) for name in feature_names}


def _sufficiency_labels(records: list[dict[str, Any]]) -> list[int]:
    label_map = {"sufficient": 1, "insufficient": 0}
    return [label_map[str(record["sufficiency_label"])] for record in records]


def _risk_labels(records: list[dict[str, Any]]) -> np.ndarray:
    label_map = {"sufficient": 0, "insufficient": 1}
    return np.array([label_map[str(record["sufficiency_label"])] for record in records], dtype=int)


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
        elif metrics["decision_accuracy"] == best_metrics["decision_accuracy"]:
            if metrics["coverage"] > best_metrics["coverage"]:
                best_tau = tau
                best_metrics = metrics
    return best_tau, best_metrics


def _find_run(
    runs: list[dict[str, Any]],
    experiment_type: str,
    estimator: str,
    feature_set: str,
    calibration_method: str,
) -> dict[str, Any]:
    for run in runs:
        if (
            run["experiment_type"] == experiment_type
            and run["estimator"] == estimator
            and run["feature_set"] == feature_set
            and run["calibration_method"] == calibration_method
        ):
            return run
    raise ValueError(f"Run not found: {experiment_type}/{estimator}/{feature_set}/{calibration_method}")


def _base_metric_row(run: dict[str, Any]) -> dict[str, Any]:
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


def _write_score_metrics(path: Path, runs: list[dict[str, Any]]) -> None:
    rows = []
    for run in runs:
        rows.append(
            {
                key: value
                for key, value in _base_metric_row(run).items()
                if key
                in {
                    "experiment_type",
                    "estimator",
                    "feature_set",
                    "calibration_method",
                    "n_features",
                    "n_test",
                    "raw_brier",
                    "raw_ece",
                    "calibrated_brier",
                    "calibrated_ece",
                    "brier_delta",
                    "ece_delta",
                }
            }
        )
    _write_csv(
        path,
        rows,
        [
            "experiment_type",
            "estimator",
            "feature_set",
            "calibration_method",
            "n_features",
            "n_test",
            "raw_brier",
            "raw_ece",
            "calibrated_brier",
            "calibrated_ece",
            "brier_delta",
            "ece_delta",
        ],
    )


def _write_decision_metrics(path: Path, runs: list[dict[str, Any]]) -> None:
    rows = []
    for run in runs:
        base = _base_metric_row(run)
        rows.append(
            {
                "experiment_type": base["experiment_type"],
                "estimator": base["estimator"],
                "feature_set": base["feature_set"],
                "calibration_method": base["calibration_method"],
                "tau_answer": base["tau_answer"],
                "valid_decision_accuracy": run["valid_decision_metrics"]["decision_accuracy"],
                "valid_coverage": run["valid_decision_metrics"]["coverage"],
                "valid_selective_accuracy": run["valid_decision_metrics"]["selective_accuracy"],
                "test_decision_accuracy": base["decision_accuracy"],
                "test_coverage": base["coverage"],
                "test_selective_accuracy": base["selective_accuracy"],
                "test_abstention_rate": base["abstention_rate"],
                "n_test": base["n_test"],
            }
        )
    _write_csv(
        path,
        rows,
        [
            "experiment_type",
            "estimator",
            "feature_set",
            "calibration_method",
            "tau_answer",
            "valid_decision_accuracy",
            "valid_coverage",
            "valid_selective_accuracy",
            "test_decision_accuracy",
            "test_coverage",
            "test_selective_accuracy",
            "test_abstention_rate",
            "n_test",
        ],
    )


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
    _write_csv(
        path,
        rows,
        [
            "experiment_type",
            "estimator",
            "feature_set",
            "calibration_method",
            "tau_answer",
            "decision_accuracy",
            "coverage",
            "selective_accuracy",
        ],
    )


def _write_main_comparison(
    path: Path,
    test_records: list[dict[str, Any]],
    raw_lr_run: dict[str, Any],
    main_run: dict[str, Any],
) -> None:
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
            "estimator": "logistic_regression",
            "feature_set": "all_features",
            "calibration_method": "identity",
            "tau_answer": raw_lr_run["tau_answer"],
            "decision_accuracy": raw_lr_run["test_decision_metrics"]["decision_accuracy"],
            "coverage": raw_lr_run["test_decision_metrics"]["coverage"],
            "selective_accuracy": raw_lr_run["test_decision_metrics"]["selective_accuracy"],
            "abstention_rate": 1.0 - raw_lr_run["test_decision_metrics"]["coverage"],
            "risk_brier": raw_lr_run["calibrated_brier"],
            "risk_ece": raw_lr_run["calibrated_ece"],
            "n_test": len(test_records),
        },
        {
            "method": "CSR-RAG",
            "estimator": "logistic_regression",
            "feature_set": "all_features",
            "calibration_method": "isotonic",
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
    _write_csv(
        path,
        rows,
        [
            "method",
            "estimator",
            "feature_set",
            "calibration_method",
            "tau_answer",
            "decision_accuracy",
            "coverage",
            "selective_accuracy",
            "abstention_rate",
            "risk_brier",
            "risk_ece",
            "n_test",
        ],
    )


def _write_model_comparison(path: Path, runs: list[dict[str, Any]]) -> None:
    rows = []
    for estimator in ESTIMATORS:
        run = _find_run(runs, "model_comparison", estimator, "all_features", "isotonic")
        rows.append(_base_metric_row(run))
    _write_csv(path, rows, list(rows[0].keys()))


def _write_feature_ablation(path: Path, runs: list[dict[str, Any]]) -> None:
    rows = []
    for feature_set in FEATURE_SETS:
        run = _find_run(runs, "feature_ablation", "logistic_regression", feature_set, "isotonic")
        rows.append(_base_metric_row(run))
    _write_csv(path, rows, list(rows[0].keys()))


def _write_question_type_breakdown(
    path: Path,
    run: dict[str, Any],
    test_retrieval_records: list[dict[str, Any]],
) -> None:
    metadata_by_id = {record["id"]: record["metadata"] for record in test_retrieval_records}
    grouped: dict[str, list[int]] = {}
    for index, record_id in enumerate(run["test_ids"]):
        question_type = metadata_by_id[record_id].get("question_type", "unknown")
        grouped.setdefault(question_type, []).append(index)

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
                "tau_answer": run["tau_answer"],
                "decision_accuracy": metrics["decision_accuracy"],
                "coverage": metrics["coverage"],
                "selective_accuracy": metrics["selective_accuracy"],
            }
        )
    _write_csv(
        path,
        rows,
        [
            "question_type",
            "n_test",
            "sufficient_count",
            "insufficient_count",
            "brier",
            "ece",
            "tau_answer",
            "decision_accuracy",
            "coverage",
            "selective_accuracy",
        ],
    )


def _write_reliability_bins(path: Path, run: dict[str, Any], n_bins: int = 10) -> None:
    labels = run["test_labels"]
    risks = run["test_risk"]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for bin_id in range(n_bins):
        left = bins[bin_id]
        right = bins[bin_id + 1]
        mask = (risks >= left) & (risks < right) if bin_id < n_bins - 1 else (risks >= left) & (risks <= right)
        n = int(mask.sum())
        if n:
            mean_predicted_risk = float(risks[mask].mean())
            observed_risk_rate = float(labels[mask].mean())
            abs_gap = abs(mean_predicted_risk - observed_risk_rate)
        else:
            mean_predicted_risk = ""
            observed_risk_rate = ""
            abs_gap = ""
        rows.append(
            {
                "estimator": run["estimator"],
                "feature_set": run["feature_set"],
                "calibration_method": run["calibration_method"],
                "bin_id": bin_id,
                "bin_left": left,
                "bin_right": right,
                "n": n,
                "mean_predicted_risk": mean_predicted_risk,
                "observed_risk_rate": observed_risk_rate,
                "abs_gap": abs_gap,
            }
        )
    _write_csv(
        path,
        rows,
        [
            "estimator",
            "feature_set",
            "calibration_method",
            "bin_id",
            "bin_left",
            "bin_right",
            "n",
            "mean_predicted_risk",
            "observed_risk_rate",
            "abs_gap",
        ],
    )


def _write_validation_summary(
    path: Path,
    runs: list[dict[str, Any]],
    train_records: list[dict[str, Any]],
    valid_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
) -> None:
    main_run = _find_run(runs, "model_comparison", "logistic_regression", "all_features", "isotonic")
    unique_scores = {round(float(score), 8) for score in main_run["test_risk"]}
    summary = {
        "splits": {
            "train_records": len(train_records),
            "valid_records": len(valid_records),
            "test_records": len(test_records),
        },
        "experiments": {
            "total_runs": len(runs),
            "estimators": ESTIMATORS,
            "calibration_methods": CALIBRATION_METHODS,
            "feature_sets": list(FEATURE_SETS.keys()),
        },
        "main_risk": {
            "unique_score_count": len(unique_scores),
            "min_risk": float(main_run["test_risk"].min()),
            "max_risk": float(main_run["test_risk"].max()),
            "all_binary": all(score in {0.0, 1.0} for score in unique_scores),
        },
        "tables": {
            "model_comparison_rows": len(ESTIMATORS),
            "feature_ablation_rows": len(FEATURE_SETS),
            "coverage_curve_rows": len(runs) * len(TAU_GRID),
        },
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_summary(path: Path, runs: list[dict[str, Any]], main_run: dict[str, Any], raw_lr_run: dict[str, Any]) -> None:
    model_runs = [_find_run(runs, "model_comparison", estimator, "all_features", "isotonic") for estimator in ESTIMATORS]
    best_model = max(model_runs, key=lambda run: run["test_decision_metrics"]["decision_accuracy"])
    ablation_runs = [
        _find_run(runs, "feature_ablation", "logistic_regression", feature_set, "isotonic")
        for feature_set in FEATURE_SETS
    ]
    best_ablation = max(ablation_runs, key=lambda run: run["test_decision_metrics"]["decision_accuracy"])

    lines = [
        "# CSR-RAG 第二轮实验结果摘要",
        "",
        "## 1. 实验定位",
        "",
        "第二轮实验用于把第一轮 credible prototype 推进为 paper-ready evidence。当前仍只研究检索充分性风险建模，不接最终 QA 生成，也不扩展 refine / clarify。",
        "",
        "## 2. 主结果",
        "",
        f"- CSR-RAG 主方法：LogisticRegression + isotonic calibration + all_features",
        f"- test decision accuracy：{main_run['test_decision_metrics']['decision_accuracy']:.4f}",
        f"- test coverage：{main_run['test_decision_metrics']['coverage']:.4f}",
        f"- test selective accuracy：{main_run['test_decision_metrics']['selective_accuracy']:.4f}",
        f"- calibrated Brier：{main_run['calibrated_brier']:.4f}",
        f"- calibrated ECE：{main_run['calibrated_ece']:.4f}",
        "",
        "## 3. 与未校准 CSR 的关系",
        "",
        f"- Uncalibrated CSR decision accuracy：{raw_lr_run['test_decision_metrics']['decision_accuracy']:.4f}",
        f"- CSR-RAG decision accuracy：{main_run['test_decision_metrics']['decision_accuracy']:.4f}",
        f"- raw ECE：{main_run['raw_ece']:.4f}",
        f"- calibrated ECE：{main_run['calibrated_ece']:.4f}",
        "",
        "如果 ECE 没有全面下降，论文表述应收敛为：calibration changes the coverage-risk trade-off，而不是 calibration always improves score quality。",
        "",
        "## 4. 模型与特征结论",
        "",
        f"- isotonic 下表现最好的 estimator：{best_model['estimator']}，decision accuracy = {best_model['test_decision_metrics']['decision_accuracy']:.4f}",
        f"- LR 消融中表现最好的 feature set：{best_ablation['feature_set']}，decision accuracy = {best_ablation['test_decision_metrics']['decision_accuracy']:.4f}",
        "",
        "非线性模型如果强于 LogisticRegression，应在论文中作为 stronger estimator variant，而不是替换 CSR-RAG 主线。主线仍然保持轻量、可解释、可复现。",
        "",
        "## 5. 当前能支撑的论文主张",
        "",
        "- 检索充分性预测比 always-answer 更可靠。",
        "- 不同 estimator 和 feature group 会明显影响选择性回答效果。",
        "- 校准模块影响 coverage 与 selective accuracy 的取舍。",
        "- 当前实验仍不能声称最终 QA 生成质量提升，因为尚未接生成评测。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_main_artifacts(artifact_dir: Path, main_run: dict[str, Any]) -> None:
    risk_records = []
    decision_records = []
    for record_id, label, sufficiency_score, raw_risk, risk in zip(
        main_run["test_ids"],
        main_run["test_labels"],
        main_run["test_sufficiency_scores"],
        main_run["test_raw_risk"],
        main_run["test_risk"],
    ):
        label_name = "insufficient" if int(label) == 1 else "sufficient"
        risk_records.append(
            {
                "id": record_id,
                "sufficiency_label": label_name,
                "sufficiency_score": float(sufficiency_score),
                "raw_risk_score": float(raw_risk),
                "risk_score": float(risk),
                "calibration_method": main_run["calibration_method"],
                "estimator": main_run["estimator"],
                "feature_set": main_run["feature_set"],
            }
        )
        decision_records.append(
            {
                "id": record_id,
                "risk_score": float(risk),
                "decision": "answer" if float(risk) <= float(main_run["tau_answer"]) else "abstain",
                "tau_answer": float(main_run["tau_answer"]),
            }
        )
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
