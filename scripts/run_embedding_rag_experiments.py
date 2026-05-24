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
from csrrag.experiments.feature_sets import (
    EMBEDDING_ABLATIONS,
    EMBEDDING_FEATURES,
    FORBIDDEN_FEATURE_FIELDS,
)
from csrrag.features.basic import extract_basic_features
from csrrag.models.baseline import train_estimator
from csrrag.utils.io import read_jsonl, write_jsonl


TAU_GRID = [round(i / 100, 2) for i in range(5, 100, 5)]
ESTIMATORS = ["logistic_regression", "random_forest", "gradient_boosting"]
CALIBRATION_METHODS = ["identity", "platt", "isotonic"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CSR-RAG experiments on the embedding-retrieval distribution.")
    parser.add_argument("--split-dir", default="data/processed/hotpotqa_embedding_splits_1800")
    parser.add_argument("--feature-dir", default="data/features/hotpotqa_embedding_1800")
    parser.add_argument("--output-dir", default="results/tables/hotpotqa_embedding_rag_1800")
    parser.add_argument("--artifact-dir", default="data/outputs/hotpotqa_embedding_rag_1800")
    args = parser.parse_args()

    split_records = {split: read_jsonl(Path(args.split_dir) / f"{split}.jsonl") for split in ("train", "valid", "test")}
    _validate_split_records(split_records)
    feature_records = _build_or_load_features(split_records, Path(args.feature_dir))
    _validate_feature_inputs(feature_records["train"] + feature_records["valid"] + feature_records["test"])

    output_dir = Path(args.output_dir)
    artifact_dir = Path(args.artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for estimator in ESTIMATORS:
        runs.append(
            _run_experiment(
                "model_comparison",
                estimator,
                "isotonic",
                "all_embedding",
                EMBEDDING_FEATURES,
                feature_records,
            )
        )
    for calibration_method in CALIBRATION_METHODS:
        runs.append(
            _run_experiment(
                "calibration_comparison",
                "logistic_regression",
                calibration_method,
                "all_embedding",
                EMBEDDING_FEATURES,
                feature_records,
            )
        )
    for feature_set_name, feature_names in EMBEDDING_ABLATIONS.items():
        runs.append(
            _run_experiment(
                "feature_ablation",
                "logistic_regression",
                "isotonic",
                feature_set_name,
                feature_names,
                feature_records,
            )
        )

    main_run = _find_run(runs, "model_comparison", "logistic_regression", "all_embedding", "isotonic")
    raw_run = _find_run(runs, "calibration_comparison", "logistic_regression", "all_embedding", "identity")

    _write_main_comparison(output_dir / "main_comparison.csv", feature_records["test"], raw_run, main_run)
    _write_run_table(output_dir / "model_comparison.csv", [run for run in runs if run["experiment_type"] == "model_comparison"])
    _write_run_table(output_dir / "calibration_comparison.csv", [run for run in runs if run["experiment_type"] == "calibration_comparison"])
    _write_run_table(output_dir / "feature_ablation.csv", [run for run in runs if run["experiment_type"] == "feature_ablation"])
    _write_coverage_curves(output_dir / "coverage_risk_curve_test.csv", runs)
    _write_reliability_bins(output_dir / "reliability_bins_test.csv", main_run)
    _write_data_summary(output_dir / "embedding_retrieval_data_summary.csv", split_records)
    _write_validation_summary(output_dir / "validation_summary.json", split_records, feature_records, runs, main_run)
    _write_summary(output_dir / "embedding_rag_summary.md", split_records, raw_run, main_run, runs)
    _write_main_artifacts(artifact_dir, main_run, feature_records["test"])

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "artifact_dir": str(artifact_dir),
                "main_decision_accuracy": main_run["test_decision_metrics"]["decision_accuracy"],
                "main_coverage": main_run["test_decision_metrics"]["coverage"],
                "main_selective_accuracy": main_run["test_decision_metrics"]["selective_accuracy"],
                "runs": len(runs),
            },
            ensure_ascii=False,
        )
    )


def _build_or_load_features(
    split_records: dict[str, list[dict[str, Any]]],
    feature_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    feature_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, list[dict[str, Any]]] = {}
    for split, records in split_records.items():
        feature_path = feature_dir / f"{split}_features.jsonl"
        rows = []
        for record in records:
            features = extract_basic_features(record)
            rows.append({"id": record["id"], **features, "sufficiency_label": record["sufficiency_label"]})
        write_jsonl(feature_path, rows)
        result[split] = rows
    return result


def _run_experiment(
    experiment_type: str,
    estimator_name: str,
    calibration_method: str,
    feature_set_name: str,
    feature_names: list[str],
    feature_records: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    train_records = feature_records["train"]
    valid_records = feature_records["valid"]
    test_records = feature_records["test"]
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


def _validate_split_records(split_records: dict[str, list[dict[str, Any]]]) -> None:
    split_ids = {split: {record["metadata"]["original_id"] for record in records} for split, records in split_records.items()}
    _require(split_ids["train"].isdisjoint(split_ids["valid"]), "train and valid original_id overlap.")
    _require(split_ids["train"].isdisjoint(split_ids["test"]), "train and test original_id overlap.")
    _require(split_ids["valid"].isdisjoint(split_ids["test"]), "valid and test original_id overlap.")
    for split, records in split_records.items():
        _require(records, f"{split} split is empty.")
        for record in records:
            _require(len(record.get("retrieved_docs", [])) == 5, f"{record['id']} does not have top_k=5.")
            _require(all("is_support" not in doc for doc in record["retrieved_docs"]), f"{record['id']} exposes is_support.")
        labels = Counter(record["sufficiency_label"] for record in records)
        _require(labels["sufficient"] > 0, f"{split} has no sufficient records.")
        _require(labels["insufficient"] > 0, f"{split} has no insufficient records.")


def _validate_feature_inputs(records: list[dict[str, Any]]) -> None:
    required = set(EMBEDDING_FEATURES)
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


def _write_main_comparison(path: Path, test_records: list[dict[str, Any]], raw_run: dict[str, Any], main_run: dict[str, Any]) -> None:
    labels = _risk_labels(test_records)
    sufficient_rate = float((labels == 0).mean()) if len(labels) else 0.0
    rows = [
        {
            "method": "Naive RAG",
            "estimator": "none",
            "feature_set": "none",
            "calibration_method": "none",
            "tau_answer": "",
            "decision_accuracy": sufficient_rate,
            "coverage": 1.0,
            "selective_accuracy": sufficient_rate,
            "abstention_rate": 0.0,
            "risk_brier": "",
            "risk_ece": "",
            "n_test": len(test_records),
        },
        _main_row("Uncalibrated CSR", raw_run),
        _main_row("CSR-RAG", main_run),
    ]
    _write_csv(path, rows, list(rows[0].keys()))


def _main_row(method: str, run: dict[str, Any]) -> dict[str, Any]:
    metrics = run["test_decision_metrics"]
    return {
        "method": method,
        "estimator": run["estimator"],
        "feature_set": run["feature_set"],
        "calibration_method": run["calibration_method"],
        "tau_answer": run["tau_answer"],
        "decision_accuracy": metrics["decision_accuracy"],
        "coverage": metrics["coverage"],
        "selective_accuracy": metrics["selective_accuracy"],
        "abstention_rate": 1.0 - metrics["coverage"],
        "risk_brier": run["calibrated_brier"],
        "risk_ece": run["calibrated_ece"],
        "n_test": run["n_test"],
    }


def _write_run_table(path: Path, runs: list[dict[str, Any]]) -> None:
    rows = []
    for run in runs:
        metrics = run["test_decision_metrics"]
        rows.append(
            {
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
                "decision_accuracy": metrics["decision_accuracy"],
                "coverage": metrics["coverage"],
                "selective_accuracy": metrics["selective_accuracy"],
                "abstention_rate": 1.0 - metrics["coverage"],
            }
        )
    _write_csv(path, rows, list(rows[0].keys()))


def _write_coverage_curves(path: Path, runs: list[dict[str, Any]]) -> None:
    rows = []
    for run in runs:
        labels = run["test_labels"]
        risk = run["test_risk"]
        for tau in TAU_GRID:
            metrics = decision_metrics_from_risk(labels, risk, tau)
            rows.append(
                {
                    "experiment_type": run["experiment_type"],
                    "estimator": run["estimator"],
                    "feature_set": run["feature_set"],
                    "calibration_method": run["calibration_method"],
                    "tau_answer": tau,
                    "decision_accuracy": metrics["decision_accuracy"],
                    "coverage": metrics["coverage"],
                    "selective_accuracy": metrics["selective_accuracy"],
                    "risk": 1.0 - metrics["selective_accuracy"] if metrics["coverage"] else "",
                }
            )
    _write_csv(path, rows, list(rows[0].keys()))


def _write_reliability_bins(path: Path, run: dict[str, Any], n_bins: int = 10) -> None:
    labels = np.asarray(run["test_labels"], dtype=float)
    risk = np.asarray(run["test_risk"], dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        left, right = bins[i], bins[i + 1]
        mask = (risk >= left) & (risk < right) if i < n_bins - 1 else (risk >= left) & (risk <= right)
        if mask.any():
            avg_risk = float(risk[mask].mean())
            empirical_risk = float(labels[mask].mean())
            count = int(mask.sum())
        else:
            avg_risk = ""
            empirical_risk = ""
            count = 0
        rows.append(
            {
                "bin": i,
                "left": left,
                "right": right,
                "count": count,
                "avg_predicted_risk": avg_risk,
                "empirical_risk": empirical_risk,
            }
        )
    _write_csv(path, rows, list(rows[0].keys()))


def _write_data_summary(path: Path, split_records: dict[str, list[dict[str, Any]]]) -> None:
    rows = []
    for split, records in split_records.items():
        labels = Counter(record["sufficiency_label"] for record in records)
        qtypes = Counter(record["metadata"].get("question_type", "unknown") for record in records)
        rows.append(
            {
                "split": split,
                "n_records": len(records),
                "sufficient": labels["sufficient"],
                "insufficient": labels["insufficient"],
                "sufficient_rate": labels["sufficient"] / len(records) if records else 0.0,
                "bridge": qtypes["bridge"],
                "comparison": qtypes["comparison"],
                "unknown": qtypes["unknown"],
            }
        )
    _write_csv(path, rows, list(rows[0].keys()))


def _write_validation_summary(
    path: Path,
    split_records: dict[str, list[dict[str, Any]]],
    feature_records: dict[str, list[dict[str, Any]]],
    runs: list[dict[str, Any]],
    main_run: dict[str, Any],
) -> None:
    labels = Counter(record["sufficiency_label"] for record in split_records["test"])
    risk_unique = len({round(float(score), 8) for score in main_run["test_risk"]})
    summary = {
        "split_counts": {split: len(records) for split, records in split_records.items()},
        "feature_counts": {split: len(records) for split, records in feature_records.items()},
        "test_label_counts": dict(labels),
        "risk_unique_count": risk_unique,
        "risk_min": float(np.min(main_run["test_risk"])),
        "risk_max": float(np.max(main_run["test_risk"])),
        "coverage_curve_expected_min_rows": len(runs) * len(TAU_GRID),
        "main_tau_answer": main_run["tau_answer"],
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_summary(path: Path, split_records: dict[str, list[dict[str, Any]]], raw_run: dict[str, Any], main_run: dict[str, Any], runs: list[dict[str, Any]]) -> None:
    labels = Counter(record["sufficiency_label"] for record in split_records["test"])
    main_metrics = main_run["test_decision_metrics"]
    raw_metrics = raw_run["test_decision_metrics"]
    best_model = max((run for run in runs if run["experiment_type"] == "model_comparison"), key=lambda run: run["test_decision_metrics"]["decision_accuracy"])
    text = f"""# Embedding RAG 实验结果摘要

## 当前定位

这一轮是 CSR-RAG 的真实检索分布实验：检索由 `text-embedding-v4` 的 embedding top-5 决定，风险模型只在 embedding retrieval 的 train split 上训练，calibration 和阈值选择只使用 valid split，test split 只用于最终报告。

## 主结果

- Test sufficient / insufficient: {labels["sufficient"]} / {labels["insufficient"]}
- Naive RAG decision accuracy: {labels["sufficient"] / sum(labels.values()):.4f}
- Uncalibrated CSR decision accuracy: {raw_metrics["decision_accuracy"]:.4f}, coverage: {raw_metrics["coverage"]:.4f}, selective accuracy: {raw_metrics["selective_accuracy"]:.4f}
- CSR-RAG decision accuracy: {main_metrics["decision_accuracy"]:.4f}, coverage: {main_metrics["coverage"]:.4f}, selective accuracy: {main_metrics["selective_accuracy"]:.4f}
- CSR-RAG Brier / ECE: {main_run["calibrated_brier"]:.4f} / {main_run["calibrated_ece"]:.4f}

## 解释

如果 CSR-RAG 高于 Naive RAG，论文主实验可以主张：真实 embedding 检索下，轻量 sufficiency risk model 能改善 answer/abstain 决策可靠性。如果提升有限，则应把结论收敛为：controlled sufficiency modeling 有效，但真实 RAG 的最终收益受到 embedding retriever 召回率和 LLM 答案行为共同限制。

当前模型对比中 test decision accuracy 最高的是 `{best_model["estimator"]}`，但主方法仍固定为 LogisticRegression + isotonic，非线性模型只作为增强变体讨论。
"""
    path.write_text(text, encoding="utf-8")


def _write_main_artifacts(artifact_dir: Path, run: dict[str, Any], test_records: list[dict[str, Any]]) -> None:
    risk_rows = []
    decision_rows = []
    for record, label, risk in zip(test_records, run["test_labels"], run["test_risk"]):
        decision = "answer" if float(risk) <= run["tau_answer"] else "abstain"
        risk_rows.append({"id": record["id"], "risk_label": int(label), "risk_score": float(risk)})
        decision_rows.append({"id": record["id"], "risk_label": int(label), "risk_score": float(risk), "tau_answer": run["tau_answer"], "decision": decision})
    write_jsonl(artifact_dir / "main_test_risk.jsonl", risk_rows)
    write_jsonl(artifact_dir / "main_test_decisions.jsonl", decision_rows)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
