from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from csrrag.calibration.methods import make_calibrator
from csrrag.decision.threshold import answer_or_abstain
from csrrag.evaluation.metrics import brier_score, decision_metrics_from_risk, ece
from csrrag.utils.io import read_jsonl, write_jsonl


TAU_GRID = [round(i / 100, 2) for i in range(5, 100, 5)]
CALIBRATION_METHODS = ["identity", "platt", "isotonic"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare calibration and selective CSR-RAG variants.")
    parser.add_argument("--valid-scores", required=True, help="Validation sufficiency score JSONL.")
    parser.add_argument("--test-scores", required=True, help="Test sufficiency score JSONL.")
    parser.add_argument(
        "--calibration-output-dir",
        required=True,
        help="Directory for calibration comparison tables.",
    )
    parser.add_argument("--main-output", required=True, help="CSV path for the main method comparison table.")
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Optional directory for per-calibrator risk and decision JSONL artifacts.",
    )
    parser.add_argument(
        "--main-calibration-method",
        default="isotonic",
        choices=CALIBRATION_METHODS,
        help="Calibrator used as the full CSR-RAG variant in the main comparison.",
    )
    args = parser.parse_args()

    valid_records = read_jsonl(args.valid_scores)
    test_records = read_jsonl(args.test_scores)

    valid_labels = _risk_labels(valid_records)
    test_labels = _risk_labels(test_records)
    valid_raw_risk = _raw_risk(valid_records)
    test_raw_risk = _raw_risk(test_records)

    calibration_rows = []
    decision_rows = []
    curve_rows = []
    per_method: dict[str, dict[str, object]] = {}
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else None

    for method in CALIBRATION_METHODS:
        calibrator = make_calibrator(method)
        calibrator.fit(valid_raw_risk, valid_labels)
        valid_risk = np.asarray(calibrator.predict(valid_raw_risk), dtype=float)
        test_risk = np.asarray(calibrator.predict(test_raw_risk), dtype=float)

        tau_answer, valid_metrics = _select_tau(valid_labels, valid_risk)
        test_metrics = decision_metrics_from_risk(test_labels, test_risk, tau_answer)

        score_row = _score_row(method, test_labels, test_raw_risk, test_risk, len(valid_records), len(test_records))
        decision_row = _decision_row(method, tau_answer, valid_metrics, test_metrics, len(test_records))
        calibration_rows.append(score_row)
        decision_rows.append(decision_row)
        curve_rows.extend(_curve_rows(method, test_labels, test_risk))

        if artifact_dir is not None:
            _write_method_artifacts(
                artifact_dir=artifact_dir,
                method=method,
                valid_records=valid_records,
                test_records=test_records,
                valid_raw_risk=valid_raw_risk,
                test_raw_risk=test_raw_risk,
                valid_risk=valid_risk,
                test_risk=test_risk,
                tau_answer=tau_answer,
                valid_metrics=valid_metrics,
                test_metrics=test_metrics,
            )

        per_method[method] = {
            "tau_answer": tau_answer,
            "test_metrics": test_metrics,
            "risk_brier": brier_score(test_labels, test_risk),
            "risk_ece": ece(test_labels, test_risk),
        }

    calibration_output_dir = Path(args.calibration_output_dir)
    _write_csv(
        calibration_output_dir / "score_metrics_test.csv",
        calibration_rows,
        [
            "method",
            "n_valid",
            "n_test",
            "raw_brier",
            "raw_ece",
            "calibrated_brier",
            "calibrated_ece",
            "brier_delta",
            "ece_delta",
        ],
    )
    _write_csv(
        calibration_output_dir / "decision_metrics_test.csv",
        decision_rows,
        [
            "method",
            "tau_answer",
            "valid_decision_accuracy",
            "valid_coverage",
            "valid_selective_accuracy",
            "test_decision_accuracy",
            "test_coverage",
            "test_selective_accuracy",
            "n_test",
        ],
    )
    _write_csv(
        calibration_output_dir / "coverage_risk_curve_test.csv",
        curve_rows,
        ["method", "tau_answer", "decision_accuracy", "coverage", "selective_accuracy"],
    )
    _write_main_comparison(
        Path(args.main_output),
        test_labels=test_labels,
        raw_method=per_method["identity"],
        calibrated_method=per_method[args.main_calibration_method],
        main_calibration_method=args.main_calibration_method,
    )

    print(
        {
            "calibration_output_dir": str(calibration_output_dir),
            "main_output": args.main_output,
            "main_calibration_method": args.main_calibration_method,
        }
    )


def _risk_labels(records: list[dict[str, object]]) -> np.ndarray:
    label_map = {"sufficient": 0, "insufficient": 1}
    return np.array([label_map[str(record["sufficiency_label"])] for record in records], dtype=int)


def _raw_risk(records: list[dict[str, object]]) -> np.ndarray:
    return np.array([1.0 - float(record["sufficiency_score"]) for record in records], dtype=float)


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


def _score_row(
    method: str,
    test_labels: Iterable[int],
    test_raw_risk: Iterable[float],
    test_risk: Iterable[float],
    n_valid: int,
    n_test: int,
) -> dict[str, object]:
    raw_brier = brier_score(test_labels, test_raw_risk)
    raw_ece = ece(test_labels, test_raw_risk)
    calibrated_brier = brier_score(test_labels, test_risk)
    calibrated_ece = ece(test_labels, test_risk)
    return {
        "method": method,
        "n_valid": n_valid,
        "n_test": n_test,
        "raw_brier": raw_brier,
        "raw_ece": raw_ece,
        "calibrated_brier": calibrated_brier,
        "calibrated_ece": calibrated_ece,
        "brier_delta": calibrated_brier - raw_brier,
        "ece_delta": calibrated_ece - raw_ece,
    }


def _decision_row(
    method: str,
    tau_answer: float,
    valid_metrics: dict[str, float],
    test_metrics: dict[str, float],
    n_test: int,
) -> dict[str, object]:
    return {
        "method": method,
        "tau_answer": tau_answer,
        "valid_decision_accuracy": valid_metrics["decision_accuracy"],
        "valid_coverage": valid_metrics["coverage"],
        "valid_selective_accuracy": valid_metrics["selective_accuracy"],
        "test_decision_accuracy": test_metrics["decision_accuracy"],
        "test_coverage": test_metrics["coverage"],
        "test_selective_accuracy": test_metrics["selective_accuracy"],
        "n_test": n_test,
    }


def _curve_rows(method: str, test_labels: Iterable[int], test_risk: Iterable[float]) -> list[dict[str, object]]:
    rows = []
    for tau in TAU_GRID:
        row = decision_metrics_from_risk(test_labels, test_risk, tau)
        rows.append({"method": method, **row})
    return rows


def _write_method_artifacts(
    artifact_dir: Path,
    method: str,
    valid_records: list[dict[str, object]],
    test_records: list[dict[str, object]],
    valid_raw_risk: np.ndarray,
    test_raw_risk: np.ndarray,
    valid_risk: np.ndarray,
    test_risk: np.ndarray,
    tau_answer: float,
    valid_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    valid_risk_records = _risk_records(valid_records, valid_raw_risk, valid_risk, method)
    test_risk_records = _risk_records(test_records, test_raw_risk, test_risk, method)
    write_jsonl(artifact_dir / f"{method}_valid_risk.jsonl", valid_risk_records)
    write_jsonl(artifact_dir / f"{method}_test_risk.jsonl", test_risk_records)
    write_jsonl(artifact_dir / f"{method}_valid_decisions.jsonl", _decision_records(valid_risk_records, tau_answer))
    write_jsonl(artifact_dir / f"{method}_test_decisions.jsonl", _decision_records(test_risk_records, tau_answer))
    metadata = {
        "method": method,
        "tau_answer": tau_answer,
        "selection_split": "valid",
        "valid_decision_accuracy": valid_metrics["decision_accuracy"],
        "valid_coverage": valid_metrics["coverage"],
        "valid_selective_accuracy": valid_metrics["selective_accuracy"],
        "test_decision_accuracy": test_metrics["decision_accuracy"],
        "test_coverage": test_metrics["coverage"],
        "test_selective_accuracy": test_metrics["selective_accuracy"],
    }
    (artifact_dir / f"{method}_threshold.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _risk_records(
    records: list[dict[str, object]],
    raw_risk: np.ndarray,
    risk: np.ndarray,
    method: str,
) -> list[dict[str, object]]:
    output = []
    for record, raw_value, risk_value in zip(records, raw_risk, risk):
        output.append(
            {
                "id": record["id"],
                "sufficiency_label": record["sufficiency_label"],
                "sufficiency_score": float(record["sufficiency_score"]),
                "raw_risk_score": float(raw_value),
                "risk_score": float(risk_value),
                "calibration_method": method,
            }
        )
    return output


def _decision_records(records: list[dict[str, object]], tau_answer: float) -> list[dict[str, object]]:
    return [
        {
            "id": record["id"],
            "risk_score": float(record["risk_score"]),
            "decision": answer_or_abstain(float(record["risk_score"]), tau_answer),
            "tau_answer": float(tau_answer),
        }
        for record in records
    ]


def _write_main_comparison(
    path: Path,
    test_labels: np.ndarray,
    raw_method: dict[str, object],
    calibrated_method: dict[str, object],
    main_calibration_method: str,
) -> None:
    naive_metrics = decision_metrics_from_risk(test_labels, np.zeros_like(test_labels, dtype=float), 0.95)
    rows = [
        {
            "method": "Naive RAG",
            "risk_source": "none",
            "calibration_method": "none",
            "tau_answer": "",
            "decision_accuracy": naive_metrics["decision_accuracy"],
            "coverage": naive_metrics["coverage"],
            "selective_accuracy": naive_metrics["selective_accuracy"],
            "abstention_rate": 1.0 - naive_metrics["coverage"],
            "risk_brier": "",
            "risk_ece": "",
            "n_test": len(test_labels),
        },
        {
            "method": "Uncalibrated CSR",
            "risk_source": "logistic_raw_risk",
            "calibration_method": "identity",
            "tau_answer": raw_method["tau_answer"],
            "decision_accuracy": raw_method["test_metrics"]["decision_accuracy"],
            "coverage": raw_method["test_metrics"]["coverage"],
            "selective_accuracy": raw_method["test_metrics"]["selective_accuracy"],
            "abstention_rate": 1.0 - raw_method["test_metrics"]["coverage"],
            "risk_brier": raw_method["risk_brier"],
            "risk_ece": raw_method["risk_ece"],
            "n_test": len(test_labels),
        },
        {
            "method": "CSR-RAG",
            "risk_source": "logistic_calibrated_risk",
            "calibration_method": main_calibration_method,
            "tau_answer": calibrated_method["tau_answer"],
            "decision_accuracy": calibrated_method["test_metrics"]["decision_accuracy"],
            "coverage": calibrated_method["test_metrics"]["coverage"],
            "selective_accuracy": calibrated_method["test_metrics"]["selective_accuracy"],
            "abstention_rate": 1.0 - calibrated_method["test_metrics"]["coverage"],
            "risk_brier": calibrated_method["risk_brier"],
            "risk_ece": calibrated_method["risk_ece"],
            "n_test": len(test_labels),
        },
    ]
    _write_csv(
        path,
        rows,
        [
            "method",
            "risk_source",
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


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
