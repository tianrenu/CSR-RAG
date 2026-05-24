from __future__ import annotations

import argparse
import json
from pathlib import Path

from csrrag.decision.threshold import answer_or_abstain
from csrrag.evaluation.metrics import decision_metrics_from_risk
from csrrag.utils.io import read_jsonl, write_jsonl


TAU_GRID = [round(x, 2) for x in [i / 100 for i in range(5, 100, 5)]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Select tau on valid and apply answer/abstain to test.")
    parser.add_argument("--valid-input", required=True, help="Validation calibrated risk JSONL.")
    parser.add_argument("--test-input", required=True, help="Test calibrated risk JSONL.")
    parser.add_argument("--output-threshold-metadata", required=True, help="Path to output selected threshold JSON.")
    parser.add_argument("--output-valid-decisions", required=True, help="Path to output validation decisions JSONL.")
    parser.add_argument("--output-test-decisions", required=True, help="Path to output test decisions JSONL.")
    args = parser.parse_args()

    valid_records = read_jsonl(args.valid_input)
    test_records = read_jsonl(args.test_input)

    chosen_tau, chosen_valid_metrics = _select_tau(valid_records)
    valid_decisions = _apply_decision(valid_records, chosen_tau)
    test_decisions = _apply_decision(test_records, chosen_tau)

    write_jsonl(args.output_valid_decisions, valid_decisions)
    write_jsonl(args.output_test_decisions, test_decisions)

    metadata = {
        "tau_answer": chosen_tau,
        "selection_split": "valid",
        "valid_decision_accuracy": chosen_valid_metrics["decision_accuracy"],
        "valid_coverage": chosen_valid_metrics["coverage"],
        "valid_selective_accuracy": chosen_valid_metrics["selective_accuracy"],
    }
    output_path = Path(args.output_threshold_metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metadata)


def _select_tau(records: list[dict[str, object]]) -> tuple[float, dict[str, float]]:
    y_true = [1 if record["sufficiency_label"] == "insufficient" else 0 for record in records]
    risk_scores = [float(record["risk_score"]) for record in records]

    best_tau = TAU_GRID[0]
    best_metrics = decision_metrics_from_risk(y_true, risk_scores, best_tau)
    for tau in TAU_GRID[1:]:
        metrics = decision_metrics_from_risk(y_true, risk_scores, tau)
        if metrics["decision_accuracy"] > best_metrics["decision_accuracy"]:
            best_tau = tau
            best_metrics = metrics
        elif metrics["decision_accuracy"] == best_metrics["decision_accuracy"]:
            if metrics["coverage"] > best_metrics["coverage"]:
                best_tau = tau
                best_metrics = metrics
    return best_tau, best_metrics


def _apply_decision(records: list[dict[str, object]], tau_answer: float) -> list[dict[str, object]]:
    output = []
    for record in records:
        decision = answer_or_abstain(float(record["risk_score"]), tau_answer)
        output.append(
            {
                "id": record["id"],
                "risk_score": float(record["risk_score"]),
                "decision": decision,
                "tau_answer": float(tau_answer),
            }
        )
    return output


if __name__ == "__main__":
    main()
