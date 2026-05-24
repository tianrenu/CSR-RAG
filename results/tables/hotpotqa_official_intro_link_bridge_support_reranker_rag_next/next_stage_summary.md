# CSR-RAG Next-Stage No-API Experiment Summary

## Purpose

This run pauses paper writing and focuses on framework/experiment improvement. It adds deployable v3 features, no-API baselines, strict valid-only policy selection, failure taxonomy, QA100 rescoring, and bootstrap confidence intervals.

## Main Findings

- Best balanced decision accuracy: `logistic_regression/v3_all/platt` with decision accuracy 0.7914, coverage 0.7077, insufficient answer rate 0.3705.
- Best non-extreme policy with test coverage >= 0.85: `logistic_regression/v3_no_query/identity` / `reliable@cov85` with coverage 0.8543, selective accuracy 0.7674, insufficient answer rate 0.6156.
- QA rescore rows: 0.
- Failure taxonomy rows: 8.

## Interpretation

Use these results to decide which framework optimization is promising. Do not treat v3 as final until it improves insufficient-answer risk under reasonable coverage and survives stronger baselines.
