# CSR-RAG Next-Stage No-API Experiment Summary

## Purpose

This run pauses paper writing and focuses on framework/experiment improvement. It adds deployable v3 features, no-API baselines, strict valid-only policy selection, failure taxonomy, QA100 rescoring, and bootstrap confidence intervals.

## Main Findings

- Best balanced decision accuracy: `logistic_regression/v3_all/identity` with decision accuracy 0.8556, coverage 0.8815, insufficient answer rate 0.5660.
- Best non-extreme policy with test coverage >= 0.85: `logistic_regression/v3_no_query/identity` / `balanced` with coverage 0.8556, selective accuracy 0.8831, insufficient answer rate 0.5094.
- QA rescore rows: 7.
- Failure taxonomy rows: 9.

## Interpretation

Use these results to decide which framework optimization is promising. Do not treat v3 as final until it improves insufficient-answer risk under reasonable coverage and survives stronger baselines.
