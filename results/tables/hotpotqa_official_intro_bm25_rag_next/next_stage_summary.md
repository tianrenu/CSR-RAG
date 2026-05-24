# CSR-RAG Next-Stage No-API Experiment Summary

## Purpose

This run pauses paper writing and focuses on framework/experiment improvement. It adds deployable v3 features, no-API baselines, strict valid-only policy selection, failure taxonomy, QA100 rescoring, and bootstrap confidence intervals.

## Main Findings

- Best balanced decision accuracy: `logistic_regression/v3_all/identity` with decision accuracy 0.8273, coverage 0.2437, insufficient answer rate 0.0679.
- Best non-extreme policy with test coverage >= 0.85: `logistic_regression/retrieval_quality_only/isotonic` / `reliable@cov85` with coverage 0.8903, selective accuracy 0.3646, insufficient answer rate 0.8375.
- QA rescore rows: 0.
- Failure taxonomy rows: 8.

## Interpretation

Use these results to decide which framework optimization is promising. Do not treat v3 as final until it improves insufficient-answer risk under reasonable coverage and survives stronger baselines.
