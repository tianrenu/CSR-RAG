# CSR-RAG Next-Stage No-API Experiment Summary

## Purpose

This run pauses paper writing and focuses on framework/experiment improvement. It adds deployable v3 features, no-API baselines, strict valid-only policy selection, failure taxonomy, QA100 rescoring, and bootstrap confidence intervals.

## Main Findings

- Best balanced decision accuracy: `logistic_regression/v3_no_query/isotonic` with decision accuracy 0.8381, coverage 0.2572, insufficient answer rate 0.0952.
- Best non-extreme policy with test coverage >= 0.85: `gradient_boosting/retrieval_quality_only/identity` / `reliable@cov85` with coverage 0.8723, selective accuracy 0.3196, insufficient answer rate 0.8271.
- QA rescore rows: 0.
- Failure taxonomy rows: 8.

## Interpretation

Use these results to decide which framework optimization is promising. Do not treat v3 as final until it improves insufficient-answer risk under reasonable coverage and survives stronger baselines.
