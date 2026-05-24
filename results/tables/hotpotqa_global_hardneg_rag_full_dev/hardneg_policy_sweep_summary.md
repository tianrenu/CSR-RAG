# Hard-Negative CSR-RAG Policy Sweep Summary

## Purpose

This stress setting adds hard-negative retrieval records that are embedding-relevant but missing at least one supporting title. It is designed to test whether CSR-RAG can reduce insufficient-answer risk under a harder retrieval distribution.

## Main Findings

- LR risk_control@cov85 coverage: 1.0000
- LR risk_control@cov85 insufficient answer rate: 1.0000
- Target coverage>=0.85 and insufficient answer rate<0.50 met: False
- Best observed insufficient answer rate under coverage>=0.85: 1.0000
- Best coverage>=0.85 non-extreme policy: `logistic_regression` / `reliable@cov85` with coverage 1.0000, selective accuracy 0.1412, insufficient answer rate 1.0000
- Best calibrated insufficient-retrieval AUPRC: `logistic_regression_balanced` with AUPRC 0.9751

## Paper Use

Use this as a stress-setting result, not as a replacement for the natural global retrieval main result. If the target is not met, report it as a failure-mode analysis of the current lightweight risk estimator and the coverage target under an insufficient-heavy stress distribution.
