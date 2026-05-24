# Estimator-Policy Sweep Summary

## Purpose

This sweep checks whether a stronger sufficiency estimator can improve the coverage-risk trade-off under the global embedding retrieval setting. It does not call the embedding API or MiniMax; QA rows are recomputed from the existing 100-sample details.

## Main Findings

- Best balanced test decision accuracy: `gradient_boosting` with decision accuracy 0.8481, coverage 0.9111, insufficient answer rate 0.6604.
- Best policy with test coverage >= 0.85 among non-extreme policies: `gradient_boosting` / `reliable@cov85` with coverage 0.8852, selective accuracy 0.8661, insufficient answer rate 0.6038.
- Best calibrated insufficient-retrieval AUPRC: `logistic_regression` with AUPRC 0.5618 and AUROC 0.8463.
- Most conservative QA100 reliability row: `logistic_regression` / `high_precision@cov50` with coverage 0.5300, answered F1 0.7635, insufficient answer rate 0.1538.

## Paper Use

Keep LogisticRegression as the lightweight CSR-RAG main method. Use RF/GB only as stronger estimator variants if they improve the trade-off under valid-selected policies. The paper claim should stay focused on controllable reliability/coverage trade-offs, not universal calibration improvement.
