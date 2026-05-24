# Support Reranker Ablation Summary

## Purpose

This summary makes the support reranker result auditable by grouping variants by feature set, estimator, and blend alpha. Each row is selected by valid top-5 sufficiency only.

## Feature Set Ablation

| Group Value | Selected Variant | Valid Top-5 | Test Top-5 | Test Gain | Train-Test Gap |
|---|---|---:|---:|---:|---:|
| all | `random_forest_balanced/all/blend1.00` | 0.6432 | 0.6772 | +0.2077 | +0.0873 |
| query_doc | `random_forest_balanced/query_doc/blend1.00` | 0.6207 | 0.6502 | +0.1808 | +0.1142 |
| bridge_context | `gradient_boosting/bridge_context/blend1.00` | 0.5892 | 0.6043 | +0.1349 | +0.0144 |
| rank_score | `logistic_regression_balanced/rank_score/blend1.00` | 0.5568 | 0.5683 | +0.0989 | +0.0085 |
| original_rank | `original_rank` | 0.4459 | 0.4694 | +0.0000 | +0.0060 |

## Estimator Ablation

| Group Value | Selected Variant | Valid Top-5 | Test Top-5 | Test Gain | Train-Test Gap |
|---|---|---:|---:|---:|---:|
| random_forest_balanced | `random_forest_balanced/all/blend1.00` | 0.6432 | 0.6772 | +0.2077 | +0.0873 |
| gradient_boosting | `gradient_boosting/all/blend1.00` | 0.6270 | 0.6457 | +0.1763 | +0.0032 |
| logistic_regression_balanced | `logistic_regression_balanced/all/blend1.00` | 0.6207 | 0.6412 | +0.1718 | +0.0034 |
| original_rank | `original_rank` | 0.4459 | 0.4694 | +0.0000 | +0.0060 |

## Blend Alpha Ablation

| Group Value | Selected Variant | Valid Top-5 | Test Top-5 | Test Gain | Train-Test Gap |
|---|---|---:|---:|---:|---:|
| 1.0 | `random_forest_balanced/all/blend1.00` | 0.6432 | 0.6772 | +0.2077 | +0.0873 |
| 0.75 | `random_forest_balanced/all/blend0.75` | 0.6324 | 0.6637 | +0.1942 | +0.1008 |
| 0.5 | `random_forest_balanced/all/blend0.50` | 0.6081 | 0.6313 | +0.1619 | +0.1331 |
| 0.25 | `random_forest_balanced/query_doc/blend0.25` | 0.5568 | 0.5890 | +0.1196 | +0.1509 |
| original_rank | `original_rank` | 0.4459 | 0.4694 | +0.0000 | +0.0060 |
| 0.0 | `random_forest_balanced/rank_score/blend0.00` | 0.4450 | 0.4694 | +0.0000 | +0.0058 |

## Interpretation

The selected production variant should be judged by valid-selected test top-5 gain and train-test gap. Large train-test gaps indicate possible overfitting even when test performance improves.
