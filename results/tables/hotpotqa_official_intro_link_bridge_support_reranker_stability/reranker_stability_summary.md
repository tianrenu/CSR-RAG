# Link-bridge Support Reranker Stability

This report summarizes existing reranker artifacts only. It does not call LLM or embedding APIs.

## Scope

- Input dir: `results/tables/hotpotqa_official_intro_link_bridge_support_reranker`
- Split counts: `{'train': 5183, 'valid': 1110, 'test': 1112}`
- Candidate counts: `{'train': 103660, 'valid': 22200, 'test': 22240}`

## Estimator Summary

| Variant | Train top-5 | Valid top-5 | Test top-5 | Test top-10 | Gain vs orig. | Train-test gap | Doc test AUPRC | Doc AUPRC gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original_rank | 0.4754 | 0.4459 | 0.4694 | 0.6646 | 0.0000 | 0.0060 |  |  |
| random_forest_balanced/all/blend1.00 | 0.7644 | 0.6432 | 0.6772 | 0.7428 | 0.2077 | 0.0873 | 0.8004 | 0.1933 |
| gradient_boosting/all/blend1.00 | 0.6489 | 0.6270 | 0.6457 | 0.7392 | 0.1763 | 0.0032 | 0.7671 | -0.0060 |
| logistic_regression_balanced/all/blend1.00 | 0.6446 | 0.6207 | 0.6412 | 0.7392 | 0.1718 | 0.0034 | 0.7134 | -0.0090 |

## Pairwise Deltas

| Comparison | Delta test top-5 | Delta test top-10 | Delta train-test gap |
|---|---:|---:|---:|
| rf_vs_gradient_boosting | 0.0315 | 0.0036 | 0.0841 |
| rf_vs_logistic_regression | 0.0360 | 0.0036 | 0.0838 |
| gb_vs_logistic_regression | 0.0045 | 0.0000 | -0.0003 |

## Research Judgment

1. RF/all/blend1.00 remains the strongest current reranker by test top-5 sufficiency.
2. RF has a clear train-test gap; it should be treated as the strongest working variant, not a frozen final method.
3. GB/LR have slightly weaker top-5 sufficiency but much cleaner train-test behavior, so they should remain mandatory robustness baselines.
4. Next no-API work should add paired question-level RF/GB/LR details, multi-seed reruns, and bootstrap confidence intervals.
