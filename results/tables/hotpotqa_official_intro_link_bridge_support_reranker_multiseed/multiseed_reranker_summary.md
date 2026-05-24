# Multi-seed Reranker Summary

This report reruns fixed RF/GB/LR all/blend1.00 rerankers with multiple seeds. It does not call LLM or embedding APIs.

## Settings

- split dir: `data/processed/hotpotqa_official_intro_link_bridge_splits_top20_full_dev`
- record kind: `official_intro_link_bridge_a0p85_p0p00_top20`
- seeds: `[13, 21, 42, 87, 100]`

## Variant Aggregate Metrics

| Variant | Test top-5 mean | Test top-5 std | Test top-10 mean | Train-test gap mean |
|---|---:|---:|---:|---:|
| gb_all | 0.6457 | 0.0000 | 0.7392 | 0.0032 |
| lr_all | 0.6412 | 0.0000 | 0.7392 | 0.0034 |
| rf_all | 0.6755 | 0.0023 | 0.7444 | 0.0889 |

## Pairwise Test Deltas

| Comparison | Top-k | Mean delta | Std | Min | Max | Positive seeds |
|---|---:|---:|---:|---:|---:|---:|
| rf_vs_gb | 5 | 0.0299 | 0.0023 | 0.0270 | 0.0324 | 5/5 |
| rf_vs_gb | 10 | 0.0052 | 0.0022 | 0.0036 | 0.0090 | 5/5 |
| rf_vs_gb | 20 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0/5 |
| rf_vs_lr | 5 | 0.0344 | 0.0023 | 0.0315 | 0.0369 | 5/5 |
| rf_vs_lr | 10 | 0.0052 | 0.0022 | 0.0036 | 0.0090 | 5/5 |
| rf_vs_lr | 20 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0/5 |
| gb_vs_lr | 5 | 0.0045 | 0.0000 | 0.0045 | 0.0045 | 5/5 |
| gb_vs_lr | 10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0/5 |
| gb_vs_lr | 20 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0/5 |

## Interpretation

1. A robust reranker advantage should remain positive across seeds and have small seed variance.
2. RF top-5 strength must be weighed against its train-test gap.
3. If RF top-10 deltas are near zero, retrieve-more policies may be less sensitive to RF than top-5 answer-only policies.
