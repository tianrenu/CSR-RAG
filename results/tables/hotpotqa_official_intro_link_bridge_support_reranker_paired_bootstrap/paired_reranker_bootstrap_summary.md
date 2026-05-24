# Paired Reranker Bootstrap Summary

This report uses existing link-bridge records and no-API reranker training. It does not call LLM or embedding APIs.

## Settings

- split dir: `data/processed/hotpotqa_official_intro_link_bridge_splits_top20_full_dev`
- record kind: `official_intro_link_bridge_a0p85_p0p00_top20`
- reranker seed: `42`
- bootstrap iters: `5000`

## Variant Summary

| Variant | Top-5 | Top-10 | Top-20 |
|---|---:|---:|---:|
| original_rank | 0.4694 | 0.6646 | 0.7716 |
| rf_all | 0.6772 | 0.7428 | 0.7716 |
| gb_all | 0.6457 | 0.7392 | 0.7716 |
| lr_all | 0.6412 | 0.7392 | 0.7716 |

## Key Pairwise Deltas

| Comparison | Top-k | Delta | Left-only | Right-only | Net left gain |
|---|---:|---:|---:|---:|---:|
| rf_vs_original | 5 | 0.2077 | 248 | 17 | 231 |
| rf_vs_original | 10 | 0.0782 | 100 | 13 | 87 |
| gb_vs_original | 5 | 0.1763 | 216 | 20 | 196 |
| gb_vs_original | 10 | 0.0746 | 98 | 15 | 83 |
| lr_vs_original | 5 | 0.1718 | 220 | 29 | 191 |
| lr_vs_original | 10 | 0.0746 | 92 | 9 | 83 |
| rf_vs_gb | 5 | 0.0315 | 57 | 22 | 35 |
| rf_vs_gb | 10 | 0.0036 | 19 | 15 | 4 |
| rf_vs_lr | 5 | 0.0360 | 59 | 19 | 40 |
| rf_vs_lr | 10 | 0.0036 | 19 | 15 | 4 |
| gb_vs_lr | 5 | 0.0045 | 42 | 37 | 5 |
| gb_vs_lr | 10 | 0.0000 | 12 | 12 | 0 |

## Bootstrap CI

| Comparison | Top-k | Estimate | 95% low | 95% high | P(delta <= 0) |
|---|---:|---:|---:|---:|---:|
| rf_vs_original | 5 | 0.2077 | 0.1817 | 0.2338 | 0.0000 |
| rf_vs_original | 10 | 0.0782 | 0.0603 | 0.0962 | 0.0000 |
| gb_vs_original | 5 | 0.1763 | 0.1520 | 0.2014 | 0.0000 |
| gb_vs_original | 10 | 0.0746 | 0.0567 | 0.0926 | 0.0000 |
| lr_vs_original | 5 | 0.1718 | 0.1466 | 0.1969 | 0.0000 |
| lr_vs_original | 10 | 0.0746 | 0.0576 | 0.0917 | 0.0000 |
| rf_vs_gb | 5 | 0.0315 | 0.0162 | 0.0468 | 0.0000 |
| rf_vs_gb | 10 | 0.0036 | -0.0063 | 0.0144 | 0.2768 |
| rf_vs_lr | 5 | 0.0360 | 0.0216 | 0.0522 | 0.0000 |
| rf_vs_lr | 10 | 0.0036 | -0.0063 | 0.0144 | 0.2746 |
| gb_vs_lr | 5 | 0.0045 | -0.0108 | 0.0207 | 0.3100 |
| gb_vs_lr | 10 | 0.0000 | -0.0081 | 0.0090 | 0.5294 |

## Interpretation

1. Use top-5 deltas to judge whether a reranker moves complete support chains into the answer context.
2. Use top-10 deltas to check whether differences remain after a moderate retrieve-more expansion.
3. If a 95% CI crosses 0, treat the paired advantage as not yet stable enough for a strong paper claim.
4. This analysis still needs multi-seed reruns before freezing the final reranker choice.
