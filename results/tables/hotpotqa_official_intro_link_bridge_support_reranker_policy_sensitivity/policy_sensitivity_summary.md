# Reranker Policy Sensitivity Summary

This no-API diagnostic compares RF/GB/LR support rerankers after the downstream answer/retrieve-more/abstain policy stage.

## Settings

- split dir: `data/processed/hotpotqa_official_intro_link_bridge_splits_top20_full_dev`
- record kind: `official_intro_link_bridge_a0p85_p0p00_top20`
- bootstrap iters: `1000`
- writes large data files: `false`
- uses LLM API: `false`
- uses embedding API: `false`

## Test Top-k Sufficiency

| Reranker | Top-5 | Top-10 | Top-20 |
|---|---:|---:|---:|
| gb_all | 0.6457 | 0.7392 | 0.7716 |
| lr_all | 0.6412 | 0.7392 | 0.7716 |
| rf_all | 0.6772 | 0.7428 | 0.7716 |

## Main Policy Rows

| Reranker | Policy | Target k | Coverage | Selective acc. | IAR | SAR | Retrieval rate |
|---|---|---:|---:|---:|---:|---:|---:|
| gb_all | `always_answer` | 5 | 1.0000 | 0.6457 | 1.0000 | 0.0000 | 0.0000 |
| gb_all | `always_answer` | 20 | 1.0000 | 0.7716 | 1.0000 | 0.0000 | 1.0000 |
| gb_all | `balanced` | 5 | 0.7734 | 0.7756 | 0.4898 | 0.0710 | 0.0000 |
| gb_all | `retrieve_more_balanced` | 20 | 0.8255 | 0.8813 | 0.4241 | 0.0538 | 0.9245 |
| gb_all | `retrieve_more_risk_control@suff_abstain15` | 10 | 0.6897 | 0.9035 | 0.2534 | 0.1549 | 0.9317 |
| gb_all | `retrieve_more@cov85` | 20 | 0.8813 | 0.8520 | 0.5642 | 0.0234 | 0.9433 |
| lr_all | `always_answer` | 5 | 1.0000 | 0.6412 | 1.0000 | 0.0000 | 0.0000 |
| lr_all | `always_answer` | 20 | 1.0000 | 0.7716 | 1.0000 | 0.0000 | 1.0000 |
| lr_all | `balanced` | 5 | 0.7311 | 0.7884 | 0.4311 | 0.1010 | 0.0000 |
| lr_all | `retrieve_more_balanced` | 20 | 0.8255 | 0.8769 | 0.4449 | 0.0618 | 1.0000 |
| lr_all | `retrieve_more_risk_control@suff_abstain15` | 20 | 0.7149 | 0.9145 | 0.2677 | 0.1527 | 1.0000 |
| lr_all | `retrieve_more@cov85` | 20 | 0.8426 | 0.8709 | 0.4764 | 0.0490 | 1.0000 |
| rf_all | `always_answer` | 5 | 1.0000 | 0.6772 | 1.0000 | 0.0000 | 0.0000 |
| rf_all | `always_answer` | 20 | 1.0000 | 0.7716 | 1.0000 | 0.0000 | 1.0000 |
| rf_all | `balanced` | 5 | 0.7077 | 0.8310 | 0.3705 | 0.1315 | 0.0000 |
| rf_all | `retrieve_more_balanced` | 20 | 0.7851 | 0.8935 | 0.3661 | 0.0909 | 1.0000 |
| rf_all | `retrieve_more_risk_control@suff_abstain15` | 20 | 0.6835 | 0.9276 | 0.2165 | 0.1783 | 0.9847 |
| rf_all | `retrieve_more@cov85` | 20 | 0.8579 | 0.8627 | 0.5157 | 0.0408 | 1.0000 |

## Interpretation

1. If policy metrics are close across RF/GB/LR, the downstream policy is robust to reranker choice.
2. RF can remain the strongest top-5 reranker while retrieve-more narrows downstream differences.
3. A lower IAR is not enough by itself; coverage and sufficient abstain rate must be checked together.
