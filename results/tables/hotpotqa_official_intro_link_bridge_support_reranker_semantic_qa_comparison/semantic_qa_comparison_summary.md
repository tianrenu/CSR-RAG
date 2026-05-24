# Semantic QA Comparison

This report combines existing semantic QA rescore artifacts. It does not call LLM or embedding APIs.

## Inputs

- natural_qa120: `results/tables/hotpotqa_official_intro_link_bridge_support_reranker_semantic_qa_rescore_natural120`
- stratified_qa90: `results/tables/hotpotqa_official_intro_link_bridge_support_reranker_semantic_qa_rescore_stratified90`

## Combined Metrics

| Sample | Policy | Coverage | Alias acc. | Judge-bad rate | Alias ISAR |
|---|---|---:|---:|---:|---:|
| natural_qa120 | naive_top5 | 1.0000 | 0.6750 | 0.0500 | 0.0750 |
| natural_qa120 | naive_top20 | 1.0000 | 0.7750 | 0.0583 | 0.1538 |
| natural_qa120 | balanced | 0.6917 | 0.7952 | 0.0602 | 0.0250 |
| natural_qa120 | retrieve_more_balanced | 0.7500 | 0.8889 | 0.0444 | 0.0385 |
| natural_qa120 | retrieve_more_risk_control@suff_abstain15 | 0.6833 | 0.8902 | 0.0488 | 0.0385 |
| natural_qa120 | retrieve_more@cov85 | 0.8167 | 0.8673 | 0.0510 | 0.0769 |
| stratified_qa90 | naive_top5 | 1.0000 | 0.4111 | 0.0556 | 0.0833 |
| stratified_qa90 | naive_top20 | 1.0000 | 0.6111 | 0.0889 | 0.1000 |
| stratified_qa90 | balanced | 0.5444 | 0.5306 | 0.0408 | 0.0500 |
| stratified_qa90 | retrieve_more_balanced | 0.6778 | 0.7377 | 0.0820 | 0.0667 |
| stratified_qa90 | retrieve_more_risk_control@suff_abstain15 | 0.4889 | 0.7727 | 0.0909 | 0.0333 |
| stratified_qa90 | retrieve_more@cov85 | 0.7667 | 0.7246 | 0.0870 | 0.0667 |

## Robustness

| Policy | Natural cov. | Natural alias acc. | Natural bad | Strat. cov. | Strat. alias acc. | Strat. bad | Max alias ISAR |
|---|---:|---:|---:|---:|---:|---:|---:|
| balanced | 0.6917 | 0.7952 | 0.0602 | 0.5444 | 0.5306 | 0.0408 | 0.0500 |
| naive_top20 | 1.0000 | 0.7750 | 0.0583 | 1.0000 | 0.6111 | 0.0889 | 0.1538 |
| naive_top5 | 1.0000 | 0.6750 | 0.0500 | 1.0000 | 0.4111 | 0.0556 | 0.0833 |
| retrieve_more@cov85 | 0.8167 | 0.8673 | 0.0510 | 0.7667 | 0.7246 | 0.0870 | 0.0769 |
| retrieve_more_balanced | 0.7500 | 0.8889 | 0.0444 | 0.6778 | 0.7377 | 0.0820 | 0.0667 |
| retrieve_more_risk_control@suff_abstain15 | 0.6833 | 0.8902 | 0.0488 | 0.4889 | 0.7727 | 0.0909 | 0.0385 |

## Interpretation

1. Retrieve-more policies consistently improve alias-corrected answer accuracy over naive top-5 and balanced answer/abstain.
2. High coverage still needs better risk control: `retrieve_more@cov85` has attractive coverage but worse max alias-corrected ISAR than stricter policies.
3. Semantic rescore is an audit layer; strict EM/F1 should remain in reports for comparability.
