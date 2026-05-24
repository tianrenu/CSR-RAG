# Semantic QA Rescore

This report combines stratified_qa90 QA details with MiniMax failure taxonomy. It does not call LLM or embedding APIs.

## Inputs

- QA dir: `results/tables/hotpotqa_official_intro_link_bridge_support_reranker_qa_eval`
- Judge dir: `results/tables/hotpotqa_official_intro_link_bridge_support_reranker_qa_failure_judge_stratified90_v2`

## Policy Rescore

| Policy | Coverage | Strict EM | F1 | Alias acc. | Lenient acc. | Judge-bad rate | Strict ISAR | Alias ISAR | Lenient ISAR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| naive_top5 | 1.0000 | 0.2667 | 0.3421 | 0.4111 | 0.4333 | 0.0556 | 0.2500 | 0.0833 | 0.0500 |
| naive_top20 | 1.0000 | 0.3889 | 0.5121 | 0.6111 | 0.6556 | 0.0889 | 0.3000 | 0.1000 | 0.0667 |
| balanced | 0.5444 | 0.3878 | 0.4538 | 0.5306 | 0.5714 | 0.0408 | 0.1167 | 0.0500 | 0.0167 |
| retrieve_more_balanced | 0.6778 | 0.5082 | 0.6285 | 0.7377 | 0.7869 | 0.0820 | 0.1333 | 0.0667 | 0.0333 |
| retrieve_more_risk_control@suff_abstain15 | 0.4889 | 0.5227 | 0.6600 | 0.7727 | 0.8182 | 0.0909 | 0.0333 | 0.0333 | 0.0000 |
| retrieve_more@cov85 | 0.7667 | 0.4493 | 0.5887 | 0.7246 | 0.7681 | 0.0870 | 0.1667 | 0.0667 | 0.0333 |

## Interpretation Guardrails

1. Alias-corrected accuracy only treats answer_alias_or_metric_mismatch as correct.
2. Lenient accuracy also treats ambiguous_or_gold_issue as non-error, so it should be reported as an audit upper bound.
3. Judge-bad rate counts generation_failure and retrieval_insufficient among answered cases.
4. Strict EM/F1 remain necessary for comparability; semantic rescore is an audit layer, not a replacement.
