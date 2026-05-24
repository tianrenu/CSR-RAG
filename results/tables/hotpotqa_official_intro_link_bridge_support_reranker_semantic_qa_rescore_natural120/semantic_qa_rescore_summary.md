# Semantic QA Rescore

This report combines natural_qa120 QA details with MiniMax failure taxonomy. It does not call LLM or embedding APIs.

## Inputs

- QA dir: `results/tables/hotpotqa_official_intro_link_bridge_support_reranker_qa_eval_natural120`
- Judge dir: `results/tables/hotpotqa_official_intro_link_bridge_support_reranker_qa_failure_judge_natural120_v2`

## Policy Rescore

| Policy | Coverage | Strict EM | F1 | Alias acc. | Lenient acc. | Judge-bad rate | Strict ISAR | Alias ISAR | Lenient ISAR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| naive_top5 | 1.0000 | 0.5083 | 0.5935 | 0.6750 | 0.6917 | 0.0500 | 0.2250 | 0.0750 | 0.0500 |
| naive_top20 | 1.0000 | 0.5250 | 0.6564 | 0.7750 | 0.7917 | 0.0583 | 0.4231 | 0.1538 | 0.1538 |
| balanced | 0.6917 | 0.6265 | 0.7122 | 0.7952 | 0.8072 | 0.0602 | 0.1000 | 0.0250 | 0.0250 |
| retrieve_more_balanced | 0.7500 | 0.5778 | 0.7286 | 0.8889 | 0.9000 | 0.0444 | 0.1923 | 0.0385 | 0.0385 |
| retrieve_more_risk_control@suff_abstain15 | 0.6833 | 0.5854 | 0.7302 | 0.8902 | 0.9024 | 0.0488 | 0.1538 | 0.0385 | 0.0385 |
| retrieve_more@cov85 | 0.8167 | 0.5714 | 0.7190 | 0.8673 | 0.8776 | 0.0510 | 0.2308 | 0.0769 | 0.0769 |

## Interpretation Guardrails

1. Alias-corrected accuracy only treats answer_alias_or_metric_mismatch as correct.
2. Lenient accuracy also treats ambiguous_or_gold_issue as non-error, so it should be reported as an audit upper bound.
3. Judge-bad rate counts generation_failure and retrieval_insufficient among answered cases.
4. Strict EM/F1 remain necessary for comparability; semantic rescore is an audit layer, not a replacement.
