# QA-aware Policy Comparison

This report reuses existing MiniMax QA artifacts only. It does not call LLM or embedding APIs.

## Inputs

- stratified_qa90: `results/tables/hotpotqa_official_intro_link_bridge_support_reranker_qa_eval`, selected `90`, completed `90`, sample counts `{'top5_sufficient': 30, 'rescued_by_top20': 30, 'unresolved_top20': 30}`
- natural_qa120: `results/tables/hotpotqa_official_intro_link_bridge_support_reranker_qa_eval_natural120`, selected `120`, completed `120`, sample counts `{'top5_sufficient': 80, 'unresolved_top20': 26, 'rescued_by_top20': 14}`

## Combined Metrics

| Sample | Policy | Coverage | Answered F1 | ISAR | False Ans. | Wrong Subst. |
|---|---|---:|---:|---:|---:|---:|
| stratified_qa90 | naive_top5 | 1.0000 | 0.3421 | 0.2500 | 60 | 20 |
| stratified_qa90 | naive_top20 | 1.0000 | 0.5121 | 0.3000 | 30 | 32 |
| stratified_qa90 | balanced | 0.5444 | 0.4538 | 0.1167 | 26 | 11 |
| stratified_qa90 | retrieve_more_balanced | 0.6778 | 0.6285 | 0.1333 | 12 | 22 |
| stratified_qa90 | retrieve_more_risk_control@suff_abstain15 | 0.4889 | 0.6600 | 0.0333 | 5 | 17 |
| stratified_qa90 | retrieve_more@cov85 | 0.7667 | 0.5887 | 0.1667 | 15 | 28 |
| natural_qa120 | naive_top5 | 1.0000 | 0.5935 | 0.2250 | 40 | 28 |
| natural_qa120 | naive_top20 | 1.0000 | 0.6564 | 0.4231 | 26 | 40 |
| natural_qa120 | balanced | 0.6917 | 0.7122 | 0.1000 | 15 | 20 |
| natural_qa120 | retrieve_more_balanced | 0.7500 | 0.7286 | 0.1923 | 7 | 34 |
| natural_qa120 | retrieve_more_risk_control@suff_abstain15 | 0.6833 | 0.7302 | 0.1538 | 6 | 30 |
| natural_qa120 | retrieve_more@cov85 | 0.8167 | 0.7190 | 0.2308 | 10 | 36 |

## Robustness Summary

| Policy | Natural cov. | Natural F1 | Natural ISAR | Strat. cov. | Strat. F1 | Strat. ISAR | Max ISAR |
|---|---:|---:|---:|---:|---:|---:|---:|
| balanced | 0.6917 | 0.7122 | 0.1000 | 0.5444 | 0.4538 | 0.1167 | 0.1167 |
| naive_top20 | 1.0000 | 0.6564 | 0.4231 | 1.0000 | 0.5121 | 0.3000 | 0.4231 |
| naive_top5 | 1.0000 | 0.5935 | 0.2250 | 1.0000 | 0.3421 | 0.2500 | 0.2500 |
| retrieve_more@cov85 | 0.8167 | 0.7190 | 0.2308 | 0.7667 | 0.5887 | 0.1667 | 0.2308 |
| retrieve_more_balanced | 0.7500 | 0.7286 | 0.1923 | 0.6778 | 0.6285 | 0.1333 | 0.1923 |
| retrieve_more_risk_control@suff_abstain15 | 0.6833 | 0.7302 | 0.1538 | 0.4889 | 0.6600 | 0.0333 | 0.1538 |

## QA-aware Selection

- Accepted policies: `balanced, retrieve_more_balanced, retrieve_more_risk_control@suff_abstain15`
- Risk-first candidate: `balanced`
- F1-first candidate: `retrieve_more_risk_control@suff_abstain15`
- High-coverage candidate: ``
- High-coverage note: No non-naive policy currently satisfies natural coverage >= 0.80 while also keeping natural insufficient-substantive answer rate no worse than naive_top5.

## Interpretation

1. `balanced` is the current risk-first policy: it gives the lowest maximum ISAR across the two QA samples under the selection filters.
2. `retrieve_more_risk_control@suff_abstain15` is the current F1-first policy among QA-safe candidates, but it has lower stratified coverage and more sufficient abstention.
3. `retrieve_more@cov85` is not yet QA-safe: its natural coverage is attractive, but its natural ISAR is slightly worse than naive top-5.
4. The next method work should improve high-coverage risk control, not merely reduce coverage.
