# Link-Bridge Reranker QA Evaluation

| Policy | Coverage | Answered F1 | Insufficient Substantive Answer Rate | Wrong Substantive Count |
|---|---:|---:|---:|---:|
| naive_top5 | 1.0000 | 0.3421 | 0.2500 | 20 |
| naive_top20 | 1.0000 | 0.5121 | 0.3000 | 32 |
| balanced | 0.5444 | 0.4538 | 0.1167 | 11 |
| retrieve_more_balanced | 0.6778 | 0.6285 | 0.1333 | 22 |
| retrieve_more_risk_control@suff_abstain15 | 0.4889 | 0.6600 | 0.0333 | 17 |
| retrieve_more@cov85 | 0.7667 | 0.5887 | 0.1667 | 28 |

This is a stratified QA sample over reranked link-bridge records. It is meant to validate whether retrieval-level reliability gains reduce unsupported substantive answers.
