# Link-Bridge Reranker QA Evaluation

| Policy | Coverage | Answered F1 | Insufficient Substantive Answer Rate | Wrong Substantive Count |
|---|---:|---:|---:|---:|
| naive_top5 | 1.0000 | 0.0000 | 0.0000 | 0 |
| naive_top20 | 1.0000 | 0.0000 | 0.0000 | 0 |
| balanced | 1.0000 | 0.0000 | 0.0000 | 0 |
| retrieve_more_balanced | 1.0000 | 0.0000 | 0.0000 | 0 |
| retrieve_more_risk_control@suff_abstain15 | 1.0000 | 0.0000 | 0.0000 | 0 |
| retrieve_more@cov85 | 1.0000 | 0.0000 | 0.0000 | 0 |

This is a stratified QA sample over reranked link-bridge records. It is meant to validate whether retrieval-level reliability gains reduce unsupported substantive answers.
