# Link-Bridge Reranker QA Evaluation

| Policy | Coverage | Answered F1 | Insufficient Substantive Answer Rate | Wrong Substantive Count |
|---|---:|---:|---:|---:|
| naive_top5 | 1.0000 | 0.5935 | 0.2250 | 28 |
| naive_top20 | 1.0000 | 0.6564 | 0.4231 | 40 |
| balanced | 0.6917 | 0.7122 | 0.1000 | 20 |
| retrieve_more_balanced | 0.7500 | 0.7286 | 0.1923 | 34 |
| retrieve_more_risk_control@suff_abstain15 | 0.6833 | 0.7302 | 0.1538 | 30 |
| retrieve_more@cov85 | 0.8167 | 0.7190 | 0.2308 | 36 |

This is a stratified QA sample over reranked link-bridge records. It is meant to validate whether retrieval-level reliability gains reduce unsupported substantive answers.
