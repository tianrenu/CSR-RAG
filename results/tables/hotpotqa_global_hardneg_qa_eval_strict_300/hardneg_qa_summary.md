# Hard-Negative QA300 Stress Evaluation

| Policy | Coverage | Answered F1 | Decision Insufficient Answer Rate | Unsupported Substantive Answer Rate |
|---|---:|---:|---:|---:|
| naive_always_answer | 1.0000 | 0.4089 | 1.0000 | 0.3350 |
| balanced | 0.1967 | 0.6161 | 0.0600 | 0.0350 |
| reliable@cov85 | 1.0000 | 0.4089 | 1.0000 | 0.3350 |
| risk_control@cov85 | 1.0000 | 0.4089 | 1.0000 | 0.3350 |
| high_precision@cov50 | 0.5833 | 0.5733 | 0.4100 | 0.2200 |

Coverage>=0.85 lower bound for decision insufficient answer rate on this QA sample: 0.7750.

This table is a stress-setting complement. It should be interpreted as reliability/coverage control, not as a free QA-quality improvement.
