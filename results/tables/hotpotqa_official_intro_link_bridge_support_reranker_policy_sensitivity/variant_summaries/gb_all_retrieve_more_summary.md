# Link-Bridge Retrieve-More Experiment Summary

## Purpose

This no-API run turns link-bridge top-10/top-20 candidates into an answer/retrieve-more/abstain policy. Thresholds are selected on valid_policy only and test is reported once.

## Test Top-k Sufficiency

- top-5 test: sufficient_rate=0.6457
- top-10 test: sufficient_rate=0.7392
- top-20 test: sufficient_rate=0.7716

## Main Rows

- `always_answer_top5` / `always_answer`: target_k=5, coverage=1.0000, IAR=1.0000, retrieval_rate=0.0000
- `always_answer_top10` / `always_answer`: target_k=10, coverage=1.0000, IAR=1.0000, retrieval_rate=1.0000
- `always_answer_top20` / `always_answer`: target_k=20, coverage=1.0000, IAR=1.0000, retrieval_rate=1.0000
- `top5/logistic_regression/v3_all/platt` / `balanced`: target_k=5, coverage=0.7734, IAR=0.4898, retrieval_rate=0.0000
- `top20/logistic_regression/v3_all/platt` / `retrieve_more_balanced`: target_k=20, coverage=0.8255, IAR=0.4241, retrieval_rate=0.9245
- `top10/logistic_regression/v3_all/isotonic` / `retrieve_more_risk_control@suff_abstain15`: target_k=10, coverage=0.6897, IAR=0.2534, retrieval_rate=0.9317
- `top20/logistic_regression/v3_no_query/platt` / `retrieve_more@cov85`: target_k=20, coverage=0.8813, IAR=0.5642, retrieval_rate=0.9433

## Interpretation

Compare retrieve-more rows against always-answer top-5/top-10/top-20 and answer/abstain top-5. A useful policy should reduce insufficient answer rate without collapsing coverage or over-abstaining on sufficient cases.
