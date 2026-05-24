# Link-Bridge Retrieve-More Experiment Summary

## Purpose

This no-API run turns link-bridge top-10/top-20 candidates into an answer/retrieve-more/abstain policy. Thresholds are selected on valid_policy only and test is reported once.

## Test Top-k Sufficiency

- top-5 test: sufficient_rate=0.4694
- top-10 test: sufficient_rate=0.6637
- top-20 test: sufficient_rate=0.7716

## Main Rows

- `always_answer_top5` / `always_answer`: target_k=5, coverage=1.0000, IAR=1.0000, retrieval_rate=0.0000
- `always_answer_top10` / `always_answer`: target_k=10, coverage=1.0000, IAR=1.0000, retrieval_rate=1.0000
- `always_answer_top20` / `always_answer`: target_k=20, coverage=1.0000, IAR=1.0000, retrieval_rate=1.0000
- `top5/logistic_regression/v3_all/platt` / `balanced`: target_k=5, coverage=0.3921, IAR=0.1407, retrieval_rate=0.0000
- `top20/logistic_regression/v3_no_query/isotonic` / `retrieve_more_balanced`: target_k=20, coverage=0.8156, IAR=0.4402, retrieval_rate=0.8984
- `top10/logistic_regression/v3_no_query/identity` / `retrieve_more_risk_control@suff_abstain15`: target_k=10, coverage=0.6421, IAR=0.2884, retrieval_rate=0.9119
- `top20/logistic_regression/v3_all/identity` / `retrieve_more@cov85`: target_k=20, coverage=0.8723, IAR=0.5709, retrieval_rate=0.8453

## Interpretation

Compare retrieve-more rows against always-answer top-5/top-10/top-20 and answer/abstain top-5. A useful policy should reduce insufficient answer rate without collapsing coverage or over-abstaining on sufficient cases.
