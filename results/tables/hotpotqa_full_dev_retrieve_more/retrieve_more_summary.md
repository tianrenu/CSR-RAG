# CSR-RAG Retrieve-More No-API Experiment Summary

## Purpose

This run tests whether lightweight top-k expansion can improve retrieval sufficiency before answer/abstain. It reads existing HotpotQA data and local embedding caches only; it does not call embedding or LLM APIs.

## Test Top-k Sufficiency Curve

- top-5: sufficient_rate=0.2824, newly_sufficient_vs_top5=0.0000
- top-8: sufficient_rate=0.3156, newly_sufficient_vs_top5=0.0464
- top-10: sufficient_rate=0.3210, newly_sufficient_vs_top5=0.0539
- top-15: sufficient_rate=0.3327, newly_sufficient_vs_top5=0.0702

## Main Retrieve-More Finding

- Best retrieve-more main row: `retrieve_more_top8/logistic_regression/v3_no_query/identity` / `retrieve_more_risk_control@suff_abstain15`, coverage=0.4793, IAR=0.3299, retrieval_rate=0.5207.

## Diagnostics

- Case study rows: 75.
- Interpret retrieve-more gains as retrieval-level evidence sufficiency gains. QA expansion should wait until these gains are stable.
- If top-k expansion only yields a small sufficiency gain, treat it as a diagnostic baseline rather than the next main CSR-RAG method.
