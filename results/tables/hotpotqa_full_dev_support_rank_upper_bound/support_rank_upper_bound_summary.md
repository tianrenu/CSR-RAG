# CSR-RAG Support-Rank Upper-Bound Summary

## Purpose

This no-API run estimates how deep the global embedding ranking must go before all HotpotQA supporting titles are present. It is an upper-bound diagnostic for retrieval refinement.

## Test Coverage by Top-k

- top-5: all_support_covered_rate=0.2824, among_ranked=0.7512, newly_vs_top5=0.0000, mean_support_title_coverage=0.5612
- top-8: all_support_covered_rate=0.3156, among_ranked=0.8397, newly_vs_top5=0.0464, mean_support_title_coverage=0.5814
- top-10: all_support_covered_rate=0.3210, among_ranked=0.8541, newly_vs_top5=0.0539, mean_support_title_coverage=0.5850
- top-15: all_support_covered_rate=0.3327, among_ranked=0.8852, newly_vs_top5=0.0702, mean_support_title_coverage=0.5926
- top-20: all_support_covered_rate=0.3381, among_ranked=0.8995, newly_vs_top5=0.0777, mean_support_title_coverage=0.5971
- top-30: all_support_covered_rate=0.3507, among_ranked=0.9330, newly_vs_top5=0.0952, mean_support_title_coverage=0.6048
- top-50: all_support_covered_rate=0.3579, among_ranked=0.9522, newly_vs_top5=0.1053, mean_support_title_coverage=0.6097
- top-100: all_support_covered_rate=0.3642, among_ranked=0.9689, newly_vs_top5=0.1140, mean_support_title_coverage=0.6156

## Test Max Support-Rank Buckets

- <=5: 0.2824
- 6-15: 0.0504
- 16-30: 0.0180
- 31-50: 0.0072
- 51-100: 0.0063
- >100: 0.0117
- missing: 0.6241

## Validation

- Global docs: 66573
- Unique titles: 66568
- Embedding API: no
- LLM API: no
- Top-5 label reconstruction: {"train": {"n": 5183, "label_match_count": 5183, "label_match_rate": 1.0}, "valid": {"n": 1110, "label_match_count": 1110, "label_match_rate": 1.0}, "test": {"n": 1112, "label_match_count": 1112, "label_match_rate": 1.0}}

## Interpretation

The overall top-k coverage is capped by corpus completeness. If many support titles are missing from the current pool, the next research step is to build or obtain a support-complete retrieval corpus before treating retrieval failures as model failures. Among questions whose full support chain exists in the pool, high top-50/top-100 coverage means reranking and bridge-aware candidate selection are promising.
