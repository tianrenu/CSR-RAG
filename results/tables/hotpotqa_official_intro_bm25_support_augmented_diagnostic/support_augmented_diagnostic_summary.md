# BM25 Support-Augmented Diagnostic Records

## Purpose

This no-API diagnostic dataset pairs natural BM25 top-5 records with oracle support-complete records built by injecting official support-title documents. It must not be used as a natural retrieval main result.

## Summary

| Split | Base questions | Paired records | Base sufficient | Paired sufficient | Mean injected support titles |
|---|---:|---:|---:|---:|---:|
| train | 5183 | 10366 | 0.3002 | 0.6501 | 0.8150 |
| valid | 1110 | 2220 | 0.2919 | 0.6459 | 0.8423 |
| test | 1112 | 2224 | 0.3246 | 0.6623 | 0.8058 |
| all | 7405 | 14810 | 0.3026 | 0.6513 | 0.8177 |

## Validation

- Input split dir: `data/processed/hotpotqa_official_intro_bm25_splits_full_dev`
- Support corpus: `data/processed/hotpotqa_support_title_intro_corpus.jsonl`
- Embedding API: no
- LLM API: no
- Oracle diagnostic: yes
