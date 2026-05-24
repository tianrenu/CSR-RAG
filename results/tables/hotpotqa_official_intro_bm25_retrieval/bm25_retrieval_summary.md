# Official HotpotQA Intro BM25 Retrieval Baseline

## Purpose

This no-API run evaluates a natural sparse retrieval baseline over the official HotpotQA introductory-paragraph Wikipedia corpus. The retrieval step does not use gold support titles; support titles are used only for sufficiency evaluation.

## Corpus

- Wiki archive: `data/external/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2`
- Documents scanned: `5233329`
- Average filtered document length: `36.58`
- Inner-file cap: `0`
- Document cap: `0`

## Test Sufficiency Curve

- top-5: sufficient_rate=0.3246, answer_present=0.5513, mean_support_coverage=0.5971
- top-10: sufficient_rate=0.4002, answer_present=0.6052, mean_support_coverage=0.6556
- top-20: sufficient_rate=0.4766, answer_present=0.6601, mean_support_coverage=0.7091
- top-50: sufficient_rate=0.5629, answer_present=0.7176, mean_support_coverage=0.7612

## Interpretation

Treat this as the sparse natural baseline for the support-complete official intro corpus. If BM25 top-k remains weak, the next retrieval work should focus on query decomposition, bridge-aware reranking, or dense retrieval over this same corpus.
