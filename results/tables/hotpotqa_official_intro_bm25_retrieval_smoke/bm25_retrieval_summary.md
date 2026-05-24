# Official HotpotQA Intro BM25 Retrieval Baseline

## Purpose

This no-API run evaluates a natural sparse retrieval baseline over the official HotpotQA introductory-paragraph Wikipedia corpus. The retrieval step does not use gold support titles; support titles are used only for sufficiency evaluation.

## Corpus

- Wiki archive: `data/external/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2`
- Documents scanned: `2000`
- Average filtered document length: `38.79`
- Inner-file cap: `0`
- Document cap: `2000`

## Test Sufficiency Curve

- top-5: sufficient_rate=0.0000, answer_present=0.0000, mean_support_coverage=0.0000
- top-10: sufficient_rate=0.0000, answer_present=0.0000, mean_support_coverage=0.0000

## Interpretation

Treat this as the sparse natural baseline for the support-complete official intro corpus. If BM25 top-k remains weak, the next retrieval work should focus on query decomposition, bridge-aware reranking, or dense retrieval over this same corpus.
