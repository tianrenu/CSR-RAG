# Official HotpotQA Intro Link-Bridge Retrieval

## Purpose

This no-API run tests whether Wikipedia links from first-hop BM25 documents can improve support-chain retrieval over the official HotpotQA intro corpus. Gold support titles are used only for evaluation and valid-only variant selection.

## Corpus

- Wiki archive: `data/external/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2`
- Documents scanned per pass: `5000`
- First-hop K: `20`
- Link source top K: `5`
- Max links per source doc: `80`

## First-Hop BM25 Test Curve

- top-5: sufficient_rate=0.0000
- top-10: sufficient_rate=0.0000
- top-5: sufficient_rate=0.0000
- top-10: sufficient_rate=0.0000
- top-5: sufficient_rate=0.0000
- top-10: sufficient_rate=0.0000
- top-5: sufficient_rate=0.0000
- top-10: sufficient_rate=0.0000
- top-5: sufficient_rate=0.0000
- top-10: sufficient_rate=0.0000
- top-5: sufficient_rate=0.0000
- top-10: sufficient_rate=0.0000
- top-5: sufficient_rate=0.0000
- top-10: sufficient_rate=0.0000
- top-5: sufficient_rate=0.0000
- top-10: sufficient_rate=0.0000
- top-5: sufficient_rate=0.0000
- top-10: sufficient_rate=0.0000
- top-5: sufficient_rate=0.0000
- top-10: sufficient_rate=0.0000
- top-5: sufficient_rate=0.0000
- top-10: sufficient_rate=0.0000
- top-5: sufficient_rate=0.0000
- top-10: sufficient_rate=0.0000
- top-5: sufficient_rate=0.0000
- top-10: sufficient_rate=0.0000

## Selected Link-Bridge Test Curve

Selected method: `link_bridge_a0p70_p0p00`

- top-5: sufficient_rate=0.0000, linked_docs_topk=0.00
- top-10: sufficient_rate=0.0000, linked_docs_topk=0.00

## Interpretation

If selected link-bridge improves top-5/top-10 sufficiency without oracle support injection, it is a promising retrieval refinement path. If it hurts top-5 while improving top-50 only, links should be treated as candidate expansion for a stronger reranker rather than direct ranking.
