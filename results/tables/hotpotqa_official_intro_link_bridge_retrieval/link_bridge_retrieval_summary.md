# Official HotpotQA Intro Link-Bridge Retrieval

## Purpose

This no-API run tests whether Wikipedia links from first-hop BM25 documents can improve support-chain retrieval over the official HotpotQA intro corpus. Gold support titles are used only for evaluation and valid-only variant selection.

## Corpus

- Wiki archive: `data/external/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2`
- Documents scanned per pass: `5233329`
- First-hop K: `50`
- Link source top K: `10`
- Max links per source doc: `80`

## First-Hop BM25 Test Curve

- top-5: sufficient_rate=0.3246
- top-10: sufficient_rate=0.4002
- top-20: sufficient_rate=0.4766
- top-50: sufficient_rate=0.5629

## Selected Link-Bridge Test Curve

Selected method: `link_bridge_a0p85_p0p00`

- top-5: sufficient_rate=0.4694, linked_docs_topk=2.47
- top-10: sufficient_rate=0.6637, linked_docs_topk=5.08
- top-20: sufficient_rate=0.7716, linked_docs_topk=8.98
- top-50: sufficient_rate=0.8552, linked_docs_topk=19.51

## Interpretation

If selected link-bridge improves top-5/top-10 sufficiency without oracle support injection, it is a promising retrieval refinement path. If it hurts top-5 while improving top-50 only, links should be treated as candidate expansion for a stronger reranker rather than direct ranking.
