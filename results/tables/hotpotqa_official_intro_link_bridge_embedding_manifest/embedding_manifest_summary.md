# Link-Bridge Embedding Manifest Summary

This is a no-API budget manifest for candidate-level embedding reranking. Token counts are approximate and use a word-token multiplier, so they should be treated as planning estimates rather than billing truth.

| Text Type | Unique Texts | Cached | Missing | Cache Coverage | Estimated Missing Tokens |
|---|---:|---:|---:|---:|---:|
| query | 7405 | 7405 | 0 | 1.0000 | 0 |
| doc | 113320 | 23503 | 89817 | 0.2074 | 6171919 |
| all | 120725 | 30908 | 89817 | 0.2560 | 6171919 |

Recommended next step: if the missing-token estimate is within budget, run candidate-level query/doc embedding only for missing manifest texts and cache the result before reranking.
