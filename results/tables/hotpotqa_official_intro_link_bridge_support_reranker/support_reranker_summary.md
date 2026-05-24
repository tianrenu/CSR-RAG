# Link-Bridge Support Reranker Summary

## Purpose

This no-API run trains a candidate-level support reranker over link-bridge top-20 records. Model selection uses valid top-5 sufficiency only; test is reported once.

## Selected Variant

- variant: `random_forest_balanced/all/blend1.00`
- valid top-5 sufficient rate: `0.6432`
- test top-5 sufficient rate: `0.6772`

## Test Sufficiency

- top-5: original `0.4694` -> reranked `0.6772`
- top-10: original `0.6646` -> reranked `0.7428`
- top-20: original `0.7716` -> reranked `0.7716`

## Interpretation

A useful reranker should move support titles already present in the top-20 candidate set into the top-5 without using test labels for model selection. This result should be followed by CSR-RAG answer/abstain evaluation on the selected top-5 records.
