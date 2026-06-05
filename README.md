# CSR-RAG Legacy Archive

> This repository is a historical archive for the early CSR-RAG research prototype.
> The active, renamed SERA-RAG project is now maintained at: https://github.com/tianrenu/sera-rag
>
> Please use the SERA-RAG repository for current code, paper-facing evidence packs, readiness audits, and reproducibility references. This CSR-RAG repository is preserved only for provenance and historical traceability.

---

# CSR-RAG

CSR-RAG is a research prototype for calibrated selective retrieval-augmented generation.

The current research line is:

```text
retrieval sufficiency -> calibrated risk -> selective answer / retrieve-more / abstain
```

The project is still in the framework-optimization and experiment-building stage. It is not yet a paper-ready frozen release.

## Current Focus

- Retrieval sufficiency estimation
- Link-bridge retrieval over HotpotQA official intro corpus
- Support-title supervised reranking
- Calibrated risk and answer / retrieve-more / abstain policy
- MiniMax QA validation and semantic failure audit

## Repository Scope

This GitHub repository is intended to track the lightweight legacy research code, small result tables, and reproducibility manifests.

Large local artifacts are intentionally not included:

- raw HotpotQA dumps
- official Wikipedia intro corpus archive
- processed retrieval splits
- embedding caches
- virtual environments
- API credentials

The old internal documentation directory has been removed from this public archive. For current documentation, paper-facing evidence packs, and reproducibility references, use the active SERA-RAG repository.

## Quick Check

```bash
PYTHONPATH=src:scripts .venv/bin/python -m compileall scripts src
```
