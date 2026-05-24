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

This GitHub repository is intended to track the lightweight research code, active documentation, small result tables, and reproducibility manifests.

Large local artifacts are intentionally not included:

- raw HotpotQA dumps
- official Wikipedia intro corpus archive
- processed retrieval splits
- embedding caches
- virtual environments
- API credentials

See `docs/README.md` for the recommended reading order and current research status.

## Quick Check

```bash
PYTHONPATH=src:scripts .venv/bin/python -m compileall scripts src
```
