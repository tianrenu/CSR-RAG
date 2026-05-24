# Risk-Sensitive Threshold Summary

## Main Finding

Risk-sensitive thresholding trades coverage for lower insufficient-answer risk. The default balanced policy optimizes valid decision accuracy, while risk-control policies select stricter thresholds using only valid split metrics.

## Test Split

- Balanced tau: 0.5, coverage: 0.9185, selective accuracy: 0.8468, insufficient answer rate: 0.7170
- Risk-control tau: 0.3, coverage: 0.7593, selective accuracy: 0.9171, insufficient answer rate: 0.3208
- High-precision tau: 0.05, coverage: 0.5000, selective accuracy: 0.9630, insufficient answer rate: 0.0943

## QA 100 Rescoring

- Balanced coverage: 0.9600, answered F1: 0.7215, insufficient answer rate: 0.8462
- Risk-control coverage: 0.8100, answered F1: 0.7605, insufficient answer rate: 0.3077
- High-precision coverage: 0.5300, answered F1: 0.7635, insufficient answer rate: 0.1538

## Paper Interpretation

Use balanced CSR-RAG as the default method and risk-sensitive CSR-RAG as the reliability-oriented variant. The reliability variant should be presented as a coverage-risk control mechanism, not as a free improvement.
