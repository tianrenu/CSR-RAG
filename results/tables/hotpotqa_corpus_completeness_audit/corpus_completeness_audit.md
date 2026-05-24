# HotpotQA Corpus Completeness Audit

## Purpose

This no-API audit separates per-question context completeness from global context-pool completeness for CSR-RAG retrieval experiments.

## Summary

| Split | N | Context all-support present | Global-pool all-support present | Context docs median | Missing context but global present |
|---|---:|---:|---:|---:|---:|
| train | 5183 | 0.2738 | 0.3436 | 10.0 | 0.0698 |
| valid | 1110 | 0.2802 | 0.3450 | 10.0 | 0.0649 |
| test | 1112 | 0.3228 | 0.3759 | 10.0 | 0.0531 |
| all | 7405 | 0.2821 | 0.3487 | 10.0 | 0.0666 |

## Validation

- Raw rows: 7405
- Global pool unique titles: 66568
- Split counts: {"train": 5183, "valid": 1110, "test": 1112}
- Embedding API: no
- LLM API: no

## Example Missing-Support Cases

- `train` / `5a70eee85542994082a3e3f0`: missing context support `ida (sword)`; missing global support `ida (sword)`; question: Which group of people in West Africa with significant populations in Ghana, Ivory Coast, Liberia and Sierra Leone use the Ida?
- `train` / `5a70f0a75542994082a3e403`: missing context support `scanian dialect`; missing global support `scanian dialect`; question: Which dialect spoken in the province of Scania calls Spettekaka, "spiddekaga"?
- `train` / `5a70f11a5542994082a3e40b`: missing context support `bull terrier (miniature) || schapendoes`; missing global support `bull terrier (miniature) || schapendoes`; question: Which dog breed, the Schapendoes or the Bull Terrier, has its origins in a greater number of other dog breeds?
- `train` / `5a70f6425542994082a3e44c`: missing context support `chevrolet corvette (c7)`; missing global support `chevrolet corvette (c7)`; question: Renamed in 2014, what was the vehicle offered as a prize to contestants on the first season of The Amazing Race Canada?
- `train` / `5a70f9335542994082a3e46a`: missing context support `essence (magazine)`; missing global support `essence (magazine)`; question: Which magazine is published more in a year, Essence or Alt for Damerne?
- `train` / `5a70ff335542994082a3e49c`: missing context support `kabhi khushi kabhie gham...`; missing global support ``; question: Which Indian actress who earned filmfare award for Best female debut in "Refugee" acted in K3G movie?
- `train` / `5a7100435542994082a3e4a3`: missing context support `shikashika`; missing global support `shikashika`; question:  Are Finding Kraftland and Shikashika both frozen drinks?
- `train` / `5a7108fa5542994082a3e4ef`: missing context support `battle of cold harbor`; missing global support `battle of cold harbor`; question: Which battle, the Battle of Cold Harbor, or the Second Battle of Bull Run, was fought first?
- `train` / `5a710a1e5542994082a3e4fd`: missing context support `flavivirus`; missing global support `flavivirus`; question: What is the genus of the viral disease that has symptoms such as fever, chills, loss of appetite, nausea, muscle pains, and headaches, and has a chance of causing liver damage?
- `train` / `5a71148b5542994082a3e567`: missing context support `searsport, maine`; missing global support `searsport, maine`; question: What was the population of the city where Penobscot Marine Museum is located?

## Interpretation

If global-pool all-support presence is low, current retrieval experiments are corpus-limited. A support-complete corpus is required before treating low sufficiency as a pure retriever or estimator failure.
