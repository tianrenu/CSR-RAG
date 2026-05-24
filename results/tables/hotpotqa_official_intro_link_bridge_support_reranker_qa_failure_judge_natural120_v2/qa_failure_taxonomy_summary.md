# QA Failure Taxonomy

This audit uses MiniMax chat judgments over existing QA answers and retrieved contexts. It does not call embedding APIs.

## Scope

- QA dir: `results/tables/hotpotqa_official_intro_link_bridge_support_reranker_qa_eval_natural120`
- Split dir: `data/processed/hotpotqa_official_intro_link_bridge_support_reranker_splits`
- Selected substantive EM-failure cases: `68`
- Max context chars per case: `5000`

## Taxonomy Counts

| Taxonomy | Count | Share |
|---|---:|---:|
| ambiguous_or_gold_issue | 4 | 0.0588 |
| answer_alias_or_metric_mismatch | 50 | 0.7353 |
| generation_failure | 8 | 0.1176 |
| retrieval_insufficient | 5 | 0.0735 |
| uncertain | 1 | 0.0147 |

## Conditional Counts

| Group | Taxonomy | Count | Share |
|---|---|---:|---:|
| taxonomy_given_insufficient | ambiguous_or_gold_issue | 1 | 0.0833 |
| taxonomy_given_insufficient | answer_alias_or_metric_mismatch | 5 | 0.4167 |
| taxonomy_given_insufficient | generation_failure | 1 | 0.0833 |
| taxonomy_given_insufficient | retrieval_insufficient | 5 | 0.4167 |
| taxonomy_given_sufficient | ambiguous_or_gold_issue | 3 | 0.0536 |
| taxonomy_given_sufficient | answer_alias_or_metric_mismatch | 45 | 0.8036 |
| taxonomy_given_sufficient | generation_failure | 7 | 0.1250 |
| taxonomy_given_sufficient | uncertain | 1 | 0.0179 |

## Interpretation Guardrails

1. These are LLM audit labels, not ground-truth labels.
2. The audit is meant to separate retrieval failure from generation/evaluation failure before larger QA runs.
3. Any paper claim must still be backed by the original QA tables and selected case examples.
