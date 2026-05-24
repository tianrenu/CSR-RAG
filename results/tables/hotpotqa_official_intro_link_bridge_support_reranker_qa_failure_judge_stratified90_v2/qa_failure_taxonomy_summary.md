# QA Failure Taxonomy

This audit uses MiniMax chat judgments over existing QA answers and retrieved contexts. It does not call embedding APIs.

## Scope

- QA dir: `results/tables/hotpotqa_official_intro_link_bridge_support_reranker_qa_eval`
- Split dir: `data/processed/hotpotqa_official_intro_link_bridge_support_reranker_splits`
- Selected substantive EM-failure cases: `52`
- Max context chars per case: `5000`

## Taxonomy Counts

| Taxonomy | Count | Share |
|---|---:|---:|
| ambiguous_or_gold_issue | 6 | 0.1154 |
| answer_alias_or_metric_mismatch | 33 | 0.6346 |
| generation_failure | 8 | 0.1538 |
| retrieval_insufficient | 5 | 0.0962 |

## Conditional Counts

| Group | Taxonomy | Count | Share |
|---|---|---:|---:|
| taxonomy_given_insufficient | ambiguous_or_gold_issue | 3 | 0.2308 |
| taxonomy_given_insufficient | answer_alias_or_metric_mismatch | 5 | 0.3846 |
| taxonomy_given_insufficient | generation_failure | 1 | 0.0769 |
| taxonomy_given_insufficient | retrieval_insufficient | 4 | 0.3077 |
| taxonomy_given_sufficient | ambiguous_or_gold_issue | 3 | 0.0769 |
| taxonomy_given_sufficient | answer_alias_or_metric_mismatch | 28 | 0.7179 |
| taxonomy_given_sufficient | generation_failure | 7 | 0.1795 |
| taxonomy_given_sufficient | retrieval_insufficient | 1 | 0.0256 |

## Interpretation Guardrails

1. These are LLM audit labels, not ground-truth labels.
2. The audit is meant to separate retrieval failure from generation/evaluation failure before larger QA runs.
3. Any paper claim must still be backed by the original QA tables and selected case examples.
