"""Shared feature sets for controlled and embedding RAG experiments."""

from __future__ import annotations


V1_QUERY = ["query_length", "has_time_constraint", "has_constraint_term"]
V1_RETRIEVAL = ["topk_score_mean", "topk_score_std", "top1_top2_gap", "doc_count", "doc_redundancy"]
V1_LEXICAL = ["title_overlap_max", "title_overlap_mean", "text_overlap_max", "text_overlap_mean"]

RANK_AWARE = ["top1_score", "top3_score_mean", "top5_score_min", "top1_top5_gap", "score_entropy"]
COVERAGE = [
    "query_token_coverage_union",
    "query_token_coverage_top1",
    "query_token_coverage_top3",
    "uncovered_query_token_ratio",
]
DIVERSITY = [
    "pairwise_doc_overlap_mean",
    "pairwise_doc_overlap_max",
    "unique_title_token_ratio",
    "doc_text_length_mean",
    "doc_text_length_std",
]
QUESTION_FORM = [
    "is_comparison_question",
    "is_bridge_like_question",
    "wh_who",
    "wh_what",
    "wh_when",
    "wh_where",
    "wh_which",
    "wh_how",
    "wh_other",
]
EMBEDDING_SCORE = [
    "embedding_topk_score_mean",
    "embedding_topk_score_std",
    "embedding_top1_top2_gap",
    "embedding_top1_top5_gap",
    "embedding_score_entropy",
]

FEATURE_GROUPS_V2 = {
    "query": V1_QUERY + QUESTION_FORM,
    "retrieval": V1_RETRIEVAL + RANK_AWARE,
    "lexical": V1_LEXICAL,
    "coverage": COVERAGE,
    "diversity": DIVERSITY,
}
FEATURE_V1 = V1_QUERY + V1_RETRIEVAL + V1_LEXICAL
FEATURE_V2 = (
    FEATURE_GROUPS_V2["query"]
    + FEATURE_GROUPS_V2["retrieval"]
    + FEATURE_GROUPS_V2["lexical"]
    + FEATURE_GROUPS_V2["coverage"]
    + FEATURE_GROUPS_V2["diversity"]
)

EMBEDDING_FEATURE_GROUPS = {**FEATURE_GROUPS_V2, "embedding_score": EMBEDDING_SCORE}
EMBEDDING_FEATURES = FEATURE_V2 + EMBEDDING_SCORE

EMBEDDING_ABLATIONS = {
    "all_embedding": EMBEDDING_FEATURES,
    "controlled_v2_only": FEATURE_V2,
    "embedding_score_only": EMBEDDING_SCORE,
    "no_query": FEATURE_GROUPS_V2["retrieval"]
    + FEATURE_GROUPS_V2["lexical"]
    + FEATURE_GROUPS_V2["coverage"]
    + FEATURE_GROUPS_V2["diversity"]
    + EMBEDDING_SCORE,
    "no_retrieval": FEATURE_GROUPS_V2["query"]
    + FEATURE_GROUPS_V2["lexical"]
    + FEATURE_GROUPS_V2["coverage"]
    + FEATURE_GROUPS_V2["diversity"]
    + EMBEDDING_SCORE,
    "no_lexical": FEATURE_GROUPS_V2["query"]
    + FEATURE_GROUPS_V2["retrieval"]
    + FEATURE_GROUPS_V2["coverage"]
    + FEATURE_GROUPS_V2["diversity"]
    + EMBEDDING_SCORE,
    "no_coverage": FEATURE_GROUPS_V2["query"]
    + FEATURE_GROUPS_V2["retrieval"]
    + FEATURE_GROUPS_V2["lexical"]
    + FEATURE_GROUPS_V2["diversity"]
    + EMBEDDING_SCORE,
    "no_diversity": FEATURE_GROUPS_V2["query"]
    + FEATURE_GROUPS_V2["retrieval"]
    + FEATURE_GROUPS_V2["lexical"]
    + FEATURE_GROUPS_V2["coverage"]
    + EMBEDDING_SCORE,
    "no_embedding_score": FEATURE_V2,
}

FORBIDDEN_FEATURE_FIELDS = {"support_doc_ids", "dropped_support_doc_id", "is_support", "supporting_facts"}
