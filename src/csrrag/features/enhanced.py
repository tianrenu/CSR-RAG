"""Enhanced non-oracle features and audit-only signals for next-stage CSR-RAG."""

from __future__ import annotations

from itertools import combinations
from statistics import mean, pstdev
from typing import Any

from csrrag.features.basic import extract_basic_features
from csrrag.utils.text import token_set, tokenize


V3_EVIDENCE_FEATURES = [
    "title_token_coverage_union",
    "title_token_coverage_top1",
    "title_token_coverage_top3",
    "title_bigram_coverage_union",
    "text_bigram_coverage_union",
    "top1_text_to_other_title_overlap_max",
    "top1_text_to_other_title_overlap_mean",
    "query_title_bridge_doc_count",
    "query_text_bridge_doc_count",
    "multi_doc_query_coverage_count",
    "top3_unique_query_token_hits",
    "top5_unique_query_token_hits",
    "top3_query_token_hit_ratio",
    "top5_query_token_hit_ratio",
    "title_chain_overlap_mean",
    "title_chain_overlap_max",
    "evidence_text_to_title_overlap_mean",
    "evidence_text_to_title_overlap_max",
]

V3_RETRIEVAL_INTERACTION_FEATURES = [
    "embedding_top1_share",
    "embedding_top3_share",
    "embedding_gap_to_std",
    "lexical_gap_to_std",
    "coverage_embedding_product",
    "coverage_margin_product",
    "top1_coverage_embedding_product",
    "diversity_adjusted_coverage",
    "redundancy_adjusted_coverage",
    "score_entropy_coverage_product",
]

V3_FEATURES = V3_EVIDENCE_FEATURES + V3_RETRIEVAL_INTERACTION_FEATURES

AUDIT_FEATURES = [
    "audit_support_title_count",
    "audit_retrieved_support_title_count",
    "audit_missing_support_title_count",
    "audit_support_title_coverage",
    "audit_gold_answer_in_top1",
    "audit_gold_answer_in_top3",
    "audit_gold_answer_in_top5",
    "audit_support_present_in_topk",
]


def extract_enhanced_features(record: dict[str, Any]) -> dict[str, float]:
    """Return deployable v2 + v3 features.

    This function intentionally avoids gold answers, support titles, and labels.
    Those fields are useful for diagnosis but would leak unavailable inference-time
    information into the sufficiency estimator.
    """

    features = extract_basic_features(record)
    features.update(_v3_features(record, features))
    return features


def extract_audit_features(record: dict[str, Any]) -> dict[str, float]:
    """Return oracle audit signals for failure analysis only."""

    docs = record.get("retrieved_docs", [])
    metadata = record.get("metadata", {})
    support_titles = {_normalize_title(title) for title in metadata.get("support_titles", [])}
    retrieved_titles = [_normalize_title(str(doc.get("title", ""))) for doc in docs]
    retrieved_support = support_titles & set(retrieved_titles)
    missing_support = support_titles - set(retrieved_titles)
    answer = str(record.get("gold_answer", "")).strip()
    top1_text = _concat_docs(docs[:1])
    top3_text = _concat_docs(docs[:3])
    top5_text = _concat_docs(docs[:5])
    return {
        "audit_support_title_count": float(len(support_titles)),
        "audit_retrieved_support_title_count": float(len(retrieved_support)),
        "audit_missing_support_title_count": float(len(missing_support)),
        "audit_support_title_coverage": float(len(retrieved_support) / len(support_titles)) if support_titles else 0.0,
        "audit_gold_answer_in_top1": float(_contains_answer(top1_text, answer)),
        "audit_gold_answer_in_top3": float(_contains_answer(top3_text, answer)),
        "audit_gold_answer_in_top5": float(_contains_answer(top5_text, answer)),
        "audit_support_present_in_topk": float(bool(metadata.get("support_present_in_topk", False))),
    }


def _v3_features(record: dict[str, Any], basic: dict[str, float]) -> dict[str, float]:
    query = str(record.get("query", ""))
    docs = record.get("retrieved_docs", [])
    query_tokens = set(tokenize(query))
    query_bigrams = _bigrams(tokenize(query))
    doc_title_token_lists = [tokenize(str(doc.get("title", ""))) for doc in docs]
    doc_text_token_lists = [tokenize(str(doc.get("text", ""))) for doc in docs]
    doc_title_sets = [set(tokens) for tokens in doc_title_token_lists]
    doc_text_sets = [set(tokens) for tokens in doc_text_token_lists]
    doc_all_sets = [title | text for title, text in zip(doc_title_sets, doc_text_sets)]
    title_union = set().union(*doc_title_sets) if doc_title_sets else set()
    text_union = set().union(*doc_text_sets) if doc_text_sets else set()
    top1_all = doc_all_sets[0] if doc_all_sets else set()
    top3_title_union = set().union(*doc_title_sets[:3]) if doc_title_sets else set()
    top3_text_union = set().union(*doc_text_sets[:3]) if doc_text_sets else set()
    title_bigrams = set().union(*[_bigrams(tokens) for tokens in doc_title_token_lists]) if doc_title_token_lists else set()
    text_bigrams = set().union(*[_bigrams(tokens) for tokens in doc_text_token_lists]) if doc_text_token_lists else set()

    embedding_scores = [max(float(doc.get("embedding_score", 0.0)), 0.0) for doc in docs]
    lexical_scores = [max(float(doc.get("score", 0.0)), 0.0) for doc in docs]
    embedding_total = sum(embedding_scores)
    embedding_std = pstdev(embedding_scores) if len(embedding_scores) > 1 else 0.0
    lexical_std = pstdev(lexical_scores) if len(lexical_scores) > 1 else 0.0
    embedding_gap = float(embedding_scores[0] - embedding_scores[1]) if len(embedding_scores) > 1 else 0.0
    lexical_gap = float(lexical_scores[0] - lexical_scores[1]) if len(lexical_scores) > 1 else 0.0

    query_hits_by_doc = [len(query_tokens & tokens) for tokens in doc_all_sets]
    query_title_bridge = [len(query_tokens & title) > 0 and len(query_tokens & text) > 0 for title, text in zip(doc_title_sets, doc_text_sets)]
    query_text_bridge = [len(query_tokens & text) >= 2 for text in doc_text_sets]

    top1_to_other_title = []
    if doc_all_sets:
        top1_text = doc_text_sets[0]
        for title_tokens in doc_title_sets[1:]:
            top1_to_other_title.append(_jaccard(top1_text, title_tokens))

    title_chain_overlaps = [_jaccard(left, right) for left, right in combinations(doc_title_sets, 2)]
    text_to_title_overlaps = [_jaccard(text, title) for text, title in zip(doc_text_sets, doc_title_sets)]

    union_coverage = float(basic.get("query_token_coverage_union", 0.0))
    top1_coverage = float(basic.get("query_token_coverage_top1", 0.0))
    pairwise_overlap = float(basic.get("pairwise_doc_overlap_mean", 0.0))
    redundancy = float(basic.get("doc_redundancy", 0.0))
    score_entropy = float(basic.get("score_entropy", 0.0))
    embedding_mean = float(basic.get("embedding_topk_score_mean", 0.0))

    return {
        "title_token_coverage_union": _coverage(query_tokens, title_union),
        "title_token_coverage_top1": _coverage(query_tokens, doc_title_sets[0] if doc_title_sets else set()),
        "title_token_coverage_top3": _coverage(query_tokens, top3_title_union),
        "title_bigram_coverage_union": _coverage(query_bigrams, title_bigrams),
        "text_bigram_coverage_union": _coverage(query_bigrams, text_bigrams),
        "top1_text_to_other_title_overlap_max": float(max(top1_to_other_title)) if top1_to_other_title else 0.0,
        "top1_text_to_other_title_overlap_mean": float(mean(top1_to_other_title)) if top1_to_other_title else 0.0,
        "query_title_bridge_doc_count": float(sum(query_title_bridge)),
        "query_text_bridge_doc_count": float(sum(query_text_bridge)),
        "multi_doc_query_coverage_count": float(sum(1 for hits in query_hits_by_doc if hits > 0)),
        "top3_unique_query_token_hits": float(len(query_tokens & (top3_title_union | top3_text_union))),
        "top5_unique_query_token_hits": float(len(query_tokens & (title_union | text_union))),
        "top3_query_token_hit_ratio": _coverage(query_tokens, top3_title_union | top3_text_union),
        "top5_query_token_hit_ratio": _coverage(query_tokens, title_union | text_union),
        "title_chain_overlap_mean": float(mean(title_chain_overlaps)) if title_chain_overlaps else 0.0,
        "title_chain_overlap_max": float(max(title_chain_overlaps)) if title_chain_overlaps else 0.0,
        "evidence_text_to_title_overlap_mean": float(mean(text_to_title_overlaps)) if text_to_title_overlaps else 0.0,
        "evidence_text_to_title_overlap_max": float(max(text_to_title_overlaps)) if text_to_title_overlaps else 0.0,
        "embedding_top1_share": float(embedding_scores[0] / embedding_total) if embedding_total > 0.0 and embedding_scores else 0.0,
        "embedding_top3_share": float(sum(embedding_scores[:3]) / embedding_total) if embedding_total > 0.0 else 0.0,
        "embedding_gap_to_std": float(embedding_gap / (embedding_std + 1e-8)),
        "lexical_gap_to_std": float(lexical_gap / (lexical_std + 1e-8)),
        "coverage_embedding_product": float(union_coverage * embedding_mean),
        "coverage_margin_product": float(union_coverage * embedding_gap),
        "top1_coverage_embedding_product": float(top1_coverage * (embedding_scores[0] if embedding_scores else 0.0)),
        "diversity_adjusted_coverage": float(union_coverage * (1.0 - pairwise_overlap)),
        "redundancy_adjusted_coverage": float(union_coverage * (1.0 - redundancy)),
        "score_entropy_coverage_product": float(score_entropy * union_coverage),
    }


def _bigrams(tokens: list[str]) -> set[str]:
    return {f"{left} {right}" for left, right in zip(tokens, tokens[1:])}


def _coverage(query_items: set[str], candidate_items: set[str]) -> float:
    if not query_items:
        return 0.0
    return float(len(query_items & candidate_items) / len(query_items))


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return float(len(left & right) / len(union))


def _normalize_title(title: str) -> str:
    return " ".join(tokenize(title))


def _contains_answer(text: str, answer: str) -> bool:
    if not answer:
        return False
    return " ".join(tokenize(answer)) in " ".join(tokenize(text))


def _concat_docs(docs: list[dict[str, Any]]) -> str:
    return " ".join(f"{doc.get('title', '')} {doc.get('text', '')}" for doc in docs)
