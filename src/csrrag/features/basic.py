"""Feature extraction for the first credible CSR-RAG experiment."""

from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Any

from csrrag.utils.text import lexical_overlap_features, token_set, tokenize


TIME_TERMS = {
    "today",
    "yesterday",
    "tomorrow",
    "year",
    "month",
    "date",
    "recent",
    "latest",
    "current",
}

CONSTRAINT_TERMS = {
    "before",
    "after",
    "between",
    "only",
    "first",
    "last",
    "most",
    "least",
}


def extract_basic_features(record: dict[str, Any]) -> dict[str, float]:
    query = record.get("query", "")
    docs = record.get("retrieved_docs", [])
    scores = [float(doc.get("score", 0.0)) for doc in docs]
    embedding_scores = [float(doc.get("embedding_score", 0.0)) for doc in docs]
    doc_texts = [str(doc.get("text", "")) for doc in docs]
    doc_titles = [str(doc.get("title", "")) for doc in docs]
    unique_doc_texts = set(doc_texts)

    top1 = scores[0] if scores else 0.0
    top2 = scores[1] if len(scores) > 1 else 0.0
    top5 = scores[4] if len(scores) > 4 else 0.0
    embedding_top1 = embedding_scores[0] if embedding_scores else 0.0
    embedding_top2 = embedding_scores[1] if len(embedding_scores) > 1 else 0.0
    embedding_top5 = embedding_scores[4] if len(embedding_scores) > 4 else 0.0
    redundancy = 1.0 - (len(unique_doc_texts) / len(doc_texts)) if doc_texts else 0.0

    title_overlaps = []
    text_overlaps = []
    doc_token_sets = []
    doc_text_lengths = []
    for doc in docs:
        title = str(doc.get("title", ""))
        text = str(doc.get("text", ""))
        overlap_features = lexical_overlap_features(
            query=query,
            title=title,
            text=text,
        )
        title_overlaps.append(overlap_features["title_overlap"])
        text_overlaps.append(overlap_features["text_overlap"])
        doc_token_sets.append(token_set(f"{title} {text}"))
        doc_text_lengths.append(len(tokenize(text)))

    query_tokens = tokenize(query)
    query_token_set = set(query_tokens)
    union_doc_tokens = set().union(*doc_token_sets) if doc_token_sets else set()
    top1_doc_tokens = doc_token_sets[0] if doc_token_sets else set()
    top3_doc_tokens = set().union(*doc_token_sets[:3]) if doc_token_sets else set()

    title_tokens = []
    for title in doc_titles:
        title_tokens.extend(tokenize(title))

    features = {
        "query_length": float(len(query_tokens)),
        "has_time_constraint": float(_contains_any(query, TIME_TERMS)),
        "has_constraint_term": float(_contains_any(query, CONSTRAINT_TERMS)),
        "topk_score_mean": float(mean(scores)) if scores else 0.0,
        "topk_score_std": float(pstdev(scores)) if len(scores) > 1 else 0.0,
        "top1_top2_gap": float(top1 - top2),
        "doc_count": float(len(docs)),
        "doc_redundancy": float(redundancy),
        "title_overlap_max": float(max(title_overlaps)) if title_overlaps else 0.0,
        "title_overlap_mean": float(mean(title_overlaps)) if title_overlaps else 0.0,
        "text_overlap_max": float(max(text_overlaps)) if text_overlaps else 0.0,
        "text_overlap_mean": float(mean(text_overlaps)) if text_overlaps else 0.0,
        "top1_score": float(top1),
        "top3_score_mean": float(mean(scores[:3])) if scores else 0.0,
        "top5_score_min": float(min(scores[:5])) if scores else 0.0,
        "top1_top5_gap": float(top1 - top5),
        "score_entropy": _normalized_entropy(scores),
        "query_token_coverage_union": _coverage(query_token_set, union_doc_tokens),
        "query_token_coverage_top1": _coverage(query_token_set, top1_doc_tokens),
        "query_token_coverage_top3": _coverage(query_token_set, top3_doc_tokens),
        "uncovered_query_token_ratio": 1.0 - _coverage(query_token_set, union_doc_tokens),
        "pairwise_doc_overlap_mean": _pairwise_overlap_mean(doc_token_sets),
        "pairwise_doc_overlap_max": _pairwise_overlap_max(doc_token_sets),
        "unique_title_token_ratio": (len(set(title_tokens)) / len(title_tokens)) if title_tokens else 0.0,
        "doc_text_length_mean": float(mean(doc_text_lengths)) if doc_text_lengths else 0.0,
        "doc_text_length_std": float(pstdev(doc_text_lengths)) if len(doc_text_lengths) > 1 else 0.0,
        "embedding_topk_score_mean": float(mean(embedding_scores)) if embedding_scores else 0.0,
        "embedding_topk_score_std": float(pstdev(embedding_scores)) if len(embedding_scores) > 1 else 0.0,
        "embedding_top1_top2_gap": float(embedding_top1 - embedding_top2),
        "embedding_top1_top5_gap": float(embedding_top1 - embedding_top5),
        "embedding_score_entropy": _normalized_entropy(embedding_scores),
    }
    features.update(_question_form_features(query_tokens, query))
    return features


def _contains_any(text: str, terms: set[str]) -> bool:
    lower_text = text.lower()
    return any(term in lower_text for term in terms)


def _coverage(query_tokens: set[str], candidate_tokens: set[str]) -> float:
    if not query_tokens:
        return 0.0
    return float(len(query_tokens & candidate_tokens) / len(query_tokens))


def _normalized_entropy(scores: list[float]) -> float:
    positive_scores = [max(float(score), 0.0) for score in scores]
    total = sum(positive_scores)
    if total <= 0.0 or len(positive_scores) <= 1:
        return 0.0
    probs = [score / total for score in positive_scores if score > 0.0]
    entropy = -sum(prob * math.log(prob) for prob in probs)
    return float(entropy / math.log(len(positive_scores)))


def _pairwise_overlap_mean(token_sets: list[set[str]]) -> float:
    overlaps = _pairwise_jaccards(token_sets)
    return float(mean(overlaps)) if overlaps else 0.0


def _pairwise_overlap_max(token_sets: list[set[str]]) -> float:
    overlaps = _pairwise_jaccards(token_sets)
    return float(max(overlaps)) if overlaps else 0.0


def _pairwise_jaccards(token_sets: list[set[str]]) -> list[float]:
    overlaps = []
    for i, left in enumerate(token_sets):
        for right in token_sets[i + 1:]:
            union = left | right
            if not union:
                overlaps.append(0.0)
            else:
                overlaps.append(len(left & right) / len(union))
    return overlaps


def _question_form_features(query_tokens: list[str], query: str) -> dict[str, float]:
    token_set_value = set(query_tokens)
    lower_query = query.lower()
    comparison_terms = {
        "same",
        "both",
        "either",
        "older",
        "younger",
        "larger",
        "smaller",
        "more",
        "less",
        "earlier",
        "later",
    }
    bridge_terms = {"who", "what", "where", "when", "which"}
    wh_words = ["who", "what", "when", "where", "which", "how"]
    features = {
        "is_comparison_question": float(any(term in token_set_value for term in comparison_terms) or " or " in lower_query),
        "is_bridge_like_question": float(any(term in token_set_value for term in bridge_terms) and not any(term in token_set_value for term in comparison_terms)),
    }
    first_token = query_tokens[0] if query_tokens else ""
    for wh_word in wh_words:
        features[f"wh_{wh_word}"] = float(first_token == wh_word or wh_word in token_set_value)
    features["wh_other"] = float(not any(features[f"wh_{wh_word}"] for wh_word in wh_words))
    return features
