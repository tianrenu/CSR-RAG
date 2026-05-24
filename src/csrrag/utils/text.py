"""Text normalization and lexical overlap helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def overlap_ratio(query_tokens: Iterable[str], candidate_tokens: Iterable[str]) -> float:
    query_set = set(query_tokens)
    if not query_set:
        return 0.0
    candidate_set = set(candidate_tokens)
    return len(query_set & candidate_set) / len(query_set)


def lexical_overlap_features(query: str, title: str, text: str) -> dict[str, float]:
    query_tokens = tokenize(query)
    title_tokens = tokenize(title)
    text_tokens = tokenize(text)
    return {
        "title_overlap": overlap_ratio(query_tokens, title_tokens),
        "text_overlap": overlap_ratio(query_tokens, text_tokens),
    }


def lexical_score(query: str, title: str, text: str) -> float:
    features = lexical_overlap_features(query, title, text)
    return 0.6 * features["title_overlap"] + 0.4 * features["text_overlap"]
