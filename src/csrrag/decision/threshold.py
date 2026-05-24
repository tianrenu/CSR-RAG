"""Threshold based selective decision policies."""

from __future__ import annotations


def answer_or_abstain(risk_score: float, tau_answer: float) -> str:
    return "answer" if risk_score <= tau_answer else "abstain"


def answer_refine_or_abstain(
    risk_score: float,
    tau_answer: float,
    tau_refine: float,
) -> str:
    if risk_score <= tau_answer:
        return "answer"
    if risk_score <= tau_refine:
        return "refine"
    return "abstain"
