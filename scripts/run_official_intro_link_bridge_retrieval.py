from __future__ import annotations

import argparse
import bz2
import csv
import heapq
import html
import json
import math
import re
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote

from csrrag.utils.io import write_jsonl
from csrrag.utils.text import lexical_score, tokenize
from run_official_intro_bm25_retrieval import (
    _answer_present,
    _bm25_idf,
    _bm25_term_score,
    _build_query_terms,
    _doc_tokens,
    _load_raw_rows,
    _load_split_ids,
    _mean,
    _normalize_title,
    _parse_int_list,
    _question_records,
    _require,
)


LINK_RE = re.compile(r"<a\s+href=\"([^\"]+)\"", flags=re.IGNORECASE)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a no-API link-bridge retrieval experiment over the official HotpotQA intro corpus. "
            "It uses first-hop BM25 plus Wikipedia links from retrieved pages; gold support is used only for evaluation."
        )
    )
    parser.add_argument("--raw-hotpot", default="data/raw/hotpotqa/hotpot_dev_fullwiki_v1.json")
    parser.add_argument(
        "--wiki-archive",
        default="data/external/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2",
    )
    parser.add_argument("--split-dir", default="data/processed/hotpotqa_global_hardneg_splits_full_dev")
    parser.add_argument("--record-kind-filter", default="natural_global_top5")
    parser.add_argument("--output-dir", default="results/tables/hotpotqa_official_intro_link_bridge_retrieval")
    parser.add_argument("--output-split-dir", default="data/processed/hotpotqa_official_intro_link_bridge_splits_full_dev")
    parser.add_argument("--top-ks", default="5,10,20,50")
    parser.add_argument("--write-record-top-ks", default="5")
    parser.add_argument("--first-hop-k", type=int, default=50)
    parser.add_argument("--link-source-top-k", type=int, default=10)
    parser.add_argument("--max-links-per-doc", type=int, default=80)
    parser.add_argument("--link-alphas", default="0.70,0.85,1.00,1.15")
    parser.add_argument("--link-position-penalties", default="0.00,0.03,0.06")
    parser.add_argument("--query-title-boost", type=float, default=3.0)
    parser.add_argument("--title-weight", type=int, default=3)
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--min-query-token-len", type=int, default=2)
    parser.add_argument("--max-queries-per-token", type=int, default=300)
    parser.add_argument("--max-questions-per-split", type=int, default=0, help="Debug only. 0 means all questions.")
    parser.add_argument("--max-inner-files", type=int, default=0, help="Debug only. 0 means all wiki files.")
    parser.add_argument("--max-docs", type=int, default=0, help="Debug only. 0 means all docs.")
    parser.add_argument("--progress-every-docs", type=int, default=500000)
    args = parser.parse_args()

    started = time.time()
    raw_rows = _load_raw_rows(Path(args.raw_hotpot))
    split_ids = _load_split_ids(Path(args.split_dir), args.record_kind_filter, args.max_questions_per_split)
    questions = _question_records(raw_rows, split_ids)
    top_ks = _parse_int_list(args.top_ks)
    write_record_top_ks = set(_parse_int_list(args.write_record_top_ks))
    link_alphas = _parse_float_list(args.link_alphas)
    penalties = _parse_float_list(args.link_position_penalties)
    _require(write_record_top_ks.issubset(set(top_ks)), "write-record-top-ks must be a subset of top-ks.")
    max_k = max(max(top_ks), args.first_hop_k)

    query_terms, token_to_query_ids = _build_query_terms(
        questions=questions,
        min_len=args.min_query_token_len,
        max_queries_per_token=args.max_queries_per_token,
    )
    first_pass = _first_pass_df(
        archive_path=Path(args.wiki_archive),
        query_vocab=set(token_to_query_ids),
        title_weight=args.title_weight,
        max_inner_files=args.max_inner_files,
        max_docs=args.max_docs,
        progress_every_docs=args.progress_every_docs,
    )
    idf = _bm25_idf(first_pass["doc_count"], first_pass["df"])
    first_hop_docs = _first_hop_rank(
        archive_path=Path(args.wiki_archive),
        questions=questions,
        query_terms=query_terms,
        token_to_query_ids=token_to_query_ids,
        idf=idf,
        avgdl=first_pass["avgdl"],
        title_weight=args.title_weight,
        k1=args.k1,
        b=args.b,
        max_k=max_k,
        max_inner_files=args.max_inner_files,
        max_docs=args.max_docs,
        progress_every_docs=args.progress_every_docs,
    )
    link_requests = _build_link_requests(
        first_hop_docs=first_hop_docs,
        source_top_k=args.link_source_top_k,
        max_links_per_doc=args.max_links_per_doc,
    )
    linked_docs_by_qid = _collect_linked_docs(
        archive_path=Path(args.wiki_archive),
        link_requests=link_requests,
        max_inner_files=args.max_inner_files,
        max_docs=args.max_docs,
        progress_every_docs=args.progress_every_docs,
    )

    variants = []
    for alpha in link_alphas:
        for penalty in penalties:
            method = _variant_name(alpha, penalty)
            final_docs = _rerank_with_links(
                questions=questions,
                first_hop_docs=first_hop_docs,
                linked_docs_by_qid=linked_docs_by_qid,
                link_alpha=alpha,
                link_position_penalty=penalty,
                query_title_boost=args.query_title_boost,
                max_k=max(top_ks),
            )
            variants.append(
                {
                    "method": method,
                    "link_alpha": alpha,
                    "link_position_penalty": penalty,
                    "ranked_docs": final_docs,
                    "topk_rows": _topk_curve_rows(questions, final_docs, top_ks, method, alpha, penalty),
                }
            )
    baseline_rows = _topk_curve_rows(questions, first_hop_docs, top_ks, "firsthop_bm25", 0.0, 0.0)
    selected = _select_variant(variants)
    selected_docs = selected["ranked_docs"]
    split_records = _build_split_records(raw_rows, questions, selected_docs, top_ks, write_record_top_ks, selected["method"])
    detail_rows = _question_detail_rows(questions, selected_docs, top_ks, selected["method"])
    link_stats_rows = _link_stats_rows(questions, first_hop_docs, linked_docs_by_qid, args.link_source_top_k)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_topk_rows = list(baseline_rows)
    for variant in variants:
        all_topk_rows.extend(variant["topk_rows"])
    _write_csv(output_dir / "topk_sufficiency_curve.csv", all_topk_rows)
    _write_csv(output_dir / "selected_question_details.csv", detail_rows)
    _write_csv(output_dir / "link_candidate_stats.csv", link_stats_rows)
    _write_csv(output_dir / "variant_selection.csv", _variant_selection_rows(variants, selected))
    _write_summary(output_dir / "link_bridge_retrieval_summary.md", baseline_rows, selected, first_pass, args)
    _write_validation(
        output_dir / "validation_summary.json",
        args=args,
        split_ids=split_ids,
        questions=questions,
        top_ks=top_ks,
        write_record_top_ks=write_record_top_ks,
        first_pass=first_pass,
        link_requests=link_requests,
        linked_docs_by_qid=linked_docs_by_qid,
        selected=selected,
        elapsed_seconds=time.time() - started,
    )

    output_split_dir = Path(args.output_split_dir)
    output_split_dir.mkdir(parents=True, exist_ok=True)
    for split, records in split_records.items():
        write_jsonl(output_split_dir / f"{split}.jsonl", records)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "output_split_dir": str(output_split_dir),
                "selected_method": selected["method"],
                "questions": len(questions),
                "doc_count": first_pass["doc_count"],
                "linked_titles_requested": len(link_requests),
                "uses_embedding_api": False,
                "uses_llm_api": False,
            },
            ensure_ascii=False,
        )
    )


def _first_pass_df(
    archive_path: Path,
    query_vocab: set[str],
    title_weight: int,
    max_inner_files: int,
    max_docs: int,
    progress_every_docs: int,
) -> dict[str, Any]:
    started = time.time()
    doc_count = 0
    total_len = 0
    df: Counter[str] = Counter()
    for doc in _iter_wiki_docs(archive_path, max_inner_files, max_docs):
        doc_count += 1
        tokens = _doc_tokens(doc, title_weight)
        total_len += len(tokens)
        for term in set(tokens) & query_vocab:
            df[term] += 1
        if progress_every_docs > 0 and doc_count % progress_every_docs == 0:
            print(
                json.dumps(
                    {
                        "pass": "df",
                        "docs": doc_count,
                        "query_vocab_terms_seen": len(df),
                        "elapsed_seconds": round(time.time() - started, 1),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    _require(doc_count > 0, "No wiki documents were read.")
    return {
        "doc_count": doc_count,
        "avgdl": total_len / doc_count,
        "df": dict(df),
        "elapsed_seconds": round(time.time() - started, 2),
    }


def _first_hop_rank(
    archive_path: Path,
    questions: list[dict[str, Any]],
    query_terms: list[set[str]],
    token_to_query_ids: dict[str, list[int]],
    idf: dict[str, float],
    avgdl: float,
    title_weight: int,
    k1: float,
    b: float,
    max_k: int,
    max_inner_files: int,
    max_docs: int,
    progress_every_docs: int,
) -> list[list[dict[str, Any]]]:
    started = time.time()
    heaps: list[list[tuple[float, int, dict[str, Any]]]] = [[] for _ in questions]
    doc_count = 0
    matched_docs = 0
    for doc in _iter_wiki_docs(archive_path, max_inner_files, max_docs):
        doc_count += 1
        counts = Counter(_doc_tokens(doc, title_weight))
        doc_len = sum(counts.values())
        scores: dict[int, float] = defaultdict(float)
        for term, tf in counts.items():
            qids = token_to_query_ids.get(term)
            if not qids:
                continue
            contribution = _bm25_term_score(tf=tf, doc_len=doc_len, avgdl=avgdl, idf=idf.get(term, 0.0), k1=k1, b=b)
            if contribution <= 0.0:
                continue
            for qid in qids:
                if term in query_terms[qid]:
                    scores[qid] += contribution
        if scores:
            matched_docs += 1
        for qid, score in scores.items():
            if score > 0.0:
                _push_doc(heaps[qid], max_k, score, doc_count, doc)
        if progress_every_docs > 0 and doc_count % progress_every_docs == 0:
            print(
                json.dumps(
                    {
                        "pass": "first_hop_rank",
                        "docs": doc_count,
                        "matched_docs": matched_docs,
                        "queries_with_hits": sum(1 for heap in heaps if heap),
                        "elapsed_seconds": round(time.time() - started, 1),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return _ranked_docs_from_heaps(heaps)


def _build_link_requests(
    first_hop_docs: list[list[dict[str, Any]]],
    source_top_k: int,
    max_links_per_doc: int,
) -> dict[str, dict[int, dict[str, Any]]]:
    requests: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for qid, docs in enumerate(first_hop_docs):
        for source_rank, doc in enumerate(docs[:source_top_k], start=1):
            source_title = _normalize_title(str(doc.get("title", "")))
            source_score = float(doc.get("bm25_score", 0.0))
            for link_position, linked_title in enumerate(doc.get("links", [])[:max_links_per_doc], start=1):
                if not linked_title or linked_title == source_title:
                    continue
                existing = requests[linked_title].get(qid)
                candidate = {
                    "source_score": source_score,
                    "source_rank": source_rank,
                    "source_title": doc.get("title", ""),
                    "link_position": link_position,
                }
                if existing is None or _link_source_key(candidate) > _link_source_key(existing):
                    requests[linked_title][qid] = candidate
    return dict(requests)


def _collect_linked_docs(
    archive_path: Path,
    link_requests: dict[str, dict[int, dict[str, Any]]],
    max_inner_files: int,
    max_docs: int,
    progress_every_docs: int,
) -> list[list[dict[str, Any]]]:
    started = time.time()
    needed_titles = set(link_requests)
    linked_docs_by_qid: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    doc_count = 0
    matched_titles = set()
    for doc in _iter_wiki_docs(archive_path, max_inner_files, max_docs):
        doc_count += 1
        title = _normalize_title(str(doc.get("title", "")))
        if title in needed_titles:
            matched_titles.add(title)
            for qid, link_info in link_requests[title].items():
                existing = linked_docs_by_qid[qid].get(title)
                candidate = {**doc, "link_info": link_info}
                if existing is None or _link_source_key(link_info) > _link_source_key(existing["link_info"]):
                    linked_docs_by_qid[qid][title] = candidate
        if progress_every_docs > 0 and doc_count % progress_every_docs == 0:
            print(
                json.dumps(
                    {
                        "pass": "collect_linked_docs",
                        "docs": doc_count,
                        "matched_titles": len(matched_titles),
                        "needed_titles": len(needed_titles),
                        "elapsed_seconds": round(time.time() - started, 1),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    max_qid = max((qid for q_docs in linked_docs_by_qid.values() for qid in [0]), default=-1)
    if linked_docs_by_qid:
        max_qid = max(linked_docs_by_qid)
    result = [[] for _ in range(max_qid + 1)]
    for qid, docs_by_title in linked_docs_by_qid.items():
        result[qid] = list(docs_by_title.values())
    return result


def _rerank_with_links(
    questions: list[dict[str, Any]],
    first_hop_docs: list[list[dict[str, Any]]],
    linked_docs_by_qid: list[list[dict[str, Any]]],
    link_alpha: float,
    link_position_penalty: float,
    query_title_boost: float,
    max_k: int,
) -> list[list[dict[str, Any]]]:
    ranked_by_question = []
    for qid, question in enumerate(questions):
        candidates: dict[str, dict[str, Any]] = {}
        for doc in first_hop_docs[qid]:
            title = _normalize_title(str(doc.get("title", "")))
            score = float(doc.get("bm25_score", 0.0))
            _upsert_candidate(candidates, title, {**doc, "final_score": score, "retrieval_path": "firsthop_bm25"})
        if qid < len(linked_docs_by_qid):
            for doc in linked_docs_by_qid[qid]:
                title = _normalize_title(str(doc.get("title", "")))
                link_info = doc["link_info"]
                title_overlap = _title_overlap(question["question"], doc.get("title", ""))
                link_score = (
                    float(link_info["source_score"]) * link_alpha
                    + title_overlap * query_title_boost
                    - float(link_info["link_position"]) * link_position_penalty
                )
                candidate = {
                    **doc,
                    "bm25_score": 0.0,
                    "final_score": link_score,
                    "retrieval_path": "linked_from_firsthop",
                }
                _upsert_candidate(candidates, title, candidate)
        ranked = sorted(
            candidates.values(),
            key=lambda doc: (-float(doc.get("final_score", 0.0)), int(doc.get("doc_seq", 10**12)), str(doc.get("title", ""))),
        )
        final_docs = []
        for rank, doc in enumerate(ranked[:max_k], start=1):
            final_docs.append({**doc, "rank": rank})
        ranked_by_question.append(final_docs)
    return ranked_by_question


def _build_split_records(
    raw_rows: dict[str, dict[str, Any]],
    questions: list[dict[str, Any]],
    ranked_docs: list[list[dict[str, Any]]],
    top_ks: list[int],
    write_record_top_ks: set[int],
    method: str,
) -> dict[str, list[dict[str, Any]]]:
    split_records = {"train": [], "valid": [], "test": []}
    for question in questions:
        raw = raw_rows[question["original_id"]]
        docs = ranked_docs[question["qid"]]
        for top_k in top_ks:
            if top_k not in write_record_top_ks:
                continue
            split_records[question["split"]].append(_make_record(raw, question, docs[:top_k], top_k, method))
    return split_records


def _make_record(raw: dict[str, Any], question: dict[str, Any], docs: list[dict[str, Any]], top_k: int, method: str) -> dict[str, Any]:
    retrieved_docs = _retrieved_docs_for_record(question["question"], docs)
    support_titles = set(question["support_titles"])
    retrieved_titles = {_normalize_title(doc["title"]) for doc in retrieved_docs}
    support_present = support_titles.issubset(retrieved_titles)
    missing_support_titles = sorted(support_titles - retrieved_titles)
    record_kind = f"official_intro_{method}_top{top_k}"
    return {
        "id": f"{raw['_id']}__{record_kind}",
        "query": raw["question"],
        "gold_answer": raw["answer"],
        "sufficiency_label": "sufficient" if support_present else "insufficient",
        "retrieved_docs": retrieved_docs,
        "metadata": {
            "dataset": "hotpotqa",
            "split": question["split"],
            "original_id": raw["_id"],
            "question_type": raw.get("type", "unknown"),
            "difficulty": raw.get("level", "unknown"),
            "retriever": f"official-intro-{method}",
            "record_kind": record_kind,
            "top_k": top_k,
            "support_present_in_topk": support_present,
            "support_titles": sorted(support_titles),
            "missing_support_titles": missing_support_titles,
            "forced_missing_support_titles": [],
        },
    }


def _retrieved_docs_for_record(query: str, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    retrieved = []
    for rank, doc in enumerate(docs, start=1):
        final_score = float(doc.get("final_score", doc.get("bm25_score", 0.0)))
        retrieved.append(
            {
                "doc_id": doc["doc_id"],
                "rank": rank,
                "score": round(float(lexical_score(query, doc["title"], doc["text"])), 6),
                "embedding_score": round(final_score, 6),
                "sparse_score": round(float(doc.get("bm25_score", 0.0)), 6),
                "bridge_score": round(final_score, 6),
                "title": doc["title"],
                "text": doc["text"],
                "source": f"hotpotqa_official_intro_{doc.get('retrieval_path', 'unknown')}",
            }
        )
    return retrieved


def _topk_curve_rows(
    questions: list[dict[str, Any]],
    ranked_docs: list[list[dict[str, Any]]],
    top_ks: list[int],
    method: str,
    link_alpha: float,
    link_position_penalty: float,
) -> list[dict[str, Any]]:
    rows = []
    for split in ("train", "valid", "test", "all"):
        split_questions = [q for q in questions if split == "all" or q["split"] == split]
        for top_k in top_ks:
            detail = [_detail_for_question(q, ranked_docs[q["qid"]], top_k, method) for q in split_questions]
            sufficient = [row for row in detail if row["support_present_in_topk"]]
            rows.append(
                {
                    "method": method,
                    "link_alpha": link_alpha,
                    "link_position_penalty": link_position_penalty,
                    "split": split,
                    "top_k": top_k,
                    "n": len(detail),
                    "sufficient_count": len(sufficient),
                    "insufficient_count": len(detail) - len(sufficient),
                    "sufficient_rate": len(sufficient) / len(detail) if detail else 0.0,
                    "answer_present_in_returned_docs_rate": _mean(float(row["answer_present_in_returned_docs"]) for row in detail),
                    "mean_support_title_coverage": _mean(float(row["support_title_coverage"]) for row in detail),
                    "mean_linked_doc_count_topk": _mean(float(row["linked_doc_count_topk"]) for row in detail),
                    "questions_with_no_hits": sum(int(row["retrieved_count"] == 0) for row in detail),
                }
            )
    return rows


def _question_detail_rows(
    questions: list[dict[str, Any]],
    ranked_docs: list[list[dict[str, Any]]],
    top_ks: list[int],
    method: str,
) -> list[dict[str, Any]]:
    rows = []
    for question in questions:
        for top_k in top_ks:
            rows.append(_detail_for_question(question, ranked_docs[question["qid"]], top_k, method))
    return rows


def _detail_for_question(question: dict[str, Any], docs: list[dict[str, Any]], top_k: int, method: str) -> dict[str, Any]:
    selected = docs[:top_k]
    support_titles = set(question["support_titles"])
    title_to_rank = {_normalize_title(doc["title"]): int(doc["rank"]) for doc in selected}
    retrieved_support = sorted(support_titles & set(title_to_rank))
    missing_support = sorted(support_titles - set(title_to_rank))
    ranks = [title_to_rank[title] for title in retrieved_support]
    return {
        "method": method,
        "split": question["split"],
        "top_k": top_k,
        "original_id": question["original_id"],
        "question_type": question["question_type"],
        "difficulty": question["difficulty"],
        "question": question["question"],
        "gold_answer": question["gold_answer"],
        "retrieved_count": len(selected),
        "linked_doc_count_topk": sum(1 for doc in selected if doc.get("retrieval_path") == "linked_from_firsthop"),
        "support_title_count": len(support_titles),
        "retrieved_support_title_count": len(retrieved_support),
        "support_title_coverage": len(retrieved_support) / len(support_titles) if support_titles else 0.0,
        "support_present_in_topk": int(not missing_support),
        "missing_support_titles": " || ".join(missing_support),
        "retrieved_support_ranks": " || ".join(str(rank) for rank in sorted(ranks)),
        "max_retrieved_support_rank": max(ranks) if ranks else "",
        "answer_present_in_returned_docs": int(_answer_present(question["gold_answer"], selected)),
        "top_titles": " || ".join(doc["title"] for doc in selected[:5]),
        "top_paths": " || ".join(str(doc.get("retrieval_path", "")) for doc in selected[:5]),
    }


def _link_stats_rows(
    questions: list[dict[str, Any]],
    first_hop_docs: list[list[dict[str, Any]]],
    linked_docs_by_qid: list[list[dict[str, Any]]],
    link_source_top_k: int,
) -> list[dict[str, Any]]:
    rows = []
    for question in questions:
        qid = question["qid"]
        source_docs = first_hop_docs[qid][:link_source_top_k]
        linked_docs = linked_docs_by_qid[qid] if qid < len(linked_docs_by_qid) else []
        linked_titles = {_normalize_title(doc["title"]) for doc in linked_docs}
        support_titles = set(question["support_titles"])
        rows.append(
            {
                "split": question["split"],
                "original_id": question["original_id"],
                "source_docs": len(source_docs),
                "source_link_count": sum(len(doc.get("links", [])) for doc in source_docs),
                "linked_doc_count": len(linked_docs),
                "support_titles_in_linked_docs_count": len(support_titles & linked_titles),
                "support_titles_in_linked_docs_rate": len(support_titles & linked_titles) / len(support_titles) if support_titles else 0.0,
            }
        )
    return rows


def _select_variant(variants: list[dict[str, Any]]) -> dict[str, Any]:
    available_top_ks = sorted(
        {
            int(row["top_k"])
            for variant in variants
            for row in variant["topk_rows"]
            if row["split"] == "valid"
        }
    )
    priority_top_ks = [top_k for top_k in [5, 10, 20, 50] if top_k in available_top_ks]
    _require(bool(priority_top_ks), "No valid top-k rows are available for variant selection.")

    def valid_row(variant: dict[str, Any], top_k: int) -> dict[str, Any]:
        return next(row for row in variant["topk_rows"] if row["split"] == "valid" and int(row["top_k"]) == top_k)

    def key(variant: dict[str, Any]) -> tuple[float, ...]:
        rates = [float(valid_row(variant, top_k)["sufficient_rate"]) for top_k in priority_top_ks]
        linked_penalty = -float(valid_row(variant, priority_top_ks[0])["mean_linked_doc_count_topk"])
        return tuple(rates + [linked_penalty])

    return max(
        variants,
        key=key,
    )


def _variant_selection_rows(variants: list[dict[str, Any]], selected: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for variant in variants:
        for row in variant["topk_rows"]:
            if row["split"] in {"valid", "test"}:
                rows.append({**row, "selected_by_valid": int(variant["method"] == selected["method"])})
    return rows


def _write_summary(path: Path, baseline_rows: list[dict[str, Any]], selected: dict[str, Any], first_pass: dict[str, Any], args: argparse.Namespace) -> None:
    baseline_test = [row for row in baseline_rows if row["split"] == "test"]
    selected_test = [row for row in selected["topk_rows"] if row["split"] == "test"]
    base_lines = "\n".join(
        f"- top-{row['top_k']}: sufficient_rate={float(row['sufficient_rate']):.4f}"
        for row in baseline_test
    )
    selected_lines = "\n".join(
        f"- top-{row['top_k']}: sufficient_rate={float(row['sufficient_rate']):.4f}, "
        f"linked_docs_topk={float(row['mean_linked_doc_count_topk']):.2f}"
        for row in selected_test
    )
    text = f"""# Official HotpotQA Intro Link-Bridge Retrieval

## Purpose

This no-API run tests whether Wikipedia links from first-hop BM25 documents can improve support-chain retrieval over the official HotpotQA intro corpus. Gold support titles are used only for evaluation and valid-only variant selection.

## Corpus

- Wiki archive: `{args.wiki_archive}`
- Documents scanned per pass: `{first_pass['doc_count']}`
- First-hop K: `{args.first_hop_k}`
- Link source top K: `{args.link_source_top_k}`
- Max links per source doc: `{args.max_links_per_doc}`

## First-Hop BM25 Test Curve

{base_lines}

## Selected Link-Bridge Test Curve

Selected method: `{selected['method']}`

{selected_lines}

## Interpretation

If selected link-bridge improves top-5/top-10 sufficiency without oracle support injection, it is a promising retrieval refinement path. If it hurts top-5 while improving top-50 only, links should be treated as candidate expansion for a stronger reranker rather than direct ranking.
"""
    path.write_text(text, encoding="utf-8")


def _write_validation(
    path: Path,
    args: argparse.Namespace,
    split_ids: dict[str, list[str]],
    questions: list[dict[str, Any]],
    top_ks: list[int],
    write_record_top_ks: set[int],
    first_pass: dict[str, Any],
    link_requests: dict[str, dict[int, dict[str, Any]]],
    linked_docs_by_qid: list[list[dict[str, Any]]],
    selected: dict[str, Any],
    elapsed_seconds: float,
) -> None:
    linked_doc_count = sum(len(docs) for docs in linked_docs_by_qid)
    validation = {
        "raw_hotpot": args.raw_hotpot,
        "wiki_archive": args.wiki_archive,
        "split_dir": args.split_dir,
        "record_kind_filter": args.record_kind_filter,
        "output_split_dir": args.output_split_dir,
        "split_counts": {split: len(ids) for split, ids in split_ids.items()},
        "question_count": len(questions),
        "top_ks": top_ks,
        "write_record_top_ks": sorted(write_record_top_ks),
        "first_hop_k": args.first_hop_k,
        "link_source_top_k": args.link_source_top_k,
        "max_links_per_doc": args.max_links_per_doc,
        "link_alphas": _parse_float_list(args.link_alphas),
        "link_position_penalties": _parse_float_list(args.link_position_penalties),
        "selected_method": selected["method"],
        "doc_count": first_pass["doc_count"],
        "linked_titles_requested": len(link_requests),
        "linked_docs_collected_total": linked_doc_count,
        "uses_embedding_api": False,
        "uses_llm_api": False,
        "retrieval_uses_gold_support": False,
        "gold_support_used_for_evaluation_only": True,
        "selection_protocol": "select link-bridge reranking variant by valid top-5/top-10/top-20 sufficiency; report test only",
        "elapsed_seconds": round(float(elapsed_seconds), 2),
    }
    path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")


def _iter_wiki_docs(archive_path: Path, max_inner_files: int, max_docs: int) -> Iterable[dict[str, Any]]:
    yielded = 0
    inner_files = 0
    with tarfile.open(archive_path, mode="r:bz2") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".bz2"):
                continue
            inner_files += 1
            fileobj = tar.extractfile(member)
            if fileobj is None:
                continue
            with bz2.open(fileobj, mode="rt", encoding="utf-8", errors="replace") as bz:
                for line in bz:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    title = html.unescape(str(row.get("title", ""))).strip()
                    text = " ".join(str(sentence) for sentence in row.get("text", []))
                    normalized_title = _normalize_title(title)
                    yield {
                        "doc_id": f"hotpotqa_intro::{row.get('id', normalized_title)}",
                        "title": title,
                        "normalized_title": normalized_title,
                        "text": text,
                        "links": _extract_links(row),
                        "source": "hotpotqa_official_intro_paragraphs",
                        "source_archive_member": member.name,
                        "url": row.get("url", ""),
                    }
                    yielded += 1
                    if max_docs > 0 and yielded >= max_docs:
                        return
            if max_inner_files > 0 and inner_files >= max_inner_files:
                return


def _extract_links(row: dict[str, Any]) -> list[str]:
    links = []
    seen = set()
    for sentence in row.get("text_with_links", []):
        for match in LINK_RE.finditer(str(sentence)):
            target = match.group(1).split("#", 1)[0]
            target = unquote(target).replace("_", " ")
            title = _normalize_title(target)
            if title and title not in seen:
                seen.add(title)
                links.append(title)
    return links


def _push_doc(heap: list[tuple[float, int, dict[str, Any]]], max_k: int, score: float, doc_seq: int, doc: dict[str, Any]) -> None:
    item = (float(score), -doc_seq, doc)
    if len(heap) < max_k:
        heapq.heappush(heap, item)
        return
    if item > heap[0]:
        heapq.heapreplace(heap, item)


def _ranked_docs_from_heaps(heaps: list[list[tuple[float, int, dict[str, Any]]]]) -> list[list[dict[str, Any]]]:
    ranked = []
    for heap in heaps:
        docs = []
        for rank, (score, neg_doc_seq, doc) in enumerate(sorted(heap, key=lambda item: (-item[0], -item[1])), start=1):
            docs.append(
                {
                    **doc,
                    "rank": rank,
                    "bm25_score": round(float(score), 6),
                    "final_score": round(float(score), 6),
                    "doc_seq": -neg_doc_seq,
                    "retrieval_path": "firsthop_bm25",
                }
            )
        ranked.append(docs)
    return ranked


def _upsert_candidate(candidates: dict[str, dict[str, Any]], title: str, candidate: dict[str, Any]) -> None:
    if not title:
        return
    existing = candidates.get(title)
    if existing is None or float(candidate.get("final_score", 0.0)) > float(existing.get("final_score", 0.0)):
        candidates[title] = candidate


def _link_source_key(info: dict[str, Any]) -> tuple[float, int, int]:
    return (float(info.get("source_score", 0.0)), -int(info.get("source_rank", 10**6)), -int(info.get("link_position", 10**6)))


def _title_overlap(question: str, title: str) -> float:
    query_tokens = set(tokenize(question))
    title_tokens = set(tokenize(title))
    if not query_tokens:
        return 0.0
    return float(len(query_tokens & title_tokens) / len(query_tokens))


def _variant_name(alpha: float, penalty: float) -> str:
    return f"link_bridge_a{alpha:.2f}_p{penalty:.2f}".replace(".", "p")


def _parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    _require(bool(values), "Expected a non-empty float list.")
    return values


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
