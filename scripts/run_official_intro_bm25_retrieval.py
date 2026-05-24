from __future__ import annotations

import argparse
import bz2
import csv
import heapq
import html
import json
import math
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from csrrag.utils.io import read_jsonl, write_jsonl
from csrrag.utils.text import lexical_score, tokenize


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "with",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a no-API BM25 retrieval baseline over the official HotpotQA intro-paragraph "
            "Wikipedia corpus. Gold support titles are used only for evaluation labels."
        )
    )
    parser.add_argument("--raw-hotpot", default="data/raw/hotpotqa/hotpot_dev_fullwiki_v1.json")
    parser.add_argument(
        "--wiki-archive",
        default="data/external/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2",
    )
    parser.add_argument("--split-dir", default="data/processed/hotpotqa_global_hardneg_splits_full_dev")
    parser.add_argument("--record-kind-filter", default="natural_global_top5")
    parser.add_argument("--output-dir", default="results/tables/hotpotqa_official_intro_bm25_retrieval")
    parser.add_argument("--output-split-dir", default="data/processed/hotpotqa_official_intro_bm25_splits_full_dev")
    parser.add_argument("--top-ks", default="5,10,20,50")
    parser.add_argument("--write-record-top-ks", default="5")
    parser.add_argument("--title-weight", type=int, default=3)
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--min-query-token-len", type=int, default=2)
    parser.add_argument("--max-queries-per-token", type=int, default=300)
    parser.add_argument("--max-questions-per-split", type=int, default=0, help="Debug only. 0 means all questions.")
    parser.add_argument("--max-inner-files", type=int, default=0, help="Debug only. 0 means all wiki files.")
    parser.add_argument("--max-docs", type=int, default=0, help="Debug only. 0 means all docs.")
    parser.add_argument("--progress-every-docs", type=int, default=250000)
    args = parser.parse_args()

    started = time.time()
    raw_rows = _load_raw_rows(Path(args.raw_hotpot))
    split_ids = _load_split_ids(Path(args.split_dir), args.record_kind_filter, args.max_questions_per_split)
    questions = _question_records(raw_rows, split_ids)
    top_ks = _parse_int_list(args.top_ks)
    write_record_top_ks = set(_parse_int_list(args.write_record_top_ks))
    _require(write_record_top_ks.issubset(set(top_ks)), "write-record-top-ks must be a subset of top-ks.")
    max_k = max(top_ks)

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
    heaps = _second_pass_rank(
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
    ranked_docs = _ranked_docs_from_heaps(heaps)
    split_records = _build_split_records(raw_rows, questions, ranked_docs, top_ks, write_record_top_ks)
    topk_rows = _topk_curve_rows(questions, ranked_docs, top_ks)
    detail_rows = _question_detail_rows(questions, ranked_docs, top_ks)
    missing_rows = _missing_support_examples(detail_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "topk_sufficiency_curve.csv", topk_rows)
    _write_csv(output_dir / "question_details.csv", detail_rows)
    _write_csv(output_dir / "missing_support_examples.csv", missing_rows)
    _write_summary(output_dir / "bm25_retrieval_summary.md", topk_rows, first_pass, args)
    _write_validation(
        output_dir / "validation_summary.json",
        args=args,
        split_ids=split_ids,
        questions=questions,
        top_ks=top_ks,
        write_record_top_ks=write_record_top_ks,
        query_terms=query_terms,
        token_to_query_ids=token_to_query_ids,
        first_pass=first_pass,
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
                "questions": len(questions),
                "doc_count": first_pass["doc_count"],
                "top_ks": top_ks,
                "write_record_top_ks": sorted(write_record_top_ks),
                "uses_embedding_api": False,
                "uses_llm_api": False,
            },
            ensure_ascii=False,
        )
    )


def _load_raw_rows(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return {row["_id"]: row for row in json.load(f)}


def _load_split_ids(split_dir: Path, record_kind_filter: str, max_questions_per_split: int) -> dict[str, list[str]]:
    split_ids: dict[str, list[str]] = {}
    for split in ("train", "valid", "test"):
        ids = []
        seen = set()
        for record in read_jsonl(split_dir / f"{split}.jsonl"):
            if record_kind_filter and record.get("metadata", {}).get("record_kind") != record_kind_filter:
                continue
            original_id = record["metadata"]["original_id"]
            if original_id in seen:
                continue
            seen.add(original_id)
            ids.append(original_id)
            if max_questions_per_split > 0 and len(ids) >= max_questions_per_split:
                break
        _require(ids, f"No ids loaded for split={split}.")
        split_ids[split] = ids
    _validate_split_ids(split_ids)
    return split_ids


def _question_records(raw_rows: dict[str, dict[str, Any]], split_ids: dict[str, list[str]]) -> list[dict[str, Any]]:
    questions = []
    qid = 0
    for split in ("train", "valid", "test"):
        for original_id in split_ids[split]:
            raw = raw_rows[original_id]
            questions.append(
                {
                    "qid": qid,
                    "split": split,
                    "original_id": original_id,
                    "question": raw["question"],
                    "gold_answer": raw["answer"],
                    "question_type": raw.get("type", "unknown"),
                    "difficulty": raw.get("level", "unknown"),
                    "support_titles": sorted(_support_titles(raw)),
                }
            )
            qid += 1
    return questions


def _build_query_terms(
    questions: list[dict[str, Any]],
    min_len: int,
    max_queries_per_token: int,
) -> tuple[list[set[str]], dict[str, list[int]]]:
    raw_terms = [_query_tokens(question["question"], min_len) for question in questions]
    token_to_ids: dict[str, list[int]] = defaultdict(list)
    for qid, terms in enumerate(raw_terms):
        for term in sorted(terms):
            token_to_ids[term].append(qid)
    kept_tokens = {
        token
        for token, qids in token_to_ids.items()
        if len(qids) <= max_queries_per_token
    }
    query_terms = []
    for terms in raw_terms:
        kept = terms & kept_tokens
        query_terms.append(kept if kept else terms)
    filtered_token_to_ids: dict[str, list[int]] = defaultdict(list)
    for qid, terms in enumerate(query_terms):
        for term in sorted(terms):
            filtered_token_to_ids[term].append(qid)
    return query_terms, dict(filtered_token_to_ids)


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
        terms = set(tokens) & query_vocab
        for term in terms:
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


def _second_pass_rank(
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
) -> list[list[tuple[float, int, dict[str, Any]]]]:
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
            if score <= 0.0:
                continue
            _push_doc(heaps[qid], max_k, score, doc_count, doc)
        if progress_every_docs > 0 and doc_count % progress_every_docs == 0:
            covered = sum(1 for heap in heaps if heap)
            print(
                json.dumps(
                    {
                        "pass": "rank",
                        "docs": doc_count,
                        "matched_docs": matched_docs,
                        "queries_with_hits": covered,
                        "elapsed_seconds": round(time.time() - started, 1),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return heaps


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
                        "source": "hotpotqa_official_intro_paragraphs",
                        "source_archive_member": member.name,
                        "url": row.get("url", ""),
                    }
                    yielded += 1
                    if max_docs > 0 and yielded >= max_docs:
                        return
            if max_inner_files > 0 and inner_files >= max_inner_files:
                return


def _doc_tokens(doc: dict[str, Any], title_weight: int) -> list[str]:
    title_tokens = _filter_tokens(tokenize(str(doc.get("title", ""))), min_len=2)
    text_tokens = _filter_tokens(tokenize(str(doc.get("text", ""))), min_len=2)
    return title_tokens * max(1, title_weight) + text_tokens


def _query_tokens(question: str, min_len: int) -> set[str]:
    return set(_filter_tokens(tokenize(question), min_len=min_len))


def _filter_tokens(tokens: list[str], min_len: int) -> list[str]:
    return [token for token in tokens if len(token) >= min_len and token not in STOPWORDS]


def _bm25_idf(doc_count: int, df: dict[str, int]) -> dict[str, float]:
    return {
        term: math.log(1.0 + (doc_count - freq + 0.5) / (freq + 0.5))
        for term, freq in df.items()
    }


def _bm25_term_score(tf: int, doc_len: int, avgdl: float, idf: float, k1: float, b: float) -> float:
    if tf <= 0 or doc_len <= 0 or avgdl <= 0.0:
        return 0.0
    denom = tf + k1 * (1.0 - b + b * doc_len / avgdl)
    return float(idf * (tf * (k1 + 1.0)) / denom)


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
                    "doc_seq": -neg_doc_seq,
                }
            )
        ranked.append(docs)
    return ranked


def _build_split_records(
    raw_rows: dict[str, dict[str, Any]],
    questions: list[dict[str, Any]],
    ranked_docs: list[list[dict[str, Any]]],
    top_ks: list[int],
    write_record_top_ks: set[int],
) -> dict[str, list[dict[str, Any]]]:
    split_records = {"train": [], "valid": [], "test": []}
    for question in questions:
        raw = raw_rows[question["original_id"]]
        docs = ranked_docs[question["qid"]]
        for top_k in top_ks:
            if top_k not in write_record_top_ks:
                continue
            split_records[question["split"]].append(_make_record(raw, question, docs[:top_k], top_k))
    return split_records


def _make_record(raw: dict[str, Any], question: dict[str, Any], docs: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    retrieved_docs = _retrieved_docs_for_record(question["question"], docs)
    support_titles = set(question["support_titles"])
    retrieved_titles = {_normalize_title(doc["title"]) for doc in retrieved_docs}
    support_present = support_titles.issubset(retrieved_titles)
    missing_support_titles = sorted(support_titles - retrieved_titles)
    record_kind = f"official_intro_bm25_top{top_k}"
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
            "retriever": "bm25-official-intro-corpus",
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
        bm25_score = float(doc.get("bm25_score", 0.0))
        retrieved.append(
            {
                "doc_id": doc["doc_id"],
                "rank": rank,
                "score": round(float(lexical_score(query, doc["title"], doc["text"])), 6),
                "embedding_score": round(bm25_score, 6),
                "sparse_score": round(bm25_score, 6),
                "title": doc["title"],
                "text": doc["text"],
                "source": "hotpotqa_official_intro_bm25",
            }
        )
    return retrieved


def _topk_curve_rows(questions: list[dict[str, Any]], ranked_docs: list[list[dict[str, Any]]], top_ks: list[int]) -> list[dict[str, Any]]:
    rows = []
    for split in ("train", "valid", "test", "all"):
        split_questions = [q for q in questions if split == "all" or q["split"] == split]
        for top_k in top_ks:
            detail = [_detail_for_question(q, ranked_docs[q["qid"]], top_k) for q in split_questions]
            sufficient = [row for row in detail if row["support_present_in_topk"]]
            rows.append(
                {
                    "split": split,
                    "top_k": top_k,
                    "n": len(detail),
                    "sufficient_count": len(sufficient),
                    "insufficient_count": len(detail) - len(sufficient),
                    "sufficient_rate": len(sufficient) / len(detail) if detail else 0.0,
                    "answer_present_in_returned_docs_rate": _mean(float(row["answer_present_in_returned_docs"]) for row in detail),
                    "mean_support_title_coverage": _mean(float(row["support_title_coverage"]) for row in detail),
                    "questions_with_no_hits": sum(int(row["retrieved_count"] == 0) for row in detail),
                }
            )
    return rows


def _question_detail_rows(questions: list[dict[str, Any]], ranked_docs: list[list[dict[str, Any]]], top_ks: list[int]) -> list[dict[str, Any]]:
    rows = []
    for question in questions:
        for top_k in top_ks:
            rows.append(_detail_for_question(question, ranked_docs[question["qid"]], top_k))
    return rows


def _detail_for_question(question: dict[str, Any], docs: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    selected = docs[:top_k]
    support_titles = set(question["support_titles"])
    title_to_rank = {_normalize_title(doc["title"]): int(doc["rank"]) for doc in selected}
    retrieved_support = sorted(support_titles & set(title_to_rank))
    missing_support = sorted(support_titles - set(title_to_rank))
    ranks = [title_to_rank[title] for title in retrieved_support]
    support_present = not missing_support
    return {
        "split": question["split"],
        "top_k": top_k,
        "original_id": question["original_id"],
        "question_type": question["question_type"],
        "difficulty": question["difficulty"],
        "question": question["question"],
        "gold_answer": question["gold_answer"],
        "retrieved_count": len(selected),
        "support_title_count": len(support_titles),
        "retrieved_support_title_count": len(retrieved_support),
        "support_title_coverage": len(retrieved_support) / len(support_titles) if support_titles else 0.0,
        "support_present_in_topk": int(support_present),
        "missing_support_titles": " || ".join(missing_support),
        "retrieved_support_ranks": " || ".join(str(rank) for rank in sorted(ranks)),
        "max_retrieved_support_rank": max(ranks) if ranks else "",
        "answer_present_in_returned_docs": int(_answer_present(question["gold_answer"], selected)),
        "top_titles": " || ".join(doc["title"] for doc in selected[:5]),
    }


def _missing_support_examples(detail_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        row for row in detail_rows
        if row["split"] == "test" and int(row["support_present_in_topk"]) == 0
    ]
    return sorted(
        rows,
        key=lambda row: (
            int(row["top_k"]),
            -float(row["support_title_coverage"]),
            str(row["missing_support_titles"]),
        ),
    )[:200]


def _write_summary(path: Path, topk_rows: list[dict[str, Any]], first_pass: dict[str, Any], args: argparse.Namespace) -> None:
    test_rows = [row for row in topk_rows if row["split"] == "test"]
    curve = "\n".join(
        f"- top-{row['top_k']}: sufficient_rate={float(row['sufficient_rate']):.4f}, "
        f"answer_present={float(row['answer_present_in_returned_docs_rate']):.4f}, "
        f"mean_support_coverage={float(row['mean_support_title_coverage']):.4f}"
        for row in test_rows
    )
    text = f"""# Official HotpotQA Intro BM25 Retrieval Baseline

## Purpose

This no-API run evaluates a natural sparse retrieval baseline over the official HotpotQA introductory-paragraph Wikipedia corpus. The retrieval step does not use gold support titles; support titles are used only for sufficiency evaluation.

## Corpus

- Wiki archive: `{args.wiki_archive}`
- Documents scanned: `{first_pass['doc_count']}`
- Average filtered document length: `{float(first_pass['avgdl']):.2f}`
- Inner-file cap: `{args.max_inner_files}`
- Document cap: `{args.max_docs}`

## Test Sufficiency Curve

{curve}

## Interpretation

Treat this as the sparse natural baseline for the support-complete official intro corpus. If BM25 top-k remains weak, the next retrieval work should focus on query decomposition, bridge-aware reranking, or dense retrieval over this same corpus.
"""
    path.write_text(text, encoding="utf-8")


def _write_validation(
    path: Path,
    args: argparse.Namespace,
    split_ids: dict[str, list[str]],
    questions: list[dict[str, Any]],
    top_ks: list[int],
    write_record_top_ks: set[int],
    query_terms: list[set[str]],
    token_to_query_ids: dict[str, list[int]],
    first_pass: dict[str, Any],
    elapsed_seconds: float,
) -> None:
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
        "title_weight": args.title_weight,
        "k1": args.k1,
        "b": args.b,
        "query_vocab_size": len(token_to_query_ids),
        "mean_query_terms": _mean(len(terms) for terms in query_terms),
        "max_queries_per_token": args.max_queries_per_token,
        "doc_count": first_pass["doc_count"],
        "avgdl": first_pass["avgdl"],
        "max_inner_files": args.max_inner_files,
        "max_docs": args.max_docs,
        "uses_embedding_api": False,
        "uses_llm_api": False,
        "retrieval_uses_gold_support": False,
        "gold_support_used_for_evaluation_only": True,
        "elapsed_seconds": round(float(elapsed_seconds), 2),
    }
    path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_int_list(value: str) -> list[int]:
    values = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    _require(values, "Expected a non-empty integer list.")
    _require(all(item > 0 for item in values), "All integer list values must be positive.")
    return values


def _support_titles(raw: dict[str, Any]) -> set[str]:
    return {_normalize_title(title) for title, _sent_idx in raw["supporting_facts"]}


def _normalize_title(title: str) -> str:
    return html.unescape(str(title)).strip().lower()


def _answer_present(answer: str, docs: list[dict[str, Any]]) -> bool:
    answer_tokens = " ".join(tokenize(answer))
    if not answer_tokens:
        return False
    context = " ".join(" ".join(tokenize(f"{doc.get('title', '')} {doc.get('text', '')}")) for doc in docs)
    return answer_tokens in context


def _validate_split_ids(split_ids: dict[str, list[str]]) -> None:
    sets = {split: set(ids) for split, ids in split_ids.items()}
    _require(sets["train"].isdisjoint(sets["valid"]), "train and valid original_id overlap.")
    _require(sets["train"].isdisjoint(sets["test"]), "train and test original_id overlap.")
    _require(sets["valid"].isdisjoint(sets["test"]), "valid and test original_id overlap.")


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


def _mean(values) -> float:
    value_list = list(values)
    return float(sum(value_list) / len(value_list)) if value_list else 0.0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
