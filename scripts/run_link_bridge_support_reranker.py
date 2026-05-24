from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from csrrag.utils.io import read_jsonl, write_jsonl
from csrrag.utils.text import lexical_score, tokenize


TOP_KS = [5, 10, 20]
DEFAULT_BLEND_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
FEATURE_SETS = {
    "rank_score": [
        "rank",
        "reciprocal_rank",
        "score",
        "score_margin_to_top",
        "score_ratio_to_top",
        "sparse_score",
        "bridge_score",
        "is_firsthop",
        "is_linked",
    ],
    "query_doc": [
        "rank",
        "reciprocal_rank",
        "score",
        "title_query_overlap",
        "text_query_overlap",
        "lexical_score",
        "title_in_question",
        "question_in_title",
        "text_len",
        "title_len",
        "is_firsthop",
        "is_linked",
    ],
    "bridge_context": [
        "rank",
        "reciprocal_rank",
        "score",
        "is_firsthop",
        "is_linked",
        "title_query_overlap",
        "text_query_overlap",
        "title_mentioned_by_other_docs",
        "title_token_mentioned_by_other_docs",
        "mentions_other_candidate_titles",
        "mentions_query_overlap_titles",
        "candidate_title_overlap_max",
        "candidate_title_overlap_mean",
    ],
}
FEATURE_SETS["all"] = sorted({name for names in FEATURE_SETS.values() for name in names})


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a no-API candidate-level support reranker over link-bridge top-20 records. "
            "Support titles are used only as train/valid supervision and test labels."
        )
    )
    parser.add_argument("--split-dir", default="data/processed/hotpotqa_official_intro_link_bridge_splits_top20_full_dev")
    parser.add_argument("--record-kind-filter", default="official_intro_link_bridge_a0p85_p0p00_top20")
    parser.add_argument("--output-dir", default="results/tables/hotpotqa_official_intro_link_bridge_support_reranker")
    parser.add_argument("--output-split-dir", default="data/processed/hotpotqa_official_intro_link_bridge_support_reranker_splits")
    parser.add_argument("--write-record-top-ks", default="5,10,20")
    parser.add_argument("--blend-alphas", default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_records = _load_records(Path(args.split_dir), args.record_kind_filter)
    _validate_records(split_records)
    write_top_ks = _parse_ints(args.write_record_top_ks)
    blend_alphas = _parse_floats(args.blend_alphas)
    _require(set(write_top_ks).issubset(set(TOP_KS)), "write-record-top-ks must be a subset of 5,10,20.")

    candidate_rows = {
        split: _candidate_rows(records)
        for split, records in split_records.items()
    }
    variants, predictions = _train_and_score(candidate_rows, blend_alphas, args.seed)
    selected = _select_variant(variants)
    selected_records = _reranked_split_records(split_records, predictions[selected["variant_id"]], write_top_ks, selected)
    detail_rows = _question_detail_rows(split_records, predictions, variants, selected)
    topk_rows = _topk_rows(split_records, predictions, variants)
    doc_metric_rows = _doc_metric_rows(candidate_rows, predictions, variants)
    feature_rows = _feature_rows(variants)
    feature_ablation_rows = _ablation_rows(variants, topk_rows, "feature_set")
    estimator_ablation_rows = _ablation_rows(variants, topk_rows, "estimator")
    blend_ablation_rows = _ablation_rows(variants, topk_rows, "blend_alpha")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "topk_sufficiency_curve.csv", topk_rows)
    _write_csv(output_dir / "doc_prediction_metrics.csv", doc_metric_rows)
    _write_csv(output_dir / "variant_selection.csv", _variant_rows(variants, selected))
    _write_csv(output_dir / "feature_ablation.csv", feature_ablation_rows)
    _write_csv(output_dir / "estimator_ablation.csv", estimator_ablation_rows)
    _write_csv(output_dir / "blend_ablation.csv", blend_ablation_rows)
    _write_csv(output_dir / "selected_question_details.csv", detail_rows)
    _write_csv(output_dir / "feature_importance.csv", feature_rows)
    _write_summary(output_dir / "support_reranker_summary.md", topk_rows, selected)
    _write_ablation_summary(output_dir / "support_reranker_ablation_summary.md", feature_ablation_rows, estimator_ablation_rows, blend_ablation_rows)
    _write_validation(output_dir / "validation_summary.json", args, split_records, variants, selected, blend_alphas)

    output_split_dir = Path(args.output_split_dir)
    output_split_dir.mkdir(parents=True, exist_ok=True)
    for split, records in selected_records.items():
        write_jsonl(output_split_dir / f"{split}.jsonl", records)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "output_split_dir": str(output_split_dir),
                "selected_variant": selected["variant_id"],
                "selected_valid_top5_sufficient_rate": selected["valid_top5_sufficient_rate"],
                "test_top5_sufficient_rate": selected["test_top5_sufficient_rate"],
                "uses_embedding_api": False,
                "uses_llm_api": False,
            },
            ensure_ascii=False,
        )
    )


def _load_records(split_dir: Path, record_kind: str) -> dict[str, list[dict[str, Any]]]:
    records = {}
    for split in ("train", "valid", "test"):
        rows = [
            record
            for record in read_jsonl(split_dir / f"{split}.jsonl")
            if record.get("metadata", {}).get("record_kind") == record_kind
        ]
        records[split] = rows
    return records


def _candidate_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        support_titles = set(_normalize_title(title) for title in record["metadata"]["support_titles"])
        docs = record["retrieved_docs"]
        top_score = max((float(doc.get("score", 0.0)) for doc in docs), default=0.0)
        max_score = max(abs(top_score), 1e-9)
        all_title_tokens = [_token_set(doc.get("title", "")) for doc in docs]
        all_texts = [_normalize_text(doc.get("text", "")) for doc in docs]
        query_title_overlap_flags = [
            _overlap_ratio(tokenize(record["query"]), tokenize(doc.get("title", ""))) > 0
            for doc in docs
        ]
        for idx, doc in enumerate(docs):
            title = str(doc.get("title", ""))
            text = str(doc.get("text", ""))
            title_norm = _normalize_title(title)
            title_tokens = all_title_tokens[idx]
            other_title_tokens = [tokens for j, tokens in enumerate(all_title_tokens) if j != idx]
            other_texts = [other_text for j, other_text in enumerate(all_texts) if j != idx]
            source = str(doc.get("source", ""))
            score = float(doc.get("score", 0.0))
            sparse_score = float(doc.get("sparse_score", 0.0))
            bridge_score = float(doc.get("bridge_score", doc.get("embedding_score", score)))
            rank = int(doc.get("rank", idx + 1))
            features = {
                "rank": float(rank),
                "reciprocal_rank": 1.0 / max(rank, 1),
                "score": score,
                "score_margin_to_top": top_score - score,
                "score_ratio_to_top": score / max_score,
                "sparse_score": sparse_score,
                "bridge_score": bridge_score,
                "is_firsthop": float(source.endswith("firsthop_bm25")),
                "is_linked": float(source.endswith("linked_from_firsthop")),
                "title_query_overlap": _overlap_ratio(tokenize(record["query"]), tokenize(title)),
                "text_query_overlap": _overlap_ratio(tokenize(record["query"]), tokenize(text)),
                "lexical_score": float(lexical_score(record["query"], title, text)),
                "title_in_question": float(title_norm and title_norm in _normalize_text(record["query"])),
                "question_in_title": float(_normalize_text(record["query"]) and _normalize_text(record["query"]) in title_norm),
                "text_len": float(len(tokenize(text))),
                "title_len": float(len(tokenize(title))),
                "title_mentioned_by_other_docs": float(any(title_norm and title_norm in other_text for other_text in other_texts)),
                "title_token_mentioned_by_other_docs": _title_token_mention_rate(title_tokens, other_texts),
                "mentions_other_candidate_titles": _mentions_other_titles(all_texts[idx], docs, idx),
                "mentions_query_overlap_titles": _mentions_query_overlap_titles(all_texts[idx], docs, query_title_overlap_flags, idx),
                "candidate_title_overlap_max": _candidate_title_overlap(title_tokens, other_title_tokens, reducer="max"),
                "candidate_title_overlap_mean": _candidate_title_overlap(title_tokens, other_title_tokens, reducer="mean"),
            }
            rows.append(
                {
                    "original_id": record["metadata"]["original_id"],
                    "doc_id": doc["doc_id"],
                    "title": title,
                    "support_label": int(title_norm in support_titles),
                    "support_title_count": len(support_titles),
                    "features": features,
                }
            )
    return rows


def _train_and_score(
    candidate_rows: dict[str, list[dict[str, Any]]],
    blend_alphas: list[float],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, list[float]]]]]:
    variants = []
    predictions: dict[str, dict[str, dict[str, list[float]]]] = {}
    for feature_set, feature_names in FEATURE_SETS.items():
        x_train = _matrix(candidate_rows["train"], feature_names)
        y_train = _labels(candidate_rows["train"])
        models = _fit_models(x_train, y_train, seed)
        for estimator, model in models.items():
            support_scores = {
                split: np.asarray(model.predict_proba(_matrix(rows, feature_names))[:, 1], dtype=float)
                for split, rows in candidate_rows.items()
            }
            original_scores = {
                split: _normalized_original_scores(rows)
                for split, rows in candidate_rows.items()
            }
            for alpha in blend_alphas:
                variant_id = f"{estimator}/{feature_set}/blend{alpha:.2f}"
                blended = {
                    split: (alpha * support_scores[split] + (1.0 - alpha) * original_scores[split]).tolist()
                    for split in candidate_rows
                }
                predictions[variant_id] = _group_scores(candidate_rows, blended)
                metrics = _variant_metrics(candidate_rows, predictions[variant_id])
                variants.append(
                    {
                        "variant_id": variant_id,
                        "estimator": estimator,
                        "feature_set": feature_set,
                        "blend_alpha": alpha,
                        "feature_names": feature_names,
                        "model": model,
                        **metrics,
                    }
                )
    return variants, predictions


def _fit_models(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> dict[str, Any]:
    return {
        "logistic_regression_balanced": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=seed,
                        class_weight="balanced",
                    ),
                ),
            ]
        ).fit(x_train, y_train),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ).fit(x_train, y_train),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.04,
            max_depth=2,
            random_state=seed,
        ).fit(x_train, y_train),
    }


def _variant_metrics(
    candidate_rows: dict[str, list[dict[str, Any]]],
    prediction: dict[str, dict[str, list[float]]],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for split in ("train", "valid", "test"):
        labels_by_id = _group_labels(candidate_rows[split])
        support_counts_by_id = _group_support_counts(candidate_rows[split])
        for top_k in TOP_KS:
            labels = []
            for original_id, scores in prediction[split].items():
                labels.append(
                    _support_present_after_rerank(
                        labels_by_id[original_id],
                        support_counts_by_id[original_id],
                        scores,
                        top_k,
                    )
                )
            metrics[f"{split}_top{top_k}_sufficient_rate"] = float(np.mean(labels)) if labels else 0.0
    return metrics


def _select_variant(variants: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        variants,
        key=lambda row: (
            float(row["valid_top5_sufficient_rate"]),
            float(row["valid_top10_sufficient_rate"]),
            float(row["valid_top20_sufficient_rate"]),
            -float(row["blend_alpha"]),
            row["variant_id"],
        ),
    )


def _reranked_split_records(
    split_records: dict[str, list[dict[str, Any]]],
    prediction: dict[str, dict[str, list[float]]],
    write_top_ks: list[int],
    selected: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    output = {"train": [], "valid": [], "test": []}
    for split, records in split_records.items():
        for record in records:
            original_id = record["metadata"]["original_id"]
            scores = prediction[split][original_id]
            ranked_docs = _rank_docs(record["retrieved_docs"], scores)
            for top_k in write_top_ks:
                output[split].append(_make_record(record, ranked_docs[:top_k], top_k, selected))
    return output


def _make_record(record: dict[str, Any], docs: list[dict[str, Any]], top_k: int, selected: dict[str, Any]) -> dict[str, Any]:
    support_titles = set(_normalize_title(title) for title in record["metadata"]["support_titles"])
    retrieved_titles = {_normalize_title(doc["title"]) for doc in docs}
    support_present = support_titles.issubset(retrieved_titles)
    method = _record_method_name(selected)
    record_kind = f"official_intro_{method}_top{top_k}"
    retrieved_docs = []
    for rank, doc in enumerate(docs, start=1):
        doc_copy = dict(doc)
        doc_copy["rank"] = rank
        retrieved_docs.append(doc_copy)
    return {
        "id": f"{record['metadata']['original_id']}__{record_kind}",
        "query": record["query"],
        "gold_answer": record["gold_answer"],
        "sufficiency_label": "sufficient" if support_present else "insufficient",
        "retrieved_docs": retrieved_docs,
        "metadata": {
            **record["metadata"],
            "retriever": f"official-intro-{method}",
            "record_kind": record_kind,
            "top_k": top_k,
            "support_present_in_topk": support_present,
            "missing_support_titles": sorted(support_titles - retrieved_titles),
            "reranker": selected["variant_id"],
            "reranker_selected_on": "valid_top5_sufficient_rate",
            "base_record_kind": record["metadata"]["record_kind"],
        },
    }


def _question_detail_rows(
    split_records: dict[str, list[dict[str, Any]]],
    predictions: dict[str, dict[str, dict[str, list[float]]]],
    variants: list[dict[str, Any]],
    selected: dict[str, Any],
) -> list[dict[str, Any]]:
    variant_ids = ["original_rank", selected["variant_id"]]
    variant_lookup = {row["variant_id"]: row for row in variants}
    rows = []
    for split, records in split_records.items():
        for record in records:
            original_id = record["metadata"]["original_id"]
            support_titles = set(_normalize_title(title) for title in record["metadata"]["support_titles"])
            original_scores = [1.0 / max(int(doc.get("rank", idx + 1)), 1) for idx, doc in enumerate(record["retrieved_docs"])]
            score_map = {
                "original_rank": original_scores,
                selected["variant_id"]: predictions[selected["variant_id"]][split][original_id],
            }
            for variant_id in variant_ids:
                ranked_docs = _rank_docs(record["retrieved_docs"], score_map[variant_id])
                row = {
                    "split": split,
                    "original_id": original_id,
                    "variant_id": variant_id,
                    "question": record["query"],
                    "gold_answer": record["gold_answer"],
                    "support_titles": " || ".join(record["metadata"]["support_titles"]),
                    "top5_titles": " || ".join(doc["title"] for doc in ranked_docs[:5]),
                    "top10_titles": " || ".join(doc["title"] for doc in ranked_docs[:10]),
                    "top5_sufficient": int(support_titles.issubset({_normalize_title(doc["title"]) for doc in ranked_docs[:5]})),
                    "top10_sufficient": int(support_titles.issubset({_normalize_title(doc["title"]) for doc in ranked_docs[:10]})),
                    "top20_sufficient": int(support_titles.issubset({_normalize_title(doc["title"]) for doc in ranked_docs[:20]})),
                }
                if variant_id in variant_lookup:
                    row["feature_set"] = variant_lookup[variant_id]["feature_set"]
                    row["estimator"] = variant_lookup[variant_id]["estimator"]
                    row["blend_alpha"] = variant_lookup[variant_id]["blend_alpha"]
                else:
                    row["feature_set"] = "none"
                    row["estimator"] = "original"
                    row["blend_alpha"] = ""
                rows.append(row)
    return rows


def _topk_rows(
    split_records: dict[str, list[dict[str, Any]]],
    predictions: dict[str, dict[str, dict[str, list[float]]]],
    variants: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for split, records in split_records.items():
        rows.extend(_original_topk_rows(split, records))
    for variant in variants:
        variant_id = variant["variant_id"]
        for split, records in split_records.items():
            labels_by_k = {top_k: [] for top_k in TOP_KS}
            support_coverage_by_k = {top_k: [] for top_k in TOP_KS}
            for record in records:
                original_id = record["metadata"]["original_id"]
                ranked_docs = _rank_docs(record["retrieved_docs"], predictions[variant_id][split][original_id])
                support_titles = set(_normalize_title(title) for title in record["metadata"]["support_titles"])
                for top_k in TOP_KS:
                    titles = {_normalize_title(doc["title"]) for doc in ranked_docs[:top_k]}
                    labels_by_k[top_k].append(int(support_titles.issubset(titles)))
                    support_coverage_by_k[top_k].append(_support_coverage(support_titles, titles))
            for top_k in TOP_KS:
                rows.append(
                    {
                        "variant_id": variant_id,
                        "estimator": variant["estimator"],
                        "feature_set": variant["feature_set"],
                        "blend_alpha": variant["blend_alpha"],
                        "split": split,
                        "top_k": top_k,
                        "n": len(records),
                        "sufficient_count": int(sum(labels_by_k[top_k])),
                        "sufficient_rate": float(np.mean(labels_by_k[top_k])) if records else 0.0,
                        "mean_support_title_coverage": float(np.mean(support_coverage_by_k[top_k])) if records else 0.0,
                    }
                )
    return rows


def _original_topk_rows(split: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for top_k in TOP_KS:
        sufficient = []
        coverage = []
        for record in records:
            support_titles = set(_normalize_title(title) for title in record["metadata"]["support_titles"])
            titles = {_normalize_title(doc["title"]) for doc in record["retrieved_docs"][:top_k]}
            sufficient.append(int(support_titles.issubset(titles)))
            coverage.append(_support_coverage(support_titles, titles))
        rows.append(
            {
                "variant_id": "original_rank",
                "estimator": "none",
                "feature_set": "none",
                "blend_alpha": "",
                "split": split,
                "top_k": top_k,
                "n": len(records),
                "sufficient_count": int(sum(sufficient)),
                "sufficient_rate": float(np.mean(sufficient)) if records else 0.0,
                "mean_support_title_coverage": float(np.mean(coverage)) if records else 0.0,
            }
        )
    return rows


def _doc_metric_rows(
    candidate_rows: dict[str, list[dict[str, Any]]],
    predictions: dict[str, dict[str, dict[str, list[float]]]],
    variants: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for variant in variants:
        variant_id = variant["variant_id"]
        for split, records in candidate_rows.items():
            y_true = _labels(records)
            y_score = np.asarray(
                [score for scores in predictions[variant_id][split].values() for score in scores],
                dtype=float,
            )
            rows.append(
                {
                    "variant_id": variant_id,
                    "estimator": variant["estimator"],
                    "feature_set": variant["feature_set"],
                    "blend_alpha": variant["blend_alpha"],
                    "split": split,
                    "doc_count": len(records),
                    "positive_doc_count": int(y_true.sum()),
                    "support_doc_rate": float(y_true.mean()) if len(y_true) else 0.0,
                    "auroc": _safe_auc(y_true, y_score),
                    "auprc": _safe_auprc(y_true, y_score),
                }
            )
    return rows


def _feature_rows(variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for variant in variants:
        key = (variant["estimator"], variant["feature_set"])
        if key in seen:
            continue
        seen.add(key)
        model = variant["model"]
        feature_names = variant["feature_names"]
        importances = _model_importances(model)
        if importances is None:
            continue
        for feature, importance in zip(feature_names, importances):
            rows.append(
                {
                    "estimator": variant["estimator"],
                    "feature_set": variant["feature_set"],
                    "feature": feature,
                    "importance": float(importance),
                }
            )
    return sorted(rows, key=lambda row: (row["estimator"], row["feature_set"], -abs(float(row["importance"]))))


def _variant_rows(variants: list[dict[str, Any]], selected: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in variants:
        rows.append(
            {
                "selected": int(row["variant_id"] == selected["variant_id"]),
                "variant_id": row["variant_id"],
                "estimator": row["estimator"],
                "feature_set": row["feature_set"],
                "blend_alpha": row["blend_alpha"],
                "train_top5_sufficient_rate": row["train_top5_sufficient_rate"],
                "valid_top5_sufficient_rate": row["valid_top5_sufficient_rate"],
                "test_top5_sufficient_rate": row["test_top5_sufficient_rate"],
                "valid_top10_sufficient_rate": row["valid_top10_sufficient_rate"],
                "test_top10_sufficient_rate": row["test_top10_sufficient_rate"],
                "valid_top20_sufficient_rate": row["valid_top20_sufficient_rate"],
                "test_top20_sufficient_rate": row["test_top20_sufficient_rate"],
            }
        )
    return rows


def _ablation_rows(variants: list[dict[str, Any]], topk_rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    original = {
        "group": group_key,
        "group_value": "original_rank",
        "selected_variant": "original_rank",
        "estimator": "none",
        "feature_set": "none",
        "blend_alpha": "",
        "train_top5_sufficient_rate": _lookup_topk(topk_rows, "original_rank", "train", 5),
        "valid_top5_sufficient_rate": _lookup_topk(topk_rows, "original_rank", "valid", 5),
        "test_top5_sufficient_rate": _lookup_topk(topk_rows, "original_rank", "test", 5),
        "test_top10_sufficient_rate": _lookup_topk(topk_rows, "original_rank", "test", 10),
        "test_top20_sufficient_rate": _lookup_topk(topk_rows, "original_rank", "test", 20),
    }
    original["test_top5_gain_vs_original"] = 0.0
    original["train_test_gap_top5"] = original["train_top5_sufficient_rate"] - original["test_top5_sufficient_rate"]
    rows = [original]
    groups = sorted({str(row[group_key]) for row in variants})
    original_top5 = float(original["test_top5_sufficient_rate"])
    for group_value in groups:
        candidates = [row for row in variants if str(row[group_key]) == group_value]
        best = _best_variant(candidates)
        output = {
            "group": group_key,
            "group_value": group_value,
            "selected_variant": best["variant_id"],
            "estimator": best["estimator"],
            "feature_set": best["feature_set"],
            "blend_alpha": best["blend_alpha"],
            "train_top5_sufficient_rate": best["train_top5_sufficient_rate"],
            "valid_top5_sufficient_rate": best["valid_top5_sufficient_rate"],
            "test_top5_sufficient_rate": best["test_top5_sufficient_rate"],
            "test_top10_sufficient_rate": best["test_top10_sufficient_rate"],
            "test_top20_sufficient_rate": best["test_top20_sufficient_rate"],
            "test_top5_gain_vs_original": float(best["test_top5_sufficient_rate"]) - original_top5,
            "train_test_gap_top5": float(best["train_top5_sufficient_rate"]) - float(best["test_top5_sufficient_rate"]),
        }
        rows.append(output)
    return sorted(rows, key=lambda row: (-float(row["valid_top5_sufficient_rate"]), -float(row["test_top5_sufficient_rate"]), str(row["group_value"])))


def _best_variant(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        candidates,
        key=lambda row: (
            float(row["valid_top5_sufficient_rate"]),
            float(row["valid_top10_sufficient_rate"]),
            float(row["valid_top20_sufficient_rate"]),
            -float(row["blend_alpha"]),
            row["variant_id"],
        ),
    )


def _lookup_topk(topk_rows: list[dict[str, Any]], variant_id: str, split: str, top_k: int) -> float:
    for row in topk_rows:
        if row["variant_id"] == variant_id and row["split"] == split and int(row["top_k"]) == top_k:
            return float(row["sufficient_rate"])
    return 0.0


def _write_summary(path: Path, topk_rows: list[dict[str, Any]], selected: dict[str, Any]) -> None:
    def lookup(variant: str, split: str, top_k: int) -> float:
        for row in topk_rows:
            if row["variant_id"] == variant and row["split"] == split and int(row["top_k"]) == top_k:
                return float(row["sufficient_rate"])
        return 0.0

    method = selected["variant_id"]
    lines = [
        "# Link-Bridge Support Reranker Summary",
        "",
        "## Purpose",
        "",
        "This no-API run trains a candidate-level support reranker over link-bridge top-20 records. Model selection uses valid top-5 sufficiency only; test is reported once.",
        "",
        "## Selected Variant",
        "",
        f"- variant: `{method}`",
        f"- valid top-5 sufficient rate: `{selected['valid_top5_sufficient_rate']:.4f}`",
        f"- test top-5 sufficient rate: `{selected['test_top5_sufficient_rate']:.4f}`",
        "",
        "## Test Sufficiency",
        "",
    ]
    for top_k in TOP_KS:
        lines.append(
            f"- top-{top_k}: original `{lookup('original_rank', 'test', top_k):.4f}` -> reranked `{lookup(method, 'test', top_k):.4f}`"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A useful reranker should move support titles already present in the top-20 candidate set into the top-5 without using test labels for model selection. This result should be followed by CSR-RAG answer/abstain evaluation on the selected top-5 records.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_ablation_summary(
    path: Path,
    feature_rows: list[dict[str, Any]],
    estimator_rows: list[dict[str, Any]],
    blend_rows: list[dict[str, Any]],
) -> None:
    def table(rows: list[dict[str, Any]]) -> list[str]:
        lines = [
            "| Group Value | Selected Variant | Valid Top-5 | Test Top-5 | Test Gain | Train-Test Gap |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| {row['group_value']} | `{row['selected_variant']}` | "
                f"{float(row['valid_top5_sufficient_rate']):.4f} | "
                f"{float(row['test_top5_sufficient_rate']):.4f} | "
                f"{float(row['test_top5_gain_vs_original']):+.4f} | "
                f"{float(row['train_test_gap_top5']):+.4f} |"
            )
        return lines

    lines = [
        "# Support Reranker Ablation Summary",
        "",
        "## Purpose",
        "",
        "This summary makes the support reranker result auditable by grouping variants by feature set, estimator, and blend alpha. Each row is selected by valid top-5 sufficiency only.",
        "",
        "## Feature Set Ablation",
        "",
        *table(feature_rows),
        "",
        "## Estimator Ablation",
        "",
        *table(estimator_rows),
        "",
        "## Blend Alpha Ablation",
        "",
        *table(blend_rows),
        "",
        "## Interpretation",
        "",
        "The selected production variant should be judged by valid-selected test top-5 gain and train-test gap. Large train-test gaps indicate possible overfitting even when test performance improves.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_validation(
    path: Path,
    args: argparse.Namespace,
    split_records: dict[str, list[dict[str, Any]]],
    variants: list[dict[str, Any]],
    selected: dict[str, Any],
    blend_alphas: list[float],
) -> None:
    validation = {
        "split_dir": args.split_dir,
        "record_kind_filter": args.record_kind_filter,
        "split_counts": {split: len(records) for split, records in split_records.items()},
        "candidate_counts": {split: sum(len(record["retrieved_docs"]) for record in records) for split, records in split_records.items()},
        "variant_count": len(variants),
        "feature_sets": {name: FEATURE_SETS[name] for name in sorted(FEATURE_SETS)},
        "blend_alphas": blend_alphas,
        "selected_variant": selected["variant_id"],
        "selection_protocol": "train support-doc reranker on train; choose variant by valid top-5 sufficiency; report test once",
        "uses_embedding_api": False,
        "uses_llm_api": False,
    }
    path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")


def _group_scores(
    candidate_rows: dict[str, list[dict[str, Any]]],
    scores_by_split: dict[str, list[float]],
) -> dict[str, dict[str, list[float]]]:
    grouped: dict[str, dict[str, list[float]]] = {"train": {}, "valid": {}, "test": {}}
    for split, rows in candidate_rows.items():
        for row, score in zip(rows, scores_by_split[split]):
            grouped[split].setdefault(row["original_id"], []).append(float(score))
    return grouped


def _group_labels(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for row in rows:
        grouped.setdefault(row["original_id"], []).append(int(row["support_label"]))
    return grouped


def _group_support_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    grouped: dict[str, int] = {}
    for row in rows:
        grouped[row["original_id"]] = int(row["support_title_count"])
    return grouped


def _rank_docs(docs: list[dict[str, Any]], scores: list[float]) -> list[dict[str, Any]]:
    indexed = list(enumerate(zip(docs, scores)))
    indexed.sort(key=lambda item: (-float(item[1][1]), int(item[1][0].get("rank", item[0] + 1)), str(item[1][0].get("title", ""))))
    return [doc for _, (doc, _) in indexed]


def _support_present_after_rerank(labels: list[int], support_title_count: int, scores: list[float], top_k: int) -> int:
    if sum(labels) < support_title_count:
        return 0
    ranked_indices = sorted(range(len(scores)), key=lambda idx: (-scores[idx], idx))
    return int(all(labels[idx] == 0 or idx in ranked_indices[:top_k] for idx in range(len(labels))))


def _normalized_original_scores(rows: list[dict[str, Any]]) -> np.ndarray:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row["original_id"], []).append(float(row["features"]["reciprocal_rank"]))
    scores = []
    for row in rows:
        values = grouped[row["original_id"]]
        min_value = min(values)
        max_value = max(values)
        value = float(row["features"]["reciprocal_rank"])
        if max_value <= min_value:
            scores.append(0.0)
        else:
            scores.append((value - min_value) / (max_value - min_value))
    return np.asarray(scores, dtype=float)


def _matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    return np.asarray([[float(row["features"][name]) for name in feature_names] for row in rows], dtype=float)


def _labels(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["support_label"]) for row in rows], dtype=int)


def _model_importances(model: Any) -> np.ndarray | None:
    if isinstance(model, Pipeline):
        clf = model.named_steps.get("clf")
        if hasattr(clf, "coef_"):
            return np.ravel(clf.coef_)
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)
    return None


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(set(y_true.tolist())) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


def _safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(set(y_true.tolist())) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def _support_coverage(support_titles: set[str], retrieved_titles: set[str]) -> float:
    if not support_titles:
        return 0.0
    return len(support_titles & retrieved_titles) / len(support_titles)


def _record_method_name(selected: dict[str, Any]) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", selected["variant_id"]).strip("_").lower()
    return f"link_bridge_support_reranker_{safe}"


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _normalize_title(value: str) -> str:
    return " ".join(tokenize(value))


def _normalize_text(value: str) -> str:
    return " ".join(tokenize(value))


def _token_set(value: str) -> set[str]:
    return set(tokenize(value))


def _overlap_ratio(left: list[str] | set[str], right: list[str] | set[str]) -> float:
    left_set = set(left)
    if not left_set:
        return 0.0
    return len(left_set & set(right)) / len(left_set)


def _title_token_mention_rate(title_tokens: set[str], other_texts: list[str]) -> float:
    if not title_tokens:
        return 0.0
    mentioned = sum(1 for token in title_tokens if any(token in other_text.split() for other_text in other_texts))
    return mentioned / len(title_tokens)


def _mentions_other_titles(text_norm: str, docs: list[dict[str, Any]], current_idx: int) -> float:
    count = 0
    for idx, doc in enumerate(docs):
        if idx == current_idx:
            continue
        title_norm = _normalize_title(doc.get("title", ""))
        if title_norm and title_norm in text_norm:
            count += 1
    return float(count)


def _mentions_query_overlap_titles(text_norm: str, docs: list[dict[str, Any]], query_overlap_flags: list[bool], current_idx: int) -> float:
    count = 0
    for idx, doc in enumerate(docs):
        if idx == current_idx or not query_overlap_flags[idx]:
            continue
        title_norm = _normalize_title(doc.get("title", ""))
        if title_norm and title_norm in text_norm:
            count += 1
    return float(count)


def _candidate_title_overlap(title_tokens: set[str], other_title_tokens: list[set[str]], reducer: str) -> float:
    values = [_overlap_ratio(title_tokens, tokens) for tokens in other_title_tokens]
    if not values:
        return 0.0
    if reducer == "max":
        return float(max(values))
    if reducer == "mean":
        return float(sum(values) / len(values))
    raise ValueError(f"Unknown reducer: {reducer}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _validate_records(split_records: dict[str, list[dict[str, Any]]]) -> None:
    for split in ("train", "valid", "test"):
        records = split_records[split]
        _require(records, f"{split} has no records.")
        ids = [record["metadata"]["original_id"] for record in records]
        _require(len(ids) == len(set(ids)), f"{split} has duplicate original ids.")
        for record in records:
            _require(len(record.get("retrieved_docs", [])) >= 20, f"{record['id']} has fewer than 20 docs.")
            _require(record.get("metadata", {}).get("support_titles"), f"{record['id']} lacks support titles.")
    split_ids = {split: {record["metadata"]["original_id"] for record in records} for split, records in split_records.items()}
    _require(not (split_ids["train"] & split_ids["valid"]), "train and valid overlap.")
    _require(not (split_ids["train"] & split_ids["test"]), "train and test overlap.")
    _require(not (split_ids["valid"] & split_ids["test"]), "valid and test overlap.")


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


if __name__ == "__main__":
    main()
