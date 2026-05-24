from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

from csrrag.utils.env import load_dotenv


FIELDNAMES = [
    "case_id",
    "sample_name",
    "original_id",
    "top_k",
    "sample_bucket",
    "sufficiency_label",
    "question",
    "gold_answer",
    "model_answer",
    "em",
    "f1",
    "taxonomy",
    "raw_taxonomy",
    "answer_supported_by_context",
    "gold_supported_by_context",
    "confidence",
    "short_reason",
    "context_titles",
    "error",
]

TAXONOMIES = {
    "retrieval_insufficient",
    "generation_failure",
    "answer_alias_or_metric_mismatch",
    "ambiguous_or_gold_issue",
    "format_or_refusal_artifact",
    "uncertain",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Use MiniMax to classify substantive QA failures from existing link-bridge QA artifacts. "
            "This script calls the chat LLM only and never calls embedding APIs."
        )
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument(
        "--qa-dir",
        default="results/tables/hotpotqa_official_intro_link_bridge_support_reranker_qa_eval_natural120",
    )
    parser.add_argument(
        "--split-dir",
        default="data/processed/hotpotqa_official_intro_link_bridge_support_reranker_splits",
    )
    parser.add_argument(
        "--record-prefix",
        default="official_intro_link_bridge_support_reranker_random_forest_balanced_all_blend1_00_top",
    )
    parser.add_argument(
        "--output-dir",
        default="results/tables/hotpotqa_official_intro_link_bridge_support_reranker_qa_failure_judge_natural120",
    )
    parser.add_argument("--sample-name", default="natural_qa120")
    parser.add_argument("--max-cases", type=int, default=80)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-context-chars", type=int, default=5000)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    load_dotenv(args.env_file)
    client = _ChatJsonClient(
        base_url=_required_env("LLM_BASE_URL"),
        api_key=_required_env("LLM_API_KEY"),
        model=_required_env("LLM_MODEL"),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    judgments_path = output_dir / "qa_failure_judgments.csv"

    details = _read_csv(Path(args.qa_dir) / "link_bridge_qa_details.csv")
    records = _load_records(Path(args.split_dir), args.record_prefix)
    cases = _candidate_cases(args.sample_name, details, records, args.max_context_chars)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    existing = {row["case_id"]: row for row in _read_csv_if_exists(judgments_path)}
    pending = [case for case in cases if case["case_id"] not in existing]
    results = dict(existing)
    workers = max(1, args.workers)
    if workers == 1:
        for case in pending:
            results[case["case_id"]] = _judge_case(client, case, args.max_tokens)
            _write_rows(judgments_path, cases, results)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_case = {executor.submit(_judge_case, client, case, args.max_tokens): case for case in pending}
            for future in as_completed(future_to_case):
                case = future_to_case[future]
                try:
                    results[case["case_id"]] = future.result()
                except Exception as exc:  # noqa: BLE001 - keep failed judge calls as data.
                    results[case["case_id"]] = _error_row(case, type(exc).__name__)
                _write_rows(judgments_path, cases, results)

    ordered = [results[case["case_id"]] for case in cases if case["case_id"] in results]
    summary_rows = _summary_rows(ordered)
    _write_csv(output_dir / "qa_failure_taxonomy_summary.csv", summary_rows, list(summary_rows[0].keys()))
    (output_dir / "qa_failure_taxonomy_summary.md").write_text(
        _summary_markdown(ordered, summary_rows, args),
        encoding="utf-8",
    )
    (output_dir / "validation_summary.json").write_text(
        json.dumps(
            {
                "qa_dir": args.qa_dir,
                "split_dir": args.split_dir,
                "record_prefix": args.record_prefix,
                "sample_name": args.sample_name,
                "candidate_case_count": len(_candidate_cases(args.sample_name, details, records, args.max_context_chars)),
                "selected_case_count": len(cases),
                "completed_case_count": len(ordered),
                "max_cases": args.max_cases,
                "max_context_chars": args.max_context_chars,
                "llm_model": os.environ.get("LLM_MODEL", ""),
                "uses_llm_api": bool(pending),
                "uses_embedding_api": False,
                "outputs": [
                    "qa_failure_judgments.csv",
                    "qa_failure_taxonomy_summary.csv",
                    "qa_failure_taxonomy_summary.md",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "candidate_cases": len(cases),
                "completed": len(ordered),
                "pending_api_calls": len(pending),
                "taxonomy_counts": dict(Counter(row["taxonomy"] for row in ordered)),
                "uses_llm_api": bool(pending),
                "uses_embedding_api": False,
            },
            ensure_ascii=False,
        )
    )


def _candidate_cases(
    sample_name: str,
    details: list[dict[str, str]],
    records: dict[tuple[str, int], dict[str, Any]],
    max_context_chars: int,
) -> list[dict[str, Any]]:
    cases = []
    for row in details:
        for top_k in (5, 20):
            if not _to_bool(row.get(f"top{top_k}_format_ok", "")):
                continue
            if not _to_bool(row.get(f"top{top_k}_is_substantive", "")):
                continue
            if float(row.get(f"top{top_k}_em", 0.0)) != 0.0:
                continue
            original_id = row["original_id"]
            record = records.get((original_id, top_k), {})
            docs = record.get("retrieved_docs", [])
            context_titles = " || ".join(doc.get("title", "") for doc in docs)
            cases.append(
                {
                    "case_id": f"{sample_name}::{original_id}::top{top_k}",
                    "sample_name": sample_name,
                    "original_id": original_id,
                    "top_k": top_k,
                    "sample_bucket": row["sample_bucket"],
                    "sufficiency_label": row[f"label_top{top_k}"],
                    "question": row["question"],
                    "gold_answer": row["gold_answer"],
                    "model_answer": row[f"top{top_k}_answer"],
                    "em": row[f"top{top_k}_em"],
                    "f1": row[f"top{top_k}_f1"],
                    "context_titles": context_titles,
                    "context": _context_text(docs, max_context_chars),
                }
            )
    return cases


def _judge_case(client: "_ChatJsonClient", case: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    payload = client.complete_json(_messages(case), max_tokens=max_tokens)
    raw_taxonomy = str(
        payload.get("taxonomy")
        or payload.get("failure_type")
        or payload.get("category")
        or payload.get("label")
        or "uncertain"
    ).strip()
    taxonomy = _normalize_taxonomy(raw_taxonomy)
    return {
        "case_id": case["case_id"],
        "sample_name": case["sample_name"],
        "original_id": case["original_id"],
        "top_k": case["top_k"],
        "sample_bucket": case["sample_bucket"],
        "sufficiency_label": case["sufficiency_label"],
        "question": case["question"],
        "gold_answer": case["gold_answer"],
        "model_answer": case["model_answer"],
        "em": case["em"],
        "f1": case["f1"],
        "taxonomy": taxonomy,
        "raw_taxonomy": raw_taxonomy,
        "answer_supported_by_context": _bool_text(payload.get("answer_supported_by_context")),
        "gold_supported_by_context": _bool_text(payload.get("gold_supported_by_context")),
        "confidence": str(payload.get("confidence", "uncertain")).strip(),
        "short_reason": str(payload.get("short_reason", "")).replace("\n", " ").strip(),
        "context_titles": case["context_titles"],
        "error": "",
    }


def _messages(case: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "You are auditing QA failures in an academic RAG experiment. "
        "Classify why a substantive model answer was marked wrong by exact match. "
        "Use only the supplied question, gold answer, model answer, sufficiency label, and retrieved context. "
        "Return exactly one JSON object. Do not include reasoning text outside JSON. "
        "Allowed taxonomy values: retrieval_insufficient, generation_failure, "
        "answer_alias_or_metric_mismatch, ambiguous_or_gold_issue, format_or_refusal_artifact, uncertain. "
        "The taxonomy value must be exactly one of the allowed strings. "
        "Use retrieval_insufficient when the retrieved context does not contain enough evidence for the gold answer. "
        "Use generation_failure when the context supports the gold answer but the model answer is wrong or unsupported. "
        "Use answer_alias_or_metric_mismatch when the model answer is semantically equivalent to the gold answer. "
        "For yes/no questions, a longer answer that clearly means the same yes/no decision is answer_alias_or_metric_mismatch. "
        "For shortened names, initials, or common aliases that identify the same entity, use answer_alias_or_metric_mismatch. "
        "Use ambiguous_or_gold_issue when the gold answer or question appears ambiguous. "
        "The JSON keys must be taxonomy, answer_supported_by_context, gold_supported_by_context, confidence, short_reason. "
        "Example: {\"taxonomy\":\"generation_failure\",\"answer_supported_by_context\":false,"
        "\"gold_supported_by_context\":true,\"confidence\":\"medium\",\"short_reason\":\"...\"}"
    )
    user = (
        f"Sufficiency label: {case['sufficiency_label']}\n"
        f"Question: {case['question']}\n"
        f"Gold answer: {case['gold_answer']}\n"
        f"Model answer: {case['model_answer']}\n"
        f"Retrieved context:\n{case['context']}\n\n"
        "Return JSON now."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _load_records(split_dir: Path, record_prefix: str) -> dict[tuple[str, int], dict[str, Any]]:
    records: dict[tuple[str, int], dict[str, Any]] = {}
    for record in _read_jsonl(split_dir / "test.jsonl"):
        metadata = record.get("metadata", {})
        kind = metadata.get("record_kind", "")
        if not kind.startswith(record_prefix):
            continue
        top_k = int(metadata.get("top_k", 0))
        if top_k not in {5, 20}:
            continue
        records[(metadata["original_id"], top_k)] = record
    return records


def _context_text(docs: list[dict[str, Any]], max_chars: int) -> str:
    parts = []
    remaining = max_chars
    for idx, doc in enumerate(docs, start=1):
        text = str(doc.get("text", "")).strip().replace("\n", " ")
        part = f"[{idx}] {doc.get('title', '')}\n{text}"
        if len(part) > remaining:
            part = part[: max(0, remaining)]
        parts.append(part)
        remaining -= len(part)
        if remaining <= 0:
            break
    return "\n\n".join(parts)


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return [{"group": "all", "value": "all", "n": 0, "share": 0.0}]
    output = []
    for group in ("all", "top_k", "sufficiency_label"):
        if group == "all":
            counts = Counter({"all": len(rows)})
        else:
            counts = Counter(str(row[group]) for row in rows)
        total = sum(counts.values())
        for value, count in sorted(counts.items()):
            output.append({"group": group, "value": value, "n": count, "share": count / total if total else 0.0})
    taxonomy_counts = Counter(row["taxonomy"] for row in rows)
    for value, count in sorted(taxonomy_counts.items()):
        output.append({"group": "taxonomy", "value": value, "n": count, "share": count / len(rows)})
    for sufficiency in sorted({row["sufficiency_label"] for row in rows}):
        subset = [row for row in rows if row["sufficiency_label"] == sufficiency]
        counts = Counter(row["taxonomy"] for row in subset)
        for value, count in sorted(counts.items()):
            output.append(
                {
                    "group": f"taxonomy_given_{sufficiency}",
                    "value": value,
                    "n": count,
                    "share": count / len(subset) if subset else 0.0,
                }
            )
    return output


def _summary_markdown(rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], args: argparse.Namespace) -> str:
    taxonomy = [row for row in summary_rows if row["group"] == "taxonomy"]
    by_suff = [row for row in summary_rows if row["group"].startswith("taxonomy_given_")]
    lines = [
        "# QA Failure Taxonomy",
        "",
        "This audit uses MiniMax chat judgments over existing QA answers and retrieved contexts. It does not call embedding APIs.",
        "",
        "## Scope",
        "",
        f"- QA dir: `{args.qa_dir}`",
        f"- Split dir: `{args.split_dir}`",
        f"- Selected substantive EM-failure cases: `{len(rows)}`",
        f"- Max context chars per case: `{args.max_context_chars}`",
        "",
        "## Taxonomy Counts",
        "",
        "| Taxonomy | Count | Share |",
        "|---|---:|---:|",
    ]
    for row in taxonomy:
        lines.append(f"| {row['value']} | {row['n']} | {float(row['share']):.4f} |")
    lines.extend(["", "## Conditional Counts", "", "| Group | Taxonomy | Count | Share |", "|---|---|---:|---:|"])
    for row in by_suff:
        lines.append(
            f"| {row['group']} | {row['value']} | {row['n']} | {float(row['share']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "1. These are LLM audit labels, not ground-truth labels.",
            "2. The audit is meant to separate retrieval failure from generation/evaluation failure before larger QA runs.",
            "3. Any paper claim must still be backed by the original QA tables and selected case examples.",
            "",
        ]
    )
    return "\n".join(lines)


class _ChatJsonClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def complete_json(self, messages: list[dict[str, str]], max_tokens: int) -> dict[str, Any]:
        last_payload: dict[str, Any] = {}
        for attempt in range(2):
            retry_messages = messages
            if attempt > 0:
                retry_messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            "The previous response was empty or invalid. Return exactly one JSON object "
                            "with the required keys and an allowed taxonomy string."
                        ),
                    }
                ]
            body: dict[str, Any] = {
                "model": self.model,
                "messages": retry_messages,
                "temperature": 0,
                "max_tokens": max(max_tokens, 1024),
                "response_format": {"type": "json_object"},
                "reasoning_split": attempt == 0,
            }
            response = _post_with_retries(
                f"{self.base_url}/chat/completions",
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                body,
                timeout=self.timeout,
                max_retries=3,
                retry_sleep=2.0,
            )
            payload = response.json()
            content = payload["choices"][0]["message"].get("content") or "{}"
            parsed = _parse_json_object(content)
            last_payload = parsed
            if parsed and any(key in parsed for key in ("taxonomy", "failure_type", "category", "label")):
                return parsed
        return last_payload


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            value = json.loads(cleaned[start : end + 1])
            return value if isinstance(value, dict) else {}
    return {}


def _post_with_retries(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout: int,
    max_retries: int,
    retry_sleep: float,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=body, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt == max_retries - 1:
                break
            time.sleep(retry_sleep * (attempt + 1))
    if last_error is None:
        raise RuntimeError("Request failed without an exception.")
    raise last_error


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_csv_if_exists(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return _read_csv(path)


def _write_rows(path: Path, cases: list[dict[str, Any]], rows_by_id: dict[str, dict[str, Any]]) -> None:
    rows = [rows_by_id[case["case_id"]] for case in cases if case["case_id"] in rows_by_id]
    _write_csv(path, rows, FIELDNAMES)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _error_row(case: dict[str, Any], error: str) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "sample_name": case["sample_name"],
        "original_id": case["original_id"],
        "top_k": case["top_k"],
        "sample_bucket": case["sample_bucket"],
        "sufficiency_label": case["sufficiency_label"],
        "question": case["question"],
        "gold_answer": case["gold_answer"],
        "model_answer": case["model_answer"],
        "em": case["em"],
        "f1": case["f1"],
        "taxonomy": "uncertain",
        "raw_taxonomy": "uncertain",
        "answer_supported_by_context": "",
        "gold_supported_by_context": "",
        "confidence": "",
        "short_reason": "",
        "context_titles": case["context_titles"],
        "error": error,
    }


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _bool_text(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    lowered = str(value).strip().lower()
    if lowered in {"true", "false"}:
        return lowered
    if lowered in {"yes", "supported", "1"}:
        return "true"
    if lowered in {"no", "not supported", "0"}:
        return "false"
    return ""


def _normalize_taxonomy(value: str) -> str:
    lowered = value.strip().lower().replace("-", "_").replace(" ", "_")
    if lowered in TAXONOMIES:
        return lowered
    if any(token in lowered for token in ("alias", "equivalent", "same_answer", "metric", "em_mismatch", "acceptable")):
        return "answer_alias_or_metric_mismatch"
    if any(token in lowered for token in ("retrieval", "insufficient", "missing_evidence", "not_in_context", "no_evidence")):
        return "retrieval_insufficient"
    if any(token in lowered for token in ("generation", "hallucination", "wrong_answer", "unsupported_answer", "model_error")):
        return "generation_failure"
    if any(token in lowered for token in ("ambiguous", "gold", "question_issue", "annotation")):
        return "ambiguous_or_gold_issue"
    if any(token in lowered for token in ("format", "refusal", "dont_know", "unknown")):
        return "format_or_refusal_artifact"
    return "uncertain"


if __name__ == "__main__":
    main()
