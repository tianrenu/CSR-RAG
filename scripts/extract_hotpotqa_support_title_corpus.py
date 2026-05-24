from __future__ import annotations

import argparse
import bz2
import csv
import html
import json
import tarfile
import time
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract HotpotQA support-title documents from the official intro-paragraph Wikipedia corpus. "
            "This is a no-API corpus-completeness step."
        )
    )
    parser.add_argument("--raw-hotpot", default="data/raw/hotpotqa/hotpot_dev_fullwiki_v1.json")
    parser.add_argument(
        "--wiki-archive",
        default="data/external/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2",
    )
    parser.add_argument("--output-corpus", default="data/processed/hotpotqa_support_title_intro_corpus.jsonl")
    parser.add_argument("--output-dir", default="results/tables/hotpotqa_support_title_intro_corpus_audit")
    parser.add_argument("--progress-every-files", type=int, default=250)
    parser.add_argument("--max-inner-files", type=int, default=0, help="Debug only. 0 means all inner wiki files.")
    args = parser.parse_args()

    started = time.time()
    raw_rows = _load_raw_rows(Path(args.raw_hotpot))
    needed_titles = _needed_support_titles(raw_rows)
    extracted = _extract_support_docs(
        archive_path=Path(args.wiki_archive),
        needed_titles=needed_titles,
        progress_every_files=args.progress_every_files,
        max_inner_files=args.max_inner_files,
    )

    output_corpus = Path(args.output_corpus)
    output_corpus.parent.mkdir(parents=True, exist_ok=True)
    with output_corpus.open("w", encoding="utf-8") as f:
        for title in sorted(extracted):
            f.write(json.dumps(extracted[title], ensure_ascii=False) + "\n")

    detail_rows = _question_detail_rows(raw_rows, extracted)
    summary_rows = _summary_rows(detail_rows)
    missing_title_rows = _missing_title_rows(needed_titles, extracted)
    validation = {
        "raw_hotpot": args.raw_hotpot,
        "wiki_archive": args.wiki_archive,
        "output_corpus": args.output_corpus,
        "raw_rows": len(raw_rows),
        "needed_support_titles": len(needed_titles),
        "extracted_support_titles": len(extracted),
        "missing_support_titles": len(needed_titles - set(extracted)),
        "support_title_coverage": len(extracted) / len(needed_titles) if needed_titles else 0.0,
        "uses_embedding_api": False,
        "uses_llm_api": False,
        "elapsed_seconds": round(time.time() - started, 2),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "support_title_corpus_summary.csv", summary_rows)
    _write_csv(output_dir / "support_title_corpus_question_details.csv", detail_rows)
    _write_csv(output_dir / "missing_support_titles.csv", missing_title_rows)
    (output_dir / "validation_summary.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(output_dir / "support_title_intro_corpus_audit.md", summary_rows, missing_title_rows, validation)
    print(json.dumps({"output_corpus": args.output_corpus, "output_dir": args.output_dir, **validation}, ensure_ascii=False))


def _load_raw_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _needed_support_titles(raw_rows: list[dict[str, Any]]) -> set[str]:
    titles = set()
    for row in raw_rows:
        for title, _sent_idx in row["supporting_facts"]:
            titles.add(_normalize_title(title))
    return titles


def _extract_support_docs(
    archive_path: Path,
    needed_titles: set[str],
    progress_every_files: int,
    max_inner_files: int,
) -> dict[str, dict[str, Any]]:
    extracted: dict[str, dict[str, Any]] = {}
    inner_files = 0
    started = time.time()
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
                    if len(extracted) >= len(needed_titles):
                        break
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    normalized_title = _normalize_title(row.get("title", ""))
                    if normalized_title not in needed_titles or normalized_title in extracted:
                        continue
                    extracted[normalized_title] = {
                        "doc_id": f"hotpotqa_intro::{row.get('id', normalized_title)}",
                        "title": row.get("title", ""),
                        "normalized_title": normalized_title,
                        "text": " ".join(str(sentence) for sentence in row.get("text", [])),
                        "source": "hotpotqa_official_intro_paragraphs",
                        "source_archive_member": member.name,
                        "url": row.get("url", ""),
                    }
                if len(extracted) >= len(needed_titles):
                    break
            if progress_every_files > 0 and (inner_files == 1 or inner_files % progress_every_files == 0):
                print(
                    json.dumps(
                        {
                            "inner_files": inner_files,
                            "extracted_support_titles": len(extracted),
                            "needed_support_titles": len(needed_titles),
                            "elapsed_seconds": round(time.time() - started, 1),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if max_inner_files > 0 and inner_files >= max_inner_files:
                break
            if len(extracted) >= len(needed_titles):
                break
    return extracted


def _question_detail_rows(raw_rows: list[dict[str, Any]], extracted: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    extracted_titles = set(extracted)
    rows = []
    for raw in raw_rows:
        support_titles = sorted({_normalize_title(title) for title, _sent_idx in raw["supporting_facts"]})
        present = [title for title in support_titles if title in extracted_titles]
        missing = [title for title in support_titles if title not in extracted_titles]
        rows.append(
            {
                "original_id": raw["_id"],
                "question_type": raw.get("type", "unknown"),
                "difficulty": raw.get("level", "unknown"),
                "question": raw["question"],
                "gold_answer": raw["answer"],
                "support_title_count": len(support_titles),
                "support_titles": " || ".join(support_titles),
                "support_titles_in_intro_count": len(present),
                "support_titles_missing_intro_count": len(missing),
                "support_titles_missing_intro": " || ".join(missing),
                "all_support_present_in_intro": int(len(missing) == 0),
                "support_title_coverage_in_intro": len(present) / len(support_titles) if support_titles else 0.0,
            }
        )
    return rows


def _summary_rows(detail_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for group_name, predicate in [
        ("all", lambda _row: True),
        ("bridge", lambda row: row["question_type"] == "bridge"),
        ("comparison", lambda row: row["question_type"] == "comparison"),
        ("easy", lambda row: row["difficulty"] == "easy"),
        ("medium", lambda row: row["difficulty"] == "medium"),
        ("hard", lambda row: row["difficulty"] == "hard"),
    ]:
        group = [row for row in detail_rows if predicate(row)]
        if not group:
            continue
        all_present = [row for row in group if int(row["all_support_present_in_intro"]) == 1]
        rows.append(
            {
                "group": group_name,
                "n": len(group),
                "all_support_present_count": len(all_present),
                "all_support_present_rate": len(all_present) / len(group),
                "mean_support_title_coverage": sum(float(row["support_title_coverage_in_intro"]) for row in group) / len(group),
                "missing_question_count": len(group) - len(all_present),
            }
        )
    return rows


def _missing_title_rows(needed_titles: set[str], extracted: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"normalized_title": title} for title in sorted(needed_titles - set(extracted))]


def _write_markdown(
    path: Path,
    summary_rows: list[dict[str, Any]],
    missing_title_rows: list[dict[str, Any]],
    validation: dict[str, Any],
) -> None:
    lines = [
        "# HotpotQA Support-Title Intro Corpus Audit",
        "",
        "## Purpose",
        "",
        "This no-API run extracts only HotpotQA dev supporting-title documents from the official HotpotQA introductory-paragraph Wikipedia corpus.",
        "",
        "## Summary",
        "",
        "| Group | N | All support present | Mean title coverage | Missing questions |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['group']} | {row['n']} | {float(row['all_support_present_rate']):.4f} | "
            f"{float(row['mean_support_title_coverage']):.4f} | {row['missing_question_count']} |"
        )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- Needed support titles: {validation['needed_support_titles']}",
            f"- Extracted support titles: {validation['extracted_support_titles']}",
            f"- Missing support titles: {validation['missing_support_titles']}",
            f"- Support-title coverage: {float(validation['support_title_coverage']):.4f}",
            "- Embedding API: no",
            "- LLM API: no",
            "",
            "## Missing Title Examples",
            "",
        ]
    )
    for row in missing_title_rows[:30]:
        lines.append(f"- {row['normalized_title']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def _normalize_title(title: str) -> str:
    return html.unescape(str(title)).strip().lower()


if __name__ == "__main__":
    main()
