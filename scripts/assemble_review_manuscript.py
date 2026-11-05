#!/usr/bin/env python3
"""Combine existing manuscript components for review without changing analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    docs = root / "docs"
    names = (
        "MANUSCRIPT_BODY_DRAFT.md",
        "ABSTRACT_AND_CONCLUSION_DRAFT.md",
        "TITLE_PAGE_DRAFT.md",
        "DECLARATIONS_DRAFT.md",
    )
    inputs = {name: (docs / name).read_text(encoding="utf-8") for name in names}
    body = inputs["MANUSCRIPT_BODY_DRAFT.md"]
    title = body.splitlines()[0]
    sections = body.split("## 1. Introduction\n", 1)[1]
    main_text, figures = sections.split("## 5. Figures\n", 1)
    figures, references = figures.split("## 6. References\n", 1)
    abstract_source = inputs["ABSTRACT_AND_CONCLUSION_DRAFT.md"]
    abstract = abstract_source.split("## Structured abstract\n", 1)[1].split("## Keywords", 1)[0]
    keywords = abstract_source.split("## Keywords\n", 1)[1].split("## Conclusion", 1)[0]
    conclusion = abstract_source.split("## Conclusion\n", 1)[1].split(
        "## Declaration preparation note", 1
    )[0]
    declarations = inputs["DECLARATIONS_DRAFT.md"].split("## Funding\n", 1)[1]
    declarations = declarations.replace("\n## ", "\n### ")
    note = """

Ridoy Hasan

Proposed affiliation: Unaffiliated, Comilla, Bangladesh

Correspondence: hasanridoy@stud.cou.ac.bd

**For author review.** Additional coauthors remain undecided. Affiliation,
contributions, final author approval and journal formatting are not finalized.
The degree in Physics from Comilla University is recorded as education rather
than institutional affiliation for this independent project. Funding and
competing-interest statements reflect the currently named author's confirmation.
This draft is not ready for submission.

## Abstract

"""
    manuscript = (
        title
        + note
        + abstract.strip()
        + "\n\n**Keywords:** "
        + " ".join(keywords.split())
        + "\n\n## 1. Introduction\n"
        + main_text
        + "## 5. Conclusion\n\n"
        + conclusion.strip()
        + "\n\n## 6. Statements and declarations\n\n### Funding\n\n"
        + declarations.strip()
        + "\n\n## 7. Figures\n"
        + figures
        + "## 8. References\n"
        + references
    )
    manuscript = manuscript.rstrip() + "\n"
    expected_rows = [line for line in body.splitlines() if line.startswith("|")]
    actual_rows = [line for line in manuscript.splitlines() if line.startswith("|")]
    assert actual_rows == expected_rows and len(actual_rows) == 58
    assert manuscript.count("\\[") == manuscript.count("\\]")
    output = docs / "MANUSCRIPT_REVIEW_DRAFT.md"
    record = root / "audit/manuscript_review_assembly.json"
    if output.exists():
        if not record.exists():
            raise FileExistsError("Existing review manuscript has no assembly record")
        previous = json.loads(record.read_text(encoding="utf-8"))
        if digest(output) != previous["output_sha256"]:
            raise FileExistsError("Preserve the author's edits to the review manuscript")
    output.write_bytes(manuscript.encode("utf-8"))
    record.write_bytes(
        (
            json.dumps(
                {
                    "classification": "review_manuscript_assembly_no_new_experiment",
                    "input_sha256": {"docs/" + name: digest(docs / name) for name in names},
                    "output_sha256": digest(output),
                    "preserved_table_data_rows": 52,
                    "funding_and_interests": "author_confirmed_2026-11-05",
                    "author_list": "provisional_additional_coauthors_pending",
                    "word_export": "pending_unavailable_packaged_renderer",
                },
                indent=2,
            )
            + "\n"
        ).encode("utf-8")
    )
    print("Combined review manuscript; all 52 table data rows preserved.")


if __name__ == "__main__":
    main()
