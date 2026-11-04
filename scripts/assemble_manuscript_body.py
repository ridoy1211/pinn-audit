#!/usr/bin/env python3
"""Assemble existing reviewed sections without fitting or changing primary results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    docs = root / "docs"
    inputs = {
        name: (docs / name).read_text(encoding="utf-8")
        for name in (
            "INTRODUCTION_DRAFT.md",
            "METHODS_DRAFT.md",
            "RESULTS_DRAFT.md",
            "DISCUSSION_DRAFT.md",
        )
    }
    tables = json.loads((root / "audit/manuscript_table_sources.json").read_text("utf-8"))
    result_path = docs / "RESULTS_DRAFT.md"
    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == tables["results_draft_sha256"]
    for name, expected in tables["input_sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected

    intro = inputs["INTRODUCTION_DRAFT.md"]
    intro = intro[intro.index("Estimating a physical transport coefficient") :]
    methods = inputs["METHODS_DRAFT.md"]
    methods = methods[methods.index("## Problem and simulation") :]
    methods, editorial = methods.split("## Items to resolve before submission", 1)
    results = inputs["RESULTS_DRAFT.md"]
    results = results[results.index("## Completion and numerical verification") :]
    results = results.split("## Scope of the findings", 1)[0]
    discussion = inputs["DISCUSSION_DRAFT.md"]
    discussion = discussion[discussion.index("## Interpretation of the comparison") :]

    def section(number: int, title: str, body: str) -> str:
        lines = []
        subsection = 0
        for line in body.strip().splitlines():
            if line.startswith("## "):
                subsection += 1
                line = f"### {number}.{subsection}. {line[3:]}"
            lines.append(line)
        return f"## {number}. {title}\n\n" + "\n".join(lines) + "\n"

    heading = (
        (
            "# Regularization, Initial-State Representation, and Optimization Budget "
            "in a Controlled Audit of Physics-Informed Heat Inversion\n\n"
        )
        + """**Manuscript body draft — 5 November 2026.** This combines the existing sections
for review. Abstract, final conclusion, author details and journal formatting
remain to be prepared. This document is not submission-ready. Scientific claims
remain conditional on the stated simulation and procedures.

Experiment evidence: commit `4a64ed4cae0991bb957c18dd2bcea26162bd4e22`.
Section-draft checkpoint: `0977887`. No experiment or primary analysis was changed
to produce this document. Inline citations identify primary works; references
below list each cited work once.

"""
    )
    figures = """## 5. Figures

### Figure 1. Errors across all six procedures

![All-method errors](../results/broader_results_review_v1/method_errors.png)

Each point represents one design-level error. Neural points summarize the three
starts separately for each outcome; they are not ensemble-model predictions.
Errors use a logarithmic vertical scale. The six-mode family-B initial-field
floor is a representation limit. Read together with Table 1, including tails.

### Figure 2. All planned paired contrasts against CV12

![Paired contrasts](../results/broader_results_review_v1/paired_contrasts.png)

Differences are PINN minus CV12 in percentage points: negative favors PINN.
Intervals are pointwise 95% percentile bootstrap intervals for median paired
differences across ten designs per condition, with no multiplicity adjustment.
They are not error bars on the individual observations or simultaneous intervals.

"""
    references = """## 6. References

- Grossmann, T. G., Komorowska, U. J., Latz, J., and Schönlieb, C.-B. (2024).
  Can physics-informed neural networks beat the finite element method?
  *IMA Journal of Applied Mathematics*, 89(1), 143-174.
  [DOI: 10.1093/imamat/hxae011](https://doi.org/10.1093/imamat/hxae011).
- Jekic, A., Natsaridou, A., Riemer-Sørensen, S., Langseth, H., and Gundersen,
  O. E. (2025; version 2, 2026). Examining the robustness of Physics-Informed
  Neural Networks to noise for Inverse Problems. *arXiv preprint* 2509.20191v2.
  [Versioned author text](https://arxiv.org/html/2509.20191v2).
- Oliveira, I. A. R., Mendes, D. A. S., Campos Velho, H. F., and Pereira, E. S.
  (2026). Solving backward heat transfer by physics-informed machine learning.
  *Computational and Applied Mathematics*, 45, 421.
  [DOI: 10.1007/s40314-026-03818-x](https://doi.org/10.1007/s40314-026-03818-x).
- Zapf, B., Haubner, J., Kuchta, M., Ringstad, G., Eide, P. K., and Mardal,
  K.-A. (2022). Investigating molecular transport in the human brain from MRI
  with physics-informed neural networks. *Scientific Reports*, 12, 15475.
  [DOI: 10.1038/s41598-022-19157-w](https://doi.org/10.1038/s41598-022-19157-w).
- Zhao, Z.-X., Banda, M. K., and Guo, B.-Z. (2017). Simultaneous identification
  of diffusion coefficient, spacewise dependent source and initial value for
  one-dimensional heat equation. *Mathematical Methods in the Applied Sciences*,
  40, 3552-3565. [DOI: 10.1002/mma.4245](https://doi.org/10.1002/mma.4245).

"""
    pieces = [heading]
    for number, title, body in (
        (1, "Introduction", intro),
        (2, "Methods", methods),
        (3, "Results", results),
        (4, "Discussion", discussion),
    ):
        pieces.append(section(number, title, body))
    pieces.extend([figures, references])
    manuscript = "\n".join(pieces)
    # All three archived Results tables must survive assembly verbatim.
    original_rows = [line for line in results.splitlines() if line.startswith("|")]
    output_rows = [line for line in manuscript.splitlines() if line.startswith("|")]
    assert output_rows == original_rows
    assert len(original_rows) == 58  # 52 data rows + three header/separator pairs.
    for image in ("method_errors.png", "paired_contrasts.png"):
        assert (root / "results/broader_results_review_v1" / image).is_file()
    output = docs / "MANUSCRIPT_BODY_DRAFT.md"
    audit_path = root / "audit/manuscript_assembly.json"
    if output.exists():
        if not audit_path.exists():
            raise FileExistsError("Existing manuscript has no assembly record; preserve it")
        previous = json.loads(audit_path.read_text("utf-8"))
        if hashlib.sha256(output.read_bytes()).hexdigest() != previous["output_sha256"]:
            raise FileExistsError("Manuscript was edited after assembly; preserve author edits")
    output.write_bytes(manuscript.encode("utf-8"))
    evidence = {
        "classification": "manuscript_assembly_no_training_or_analysis_change",
        "input_sha256": {
            f"docs/{name}": hashlib.sha256((docs / name).read_bytes()).hexdigest()
            for name in inputs
        },
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "preserved_table_data_rows": 52,
        "reference_entries": 5,
        "figures": 2,
        "editorial_notes_from_methods": editorial.strip(),
    }
    audit_path.write_bytes(
        (json.dumps(evidence, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    )
    print(f"Assembled {output.name}; all 52 table data rows preserved.")


if __name__ == "__main__":
    main()
