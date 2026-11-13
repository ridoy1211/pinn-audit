# Integrated heat inversion manuscript

This author-review draft presents one paper connecting the original comparison,
controlled observation timing, explanatory diagnostics and the localized-field
evaluation. It does not change the completed experiments.

Read the [manuscript PDF](manuscript.pdf). Edit [main.tex](main.tex) and
[references.bib](references.bib) for the next revision. The
[evidence map](../../docs/MANUSCRIPT_INTEGRATION.md) explains each stage's role.
The [earlier manuscript](../../docs/MANUSCRIPT_REVIEW_DRAFT.md) covers the
original 400-job comparison and remains a historical draft.

## Build

From this directory, using TeX Live:

```text
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

Alternatively, upload this directory's TeX files, bibliography and figures to
Overleaf, select main.tex and compile with XeLaTeX. The matching PDF was built
with Tectonic/XeTeX and BibTeX on 14 November 2026. A generated .bbl is excluded
from this repository directory so that bibliography edits are rebuilt.

## Evidence

- Original comparison: [protocol](https://github.com/ridoy1211/pinn-audit/blob/02af8a24de9d536f5f6b68352ab8fc980a37819b/docs/BROADER_COMPARISON_PROTOCOL.md) and [results](https://github.com/ridoy1211/pinn-audit/blob/02af8a24de9d536f5f6b68352ab8fc980a37819b/results/broader_analysis_v1/summary.md).
- Controlled timing: [protocol](https://github.com/ridoy1211/pinn-audit/blob/02af8a24de9d536f5f6b68352ab8fc980a37819b/docs/TIMING_PROTOCOL.md) and [results](https://github.com/ridoy1211/pinn-audit/blob/02af8a24de9d536f5f6b68352ab8fc980a37819b/results/timing_evaluation_analysis_v1/summary.md).
- Localized initial field: [protocol](https://github.com/ridoy1211/pinn-audit/blob/02af8a24de9d536f5f6b68352ab8fc980a37819b/docs/OFFBASIS_PROTOCOL.md) and [results](https://github.com/ridoy1211/pinn-audit/blob/02af8a24de9d536f5f6b68352ab8fc980a37819b/results/offbasis_analysis_v1/REPORT.md).
- Explanatory diagnostics: [identifiability and independent propagation](https://github.com/ridoy1211/pinn-audit/blob/02af8a24de9d536f5f6b68352ab8fc980a37819b/results/timing_fix2_report_v1/REPORT.md).
- Saved observations and fits: [public evidence release](https://github.com/ridoy1211/pinn-audit/releases/tag/heat-inversion-evidence-v1.0.0).

The pinned evidence links deliberately point to the research branch's reporting
checkpoint. Dataset, block and fit counts have different meanings and are not
pooled across stages. The 29 cited references include close work on joint heat
inversion, initial-state recovery and identifiability; reference count is not a
measure of novelty.

## Revision practice

Read and approve the complete text and cited sources before submission. This is
not a peer-reviewed or accepted paper, and no arXiv posting is implied. The
source uses an arXiv-style article layout; journal-specific formatting remains
separate. Synchronize any Word review edits with this source before rebuilding
the PDF or a submission ZIP.
