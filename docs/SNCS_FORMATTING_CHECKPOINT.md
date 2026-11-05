# Manuscript formatting checkpoint — 5 November 2026

**Latest review, 6 November:** the corrected Overleaf PDF is archived unchanged
at `manuscript/reviews/overleaf_v2_2026-11-06.pdf`. Six pages are pixel-identical
to the previously inspected PDF; the five changed pages were rendered and
visually inspected. Declaration heading/numbering defects are resolved, table
headers are clearer, and all 52 numeric result rows are retained. This is a
scientific-review draft checkpoint, not final publication-layout approval. Dense
Figure 1 labels remain a candidate for improved print-size presentation. See
`audit/overleaf_pdf_v2_review_2026-11-06.json`. The current local source also has
a font fallback and compiled successfully in the desktop editor; its exact
native preview was not exported for visual comparison with Overleaf.

**Update, 6 November:** the author compiled package v1 in Overleaf. All 11 PDF
pages were rendered and inspected, and all 52 result rows were found in the PDF
text. Two converter defects affected declaration headings; the corrected source
also shortens cramped table headers. Package v2 preserves the earlier ZIP. The
new source needs recompilation and visual review; the v1 review does not certify
the v2 layout. See `audit/overleaf_pdf_review_2026-11-06.json`. Upload the updated
`manuscript/sncs_review/main.tex` into the existing Overleaf project, replace the
old file, and recompile with LuaLaTeX. The figure files are unchanged.

User checkpoint 206d808 synchronized the declaration draft and review manuscript,
but the declarations were then saved with one heading while the assembled copy
still contained two. The source hash differed from the assembly record. Regenerated
the combined manuscript from the current components without changing the author's
declaration wording. The new copy has one AI-assistance heading and all 52 original
result-table data rows.

## Prepared files

- `manuscript/sncs_review/main.tex`: editable, uncompiled LaTeX review layout.
- `manuscript/sncs_review/figures/`: two unchanged vector figures.
- `manuscript/sncs_review/README.md`: compilation notes and limitations.
- `scripts/prepare_sncs_latex.py`: reproducible Markdown-to-LaTeX builder; protects
  edited outputs and preserves earlier ZIP packages.
- `audit/sncs_latex_assembly.json`: input/output hashes.
- `audit/sncs_format_source_validation.json`: static checks and metadata sources.

The portable ZIP is outside the repository at `outputs/sncs_review_package_v1.zip`
under the workspace root. It contains main.tex, README and the two figure PDFs.

## Journal-specific preparation

The [current SN Computer Science instructions](https://link.springer.com/journal/42979/submission-guidelines)
allow LaTeX for mathematical manuscripts and recommend the publisher template.
This package uses a neutral article layout, not the publisher's official class.
It retains a 202-word structured abstract and six keywords, converts citations
to numbers, and numbers the four published references. The Jekic preprint remains
identified and linked in the text, rather than being treated as a published
bibliography entry. Its metadata/publication status was checked on arXiv; no
acceptance was established. Four publisher records were checked for bibliography
metadata. This was not a new full-text appraisal of those papers.

The formatting source has passed lint and static checks: 52 table rows, 10
displayed equations, inline math delimiters, two original figure hashes and all
15 frozen study source hashes. No evaluation was rerun. These checks are not
LaTeX compilation or visual proof of correct page layout.

## Next task for Ridoy

1. In Overleaf, choose New Project, then Upload Project, and select the ZIP.
2. Set the compiler to LuaLaTeX and the main document to main.tex.
3. Recompile and inspect every page. Send the compilation log if there is an error.
   Otherwise provide the resulting PDF for layout and scientific review.

Official help: [uploading a project](https://www.overleaf.com/learn/latex/Kb/Uploading_a_project),
[choosing a compiler](https://docs.overleaf.com/getting-started/recompiling-your-project/selecting-a-tex-live-version-and-latex-compiler).

Compilation and visual verification remain pending because no LaTeX compiler is
available in this environment. This is an author-review package, not an approved
submission. Authorship/affiliation, contributions, complete assistance disclosure,
final references and scientific review still need resolution. Current author
wording is preserved; preparation does not certify that all author checks have
been completed. No Git commit, push, journal submission or message was sent.
