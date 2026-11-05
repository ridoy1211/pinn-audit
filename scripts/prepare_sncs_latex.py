#!/usr/bin/env python3
"""Convert the reviewed Markdown into a portable, uncompiled LaTeX draft."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "manuscript" / "sncs_review"
RECORD = ROOT / "audit" / "sncs_latex_assembly.json"
SOURCE = ROOT / "docs" / "MANUSCRIPT_REVIEW_DRAFT.md"
REFERENCES = {
    "10.1002/mma.4245": "zhao2017",
    "10.1007/s40314-026-03818-x": "oliveira2026",
    "10.1093/imamat/hxae011": "grossmann2024",
    "10.1038/s41598-022-19157-w": "zapf2022",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def escape(value: str) -> str:
    translations = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(translations.get(char, char) for char in value)


def inline(value: str) -> str:
    """Preserve math and convert the small Markdown subset used by this draft."""
    pattern = r"(\\\(.*?\\\)|`[^`]+`|\[[^\]]+\]\([^)]+\)|\*\*.+?\*\*|\*[^*]+\*)"
    pieces = re.split(pattern, value, flags=re.DOTALL)
    rendered = []
    for part in pieces:
        if part.startswith(r"\(") and part.endswith(r"\)"):
            rendered.append(part)
        elif part.startswith("`") and part.endswith("`"):
            rendered.append(r"\texttt{" + escape(part[1:-1]) + "}")
        elif part.startswith("**") and part.endswith("**"):
            rendered.append(r"\textbf{" + inline(part[2:-2]) + "}")
        elif part.startswith("*") and part.endswith("*"):
            rendered.append(r"\emph{" + inline(part[1:-1]) + "}")
        elif match := re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", part):
            label, url = match.groups()
            key = next((key for doi, key in REFERENCES.items() if doi in url), None)
            if key:
                rendered.append(r"\cite{" + key + "}")
            else:
                if not url.startswith("https://"):
                    relative = (ROOT / "docs" / url).resolve().relative_to(ROOT)
                    url = "https://github.com/ridoy1211/pinn-audit/blob/main/" + relative.as_posix()
                rendered.append(r"\href{" + url + "}{" + inline(label) + "}")
        else:
            rendered.append(escape(part))
    return "".join(rendered)


def table(lines: list[str], number: int, caption: str) -> str:
    cells = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
    assert all(len(row) == len(cells[0]) for row in cells)
    widths = {
        1: [0.04, 0.07, 0.15, 0.16, 0.16, 0.17],
        2: [0.03, 0.07, 0.12, 0.12, 0.10, 0.23, 0.11],
        3: [0.05, 0.08, 0.22, 0.22, 0.22],
    }[number]
    spec = "@{}" + "".join(f"p{{{width}\\textwidth}}" for width in widths) + "@{}"
    headings = cells[0]
    if number == 2:
        headings = ["N", "Family", "Procedure", "Outcome", "Median diff.", "95% interval", "W/T/L"]
        caption += " W/T/L denotes wins/ties/losses."
    elif number == 3:
        headings = [
            "N",
            "Family",
            "CV12 selection + fit",
            "PINN-200 (3 starts)",
            "PINN-1000 (3 starts)",
        ]
    header = " & ".join(inline(cell) for cell in headings) + r" \\"
    rows = [" & ".join(inline(cell) for cell in row) + r" \\" for row in cells[2:]]
    return "\n".join(
        [
            r"\begingroup\small",
            r"\setlength{\tabcolsep}{3pt}",
            r"\begin{longtable}{" + spec + "}",
            r"\caption{" + inline(caption) + r"}\label{tab:" + str(number) + r"}\\",
            r"\toprule",
            header,
            r"\midrule\endfirsthead",
            r"\toprule",
            header,
            r"\midrule\endhead",
            r"\bottomrule\endfoot",
            *rows,
            r"\end{longtable}",
            r"\endgroup",
        ]
    )


def convert(value: str, *, numbered: bool = True) -> str:
    result: list[str] = []
    lines = value.splitlines()
    cursor = 0
    while cursor < len(lines):
        line = lines[cursor]
        if not line.strip():
            result.append("")
            cursor += 1
        elif line.strip() == r"\[":
            end = lines.index(r"\]", cursor + 1)
            result.extend([r"\begin{equation}", *lines[cursor + 1 : end], r"\end{equation}"])
            cursor = end + 1
        elif match := re.match(r"^(#{2,3}) (.*)", line):
            depth, heading = match.groups()
            heading = re.sub(r"^\d+(?:\.\d+)*\.\s*", "", heading)
            command = "section" if len(depth) == 2 else "subsection"
            if not numbered:
                command += "*"
            result.append("\\" + command + "{" + inline(heading) + "}")
            cursor += 1
        elif match := re.match(r"^Table (\d+)\. (.*)", line):
            number, caption = match.groups()
            cursor += 1
            while not lines[cursor].strip():
                cursor += 1
            start = cursor
            while cursor < len(lines) and lines[cursor].startswith("|"):
                cursor += 1
            result.append(table(lines[start:cursor], int(number), caption))
        elif line.startswith("- "):
            result.append(r"\begin{itemize}")
            while cursor < len(lines) and lines[cursor].startswith("- "):
                result.append(r"\item " + inline(lines[cursor][2:]))
                cursor += 1
            result.append(r"\end{itemize}")
        else:
            start = cursor
            cursor += 1
            while cursor < len(lines) and lines[cursor].strip():
                if re.match(r"^(#{2,3}) ", lines[cursor]):
                    break
                cursor += 1
            result.append(inline("\n".join(lines[start:cursor])))
    return "\n".join(result)


def main() -> None:
    assembly = json.loads((ROOT / "audit/manuscript_review_assembly.json").read_text())
    assert digest(SOURCE) == assembly["output_sha256"], "Preserve edited manuscript"
    assert all(digest(ROOT / name) == value for name, value in assembly["input_sha256"].items()), (
        "Run scripts/assemble_review_manuscript.py after saving component changes"
    )
    source = SOURCE.read_text(encoding="utf-8")
    assert source.count("### Use of AI assistance") == 1
    abstract = source.split("## Abstract\n", 1)[1].split("**Keywords:**", 1)[0].strip()
    keywords = source.split("**Keywords:**", 1)[1].split("## 1. Introduction", 1)[0].strip()
    main_text = "## 1. Introduction" + source.split("## 1. Introduction", 1)[1]
    main_text, remainder = main_text.split("## 6. Statements and declarations", 1)
    declarations, remainder = remainder.split("## 7. Figures", 1)
    figure_text, references = remainder.split("## 8. References", 1)
    title = source.splitlines()[0].removeprefix("# ")

    preamble = (
        r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=25mm]{geometry}
\usepackage{fontspec}
\IfFontExistsTF{TeX Gyre Termes}
  {\setmainfont{TeX Gyre Termes}}
  {\setmainfont{Times New Roman}}
\usepackage{amsmath,amssymb,booktabs,longtable,array,graphicx}
\usepackage[hidelinks]{hyperref}
\usepackage{microtype}
\setlength{\emergencystretch}{2em}
\title{"""
        + escape(title)
        + r"""}
\author{Ridoy Hasan\\Proposed affiliation: Unaffiliated, Comilla, Bangladesh\\
\href{mailto:hasanridoy@stud.cou.ac.bd}{hasanridoy@stud.cou.ac.bd}}
\date{Author-review draft --- not for submission}
\begin{document}
\maketitle
\begin{abstract}
"""
    )
    tex = preamble + inline(abstract) + "\n\\end{abstract}\n"
    tex += r"\noindent\textbf{Keywords:} " + inline(keywords) + "\n\n"
    tex += convert(main_text)
    formatted_declarations = convert(declarations, numbered=False)
    for heading in re.findall(r"^### (.+)$", declarations, flags=re.MULTILINE):
        assert formatted_declarations.count(r"\subsection*{" + inline(heading) + "}") == 1
    assert r"\#\#\#" not in formatted_declarations
    tex += "\n\\section*{Statements and declarations}\n" + formatted_declarations
    for number, chunk in enumerate(re.split(r"### Figure \d+\. ", figure_text)[1:], 1):
        heading, chunk = chunk.split("\n", 1)
        image_match = re.search(r"!\[[^\]]+\]\(([^)]+)\)", chunk)
        assert image_match is not None
        caption = re.sub(r"!\[[^\]]+\]\([^)]+\)", "", chunk).strip()
        name = "method_errors.pdf" if number == 1 else "paired_contrasts.pdf"
        tex += "\n" + r"\begin{figure}[p]\centering" + "\n"
        tex += r"\includegraphics[width=\textwidth]{figures/" + name + "}\n"
        tex += r"\caption{" + inline(heading + ". " + caption) + "}\n"
        tex += r"\label{fig:" + str(number) + r"}\end{figure}" + "\n"
    entries = re.split(r"\n- ", "\n" + references.strip())
    bibliography = {}
    for entry in entries:
        key = next((key for doi, key in REFERENCES.items() if doi in entry), None)
        if key:
            bibliography[key] = inline(" ".join(entry.split()))
    # Preserve DOI links in the bibliography rather than converting them to self-citations.
    for doi, key in REFERENCES.items():
        bibliography[key] = bibliography[key].replace(
            r"\cite{" + key + "}", r"\url{https://doi.org/" + doi + "}"
        )
    assert len(bibliography) == 4
    assert "2509.20191v2" in tex, "Keep the preprint attribution in the text"
    tex += "\n\\clearpage\n\\begin{thebibliography}{9}\n"
    for key in REFERENCES.values():
        assert r"\cite{" + key + "}" in tex
        tex += r"\bibitem{" + key + "} " + bibliography[key] + "\n\n"
    tex += "\\end{thebibliography}\n\\end{document}\n"
    assert tex.count(r"\begin{longtable}") == 3
    assert sum(line.startswith("|") for line in main_text.splitlines()) == 58

    if OUTPUT.exists():
        previous = json.loads(RECORD.read_text())
        assert all(
            digest(ROOT / name) == value for name, value in previous["output_sha256"].items()
        ), "Preserve changes to the generated LaTeX package before rebuilding"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "figures").mkdir(exist_ok=True)
    (OUTPUT / "main.tex").write_bytes(tex.encode("utf-8"))
    for name in ("method_errors.pdf", "paired_contrasts.pdf"):
        shutil.copyfile(
            ROOT / "results/broader_results_review_v1" / name, OUTPUT / "figures" / name
        )
    instructions = """# LaTeX review package

Generated from the synchronized Markdown draft. This is an uncompiled, neutral
LaTeX review layout following the main SN Computer Science author instructions;
it is not Springer Nature's official class or an approved submission.

Compile main.tex with LuaLaTeX (twice to resolve citations), for example in
Overleaf after uploading this package. All equations and tables are editable.
The two figure PDFs are copied unchanged from the archived results.

The four published references are numbered. The arXiv preprint remains credited
and linked in the text; it is omitted from the published-reference list because
the current journal instructions exclude unpublished work from that list.
Its current publication status still needs a final check before submission.

No PDF has been compiled or visually verified in this environment. Inspect every
page after compiling, especially the long tables, equations, figure placement,
references, accents and hyperlinks. Share the compiler log if a build fails.

The author list, proposed affiliation, contribution statement, disclosure wording
and final approval remain unresolved. The neutral layout does not resolve those
scientific/author issues. No new training or result selection was performed.

Journal instructions checked 5 November 2026:
https://link.springer.com/journal/42979/submission-guidelines

Do not edit generated files and then rerun the builder without preserving your
changes. For reproducible updates, edit the Markdown components, assemble the
review manuscript, then rebuild with python scripts/prepare_sncs_latex.py.
"""
    (OUTPUT / "README.md").write_bytes(instructions.encode("utf-8"))
    outputs = [
        OUTPUT / "README.md",
        OUTPUT / "main.tex",
        OUTPUT / "figures/method_errors.pdf",
        OUTPUT / "figures/paired_contrasts.pdf",
    ]
    RECORD.write_bytes(
        (
            json.dumps(
                {
                    "classification": "uncompiled_latex_formatting_draft_no_scientific_change",
                    "source_path": SOURCE.relative_to(ROOT).as_posix(),
                    "source_sha256": digest(SOURCE),
                    "output_sha256": {
                        path.relative_to(ROOT).as_posix(): digest(path) for path in outputs
                    },
                    "table_data_rows_preserved": 52,
                    "published_reference_entries": 4,
                    "preprint_attribution": "retained_in_text_2509.20191v2",
                    "compiled": False,
                    "visually_verified": False,
                    "compiler": "LuaLaTeX_required_not_available_locally",
                },
                indent=2,
            )
            + "\n"
        ).encode()
    )
    version = 1
    while True:
        package = ROOT.parents[1] / f"sncs_review_package_v{version}.zip"
        if not package.exists():
            with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as archive:
                for path in outputs:
                    archive.write(path, path.relative_to(OUTPUT).as_posix())
            break
        with zipfile.ZipFile(package) as archive:
            if sorted(archive.namelist()) == sorted(
                path.relative_to(OUTPUT).as_posix() for path in outputs
            ) and all(
                archive.read(path.relative_to(OUTPUT).as_posix()) == path.read_bytes()
                for path in outputs
            ):
                break
        version += 1
    print(f"Prepared uncompiled LaTeX draft; 52 table rows preserved. Package: {package}")


if __name__ == "__main__":
    main()
