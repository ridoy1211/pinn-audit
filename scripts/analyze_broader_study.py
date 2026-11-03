#!/usr/bin/env python3
"""Verify and summarize a completed broader study without fitting or selecting models."""

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from broader_analysis import format_markdown, summarize_study
from broader_common import write_json
from broader_provenance import check_hashes, sha256, verify_run_outputs


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                    for k, v in row.items()
                }
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/broader_study_v1"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/broader_analysis_v1"))
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("Choose a fresh analysis directory; retain earlier evidence")
    root = Path(__file__).resolve().parents[1]
    manifest = verify_run_outputs(args.results)
    check_hashes(root, manifest["source_sha256"])
    config = json.loads((args.results / "config.json").read_text(encoding="utf-8"))
    records = json.loads((args.results / "records.json").read_text(encoding="utf-8"))
    cases = json.loads((args.results / "cases.json").read_text(encoding="utf-8"))
    if len(records) != manifest["expected_jobs"] or len(cases) != manifest["expected_cases"]:
        raise RuntimeError("Run evidence does not cover the full declared schedule")
    result = summarize_study(
        records,
        cases,
        config["training_seeds"],
        bootstrap_reps=config["bootstrap_reps"],
        bootstrap_seed_offset=config["bootstrap_seed_offset"],
        min_complete_pairs=config["min_complete_pairs"],
    )
    result.update(
        mode=manifest["mode"],
        run_status=manifest["status"],
        input_manifest_sha256=sha256(args.results / "manifest.json"),
        created_utc=datetime.now(UTC).isoformat(),
    )
    args.out_dir.mkdir(parents=True)
    write_json(args.out_dir / "analysis.json", result)
    for name, key in (
        ("completeness", "completeness_ledger"),
        ("dataset_scores", "dataset_scores"),
        ("method_summary", "method_summaries"),
        ("paired_contrasts", "paired_contrasts"),
    ):
        write_csv(args.out_dir / (name + ".csv"), result[key])
    body = format_markdown(result)
    lines = [
        f"Mode: **{manifest['mode']}**. Collection status: {manifest['status']}.",
        "Median errors below are percentages. Neural case scores require all three starts.",
        "",
        "| N | family | method | complete / expected datasets | median alpha error "
        "| median field error | median IC error |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in result["method_summaries"]:
        values = [
            row["metrics"][k]["median"]
            for k in ("relative_alpha_error", "field_relative_l2", "ic_relative_l2")
        ]
        text = ["missing" if value is None else f"{100 * value:.5g}" for value in values]
        lines.append(
            f"| {row['n_obs']} | {row['family']} | {row['method']} "
            f"| {row['complete_count']} / {row['expected_count']} | {' | '.join(text)} |"
        )
    insertion = "\n".join(lines) + "\n\n"
    body = body.replace(
        "## Paired contrasts against CV12", insertion + "## Paired contrasts against CV12"
    )
    (args.out_dir / "summary.md").write_text(body, encoding="utf-8", newline="\n")
    write_json(
        args.out_dir / "manifest.json",
        {
            "classification": "derived_prespecified_analysis_no_fitting",
            "input_manifest_sha256": result["input_manifest_sha256"],
            "analysis_sources_sha256": manifest["source_sha256"],
            "output_sha256": {
                p.name: sha256(p) for p in args.out_dir.iterdir() if p.name != "manifest.json"
            },
        },
    )
    print(body)


if __name__ == "__main__":
    main()
