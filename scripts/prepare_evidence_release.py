"""Prepare small release documents from the committed evidence inventory."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    if args.out.exists():
        raise FileExistsError(f"Preserve existing release documents: {args.out}")

    def git(*parts):
        return subprocess.check_output(["git", "--git-dir", str(repo / ".git"), *parts])

    commit = git("rev-parse", "HEAD").decode().strip()
    inventory_raw = git("show", commit + ":docs/PUBLICATION_ARCHIVE_INVENTORY.json")
    inventory = json.loads(inventory_raw)
    provenance_raw = git(
        "show", commit + ":docs/source_provenance_addendum_v1/source_provenance.json"
    )
    provenance = json.loads(provenance_raw)
    if provenance["unexplained_mismatches"] != 0:
        raise ValueError("Unresolved source provenance.")
    names = ("fix1-results-b6906ae.zip", "fix2-evidence-v1.zip", "fix3-evidence-v1.zip")
    by_name = {r["filename"]: r for r in inventory["archives"]}
    assets = [by_name[n] for n in names]
    folders = [folder for r in assets for folder in r["result_folders"]]
    if len(folders) != 14 or len(set(folders)) != 14:
        raise ValueError("Unexpected or overlapping result-folder coverage.")
    if not all(r["zip_crc_verified"] for r in assets):
        raise ValueError("Missing recorded archive integrity verification.")
    manifest = {
        "status": "draft_not_uploaded_or_download_verified",
        "scope": "Completed timing, identifiability, rollout and off-basis evidence",
        "reviewed_repository_commit": commit,
        "original_results_commit": "1a4a790fc44991b47f8b815458215577e7aec259",
        "inventory_git_bytes_sha256": hashlib.sha256(inventory_raw).hexdigest(),
        "source_addendum_git_bytes_sha256": hashlib.sha256(provenance_raw).hexdigest(),
        "assets": assets,
        "note": "Archive hashes copied from the committed verified inventory; "
        "this preparation command does not rehash archives or upload files.",
    }
    lines = [
        "# Controlled heat-inversion evidence release",
        "",
        "**Draft release notes. No public download has yet been verified.**",
        "",
        "This package contains saved synthetic observations, fitted models, diagnostics and",
        "analysis outputs for the completed controlled-timing and off-basis extensions.",
        "It is an evidence release, not a claim that the portable reproduction workflow "
        "is finished.",
        "",
        f"Documentation/inventory source commit: `{commit}`.",
        "Completed results branch commit: `1a4a790fc44991b47f8b815458215577e7aec259`.",
        "The eventual release tag should identify the commit containing these release documents",
        "and the completed restore instructions; it has not been assigned by this command.",
        "",
        "## Required evidence assets",
        "",
        "| Archive | Bytes | Files |",
        "|---|---:|---:|",
    ]
    for r in assets:
        lines.append(f"| {r['filename']} | {r['bytes']} | {r['files']} |")
    lines += [
        "",
        "Download all three archives and SHA256SUMS.txt. The automatically generated",
        "GitHub source-code ZIP is not a substitute for these ignored result archives.",
        "",
        "## Contents",
        "",
    ]
    for r in assets:
        lines.append(f"### {r['filename']}")
        lines.append("")
        lines.extend(
            f"- `results/{name}/`: {count} files" for name, count in r["result_folders"].items()
        )
        lines.append("")
    lines += [
        "## Integrity and restoration",
        "",
        "SHA256SUMS.txt records the expected archive bytes. Verify downloads before extraction.",
        "Each archive stores paths beginning with results/. Use a new empty staging directory;",
        "do not extract over a working research checkout or its committed report folders.",
        "About 2.39 billion bytes are needed for extracted evidence, in addition to about",
        "1.69 billion bytes for the three downloads. Allow extra space for replay outputs.",
        "A tested restore command and a clean-checkout verification record remain required",
        "before publication. Do not treat this draft as an operational reproduction guide.",
        "",
        "## Provenance and interpretation",
        "",
        "The original freezes and their hashes remain unchanged. See the repository source",
        "provenance addendum for the demonstrated Git LF versus historical CRLF mapping.",
        "The evidence archives retain their original contents and study-stage classifications.",
        "The primary timing interaction is inconclusive; secondary results must retain their",
        "pointwise interval and multiplicity qualifications. The off-basis study is one",
        "localized initial field, not a universal robustness claim.",
        "",
        "## Scope exclusions",
        "",
        "Local Git bundles are redundant history backups and are not required evidence assets.",
        "Code history remains in the repository; preserve the bundles locally.",
        "The later publication audit and off-basis observation-tail diagnostic are not silently",
        "inserted into these historical archives. Archive those separately when finalized.",
        "No paper DOI, acceptance status, or permanent archive identifier is claimed.",
        "",
    ]
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (args.out / "SHA256SUMS.txt").write_text(
        "".join(f"{r['sha256']}  {r['filename']}\n" for r in assets), encoding="utf-8", newline="\n"
    )
    (args.out / "RELEASE_NOTES.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("RELEASE DOCUMENTS PREPARED")
    print(f"3 evidence ZIP assets; 14 result folders; {sum(r['files'] for r in assets)} files.")
    print(f"Output: {args.out.resolve()}")
    print("Nothing uploaded. Historical evidence and freezes unchanged.")


if __name__ == "__main__":
    main()
