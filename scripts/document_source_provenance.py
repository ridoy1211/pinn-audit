"""Document historical source hashes using Git objects; never rewrite freezes."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    if args.out.exists():
        raise FileExistsError(f'Preserve existing addendum: {args.out}')

    def git(*arguments):
        return subprocess.check_output(['git', '--git-dir', str(repo / '.git'), *arguments])

    head = git('rev-parse', 'HEAD').decode().strip()
    rows = []
    freezes = []
    for filename in ('TIMING_EVALUATION_FREEZE.json', 'OFFBASIS_EVALUATION_FREEZE.json'):
        freeze_path = 'docs/' + filename
        raw = git('show', head + ':' + freeze_path)
        freeze = json.loads(raw)
        implementation = freeze['implementation_commit']
        freezes.append({'path': freeze_path, 'git_bytes_sha256': sha(raw),
                        'implementation_commit': implementation})
        for name, expected in sorted(freeze['source_sha256'].items()):
            historical = git('show', implementation + ':' + name)
            normalized = historical.replace(b'\r\n', b'\n')
            if sha(historical) == expected:
                convention = 'exact_git_bytes'
            elif historical == normalized and sha(normalized.replace(b'\n', b'\r\n')) == expected:
                convention = 'LF_git_to_CRLF_reconstructs_frozen_bytes'
            else:
                raise RuntimeError(f'Unexplained historical mismatch: {filename}/{name}')
            current = git('show', head + ':' + name)
            row = {
                'freeze': freeze_path, 'source_path': name,
                'implementation_commit': implementation,
                'git_blob_oid': git('rev-parse', implementation + ':' + name).decode().strip(),
                'frozen_raw_sha256': expected,
                'git_source_sha256': sha(historical),
                'lf_normalized_sha256': sha(normalized),
                'reconstruction': convention,
                'current_committed_source_sha256': sha(current),
                'current_committed_source_matches_implementation': current == historical,
            }
            rows.append(row)
    changed = [r for r in rows if r['reconstruction'] != 'exact_git_bytes']
    result = {
        'classification': 'post-evaluation provenance addendum, not a new freeze',
        'inspected_commit': head,
        'method': 'Read Git object bytes at each frozen implementation commit; test exact equality or explicit LF-to-CRLF reconstruction against original SHA256.',
        'freeze_files': freezes,
        'sources': rows,
        'source_entries_checked': len(rows),
        'line_ending_reconstructions': len(changed),
        'unexplained_mismatches': 0,
    }
    lines = [
        '# Historical source provenance addendum', '',
        'This post-evaluation document preserves both original freezes and every recorded hash.',
        'It does not revise the protocol, certify newly edited code, or claim a new preregistration.', '',
        f'Inspected Git commit: `{head}`.',
        f'Checked {len(rows)} source entries across the two freezes; '
        f'{len(changed)} require a line-ending reconstruction; no unexplained mismatches.', '',
        '## Explanation', '',
        'Some historical working files used Windows CRLF line endings, while Git stored LF.',
        'The freeze stored raw working-file hashes. Its creation check allowed newline-normalized',
        'equivalence, but later validation demanded raw-byte equality. This caused fresh-checkout',
        'validation failures even though the affected source text was otherwise unchanged.', '',
        '## Affected entries', '',
        '| Freeze | Source | Reconstruction |', '|---|---|---|',
    ]
    for r in changed:
        lines.append(f'| {r["freeze"]} | {r["source_path"]} | LF to CRLF |')
    lines += [
        '', '## Independent verification', '',
        '`source_provenance.json` records each historical implementation commit, Git blob OID,',
        'original frozen SHA256, Git-byte SHA256 and LF-normalized SHA256.',
        'Read the specified source from that implementation commit using `git show` in a',
        'binary subprocess capture. For an affected LF file, replace byte 0A with 0D 0A',
        'and hash the resulting bytes. This reproduces the frozen raw SHA256 exactly.',
        'Do not perform this conversion on an already CRLF file, and do not rewrite the',
        'working tree or old freeze to make a guard pass.', '',
        '## Limits and next action', '',
        'This establishes a byte-level correspondence, not a complete replay interface.',
        'A maintained replay implementation must preserve the historical snapshot and',
        'explicitly validate this mapping. Future freezes should use one consistent byte',
        'policy at creation and validation. Changing source content requires a new identity.',
        'Current committed-source comparisons are recorded separately; they do not override',
        'historical hashes. No models were retrained, no result archives changed, and nothing',
        'was published by this command.', '',
    ]
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'source_provenance.json').write_text(
        json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    (args.out / 'README.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    print(f'Source entries verified: {len(rows)}')
    print(f'Line-ending reconstructions: {len(changed)}; unexplained mismatches: 0')
    for r in changed:
        print(f'  {r["freeze"]}: {r["source_path"]}')
    print(f'Current committed sources differing from historical implementations: '
          f'{sum(not r["current_committed_source_matches_implementation"] for r in rows)}')
    print(f'ADDENDUM COMPLETE: {args.out.resolve()}')
    print('Original freezes and source files were not modified.')


if __name__ == '__main__':
    main()
