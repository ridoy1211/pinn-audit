# Historical source provenance addendum

This post-evaluation document preserves both original freezes and every recorded hash.
It does not revise the protocol, certify newly edited code, or claim a new preregistration.

Inspected Git commit: `333336060f33b97548635f94df79d86f9787754d`.
Checked 47 source entries across the two freezes; 5 require a line-ending reconstruction; no unexplained mismatches.

## Explanation

Some historical working files used Windows CRLF line endings, while Git stored LF.
The freeze stored raw working-file hashes. Its creation check allowed newline-normalized
equivalence, but later validation demanded raw-byte equality. This caused fresh-checkout
validation failures even though the affected source text was otherwise unchanged.

## Affected entries

| Freeze | Source | Reconstruction |
|---|---|---|
| docs/TIMING_EVALUATION_FREEZE.json | docs/TIMING_DEVELOPMENT_DECISIONS.md | LF to CRLF |
| docs/TIMING_EVALUATION_FREEZE.json | scripts/run_timing_study.py | LF to CRLF |
| docs/TIMING_EVALUATION_FREEZE.json | scripts/timing_common.py | LF to CRLF |
| docs/OFFBASIS_EVALUATION_FREEZE.json | scripts/run_timing_study.py | LF to CRLF |
| docs/OFFBASIS_EVALUATION_FREEZE.json | scripts/timing_common.py | LF to CRLF |

## Independent verification

`source_provenance.json` records each historical implementation commit, Git blob OID,
original frozen SHA256, Git-byte SHA256 and LF-normalized SHA256.
Read the specified source from that implementation commit using `git show` in a
binary subprocess capture. For an affected LF file, replace byte 0A with 0D 0A
and hash the resulting bytes. This reproduces the frozen raw SHA256 exactly.
Do not perform this conversion on an already CRLF file, and do not rewrite the
working tree or old freeze to make a guard pass.

## Limits and next action

This establishes a byte-level correspondence, not a complete replay interface.
A maintained replay implementation must preserve the historical snapshot and
explicitly validate this mapping. Future freezes should use one consistent byte
policy at creation and validation. Changing source content requires a new identity.
Current committed-source comparisons are recorded separately; they do not override
historical hashes. No models were retrained, no result archives changed, and nothing
was published by this command.
