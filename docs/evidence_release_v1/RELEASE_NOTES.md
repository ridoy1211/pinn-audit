# Controlled heat inversion evidence release

The three ZIP archives contain the saved observations, fitted models, diagnostics,
and reports for the controlled timing and off-basis extensions. The completed
results are recorded at commit `1a4a790fc44991b47f8b815458215577e7aec259`.

| Archive | Contents | Files |
|---|---|---:|
| `controlled-timing-evidence-v1.zip` | Timing development, ridge-grid checks, frozen evaluation, and timing analysis | 5,856 |
| `identifiability-and-rollout-evidence-v1.zip` | Identifiability, basis sensitivity, independent rollout, rollout refinement, and diagnostic report | 5,076 |
| `off-basis-robustness-evidence-v1.zip` | Off-basis truth verification, development, frozen evaluation, and analysis | 1,799 |

## Verify and restore

Download the three ZIPs from the release assets into a folder named
`heat-evidence-downloads` next to the repository checkout. Use the source at tag
`heat-inversion-evidence-v1.0.0`. Its `docs/evidence_release_v1/release_manifest.json` lists archive
sizes, SHA256 digests, and folder counts. The accompanying `SHA256SUMS.txt`
provides the same archive digests for independent checks.

From the checkout root, run this command in Windows CMD to verify the downloads.
The verifier uses the Python standard library.

```bat
python -u scripts\restore_publication_evidence.py --manifest "docs\evidence_release_v1\release_manifest.json" --archives "..\heat-evidence-downloads" --verify-only
```

To verify and extract in one pass, use a new sibling directory as the destination.
Allow about 3 GB of free space for the restored files and the utility's space
margin, in addition to the downloads.

```bat
python -u scripts\restore_publication_evidence.py --manifest "docs\evidence_release_v1\release_manifest.json" --archives "..\heat-evidence-downloads" --destination "..\heat-evidence-restored"
```

The destination must be absent at the start. On success, it contains `results/`
and `RESTORE_VERIFICATION.json`. The expected counts are **12,731 archive files**,
**12 output-hash indexes**, and **11,917 indexed hash entries**. If interrupted,
keep the partial directory and choose a new destination for the next attempt.

## Verification record and scope

A complete local restoration passed on Windows using the manifest at `7b21bfc`.
The descriptive release filenames refer to the same ZIP bytes and SHA256 digests.

The verifier checks archive digests, ZIP CRCs, archive inventory, and recorded
output hashes. The [analysis replay instructions](../../README.md#replay-the-follow-up-analyses)
use the restored evidence to rebuild the tables and reports from saved fits.

The [source provenance addendum](../source_provenance_addendum_v1/README.md)
documents the correspondence between the original freeze hashes and the Git
source bytes. The primary timing interaction remains inconclusive, and the
off-basis evaluation concerns one localized initial field. The archives preserve
the original study stages and include the separately recorded rollout refinement.
