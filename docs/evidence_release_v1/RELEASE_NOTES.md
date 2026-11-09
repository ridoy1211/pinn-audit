# Controlled heat inversion evidence release

The three ZIP archives contain the saved observations, fitted models, diagnostics,
and reports for the controlled timing and off-basis extensions. The completed
results are recorded at commit `1a4a790fc44991b47f8b815458215577e7aec259`.

| Archive | Contents | Files |
|---|---|---:|
| `fix1-results-b6906ae.zip` | Timing development, ridge-grid checks, frozen evaluation, and timing analysis | 5,856 |
| `fix2-evidence-v1.zip` | Identifiability, basis sensitivity, independent rollout, rollout refinement, and diagnostic report | 5,076 |
| `fix3-evidence-v1.zip` | Off-basis truth verification, development, frozen evaluation, and analysis | 1,799 |

## Verify and restore

Download the three ZIPs from the release assets into a folder named
`heat-evidence-downloads` next to the repository checkout. Use the source at the
release tag. Its `docs/evidence_release_v1/release_manifest.json` lists archive
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

The complete local restoration passed on Windows. The restore receipt's manifest
digest was checked against the committed manifest at `7b21bfc`. The restore
utility and its four filesystem tests were also checked using the Git snapshot
at that commit. The release manifest records its preparation state, while the
verification procedure checks the bytes readers obtain from the release.

The verifier checks archive digests, ZIP CRCs, archive inventory, and recorded
output hashes. The original training and analysis scripts retain their historical
environment and path requirements. Use the restored directory to inspect the
saved evidence. Portable analysis replay is a separate step.

The [source provenance addendum](../source_provenance_addendum_v1/README.md)
documents the correspondence between the original freeze hashes and the Git
source bytes. The primary timing interaction remains inconclusive, and the
off-basis evaluation concerns one localized initial field. The archives preserve
the original study stages and include the separately recorded rollout refinement.
