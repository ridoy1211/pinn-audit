# Contribution assessment — 5 November 2026

## Supervisory judgment

The project is a completed computational audit with reproducible evidence.
Its potential journal contribution is **narrow and incremental**. Reproducibility
supports credibility; it does not independently establish novelty or acceptance.
The current evidence is sufficient to describe the fixed study, but journal
submission should follow a final scope/contribution assessment of the assembled
manuscript. No favorable result should be manufactured by tuning evaluation data.

## What existing work already covers

| Existing work | Established overlap | Relevant distinction in this study |
|---|---|---|
| [Zhao et al. (2017)](https://doi.org/10.1002/mma.4245) | Joint diffusivity/source/initial-state inversion | Sparse noisy interior-only observations and a neural comparison |
| [Grossmann et al. (2024)](https://doi.org/10.1093/imamat/hxae011) | PINN–classical accuracy/cost comparison | Joint inverse task rather than forward solution construction |
| [Zapf et al. (2022)](https://doi.org/10.1038/s41598-022-19157-w) | Sparse/noisy diffusion-coefficient PINN–FEM comparison | Unknown initial field; different boundary and measurement formulation |
| [Jekic et al., v2 (2026)](https://arxiv.org/html/2509.20191v2) | Noisy inverse benchmarking and training problems | Heat joint-state/parameter task and explicit IC basis controls |
| [Oliveira et al. (2026)](https://doi.org/10.1007/s40314-026-03818-x) | Physics-informed initial-temperature reconstruction | Diffusivity is also estimated here |
| [Marques et al., NEXT (2026)](https://arxiv.org/html/2609.31539v1) | Neuro-spectral inverse PDE experiments | Our study audits conventional procedures rather than proposing a hybrid architecture |

These distinctions are not a claim that no other paper covers their intersection.
The targeted search and reading history are in `audit/literature_review.md` and
`audit/literature_update_2026-11-05.md`. Coverage is not exhaustive.

## Candidate contribution statement

> We provide a prospectively fixed, reproducible audit of joint diffusivity and
> initial-state heat inversion, showing how the comparison of specified PINN
> procedures with spectral inversion changes across explicit regularization,
> initial-state basis capacity, observation count and optimizer-budget choices.

This statement is supported as a description of our study. Its originality and
importance relative to all prior work remain subject to critical review.
There was no formal preregistration, matched tuning-budget study, full sensitivity
analysis, new solver or general theorem.

## Why a reviewer might still find the study too limited

- The underlying equation and inverse ingredients are well established.
- The small synthetic matrix may illustrate familiar bias/regularization issues
  rather than establish a sufficiently new general result.
- The spectral truth favors the reference, while unequal tuning limits a broad
  fairness claim. Earlier proposals for matched tuning were not implemented.
- Ten designs per stratum, three neural starts and pointwise intervals limit
  inferential precision. No formal factor-interaction analysis was declared.
- The seventh-mode error floor is standard orthogonal projection, not new theory.

## Recommended action

Finish a short, candid manuscript around the audit and have its contribution
critically reviewed. Then assess journal scope and mandatory fees using current
official policies. A preprint or research report remains an option if the result
is judged too incremental for the target journal; neither is a substitute for
peer review. New experiments should address a specific unresolved scientific
question through an amended protocol and fresh data, rather than seek a preferred
winner. There is no need to repeat the completed fixed experiment merely to
write it accurately.
