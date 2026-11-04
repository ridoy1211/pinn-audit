# Targeted literature update — 5 November 2026

This supplements the earlier five-question review. It is a bounded search,
not a systematic review or proof that a combination of study features is unique.

## Search and access record

Queries included:

- `"physics informed" "diffusion coefficient" "unknown initial" comparison regularization`
- `"physics-informed" "heat" "initial condition" "variable projection"`
- `"heat" "PINN" "initial condition" "diffusivity" comparison regularized`
- `"simultaneous" "initial condition" "PINN" "diffusion coefficient"`
- `"unknown initial" "PINNs" "classical" heat`

Primary-source checks revisited Grossmann's arXiv text (2302.04107v1), Zhao's
author-hosted PDF (DOI 10.1002/mma.4245), Jekic's arXiv v2 (2509.20191v2),
Oliveira's publisher text (10.1007/s40314-026-03818-x), and Zapf's arXiv v2
(2205.02592v2). Their five-question entries remain in the original review.
Nature's HTML route was inaccessible during this update; the arXiv author text
was available. The new paper below was screened at targeted-section level.

An F1000Research search lead at `/articles/15-1789` was inaccessible (403).
It was not read or used to support the gap claim. Other search-result leads
were not promoted to evidence without primary-text verification.

## New relevant paper: five questions

**Marques et al. (2026), NEXT: Physics-Informed Neuro-Spectral Exponential Time
Differencing Architectures.** [arXiv v1](https://arxiv.org/html/2609.31539v1).
Preprint submitted 25 September 2026. Consulted introduction, formulation,
inverse experiments and relevant appendix details; not a full replication.

1. **Problem:** Stable neural representations of stiff PDE dynamics and inverse tasks.
2. **Difference:** Combines a spectral neural formulation with exponential time differencing.
3. **Testing:** Forward PDE benchmarks, Burgers coefficient recovery from two snapshots,
   and boundary heat-flux inversion from two sensor locations.
4. **Evidence:** Table 2 reports Burgers coefficient estimates over ten starts;
   Figure 7 reports heat-flux reconstruction, with relative heat-flux RMSE 0.035.
5. **Limits/questions:** The heat example has a prescribed uniform initial temperature
   and estimates boundary flux. It does not directly test our unknown-IC/scalar-
   diffusivity comparison. Its results do not establish performance for that task.

**Implication:** Do not claim combining spectral physics and neural inversion
is new. This paper does not, by itself, certify or eliminate our narrower audit's
novelty. The comparison matrix and limitations in `docs/NOVELTY_ASSESSMENT.md`
must guide manuscript wording.
