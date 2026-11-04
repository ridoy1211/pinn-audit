# Assessment of the supplied SciSpace review — 5 November 2026

## Verdict in simple language

The report is useful for orientation. It names relevant papers and recognizes
why regularization, representation and training settings matter. It does not
prove that our question is a significant unfilled research gap, and its text
should not be copied into the manuscript without correction. It supplies no
new experimental validation of our results.

This assessment checks the supplied text against the five named primary studies
and our frozen study configuration. It is not a new exhaustive literature search.
Existing five-question paper notes remain in `audit/literature_review.md`.

## 1. The gap claim is too certain

Not finding the exact combination in these five papers shows a difference from
those papers. It does not establish absence across the literature, scientific
importance, or journal-level novelty. Adding familiar ingredients is not enough
unless the resulting study provides useful new evidence or understanding.

The candidate contribution remains a narrow comparative audit. It is reasonable
to investigate the question, but publication suitability is still unresolved.
Our existing `NOVELTY_ASSESSMENT.md` should retain its cautious wording.

## 2. Basis capacity and network architecture are different here

The report equates basis capacity with network architecture. Our declared
capacity control is **six versus twelve classical sine modes**. Both neural
budgets use **three hidden layers of 32 units**; the network size is fixed.
We varied the L-BFGS iteration cap, not neural capacity. We cannot claim a
network-width/depth study, an optimal crossover threshold, or a general functional
relationship from these few settings.

Our classical estimator is spectral variable projection with an explicit
initial-field penalty. It is not FEM or FDM. The study is source-free, with
zero Dirichlet boundaries and an unknown initial field. The report's general
source-term equation and full data/PDE/BC/IC loss are background formulas, not
the actual specification of our experiment. Our PINN has exact zero boundaries
by construction and no initial-condition loss.

## 3. Several conclusions cross between different problems

Grossmann's comparison concerns forward Poisson, Allen–Cahn and Schrödinger
problems; its relative errors and construction/evaluation timings do not settle
joint heat inversion. [Primary author text](https://arxiv.org/html/2302.04107)

Jekic's inverse examples concern fluid viscosity. The quoted mean prediction
RMSE values 0.0013 and 0.01 are reported for 2D Taylor–Green with sigma=1.
They are not diffusivity-recovery errors or heat initial-field errors. They
cannot establish classical superiority for our task, or general dimensional
scaling from a controlled complexity study. [Primary v2](https://arxiv.org/html/2509.20191v2)

The report's conclusion that classical methods are already more accurate and
efficient for the specific joint problem is therefore unsupported by its cited
comparisons. It also sits uneasily beside its claim that direct comparisons
are missing. Our own results must answer our comparison, condition by condition.

## 4. Zhao is not just a theoretical precedent

Zhao and colleagues supply an algorithm and numerical simulations, using
boundary control, boundary measurement and a distributed temperature snapshot.
This is important prior work on joint recovery under different information.
Describing it as theoretical only, or as using only boundary measurements,
misses material parts of the study. [Author PDF](https://lsc.amss.ac.cn/~bzguo/papers/Zhaozx5heat.pdf)

## 5. Physics penalties are not guarantees

A neural field trained to reduce a PDE residual does not automatically satisfy
the equation everywhere, recover the correct coefficient or remove inverse
instability. In our network, exact boundary enforcement is distinct from the
approximately minimized PDE residual. Finite architecture, optimization, loss
weights and stopping choices also constrain the fitted solution.

Oliveira reports reconstruction without an added explicit regularization term
for a specified-diffusivity initial-state problem. That is evidence for its
examples, not proof that every PINN needs no stabilization or that joint
diffusivity/initial-state recovery is generally reliable.
[Publisher text, Sections 2–3](https://doi.org/10.1007/s40314-026-03818-x)

Zapf investigates PDE weights, residual norms and adaptive residual sampling.
Its noisy examples can improve with increased PDE weight or an L1 PDE loss;
do not describe L1 and adaptive weighting as universally required. The PINN
uses the first image through data loss, whereas FEM fixes the initial state
to zero and estimates a regularized boundary field. Agreement with FEM on
real MRI is not independent knowledge of true diffusivity.
[Author text, Figures 3 and Methods 5.2–5.3](https://arxiv.org/html/2205.02592v2)

## 6. The reference system needs repair

The body lists nine labels but repeatedly lists the same five studies, sometimes
as parsed full text. The final six-entry bibliography also duplicates Grossmann.
For example, body citation 2 denotes Oliveira while final reference 2 denotes
Grossmann. Body citation 3 has the opposite mismatch. Citation 68 has no listed
entry. These problems make the current numerical citations unreliable even
where a sentence's underlying idea is sound.

Use one reference per distinct work/version, with consistent author, year, title,
publication status and DOI or versioned arXiv link. Cite the original study,
rather than the SciSpace summary, for scientific claims. Do not add the summary
as an independent study in our evidence count.

## What to retain

- Separate diffusivity recovery, initial-state recovery and field prediction.
- State regularization, representation assumptions and optimizer budgets.
- Preserve unfavorable outcomes and report uncertainty and timing definitions.
- Position our work against joint classical inversion, noisy inverse benchmarking
  and neural initial-condition reconstruction, rather than claiming these are new.

## Safer manuscript wording

> The reviewed studies establish precedents for joint heat inversion, neural
> initial-state reconstruction and PINN–classical comparisons, but investigate
> different measurement and inference settings. We examine a controlled joint
> heat-inversion benchmark using explicit initial-field regularization, two
> classical basis widths and two fixed neural optimization budgets. We report
> diffusivity, initial-field and full-field errors separately. The contribution
> is a limited audit of comparative sensitivity, not a claim that these ingredients
> are new or that no prior study considers their intersection.

## Action for this project

Keep the completed experiment and archived results unchanged. Use the report as
a reading aid, with the corrections above. Continue the manuscript with accurate
citations and bounded claims. This review alone justifies neither a publication
promise nor additional experiments aimed at obtaining a preferred winner.
