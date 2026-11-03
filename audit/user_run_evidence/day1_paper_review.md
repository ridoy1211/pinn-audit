# Supervisor review of the user's two paper-reading records

Reviewed 3 November 2026. The user reports reading both papers and supplied
five-question answers in chat. Those answers are consolidated in DAY1_NOTES.md
with attribution. This review checks important methodological claims; it does not
claim every supplementary detail has been independently reverified in this pass.

## Grossmann et al.

Checked against Section 3.1 of the [arXiv v1 HTML](https://arxiv.org/html/2302.04107).
It distinguishes analytical Poisson references from fine-FEM references for the
time-dependent equations, separates construction and evaluation time, and reports
ten timing executions. The note correctly avoids treating these as ten explicitly
independent accuracy seeds. CPU FEM and GPU PINN timings are implementation-specific.

Record the exact version used. The accessible HTML labels itself arXiv v1 (2023);
the [published article](https://doi.org/10.1093/imamat/hxae011) is dated 2024. Check
figure numbering in the cited version before manuscript use.

## Zapf et al.

The [published main text](https://pmc.ncbi.nlm.nih.gov/articles/PMC9474534/) confirms
the information distinction: the PINN includes the initial image in data loss and
omits an explicit boundary residual; FEM fixes the initial state at zero and
regularizes an optimized boundary field (Equations 4-9). Table 1 confirms four
trainings and the stated 72% versus 2% coefficient-error example. This is a
particular configured experiment, not a transferable accuracy guarantee.

The supplement did not load through the research tool in this review. Its detailed
noise, architecture, discretization, runtime, and repeat counts remain recorded as
user-reported supplementary readings, pending archival/page-level verification.

## Corrections to earlier shorthand in the assistant's material

The observed initial image describes Zapf's PINN data treatment, not the FEM
initial condition. The FEM formulation fixes c(x,0)=0. Summaries that grouped both
approaches under "observed initial field" should use this more precise distinction.
The conclusion that this paper does not test our joint unknown-IC comparison is
unchanged.

## Decisions for our study

1. Match information, boundary/initial assumptions, parameter bounds, and tuning
   access explicitly. Giving the same observations alone does not establish fairness.
2. Separate dataset/noise replication, neural training restarts, and timing repeats.
3. Label exact truth and numerical reference solutions distinctly.
4. Use independent parameter/field errors, profiles, failures, and computational cost.
   A small PDE residual does not establish a unique or accurate inverse solution.
5. In our exact spectral heat family, every candidate alpha with its amplitudes
   satisfies its corresponding heat PDE. The residual alone therefore cannot
   certify the recovered alpha. Observation information and prior assumptions matter.
6. Continue the Day 2 bounds/search diagnostic before selecting final priors or
   running confirmation data. It remains post-hoc exploratory work.

## Citation cleanup

Replace generic trailing labels such as "PMC" or "Oxford Academic" with a paper DOI
and a section/table/figure locator. Keep quotations brief and identify the version.
Do not convert agreement with a numerical comparator on real data into known truth.
