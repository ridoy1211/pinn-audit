# Reproducibility audit notes - 2026-11-03

Filename note: DAY1 refers to the first session of the current supervision plan,
not the beginning of the existing PINN Audit project. The October lab log and
original experiment history remain in place.

Review date: 3 November 2026 (Bangladesh).
This template, verified execution facts, and draft interpretation were prepared by
the research assistant. The user subsequently reported reading both papers and
provided five-question answers in chat on 3 November 2026; those are consolidated
below with attribution. Personal mathematical explanations remain to be recorded.

## Verified execution record

- Python reported in the user's CMD transcript: 3.14.6.
- Project dependency installation: completed successfully in .venv.
- Environment snapshot: day1_environment.txt.
- Tests: 180 passed in 55.22 seconds, according to the supplied transcript.
- Pilot: results/my_classical_pilot_v1.
- Pilot records checked by the assistant: 50 fits, ten observation files, forty
  parameter profiles, no missing or duplicate jobs.
- The pilot contains classical methods only; no PINNs were trained.
- Data seeds: 0 through 4, all development datasets.
- Full review: audit/user_run_evidence/day1_pilot_review.md.

## Assistant-drafted interpretation of the pilot

The pilot reproduced successfully. At N=10, lambda=1e-6 reduced median relative
diffusivity error from 25.40% to 12.10% compared with the unregularized spectral
estimator. However, mean error increased from 21.71% to 27.16%, and the largest
observed error increased from 40.71% to 98.00%. The regularized estimator improved
on three of five datasets and worsened on two. One estimate reached the lower
diffusivity search bound. Therefore, the lower median does not demonstrate
consistently better recovery. These are exploratory development results; no PINN
superiority, independent confirmation, or optimal regularization claim is supported.

## My understanding of the mathematics -- fill after studying

1. What does u_t = alpha*u_xx describe, and what does alpha mean?

My explanation:

2. Why do higher-frequency sine modes decay faster?

My explanation:

3. What are the seven unknown parameters in the classical pilot?

My explanation:

4. Which quantities does the regularization penalty act on, and how can it help or hurt?

My explanation:

5. Why can a single-time snapshot leave alpha ambiguous when initial amplitudes are free?

My explanation:

## Paper 1: Grossmann et al. -- user-reported reading notes

Reading reported by the user in chat on 3 November 2026. Answers below are
consolidated from that submission by the assistant, rather than verbatim quotations.

Title: Can physics-informed neural networks beat the finite element method?
Published DOI: https://doi.org/10.1093/imamat/hxae011
Preprint used: https://arxiv.org/html/2302.04107 (this HTML identifies itself as v1).

1. What problem does it investigate?

Whether standard PINNs compete with FEM on forward PDE accuracy, solution
construction time, and subsequent evaluation time.

2. What did it do differently from earlier work?

It contributes a systematic numerical comparison across PDE families, mesh
resolutions, and network architectures, rather than a new neural architecture.

3. How did the researchers test their approach?

Poisson in 1D/2D/3D, Allen-Cahn in 1D, and semilinear Schrodinger in 1D/2D;
FEniCS FEM versus dense PINNs using Adam then L-BFGS. Poisson has analytical
references; time-dependent cases use fine-FEM references. Common evaluation
meshes and relative discrete L2 errors are used. Ten executions support averaged
timings, not a documented set of ten independently seeded accuracy trials.
FEM uses CPU and PINN training uses GPU (Section 3.1).

4. What evidence supports the conclusions?

The user points to Figure 2(a) for 1D Poisson construction time versus error and
Figure 6(a-b) for 3D Poisson construction versus evaluation behavior. Verify those
locators in the exact version cited in the future manuscript.

5. What limitations or remaining questions exist?

Selected low-dimensional forward problems, primarily vanilla PINNs, hardware and
interpolation dependence, and numerical references in some cases. It does not
answer our joint diffusivity/unknown-initial-field inverse question.

Connection to our project: strong numerical baselines, common scoring, and separate
construction/evaluation costs are essential.

## Paper 2: Zapf et al. -- user-reported reading notes

Reading reported by the user in chat on 3 November 2026. Answers below are
consolidated from that submission by the assistant. Detailed supplementary claims
are recorded separately below and are not all independently rechecked here.

Title: Investigating molecular transport in the human brain from MRI with
physics-informed neural networks.
Published DOI: https://doi.org/10.1038/s41598-022-19157-w
Published full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC9474534/

1. What problem does it investigate?

Estimate constant apparent diffusion D in c_t = D*Laplacian(c) using tracer MRI
at 0, 7, 24, and 46 hours in a 3D brain subregion.

2. What did it do differently from earlier work?

The user identifies the application to temporally sparse, unsmoothed measurements
in complex geometry, compared with earlier ODE/1D approaches and the Valnes FEM
inference precedent. The authors qualify their novelty claim.

3. How did the researchers test their approach?

Synthetic clean/noisy observations and real MRI; PINN loss-norm/weight choices and
adaptive residual-point sampling; FEM inference with a regularized unknown boundary.
PINN uses the first image through data loss and no explicit boundary residual.
FEM fixes c(x,0)=0 while optimizing D and boundary g(x,t) (Equations 4-9).
These are different information formulations, not unrestricted unknown-IC recovery.
Table 1 averages four trainings; do not infer independent noise realizations.

4. What evidence supports the conclusions?

For the specified bounded-D, w_r=1, decaying-learning-rate example, Table 1 gives
72% coefficient error for p=2 versus 2% for p=1, with reported residual norms
0.77 versus 0.0057. Figures 5-6 show observation-time residual spikes and adaptive
sampling behavior. This demonstrates configured sensitivity, not general superiority.

5. What limitations or remaining questions exist?

A simplified scalar diffusion model, limited patient evidence, tuning sensitivity,
and hardware-specific timing. Agreement with FEM for MRI is not independently
known parameter accuracy; a small residual does not prove inverse uniqueness.
Our joint unknown-initial-field task is different.

User-reported supplementary details to retain and verify by page/table before reuse:

- Synthetic FEM truth D0=0.36 mm^2/h, about one million cells and 16-minute steps.
- Clipped additive noise max(0,c+eta), eta Gaussian with SD 0.05, described relative
  to simulated peak concentration rather than each voxel value.
- PINN: nine hidden layers, 64 neurons each, tanh, Adam.
- FEM: linear elements, Crank-Nicolson, adjoint gradients, L-BFGS.
- Supplement S1 timings: FEM about 45-50 h; PINN refinement 20 h; exchange 15 h,
  on different CPU/GPU hardware. These are not portable method speed ratios.
- Five-run clean/MRI subsets versus four trainings in main Table 1.

Connection to our project: explicitly match priors and information, record noise
scaling and replication types, and evaluate physical parameters independently of
training losses. Supervisor checks: audit/user_run_evidence/day1_paper_review.md.

## Questions for the next supervision discussion

-

## Decisions and next work

- Preserve the original pilot folder and report every declared setting.
- Do not choose the final lambda by minimizing error against the known alpha.
- Investigate the bound-limited estimate and observation-design sensitivity.
- Establish a proper Git checkout and freeze the final protocol before confirmation.
- Additional decisions made today, with reasons:
