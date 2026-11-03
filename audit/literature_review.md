# Literature review: PINNs for sparse, noisy heat diffusivity inversion with an unknown initial condition

**Evidence cutoff:** 3 November 2026. **Prepared for:** proposed extension of `ridoy1211/pinn-audit` (smooth 1D heat/wave forward tests and inverse diffusivity with known/unknown IC).

## Scope and reading note

The target claim is narrow: *does a PINN recover constant heat diffusivity better than a carefully regularized classical inverse solver when the temperature observations are sparse and noisy and the initial field is unknown?* The existing project appears to compare PINN and finite difference in smooth 1D forward heat/wave cases, and to invert diffusivity with known/unknown initial conditions. That gives a useful starting scaffold, but it does not by itself answer the harder inverse benchmark question.

I searched primary papers and official publisher/proceedings pages, including 2026 material indexed by the cutoff. Full text read: the two Raissi companion preprints corresponding to the 2019 JCP framework (Parts I and II); full arXiv HTML for Grossmann, Rathore, PINNacle benchmark, Jekic et al. (including appendices/results), Lau et al. (PINNACLE point selection), and Salmaniw & Browning; full publisher HTML for Oliveira et al. (2026); author-hosted complete PDF text for Zhao et al. (2017, 2021); and full article text for Zapf et al. (2022) and Li et al. (2024). The 2022 PMC page intermittently served a CAPTCHA, so Zapf's article text was read from the primary publisher/arXiv-indexed article and author-hosted material rather than the PMC page itself. The 1999 classical IC paper was available at abstract level only. I also checked the primary publisher record for the 2026 heat sensor-placement paper. Where a lead was abstract-only or inaccessible, this is stated. Coverage is targeted, not an exhaustive census of all 2026 work.

## Executive assessment

There is strong direct precedent for estimating scalar diffusivity from sparse/noisy data with PINN and FEM (Zapf et al., 2022), and separate heat-equation work reconstructs unknown initial fields with Tikhonov/other regularizers and, now, PINN variants. There is also classical 1D work that jointly identifies diffusivity and the initial field (Zhao et al., 2017) under designed boundary excitation. These substantially narrow the gap: a general claim that PINNs uniquely enable noisy diffusivity inversion, unknown-initial-condition recovery, or heat-equation inversion is not defensible. In this targeted search, I did not find a head-to-head study of the **specific joint task**—1D constant diffusivity plus unknown initial field, sparse noisy interior temperature observations, and explicit classical IC regularization—with matched tuning/compute budgets. This remains a bounded search finding, not a novelty certification.

Given the stated one-week / 56-hour project window, 16 GB RAM, CPU-only Intel UHD 620 laptop, and optional Colab access, a defensible contribution is a tightly scoped controlled comparison, not a broad solver survey. Start with one low-order IC basis, a small prespecified set of sensor/noise cases, a forward finite-difference solver plus a regularized coefficient fit, and the existing PINN. Reserve Adam→L-BFGS as the sole PINN optimizer sequence; NNCG is not required. Keep the data, noise draws, parameter bounds, and a small declared tuning budget matched. Report diffusivity error, IC/prediction error, uncertainty or sensitivity, convergence failures, and elapsed time. Expand only if the core comparison completes within the time budget.

## Closest papers (five questions per paper)

### 1. Foundational inverse-PINN framing

**Maziar Raissi, Paris Perdikaris, George Em Karniadakis (2019), “Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.”** *Journal of Computational Physics* 378, 686–707. [DOI](https://doi.org/10.1016/j.jcp.2018.10.045) · [author PDF](https://faculty.sites.iastate.edu/hliu/files/inline-files/PINN_RPK_2019_1.pdf)

1. **Problem:** Introduces PINNs for PDE solution and unknown-physics identification from incomplete data.
2. **Difference:** Part I presents PDE/data losses; Part II makes physical coefficients trainable alongside the field. Neither compares regularized inverse heat solvers.
3. **Testing:** Part II’s Burgers inverse case uses 2,000 random space-time samples, a 9×20 network, L-BFGS, and varies 500–2,000 observations and noise up to 10%; it also treats 2D Navier–Stokes. Part I tests forward Burgers and nonlinear Schrödinger systems.
4. **Evidence:** For Burgers at 2,000 points and 1% noise, coefficient errors are 0.039% and 0.008%; at 10% noise they are 0.101% and 6.391%. Their accuracy is example- and noise-dependent.
5. **Limitations/reading:** Prescribed initial/boundary data and Burgers physics do not establish unknown-IC heat superiority. **Read:** full HTML of both companion preprints.

### 2. Classical joint heat coefficient/initial-state identification

**Zhi-Xue Zhao, Mapundi K. Banda, Bao-Zhu Guo (2017), “Simultaneous identification of diffusion coefficient, spacewise dependent source and initial value for one-dimensional heat equation.”** *Mathematical Methods in the Applied Sciences* 40, 3552–3565. [DOI](https://doi.org/10.1002/mma.4245) · [author PDF](https://lsc.amss.ac.cn/~bzguo/papers/Zhaozx5heat.pdf)

1. **Problem:** Jointly recover constant 1D diffusivity, a spatial source, and an unknown initial state.
2. **Difference:** This classical inverse/control method estimates diffusivity spectrally, represents the IC by Fourier modes, and uses finite differences plus truncated SVD/GCV for source reconstruction. It shows regularized numerical inversion predates PINN comparisons.
3. **Testing:** Uses designed boundary input and boundary trace plus a distributed temperature snapshot; Crank–Nicolson simulations test 1% and 3% multiplicative errors.
4. **Evidence:** Reconstructions are accurate in the designed setup. The authors show a boundary trace alone cannot separate source and initial-state effects; distributed data and control matter. Thus the study is direct precedent for joint D/IC recovery, while leaving sparse interior-only performance open.
5. **Limitations/reading:** Rich measurements, designed excitation, and an unknown source differ from sparse interior-only data. **Read:** full author PDF, including analysis and numerical examples.

### 3. PINN versus FEM (forward comparison)

**Tamara G. Grossmann, Urszula Julia Komorowska, Jonas Latz, Carola-Bibiane Schönlieb (2024), “Can physics-informed neural networks beat the finite element method?”** *IMA Journal of Applied Mathematics* 89(1), 143–174. [DOI](https://doi.org/10.1093/imamat/hxae011) · [full arXiv HTML](https://arxiv.org/html/2302.04107)

1. **Problem:** Compare PINNs and FEM for forward PDE accuracy and computational cost.
2. **Difference:** Covers 1D/2D/3D Poisson, 1D Allen–Cahn, and semilinear Schrödinger problems, separating solve/training time from evaluation time.
3. **Testing:** Compares FEniCS mesh refinement with tanh MLPs, multiple architectures/residual samples, Adam then L-BFGS, and common evaluation grids.
4. **Evidence:** FEM is generally faster and at least as accurate for solution construction; PINN evaluation can be faster in some cases. Allen–Cahn fronts need larger networks, illustrating a problem-dependent trade-off. Their common-grid errors and separate timing measures make clear that query speed and solve speed answer different questions.
5. **Limitations/reading:** Forward tests do not answer noisy inverse diffusivity or unknown ICs; they motivate strong numerical baselines and clear timing. **Read:** full arXiv HTML.

### 4. PINN optimizer quality

**Pratik Rathore, Weimu Lei, Zachary Frangella, Lu Lu, Madeleine Udell (2024), “Challenges in Training PINNs: A Loss Landscape Perspective.”** *ICML 2024*, PMLR 235, 42159–42191. [PMLR](https://proceedings.mlr.press/v235/rathore24a.html) · [full arXiv HTML](https://arxiv.org/html/2402.01868)

1. **Problem:** Study conditioning and optimization difficulties in PINN training.
2. **Difference:** Compares Adam, L-BFGS, their sequence, and NysNewton-CG; relates differential-operator conditioning to loss behavior and the PINN loss landscape.
3. **Testing:** Experiments cover convection, reaction, and wave equations, optimizer/width choices, Hessian spectral analyses, and NNCG fine-tuning; no inverse heat case.
4. **Evidence:** In the wave table, L2RE is 0.349 for Adam, 0.335 for L-BFGS, and 0.0552 for Adam→L-BFGS; NNCG lowers it to 0.0127 but costs 322× per iteration there. Their analysis attributes optimization difficulty partly to differential-operator conditioning and uneven gradients across loss components.
5. **Limitations/reading:** These results motivate recording optimization cost and errors, not requiring NNCG in the proposed small study. Their demonstrations are forward PDE tasks, so do not assume the gains transfer quantitatively to inverse heat. **Read:** full HTML, including optimizer experiments.

### 5. PINN benchmark coverage

**Zhongkai Hao, Jiachen Yao, Chang Su, Hang Su, Ziao Wang, Fanzhi Lu, Zeyu Xia, Yichi Zhang, Songming Liu, Lu Lu, Jun Zhu (2024), “PINNacle: A Comprehensive Benchmark of Physics-Informed Neural Networks for Solving PDEs.”** *NeurIPS 2024*, Datasets and Benchmarks Track. [Official paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8c63299fb2820ef41cb05e2ff11836f5-Abstract-Datasets_and_Benchmarks_Track.html) · [arXiv](https://arxiv.org/abs/2306.08827)

1. **Problem:** Benchmark PINN variants across PDEs, scales, and geometries.
2. **Difference:** Benchmarks 10 PINN variants across 22 tasks, including HInv, a 2D inverse diffusion coefficient-field problem.
3. **Testing:** HInv uses a manufactured source/solution and 2,500 random space-time observations with Gaussian noise N(0, 0.1); the benchmark compares PINN variants, not regularized FEM.
4. **Evidence:** HInv coefficient-field L2RE is 1.57 for vanilla PINN, 0.0526 with loss reweighting, and 0.0119 for vPINN. This large spread shows that a default PINN result is not a reliable stand-in for a tuned neural baseline.
5. **Limitations/reading:** It has known manufactured IC/BC, a 2D variable coefficient, and dense data; no unknown IC or classical comparator. Its strong vPINN result also means a vanilla-PINN-only comparison would be inadequate. **Read:** full arXiv HTML, including appendix/results.

### 6. Noisy inverse PINN/FEM comparison (fluid systems)

**Aleksandra Jekic, Afroditi Natsaridou, Signe Riemer-Sørensen, Helge Langseth, Odd Erik Gundersen (2025; v2, 28 Aug 2026), “Examining the robustness of Physics-Informed Neural Networks to noise for Inverse Problems.”** [Full arXiv HTML](https://arxiv.org/html/2509.20191) · [record](https://arxiv.org/abs/2509.20191)

1. **Problem:** Test whether PINN noise robustness holds against numerical inverse baselines.
2. **Difference:** Compares PINN, PINN/FEM hybrid, and FEM/SLSQP for viscosity identification in Burgers and 2D/3D Taylor–Green/Navier–Stokes.
3. **Testing:** Reports multiple noise levels, 30 starts for 1D Burgers and 2D vortex, error/spread, parameter bias, and runtime; uses adjoints for 1D FEM and finite-difference gradients in 2D/3D.
4. **Evidence:** At 2D Taylor–Green and σ=1, FEM/SLSQP mean RMSE is 0.0013 versus 0.01 for the best PINN. About 20% of 3D FEM runs fail catastrophically. Their results therefore include both accuracy and robustness/runtime trade-offs rather than a uniform winner.
5. **Limitations/reading:** Fluid equations and known IC/BC differ; authors call their baseline not extensively tuned, so it is not a strongest-baseline verdict. **Read:** full v2 HTML and appendices.

### 7. Initial-condition-dependent identifiability

**Yurij Salmaniw, Alexander P. Browning (2025), “Structural identifiability of linear-in-parameter parabolic PDEs through auxiliary elliptic operators.”** *Journal of Mathematical Biology* 91(1), article 4. [DOI/full text](https://doi.org/10.1007/s00285-025-02225-w) · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12122637/)

1. **Problem:** Analyze structural and practical identifiability of parabolic PDE parameters with unknown IC components.
2. **Difference:** Relates identifiability to an auxiliary elliptic problem and profiles nuisance IC coefficients to study their effect on parameter inference.
3. **Testing:** Gives analytical results and noisy likelihood/profile-likelihood analysis for reaction–diffusion with an unknown eight-term eigenfunction IC and physical parameters.
4. **Evidence:** In that example, a special dominant eigenfunction IC causes structural non-identifiability; perturbation restores it, while nearby cases remain practically weakly identifiable. Diffusion damps higher modes, and profile likelihoods show nuisance IC coefficients widening uncertainty in physical parameters.
5. **Limitations/reading:** The reaction parameter and full-grid data make this result specific to reaction–diffusion; do not transfer structural non-identifiability to pure heat. Use it to motivate a separate heat sensitivity/profile analysis. **Read:** full PMC text.

### 8. Closest scalar-diffusivity PINN/FEM precedent

**Bastian Zapf, Johannes Haubner, Miroslav Kuchta, Geir Ringstad, Per Kristian Eide, Kent-Andre Mardal (2022), “Investigating molecular transport in the human brain from MRI with physics-informed neural networks.”** *Scientific Reports* 12, 15475. [DOI/full article](https://doi.org/10.1038/s41598-022-19157-w) · [arXiv](https://arxiv.org/abs/2205.02592)

1. **Problem:** Estimate constant apparent diffusivity from sparse MRI observations of tracer spread in a 3D brain domain.
2. **Difference:** Direct PINN/FEM comparison; PINN uses data fit and PDE residual, takes measured t=0 MRI as initial data, and has no separate boundary-condition residual in its loss.
3. **Testing:** Uses four snapshots (0, 7, 24, 46 h), noisy/noiseless synthetic cases, loss-weight/norm tuning, adaptive residual refinement, and real MRI.
4. **Evidence:** When tuned, both recover synthetic D within a few percent from three images. Real-data PINN gives 0.41 mm²/h versus FEM 0.50; estimates align when PINN PDE residual is small. PINN runtime is about 2× faster in their GPU setup.
5. **Limitations/reading:** The PINN uses initial-image data; FEM fixes a zero initial state and regularizes an unknown boundary field. This differs from joint unknown-IC inversion. The 3D clinical setting and unmatched formulations limit transfer. **Read:** full publisher/article text; PMC access intermittently CAPTCHA-blocked.

## Additional selected papers (five questions each)

### 9. PINN for composition-dependent diffusion

**Dongchen Li, Bin Yan, Tianya Gao, Guowei Li, You Wang (2024), “PINN Model of Diffusion Coefficient Identification Problem in Fick’s Laws.”** *ACS Omega* 9(3), 3846–3857. [Full PMC text](https://pmc.ncbi.nlm.nih.gov/articles/PMC10809368/)

1. **Problem:** Identify concentration-dependent diffusion coefficients in Fick-law systems.
2. **Difference:** Couples concentration/flux networks to Fick-law residuals; the paper fits a spatially varying coefficient and is not heat conduction or a classical-solver comparison.
3. **Testing:** One example uses 505 flux and 505 concentration samples with an 80/20 split; others vary missing flux/gradient data, sample size, data errors, and initial guesses.
4. **Evidence:** Estimates depend on observations and sample choice; reported runs take about 2,000–3,000 iterations, and one invalid-data-heavy sampling strategy fails tolerance. In one case a cubic spatial coefficient representation is fitted from measured flux and concentration profiles.
5. **Limitations/reading:** Constitutive diffusion differs from heat diffusivity and unknown temperature IC, and its input data include flux/concentration rather than sparse temperature snapshots. **Read:** full PMC text.

### 10. Classical regularization for backward heat recovery

**Wagner Barbosa Muniz, Haroldo Fraga de Campos Velho, Fernando Manuel Ramos (1999), “A comparison of some inverse methods for estimating the initial condition of the heat equation.”** *Journal of Computational and Applied Mathematics* 103(1), 145–163. [Publisher record](https://doi.org/10.1016/S0377-0427(98)00249-0)

1. **Problem:** Compare methods for recovering an initial condition from heat-equation data.
2. **Difference:** Abstract describes two explicit methods and least-squares methods using finite-difference and integral forward formulations, then examines Tikhonov regularization.
3. **Testing:** Abstract indicates a numerical comparison of these formulations; full article was inaccessible in this review.
4. **Evidence:** The abstract says explicit methods produce disastrous results without some regularization, supporting classical IC-regularization precedent.
5. **Limitations/reading:** No PINN or joint D/IC conclusion can be inferred. **Read:** abstract only; this is a lead, not deeply reviewed evidence.

### 11. PINN reconstruction of a backward heat initial condition

**Igor A. R. Oliveira, Daniel A. S. Mendes, Haroldo F. Campos Velho, Eduardo S. Pereira (2026), “Solving backward heat transfer by physics-informed machine learning.”** *Computational and Applied Mathematics* 45, 421. [Full publisher article](https://doi.org/10.1007/s40314-026-03818-x)

1. **Problem:** Recover a 1D initial temperature from one future-time profile for insulated heat flow.
2. **Difference:** Compares standard PINN, multiscale Fourier PINN, and KAN without an explicit regularizer; diffusivity stays fixed.
3. **Testing:** Uses the same final-state observations at 0%, 1%, 5%, and 10% multiplicative noise; reports training time rather than matched parameter counts.
4. **Evidence:** For a triangular IC, KAN MSE is about 4×10⁻⁴ across noise cases; standard PINN is less stable. The study uses 20,000 Adam epochs for PINN/MsFF and 50 KAN epochs selected for similar total training time.
5. **Limitations/reading:** No joint D estimate or classical regularized comparator; one-time heat inversion is smoothing. The result supports IC recovery precedent, not a PINN advantage over classical regularization. **Read:** full publisher HTML, including methods and results.

### 12. PINN point selection and experimental design

**Gregory Kang Ruey Lau, Apivich Hemachandra, See-Kiong Ng, Bryan Kian Hsiang Low (2024), “PINNACLE: PINN Adaptive ColLocation and Experimental points selection.”** [Full arXiv HTML](https://arxiv.org/html/2404.07662)

1. **Problem:** Select PINN data, PDE, IC, and BC points to improve training and parameter estimates.
2. **Difference:** Studies adaptive selection of observation and collocation points for inverse Navier–Stokes constants and an Eikonal coefficient field. It treats measurement and residual-point placement as related design choices.
3. **Testing:** Compares acquisition/selection strategies with pseudorandom sampling in the PDE examples and tracks parameter error as observations are added.
4. **Evidence:** In a Navier–Stokes case, reports 5% parameter accuracy using 10× fewer points. The study jointly adapts observation and physics collocation points, so its result concerns information allocation within the PINN objective.
5. **Limitations/reading:** Result concerns point savings within PINN workflows, not classical experimental design or regularized heat inversion. It motivates sensor-placement controls but cannot imply a benefit for unknown-IC heat estimates. **Read:** full arXiv HTML.

### 13. Classical boundary control for heat coefficient inversion

**Zhi-Xue Zhao, Bao-Zhu Guo, Zhong-Jie Han (2021), “Boundary Control and Observation to Inverse Coefficient Problem for Heat Equation With Unknown Source and Initial Value.”** *IEEE Transactions on Automatic Control* 66(12), 6003–6010. [Author PDF](https://lsc.amss.ac.cn/~bzguo/papers/Zhaozx6.pdf)

1. **Problem:** Identify a spatially varying heat coefficient with unknown source and initial value.
2. **Difference:** Uses designed switch-ON/OFF boundary control and a single-end boundary trace despite unknown source and IC.
3. **Testing:** Proves coefficient uniqueness under the protocol and proposes matrix-pencil reconstruction with optimal-perturbation regularization; simulations assess recovery under that input/trace design.
4. **Evidence:** Numerical examples demonstrate recovery under designed excitation, with regularization addressing the ill-posed inverse reconstruction. The setup deliberately uses one boundary input and one boundary output despite the additional unknown source and IC, showing the role of control design.
5. **Limitations/reading:** Not scalar diffusivity or a PINN comparison; source and IC are also unknown. Its control-driven boundary data differ from target sensors and the numerical method estimates a spatial coefficient profile. **Read:** full author PDF.
## Unreviewed search leads (excluded from the evidence synthesis)

- **M. S. Bidou, J.-G. Bauzin, N. Laraqi (2026), “Physics-informed neural networks for 2D-transient inverse heat conduction problem involving static and moving heat source.”** *International Journal of Heat and Mass Transfer* 259, 128349. [Publisher record](https://doi.org/10.1016/j.ijheatmasstransfer.2026.128349). Metadata/title and abstract were checked; full text was not read for this review. No findings from this paper are used in the assessment.
- **Dzimah et al. (2026), “A Unified Benchmark of Physics-Informed Neural Networks and Kolmogorov-Arnold Networks for Ordinary and Partial Differential Equations.”** [arXiv record](https://arxiv.org/abs/2602.15068). Abstract/metadata only; excluded from evidence synthesis.
- **“Inverse problems for the diffusion equation with one time observation” (2026).** [ScienceDirect abstract record](https://www.sciencedirect.com/science/article/abs/pii/S0362546X26001756). Abstract metadata only; full text unavailable, and its relevance to joint scalar diffusivity/unknown-IC estimation remains unverified. Excluded from evidence synthesis.

### Search log (3 November 2026; primary/official sources prioritized)

Representative exact queries used in the focused search:

- `inverse heat equation unknown initial condition diffusivity Tikhonov regularization simultaneous identification heat coefficient initial data`
- `"unknown initial condition" "thermal diffusivity" inverse heat equation Tikhonov`
- `PINN "Tikhonov" inverse heat equation diffusivity initial condition comparison`
- `"variable projection" inverse heat equation initial condition diffusivity`
- `"Investigating molecular transport in the human brain from MRI" PINN FEM Tikhonov authors`
- `"inverse heat equation" unknown initial state diffusion coefficient PINN FEM noisy data`
- `"simultaneous identification" heat equation diffusion coefficient initial value finite difference regularization`
- `"inverse diffusivity" PINN finite element noisy sparse observations heat equation`
- `PINNs versus classical methods inverse heat diffusion unknown initial condition regularization`
- `"PINNacle" inverse diffusion coefficient noisy data HInv`
- `PINN heat sensor placement sensitivity inverse diffusivity sparse observations 2026`
- `Dzimah 2026 PIKAN MLP heat equation matched parameter budgets`

The variable-projection/Tikhonov queries yielded classical initial-condition inversion literature and explicit regularization precedents, but no directly retrieved variable-projection paper benchmarking PINN against regularized classical inversion for joint scalar diffusivity plus unknown initial field. This is a search result, not evidence that no such paper exists. Zapf et al. (2022) is a stronger direct comparator: it compares PINN/FEM scalar-diffusivity recovery on sparse/noisy MRI observations, but uses different IC/BC formulations: initial-image data for the PINN, versus zero initial state and an optimized regularized boundary for FEM (published Methods, Equations 4-9).

## What the nearest prior work means for a credible research question

A precise and supportable question is: **For a 1D heat equation with unknown constant diffusivity and a parameterized unknown initial temperature, how do a PINN and an explicitly regularized PDE-constrained classical estimator compare across sensor counts/placements, noise levels, and IC spectral content, under a declared matched tuning and computational budget?**

This frames the paper around the intersection of existing lines—sparse/noisy scalar D PINN/FEM comparison (Zapf et al.), classical simultaneous D/IC identification (Zhao et al.), noisy inverse benchmarking (Jekic), unknown-IC heat reconstruction (Oliveira et al.), and IC-dependent practical identifiability (Salmaniw & Browning)—rather than claiming any ingredient is individually new. The key comparative extension would be an explicit unknown-IC regularized numerical baseline under the same sparse interior-data design and matched tuning budget.

### Recommended minimum experiment within the stated resources

- **Fix the problem size:** use one 1D heat equation, one boundary-condition choice, and a low-order (for example, 3–5 coefficient) Fourier IC basis. Estimate diffusivity and IC coefficients jointly; first run a synthetic case where identifiability can be checked from a sensitivity matrix.
- **Use two solvers:** the existing PINN with Adam→L-BFGS, and a finite-difference forward model with weighted least squares plus a quadratic IC smoothness penalty. Select the penalty from a small, prespecified grid using held-out synthetic observations or a discrepancy rule. No NNCG, broad architecture sweep, or additional model family is needed for the core result.
- **Limit the matrix:** one moderate sensor layout and one deliberately weak late-time/sparse layout; noiseless plus one stated noise level initially. If these complete, add a few independent noise realizations to the sparse case. This directly probes the gap without an exhaustive design study.
- **Keep the comparison auditable:** same synthetic truth, observation locations, noise draws, D bounds, evaluation grid, and tuning-trial or wall-clock allowance. Report D error, IC relative error, held-out field error, residuals, convergence/failure rate, and elapsed time. Avoid claims based on a single lucky initialization.
- **Treat device choice as an execution option:** keep the classical baseline and small PINN runnable on CPU within 16 GB; use Colab only if the PINN runtime blocks iteration, and record runtime/platform. A GPU should not be necessary for the low-dimensional classical fit. Cap the core study at the available 56 project hours and reserve the final block for plots, failure review, and writing.
- **Defer rather than expand:** profile likelihood/bootstrapping over many noise draws, sensor optimization, high-order IC bases, and optimizer alternatives are follow-up work only if the core comparison finishes early. The main conclusion should remain conditional on the chosen IC basis, noise level, and observation layout.

## Reference links (primary sources)

1. [Raissi, Perdikaris & Karniadakis (2019), JCP](https://doi.org/10.1016/j.jcp.2018.10.045)
2. [Zhao, Banda & Guo (2017), Math. Methods Appl. Sci.](https://doi.org/10.1002/mma.4245)
3. [Grossmann et al. (2024), IMAJAM](https://doi.org/10.1093/imamat/hxae011)
4. [Rathore et al. (2024), ICML/PMLR](https://proceedings.mlr.press/v235/rathore24a.html)
5. [Hao et al. (2024), NeurIPS PINNacle](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8c63299fb2820ef41cb05e2ff11836f5-Abstract-Datasets_and_Benchmarks_Track.html)
6. [Jekic et al. (2025; v2 2026), noisy inverse comparisons](https://arxiv.org/html/2509.20191)
7. [Salmaniw & Browning (2025), J Math Biol](https://doi.org/10.1007/s00285-025-02225-w)
8. [Zapf et al. (2022), sparse/noisy scalar D PINN/FEM comparison](https://doi.org/10.1038/s41598-022-19157-w)


## Clarification after the user's reading review (3 November 2026)

The published Zapf Methods confirms distinct initial/boundary treatment for PINN and FEM. Earlier shorthand about a measured IC must not be read as a common FEM initial condition. Grossmann Section 3.1 reports ten timing executions, not an explicitly documented independent accuracy-seed study. The linked arXiv HTML is v1; check figure locators against the cited publication version.
