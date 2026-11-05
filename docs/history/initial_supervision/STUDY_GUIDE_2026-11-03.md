# Reproducibility audit study guide (supervision session 1)

Filename note: DAY1 refers to the first session of the current supervision plan,
not the beginning of the existing PINN Audit project. The October lab log and
original experiment history remain in place.

Prepared 3 November 2026. This guide was prepared by the research assistant.
Read it in Notepad or a Markdown viewer. Write your own understanding and
questions in DAY1_NOTES.md. No further Python experiment is required for this guide.

## Where to do each task

- CMD: launch commands and open files.
- Notepad: read this guide and edit DAY1_NOTES.md.
- Your browser or PDF reader: read the two actual research papers.
- Paper and pen, if helpful: work through the short derivation below.

## A. Read the saved pilot review (15-20 minutes)

From CMD in the project folder:

    notepad audit\user_run_evidence\day1_pilot_review.md

Read the whole file, especially "Individual sparse-data outcomes". Concentrate on:

1. A relative error of 0.254 means approximately 25.4%.
2. A lower median can coexist with a worse mean and a severe individual error.
3. At N=10 and lambda=1e-6, one estimate is alpha=0.001, the lower search bound.
   Against alpha_true=0.05, its error is abs(0.001-0.05)/0.05 = 0.98 = 98%.
4. These are five development datasets. We did not train a PINN in this pilot.

This is reproduction of the same saved data-generating design, not an independent
confirmation with new datasets. Keep every result, not only the favorable ones.

## B. Understand the mathematics (about 60-90 minutes)

### 1. What is modeled?

The heat equation is:

    du/dt = alpha * d2u/dx2

Here, u(x,t) is temperature relative to a reference, x is position, t is time, and
alpha is positive thermal diffusivity. In this nondimensional benchmark, position
and time both range from 0 to 1, and the temperature at each end of the interval
is fixed at zero. We are not claiming to model a particular physical material.

The inverse task estimates alpha and the starting temperature shape from noisy
temperatures measured at selected positions and times. Alpha=0.05 is the synthetic
truth used by the generator/scorer; the estimator must not use it for selection.

### 2. Why does a sine mode decay?

Take one possible component:

    u_j(x,t) = a_j(t) * sin(j*pi*x)

Differentiate:

    du_j/dt   = a_j'(t) * sin(j*pi*x)
    d2u_j/dx2 = -(j*pi)^2 * a_j(t) * sin(j*pi*x)

Substitution in the heat equation gives:

    a_j'(t) = -alpha * (j*pi)^2 * a_j(t)

Solving this ordinary differential equation gives:

    a_j(t) = a_j(0) * exp[-alpha * (j*pi)^2 * t]

The negative exponent makes the mode decay. Mode 3 has nine times the decay rate
of mode 1. Fine spatial structure therefore becomes harder to infer from late data.

Short exercise: at alpha=0.05 and t=1, evaluate the decay factors for j=1 and j=3
using a calculator. They are approximately 0.6105 and 0.0118. These are fractions
of their respective starting amplitudes, not the full temperature field.

### 3. What is unknown about the initial condition?

The classical model represents the starting field as:

    u(x,0) = sum from j=1 to M of a_j * sin(j*pi*x)

In the pilot M=6. The six amplitudes and alpha are unknown: seven fitted parameters.
Ten measurements exceed seven in count, but noise and nearly insensitive modes
can still make estimation unreliable. A parameter count alone does not establish
identifiability or numerical stability.

### 4. What does regularization do?

The spectral estimator minimizes:

    mean(squared prediction errors) + lambda * sum(j^4 * a_j^2)

The first term rewards fitting the measured temperatures. The second discourages
large initial-condition coefficients, especially high-frequency ones. Lambda
controls the trade-off. It penalizes initial-field coefficients, not alpha directly.

Regularization introduces a preference for smoother starting fields. It can help
stabilize recovery, but it can also bias the inferred field and alpha. We need a
selection procedure based on available data, not on the known simulation answer.

### 5. Why can one snapshot be insufficient?

At a single time t_star, an observed modal amplitude is:

    b_j = a_j * exp[-alpha * (j*pi)^2 * t_star]

If a_j is free, a different alpha can be compensated by a different a_j. Thus that
snapshot alone cannot separate alpha from the unknown amplitudes in this setting.
Measurements over multiple times help within a restricted initial-field model,
but noise and weak sensitivity can still limit recovery.

After reading, use DAY1_NOTES.md to explain the equation, mode decay, penalty,
and single-snapshot ambiguity in short sentences. Mark uncertainty explicitly.

## C. Read the two papers (about 45-60 minutes each for a first pass)

Do not aim to understand every technical detail in the first pass. Read the abstract,
the research question/introduction, relevant result figures/tables, methods used
for the comparison, and conclusion/limitations. Note the section or figure that
supports each answer. Record unfinished sections rather than claiming a full read.

### Paper 1: Grossmann and colleagues (2024)

Title: Can physics-informed neural networks beat the finite element method?

Free author-preprint HTML:
https://arxiv.org/html/2302.04107

Published article:
https://doi.org/10.1093/imamat/hxae011

Reading purpose: understand why a credible numerical comparator, common accuracy
evaluation, and separate construction/evaluation times matter. This is a forward
comparison; it does not directly answer our unknown-initial-condition inverse task.

### Paper 2: Zapf and colleagues (2022)

Title: Investigating molecular transport in the human brain from MRI with
physics-informed neural networks.

Full publisher article:
https://www.nature.com/articles/s41598-022-19157-w

Author-preprint record, with a PDF link:
https://arxiv.org/abs/2205.02592

Reading purpose: understand a direct PINN/FEM diffusion-inference precedent. Track
what measurements and initial-state information each approach receives. Their
measured initial image differs from our unknown-initial-field setting.

For EACH paper, answer in DAY1_NOTES.md:

1. What problem does it investigate?
2. What did it do differently from earlier work? Identify the comparison in its text.
3. How was the approach tested: equations/data, methods, noise, metrics, and repeats?
4. What specific figure, table, or calculation supports the conclusions?
5. What limitations remain, and how does its setup differ from our project?

Use 2-4 sentences per question initially. Write "not yet located" when necessary.
After your own first pass, compare with entries 3 and 8 in audit/literature_review.md.
Do not substitute the assistant's notes for reading the paper itself.

## D. Complete the record (20-30 minutes)

Open:

    notepad DAY1_NOTES.md

Verified execution facts and a clearly labeled assistant-drafted interpretation
are already included. Add your reading status, explanations, questions, and any
corrections. Save with Ctrl+S. No GitHub upload has occurred automatically.

Day 1 is finished when the preserved execution results, your preliminary scientific
explanations, and honest reading notes are together in this project folder.
