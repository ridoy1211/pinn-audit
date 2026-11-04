# What we found — a plain-language guide

This guide explains `RESULTS_DRAFT.md` at about a Grade 10 reading level.
Read `methods.md` first. The numbers come from the completed experiment archived
at Git commit `4a64ed4`; this guide does not introduce new experiments.

## 1. Did the experiment finish?

Yes. All **400 scheduled jobs** completed:

- 160 reported wave-based fits: four procedures for each of 40 cases.
- 240 neural fits: two budgets, three starts and 40 cases.

The wave-based cross-validation procedures also performed many smaller internal
fits. The 400 count refers to the main job schedule, not every internal calculation.

The jobs came from **20 independent observation/noise designs**. A and B shared
a design, and three neural starts reused the same observations. We must not
treat all 400 jobs as independent experiments.

No job failed or timed out. However, finishing a calculation does not mean it
found the right physical answer. Every neural fit reached its allowed L-BFGS
iteration limit; we cannot claim that training reached the best possible solution.

## 2. What do A, B, 10 and 50 mean?

| Label | Meaning |
|---|---|
| A | Starting temperature pattern with wave modes 1 and 3 |
| B | Pattern A plus a smaller mode-7 component |
| 10 | Ten noisy temperature readings per case |
| 50 | Fifty noisy temperature readings per case |

CV12 is the twelve-wave method whose penalty was chosen by cross-validation.
PINN-1000 is the neural procedure with the longer second training stage.

## 3. Where did the longer-trained PINN do better?

For **family B with 10 readings**, PINN-1000 had lower diffusivity error than
CV12 in **9 out of 10 cases**. It also had lower initial-field error in 9 cases
and lower full-field error in 8 cases.

Here are the middle, or median, errors for that condition:

| What we checked | CV12 error | PINN-1000 error |
|---|---:|---:|
| Diffusivity | 11.75% | 3.40% |
| Starting temperature pattern | 26.11% | 23.37% |
| Temperatures over position and time | 7.40% | 6.07% |

Lower values are better. Each neural case first uses the middle score of its
three starts. Each displayed number then takes the middle of ten case scores.

These are useful results for this specific setting: few readings and the
starting pattern containing the seventh wave. They do not show that PINNs win
for every heat problem.

## 4. Where did the classical method do better?

For **family B with 50 readings**, CV12 had lower full-field error than
PINN-1000 in **8 out of 10 cases**. Its full-field error was also lower than
the shorter-trained PINN in **all 10 cases**.

| What we checked | CV12 error | PINN-200 error | PINN-1000 error |
|---|---:|---:|---:|
| Diffusivity | 1.90% | 5.36% | 2.33% |
| Starting temperature pattern | 10.88% | 22.35% | 16.48% |
| Temperatures over position and time | 2.19% | 4.93% | 3.70% |

These are again median errors. CV12 has smaller values in this table, but the
uncertainty matters. Against PINN-1000, the paired diffusivity and initial-field
comparisons did not clearly settle a direction. The full-field comparison
favored CV12 more clearly in this sample.

For the simpler A pattern, the diffusivity comparisons were also uncertain.
One method having a smaller median alone is not enough to establish a clear winner.
The academic Results draft retains all methods and all 24 planned comparisons.

## 5. Why does the number of wave shapes matter?

Family B contains the seventh wave shape. A six-mode method can use only shapes
1 through 6. No choice of their sizes can exactly reproduce that missing shape
over the whole rod. This is a limit of what the method can represent, even if
the computer performs the fitting perfectly.

For our particular family B, the missing wave alone gives a minimum possible
initial-field relative error of about **21.82%**. Actual errors can be larger.
This number applies to the whole starting pattern under our error formula;
it is not a claim that every position is wrong by 21.82%.

CV12 can include the seventh wave. With 50 readings, its median initial-field
error was **10.88%**, compared with **24.98%** for CV6. Using only CV6 as the
classical competitor would therefore give a different impression of neural
performance.

But allowing more shapes does not guarantee better estimates. With only 10
readings, the extra flexibility and limited noisy information can still lead
to poor reconstruction. Our results do not prove one single cause for every error.

## 6. Did longer training always help?

No. For family B with 50 readings, the longer PINN budget reduced the median
initial-field error from **22.35% to 16.48%**.

For family A with 50 readings, that median instead increased from **1.57% to
2.37%**. Its full-field error also increased slightly.

These describe the observed results. They do not prove why the changes occurred.
Training tries to reduce a weighted combination of reading error and physics
error. That training target is different from the hidden-answer scores we use
afterward. More training does not guarantee improvement in every recovery score.

## 7. Why can two kinds of median difference disagree?

There are two different calculations:

1. Find each method's median error, then subtract the two medians.
2. Subtract the methods' errors within each case, then find the median difference.

Our planned comparison used the second calculation, because both methods faced
the same case. These calculations need not give the same answer.

For example, the N=10/B diffusivity medians are about 3.40% and 11.75%.
Their subtraction is about -8.35 percentage points. The median of the ten
case-by-case differences is instead **-4.45 percentage points**. Neither is an
arithmetic mistake; they summarize different things.

Our uncertainty intervals describe the case-by-case comparison. They are based
on only ten designs in each condition and apply separately to each comparison.
They do not establish that one method wins generally across all conditions.

## 8. Were any answers very poor?

Yes. Although every job finished, an individual unregularized fit reached about
**645.5% full-field error** and **2622.4% initial-field error**. An individual
short-budget PINN fit reached about **107.8% diffusivity error**.

These are individual-fit extremes, not the median scores in the earlier tables.
We kept them in the evidence. Looking only at typical scores can hide occasional
large mistakes.

## 9. How much computation did it take?

The complete collection took about **2 hours and 40 minutes** on the recorded
CPU setup.

Across the four conditions, median CV12 selection-plus-final-fit costs were
about **7–11 seconds per case**. Training all three neural starts took about
**74–124 seconds per case**, depending on the budget and condition.

Those times exclude evaluation. The CV12 figure also excludes extra full-data
fits saved for examining nonselected candidates. We retained separate cost
records. The figures describe these implementations on this machine; they
are not a universal speed ranking.

## 10. How did we check that the calculations were trustworthy?

We checked file fingerprints, recomputed validation predictions and reloaded
saved neural models to check their predictions. We also checked field scoring
with a finer grid and with exact calculations available for the wave-based fits.

The checks supported the saved calculations. They do not prove that our methods
will work equally well on other equations or real laboratory measurements.

## 11. What conclusion can we defend?

**In this controlled heat-inversion study, the comparison changed with the
number of readings, the starting pattern, the wave-based method's allowed
shapes and regularization, the neural training budget, and the error being measured.**

The longer PINN had a favorable result in the sparse family-B setting. CV12
had a favorable full-field result in the denser family-B setting. Several
other comparisons remained uncertain.

We tested one diffusivity, one noise level, one domain and two starting patterns.
The classical representation was well suited to the simulated truths, and the
neural settings were fixed rather than equally tuned. We can report this study
carefully; whether its contribution is enough for a journal still requires
comparison with existing research.
