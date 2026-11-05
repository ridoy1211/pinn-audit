# How we did the study — a plain-language guide

This guide explains our completed study at about a Grade 10 reading level.
It follows `METHODS_DRAFT.md`. The research settings and results are unchanged.

## 1. What were we trying to find?

Imagine a thin rod. Different parts start at different temperatures. Over time,
heat spreads and the temperature pattern becomes smoother. Both ends are kept
at a reference temperature called zero.

Our computer model represents this situation. We wanted to recover two unknowns
from a small number of imperfect temperature readings:

- **Diffusivity:** a number that controls how quickly temperature differences
  smooth out. We call it **alpha**.
- **The initial temperature pattern:** the temperature at every position when
  the process started. We also call this the **initial field** or **initial condition**.

We also checked the predicted temperatures throughout the rod over the whole
time period. This is the **full temperature field**.

Finding the starting pattern and diffusivity from later observations is an
**inverse problem**. We see some effects and try to work backward to their causes.

## 2. Where did the readings come from?

We used a mathematical heat model to generate the readings. They were not
measurements from a laboratory experiment.

This gives us a known answer against which to check every estimate. The true
diffusivity was **0.05**. We used a position range and time range from 0 to 1;
these were normalized model coordinates, not a claim about a particular metal rod.

We tested two starting patterns:

| Pattern | What it contains | Why we used it |
|---|---|---|
| Family A | Two smooth sine-wave parts, called modes 1 and 3 | A simpler starting pattern |
| Family B | The same parts, plus a smaller mode-7 part | A pattern with extra fine detail |

A **mode** is one particular wave shape. Adding several wave shapes together
can build a more detailed temperature pattern. The formulas are:

```text
A = sin(pi*x) + 0.5*sin(3*pi*x)
B = A + 0.25*sin(7*pi*x)
```

Here, `x` means position and `pi` is approximately 3.14159. You do not need to
calculate these formulas to follow the experiment. The important point is that
B contains a wave shape that A does not.

We sampled either **10 or 50 temperature readings**. Each reading had a randomly
chosen position and time. We added random measurement error, called **noise**,
with standard deviation 0.01 in our model's temperature units. This is an
absolute noise size, not 1% of every reading.

For each reading count, we made ten independent sets of positions, times and
noise. We used each set with both A and B, so their comparison shared the same
measurement conditions. That produced:

```text
2 reading counts × 10 random designs × 2 starting patterns = 40 cases
```

There were **20 independent random designs**, not 40 independent designs,
because A and B reused a design. Each of the four conditions—10/A, 10/B, 50/A
and 50/B—contained ten designs.

## 3. Which methods did we compare?

We compared six procedures. Four used wave-based mathematics; two used the same
neural network with different amounts of training.

### The wave-based methods

These methods describe the temperature pattern by adding sine waves. Their
formula follows the heat equation exactly for the diffusivity they estimate.
They still need to estimate the right diffusivity and the right amount of each wave.

A six-mode method can use wave shapes 1 through 6. A twelve-mode method can use
shapes 1 through 12.

**Regularization** adds a cost for large fine-scale wave components. This helps
discourage a method from following random noise too closely. Too much can also
remove real detail, so its strength matters.

| Name in our files | Plain-language meaning |
|---|---|
| unreg6 | Six wave shapes, with no extra regularization penalty |
| fixed6 | Six wave shapes, with one small penalty chosen before evaluation |
| CV6 | Six wave shapes, with penalty strength chosen using the observations |
| CV12 | Twelve wave shapes, with penalty strength chosen using the observations |

For CV6 and CV12, we used **cross-validation**. We divided readings into five
groups, fitted on four groups, and checked predictions on the group left out.
We repeated this so each group was left out once. We compared five penalty
strengths, chose the one with the lowest combined validation error, and fitted
again using all readings.

The known true diffusivity and starting pattern were not used to choose the
penalty. Predicting left-out noisy readings well does not guarantee that the
recovered diffusivity or starting pattern is correct.

### The physics-informed neural network

A **neural network** is an adjustable mathematical function. Ours took position
and time as inputs and predicted temperature.

A **physics-informed neural network**, or **PINN**, adjusts its parameters to do
two things:

1. Match the available noisy temperature readings.
2. Approximately obey the heat equation at additional positions and times.

Those extra points are physics-check locations, not extra measured temperatures.
We used 512 fixed locations during training and 2048 separate locations for a
later check. We gave the reading-matching error ten times the weight of the
physics error in the training formula.

The network had three hidden layers, each containing 32 units. Its design made
the predicted temperature exactly zero at both ends. It was **not given the
correct starting temperature pattern**. It learned the temperature function
and diffusivity together. Diffusivity stayed between 0.00001 and 1, with an
initial guess of 0.2. The wave-based methods used the same diffusivity bounds.

Training used two adjustment algorithms, called **Adam** and **L-BFGS**:

| Procedure | Declared training budget |
|---|---|
| PINN-200 | 2000 Adam steps, followed by up to 200 L-BFGS iterations |
| PINN-1000 | 2000 Adam steps, followed by up to 1000 L-BFGS iterations |

An iteration is an adjustment step. These two algorithms' steps are not directly
equivalent. More steps give an algorithm more opportunity to adjust; they do
not guarantee a more accurate physical answer.

For each case and budget, we trained three times with different random starting
weights. These are three attempts on the same observations, not three new datasets.
We kept both budgets and every attempt.

## 4. How did we judge the answers?

We checked three separate errors against the known mathematical answer:

| Error | Question it answers |
|---|---|
| Diffusivity error | How far is the estimated alpha from 0.05? |
| Initial-field error | How wrong is the predicted starting temperature pattern? |
| Full-field error | How wrong are predicted temperatures across position and time? |

Smaller is better. Zero means an exact match for that score. For example,
an alpha estimate of 0.055 has a relative error of 10%, because its difference
from 0.05 is one tenth of 0.05.

For field errors, we combined squared differences over many positions, or over
positions and times, and scaled by the size of the true field. A 10% field error
does **not** mean that exactly 90% of individual readings are correct. Relative
errors can exceed 100% when an estimate is very poor.

We checked fields on a grid with 401 positions and 401 times. We also used a
finer grid on the first design in each condition to check the scoring calculation.

For each neural method and case, we took the **median** of its three errors:
the middle value after sorting. We did this separately for each score. We did
not choose the best attempt. A later summary took the median over ten cases.
These middle scores describe performance; they are not a combined neural model.

## 5. How did we compare methods fairly?

Every method received the same observations for a case. We compared each PINN
budget with CV12 **within that same case**, then examined the ten differences
in each condition.

Suppose one method has 8% error and another has 5% error. The difference is
3 **percentage points**. We used PINN error minus CV12 error, so a negative
difference favors the PINN.

We also used repeated resampling, called a **bootstrap**, to show uncertainty
in the typical difference. With only ten designs per condition, this uncertainty
can be large. An interval crossing zero means the comparison does not clearly
settle the direction; it does not prove the methods are equal. We examined many
comparisons, so one favorable interval cannot establish a universal winner.

The methods did not receive identical kinds of prior knowledge or tuning.
The wave-based formula is especially suitable for our simulated sine-wave
truths. Its penalty was tuned by cross-validation; the PINN settings were fixed.
The methods also used different computation amounts. Our conclusions must
therefore describe these particular procedures and conditions.

## 6. What made the work reproducible?

We saved the experiment rules and code version before generating evaluation
data. Earlier small runs helped development and were kept separate. The plan
was versioned on Git; it was not a formal preregistration.

We saved observations, settings, random seeds, model states, scores and training
records. A seed lets us repeat a particular random draw. File fingerprints,
called **hashes**, help detect changes to saved evidence. We also checked the
saved model predictions and calculations. These practices let someone inspect
how a result was produced, including unfavorable outcomes.

The reading order from here is `result.md`, followed by the academic drafts if
you want the complete equations and settings.
