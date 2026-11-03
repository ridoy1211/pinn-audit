# Bounded PINN feasibility pilot

This is the first neural development stage of the regularized inverse audit.
It checks implementation, runtime, training behavior, and the ability to retain
fields/checkpoints. It is not a tuned PINN comparison or held-out confirmation.

## Fixed cases and information

Reuse N=10 seed 1000 and N=50 seed 2000: the first dataset in each development
seed range, chosen by order rather than error. Use their saved x, t, noisy y
without regenerating anything. Repeat each with training seeds 0, 1, 2. These
seeds vary network initialization only; all six fits use identical fixed PDE
collocation points. There are two datasets and three initializations per dataset,
not six independent datasets or an estimated population success rate.

The trainer receives no true alpha, clean observations, initial-condition values,
or problem object containing the true solution. The known unit domain and zero
Dirichlet boundary values are supplied physical information. With u=x*(1-x)*NN,
the network enforces both boundary values exactly, while its initial trace is
free. There is no initial-condition loss. Alpha is exp(log(lo)+(log(hi)-log(lo))
*sigmoid(raw)), constrained to the same [1e-5,1] interval as the classical fits,
and initialized at 0.2. Bounds and boundary information are thus shared.

The network and six-mode classical family have different representational priors.
The true IC lies exactly in the classical family. This pilot does not control that
advantage or establish a matched-prior comparison. Neither method's recovery
should be judged using PDE residual alone.

## Training and stopping

The complete numeric settings are in `configs/inverse/bounded_pinn_pilot.json`.
CPU, float64, one Torch thread, deterministic Torch algorithms; three hidden tanh
layers of width 32, normalized unit-domain inputs. Full-batch objective is
10*mean(data residual squared)+mean(heat PDE residual squared). No tuning occurs
in this pilot, and no truth-based early stopping or best-checkpoint selection is
allowed. The reported model is the final model after the declared budget.

Adam: 2000 steps, network learning rate 0.001, alpha-coordinate rate 0.01.
Then L-BFGS: up to 200 iterations, max_eval=250, history=50, learning rate 1,
strong-Wolfe search, tolerance_grad=1e-12, tolerance_change=1e-14. Actual closure
counts and optimizer iteration counts are saved; line search can exceed a nominal
function-evaluation budget within its final iteration. Neither stopping nor finite
loss alone certifies optimization convergence.

Use 512 fixed uniformly sampled interior space-time collocation points from seed
314159. They remain fixed during Adam and all L-BFGS closures. Independently
sample 2048 residual-evaluation points from seed 271828 and keep them fixed across
runs; these are never training points or a tuning criterion. Save both point sets.
There are no deliberately supplied t=0 measurements; archived random observation
times are retained. Initial-field recovery is evaluated, not assumed.

Trace losses and alpha every 250 Adam steps, at stage transitions, and at the end.
Save model-parameter snapshots (these omit optimizer states and do not support
exact optimizer-state resumption),
trace data, original input hashes, source/config hashes, settings, versions, actual
training seeds, statuses, and timings. A 600-second training cap per fit is checked
between optimizer evaluations; a single evaluation can overrun the cap slightly.
Checkpoint writing is included, while import and post-training scoring are outside
this training timer. Timings here are implementation-specific, not an equal-cost
comparison with the classical study.

Nonfinite loss/gradients, other exceptions, or timeouts are retained as failed or
timeout runs. A separate failed-state snapshot preserves the latest periodic
checkpoint. Continue to the other cases, list every status, and exit nonzero if
any case fails. Do not replace a failed initialization or silently enlarge budgets.

## Evaluation and limits

After training, evaluate coefficient error and relative L2 errors for the complete
temperature field and t=0 trace using the same analytical reference and 401-point
trapezoidal quadrature as the classical study. Save the whole predicted grid and
model state so evaluation can be reproduced. Also report data MSE, separate PDE
training and independent residual errors, exact boundary error, and loss traces.
Check that the saved state reproduces data predictions before accepting a run.

Retain the unregularized, fixed-lambda, and CV-selected classical results for these
exact two datasets as context only. Do not infer superiority, publication novelty,
or generalization from six untuned neural runs. Possible next changes must be
justified as development decisions and followed by a new frozen comparison design.

The assistant verifies a separate smoke configuration (archived software-test
dataset seed 987000, training seed 91, width 8, depth 2, three Adam steps, two
L-BFGS iterations, 16 collocation points, a 21-point scoring grid). It is excluded
from the six-run pilot. The default pilot is left for the user's environment.

## Run

```cmd
python scripts/pilot_bounded_pinn.py --out-dir results\bounded_pinn_pilot_v1
```

Use the existing environment and a new output directory. The earlier inverse
configs and solver are preserved for historical reproduction.
