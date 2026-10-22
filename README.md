# pinn-audit

![ci](https://github.com/ridoy1211/pinn-audit/actions/workflows/ci.yml/badge.svg)

**Neural networks are being sold as PDE solvers. This repo checks whether you should buy one.**

![Accuracy against cost for classical and neural solvers](figures/accuracy_vs_time.png)

## The problem

Weather forecasts, heat flow in a chip, air over a wing: all of it comes down
to solving partial differential equations, and numerical solvers have been
doing that, reliably, for about seventy years.

Since 2019 there has been a rival pitch. A *physics-informed neural network*
(PINN) skips the mesh and the time-stepping, and learns the solution instead.
The promise is flexibility, and in particular that it copes with messy,
sparse measurements where classical methods struggle.

The published evidence is scattered. Some papers show spectacular results, a
few show PINNs failing on problems a first-year numerics course can solve, and
each uses its own problems, metrics and baselines. If you have to choose a
tool, there is no single place that answers the simple question: **when does
the neural network actually win?**

## The bet

> For **forward** problems, a well-built classical solver beats a PINN by
> orders of magnitude in accuracy per second. A PINN earns its place only in
> **inverse** problems (recovering unknown physical parameters) or when data
> is sparse and noisy.

So far the first half holds on the two problems tested. The second half is
untested. The most recent comparison I found (Jekic et al., 2025) has the
classical method winning on noisy inverse problems too, so this is a test I can
lose, and the repo is built so that anyone can see if I do.

## How the test works

A fair fight needs a referee nobody can argue with.

- **Exact answers.** Every test problem has a closed-form solution, so error
  is measured against the truth, not against another approximation.
- **Strong baselines.** Each classical scheme must reproduce its theoretical
  convergence order before it is allowed to judge anything, and is run at five
  grid sizes so the PINN is compared with a cost/accuracy curve, not one point.
- **One scoreboard.** Relative L2 error and wall-clock time, the same for every
  method, on one CPU thread.
- **Repeats.** Every neural result is five seeds, reported with spread.
- **Controls.** A network with no physics term, and a PINN given the same
  observations, show what the physics is actually worth.
- **One config per run.** A result is fully determined by a YAML file and a
  seed, and finished runs are skipped on re-run.

## What I found so far

Mean over five seeds, neural solvers scored on a 41-point grid:

| Method | Heat error | Heat time | Wave error | Wave time |
|---|---|---|---|---|
| Finite differences, 41 points | 5.3e-4 | 1.0 ms | 4.2e-3 | 0.7 ms |
| PINN | 3.1e-3 | 147 s | 5.1e-1 | 230 s |
| PINN + L-BFGS | 4.1e-4 | 175 s | 3.6e-1 | 268 s |
| PINN + 200 observations | 3.8e-3 | 180 s | 4.4e-1 | 234 s |
| PINN + loss weight 100 + L-BFGS | 1.2e-3 | 174 s | 3.3e-2 | 255 s |
| Network with no physics, 200 observations | 2.3e-2 | 27 s | 8.4e-2 | 36 s |

- **The classical solver wins on both forward problems, even against a tuned
  PINN.** On the wave equation the best tuned PINN reaches an error about twice
  that of the coarsest finite-difference grid I ran (21 points, 0.28 ms) and
  takes about a million times longer. On heat, the best PINN is matched by a
  41-point grid that runs in about a millisecond, against 175 s of training.
- **Tuning matters enormously, and it does not transfer.** On the wave
  equation, rebalancing the loss weights and adding L-BFGS together cut the
  error 15 times (0.51 to 0.033), far more than either change alone (2.9 and
  1.4 times). The same weights made the heat PINN 2.8 times worse than plain
  L-BFGS. Each problem needs its own tuning, and the classical solvers needed
  none.
- **Why the wave PINN failed: loss balance, not the fast mode.** Splitting the
  error into Fourier modes shows the plain PINN misses mode 3 almost completely
  (share 0.44). With balanced weights and L-BFGS the same network learns it
  (share 0.013, better than the no-physics network's 0.043).
- **Exact data did not help the heat PINN,** and at equal data the physics term
  helps on heat but hurts on the wave equation until the loss is rebalanced.

![Which Fourier modes carry the error on the wave problem](figures/wave_modes_weights.png)

Caveats: the loss weights were chosen using the exact error, which a real user
would not have; five seeds; one architecture; wall-clock times are noisy, so
read time ratios as orders of magnitude. I wrote predictions down and several
turned out wrong. Details and dead ends are in [`docs/LOG.md`](docs/LOG.md).

## Roadmap

| Milestone | Status |
|---|---|
| Exact solutions and classical baselines, verified | done |
| PINN and no-physics control, five-seed sweeps | done |
| Give the PINN its best shot (loss weights, L-BFGS) | done |
| Inverse problems under noise and sparse data | next |
| Burgers' equation (shocks) and 2D Poisson | planned |
| Written report | planned |

## Try it

Python 3.11 or newer.

```bash
git clone https://github.com/ridoy1211/pinn-audit
cd pinn-audit
python -m venv .venv
.venv\Scripts\activate            # macOS / Linux: source .venv/bin/activate
pip install -e ".[dev,nn]"

pytest                            # tests, including convergence-order and autograd checks
pinnaudit convergence --config configs/heat_fd.yaml
pinnaudit sweep --config configs/sweeps/forward_heat.yaml --n-seeds 1
pinnaudit summary --name forward_heat
python scripts/plot_pareto.py --name forward_heat forward_wave
```

The classical runs take milliseconds. Each neural run takes minutes on a laptop
CPU, so a full five-seed sweep takes a while. Sweeps are resumable, so you can
stop and restart them.

## What's in the repo

```
src/pinnaudit/
  problems/     PDEs with exact solutions (the ground truth)
  solvers/      finite differences, PINN, no-physics control, one shared interface
  metrics.py    error measures and observed convergence order
  modes.py      Fourier-mode breakdown of a solver's error
  sweep.py      resumable multi-seed experiments
  config.py     validated run configuration
  cli.py        pinnaudit run | sweep | summary | convergence | list
configs/        one YAML file per experiment, and configs/sweeps/ for the sweeps
tests/          exact solutions, convergence order, autograd residuals, sweeps
scripts/        figure generation
docs/LOG.md     dated lab notes, including dead ends
docs/REFERENCES.md  the five papers behind the design
```

## What this is not

It is not a new method and not a new discovery. The failure modes of PINNs are
documented (start with Krishnapriyan et al., 2021, in
[`docs/REFERENCES.md`](docs/REFERENCES.md)). What I'm adding is a small,
strict, reproducible testbed that puts the claims side by side.

## Why I built it

I finished a physics degree this year and spent the summer on machine-learning
pipelines. PINNs sit exactly between those two, and I wanted to understand
where the hype ends by measuring it myself. Disagree with a baseline or think
a test problem is missing? Open an issue.

Ridoy Hasan · [LinkedIn](https://www.linkedin.com/in/ridoy-hasan7/) · MIT licence
