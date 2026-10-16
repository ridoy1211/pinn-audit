"""The single code path that turns a config into a scored measurement.

Both the CLI and the test suite go through ``run_once``, so there is exactly
one definition of what "running an experiment" means.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pinnaudit import metrics as M
from pinnaudit.config import RunConfig
from pinnaudit.problems import build_problem
from pinnaudit.problems.base import Problem
from pinnaudit.solvers import build_solver
from pinnaudit.solvers.base import Solution


@dataclass
class RunResult:
    config: RunConfig
    problem: Problem
    solution: Solution
    metrics: dict[str, float]

    def summary(self) -> str:
        m = self.metrics
        return (
            f"{self.config.problem.name:>8} | {self.solution.solver_name:<18} | "
            f"nx={len(self.solution.x):<5} nt={len(self.solution.t):<6} | "
            f"relL2={m['relative_l2']:.3e} | maxerr={m['max_abs_error']:.3e} | "
            f"{self.solution.wall_time_s * 1e3:.2f} ms"
        )


def run_once(cfg: RunConfig) -> RunResult:
    """Build problem and solver, solve, and score against the analytic solution."""
    problem = build_problem(cfg.problem)
    solver = build_solver(cfg.solver, seed=cfg.seed)

    solution = solver.solve(problem, nx=cfg.solver.nx, nt=cfg.solver.nt)
    reference = problem.exact(solution.x, solution.t)

    scores: dict[str, Any] = {
        "relative_l2": M.relative_l2(solution.u, reference),
        "max_abs_error": M.max_abs_error(solution.u, reference),
        "rmse": M.rmse(solution.u, reference),
        "final_time_relative_l2": M.final_time_relative_l2(solution.u, reference),
    }
    return RunResult(config=cfg, problem=problem, solution=solution, metrics=scores)
