"""Verify evaluation artifacts and perform the prespecified paired analysis."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TIMINGS = ('early', 'mixed', 'late')
METHODS = ('cv12', 'pinn_1000')
METRICS = {
    'diffusivity': 'relative_alpha_error',
    'initial_field': 'ic_relative_l2',
    'full_field': 'field_relative_l2',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interval(values, indices):
    values = np.asarray(values, dtype=float)
    bootstrap = np.median(values[indices], axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return float(np.median(values)), float(low), float(high)


def main():
    parser = argparse.ArgumentParser(description="Replay the timing analysis from saved results.")
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "results/timing_evaluation_v1",
        help="Saved timing evaluation directory containing manifest.json and records.json.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results/timing_evaluation_analysis_v1",
        help="New output directory outside the saved evaluation directory.",
    )
    args = parser.parse_args()
    source = args.results.resolve()
    out = args.out_dir.resolve()
    if out.is_relative_to(source):
        parser.error("--out-dir must be outside --results.")
    if out.exists():
        parser.error(f"--out-dir already exists: {out}. Choose a new directory.")
    manifest = read(source / 'manifest.json')
    if manifest['status'] != 'complete':
        raise RuntimeError('This analysis requires the reported complete, failure-free run.')
    print('Checking saved artifact hashes...', flush=True)
    for name, expected in manifest['output_sha256'].items():
        if digest(source / name) != expected:
            raise RuntimeError(f'Changed artifact: {name}')
    cfg = read(source / 'config.json')
    if cfg != manifest['freeze']['config']:
        raise RuntimeError('Saved configuration differs from the freeze.')
    jobs = read(source / 'schedule.json')
    records = read(source / 'records.json')
    if len(records) != 720 or len(jobs) != 720:
        raise RuntimeError('Expected 720 scheduled and recorded jobs.')
    indexed = {r['job_id']: r for r in records}
    if len(indexed) != 720 or set(indexed) != {j['job_id'] for j in jobs}:
        raise RuntimeError('Duplicate, missing or unexpected jobs.')
    for job in jobs:
        r = indexed[job['job_id']]
        if any(r[k] != v for k, v in job.items()) or r['status'] != 'completed':
            raise RuntimeError('Job identity or completion check failed.')
        for key in METRICS.values():
            if not np.isfinite(r[key]):
                raise RuntimeError('Nonfinite score; conditional analysis required.')
        expected = abs(r['alpha_hat'] - cfg['alpha_true_generation_and_scoring_only'])
        expected /= cfg['alpha_true_generation_and_scoring_only']
        if not np.isclose(r['relative_alpha_error'], expected, rtol=1e-12, atol=1e-14):
            raise RuntimeError('Diffusivity error does not match the saved estimate.')

    lookup = {(r['n_obs'], r['family'], r['data_seed'], r['timing'], r['method']): r
              for r in records}
    rng = np.random.default_rng(cfg['analysis_plan']['bootstrap_seed'])
    absolute, differences, interactions, blocks = [], [], [], []
    plotting = {}
    for n in (10, 50):
        seeds = cfg['designs'][str(n)]
        if len(seeds) != 30:
            raise RuntimeError('Expected 30 blocks per observation count.')
        # The same resampled block indices apply to both families, all schedules,
        # methods and outcomes within this observation-count stratum.
        indices = rng.integers(0, len(seeds), size=(cfg['analysis_plan']['bootstrap_reps'], len(seeds)))
        for family in ('A', 'B'):
            for metric, key in METRICS.items():
                deltas = {}
                for timing in TIMINGS:
                    values = {}
                    for method in METHODS:
                        group = [lookup[n, family, s, timing, method] for s in seeds]
                        values[method] = np.array([100 * r[key] for r in group])
                        absolute.append({
                            'n_obs': n, 'family': family, 'timing': timing,
                            'metric': metric, 'method': method, 'blocks': len(seeds),
                            'median_error_percent': float(np.median(values[method])),
                            'q25_error_percent': float(np.quantile(values[method], .25)),
                            'q75_error_percent': float(np.quantile(values[method], .75)),
                        })
                    delta = values['pinn_1000'] - values['cv12']
                    deltas[timing] = delta
                    estimate, low, high = interval(delta, indices)
                    differences.append({
                        'n_obs': n, 'family': family, 'timing': timing, 'metric': metric,
                        'blocks': len(seeds), 'median_difference_pp': estimate,
                        'ci95_low_pp': low, 'ci95_high_pp': high,
                        'pinn_lower_error_blocks': int(np.sum(delta < 0)),
                    })
                    for i, seed in enumerate(seeds):
                        blocks.append({'n_obs': n, 'family': family, 'seed': seed,
                                       'timing': timing, 'metric': metric,
                                       'cv12_error_percent': float(values['cv12'][i]),
                                       'pinn_error_percent': float(values['pinn_1000'][i]),
                                       'difference_pp': float(delta[i])})
                interaction = deltas['mixed'] - deltas['late']
                estimate, low, high = interval(interaction, indices)
                interactions.append({
                    'n_obs': n, 'family': family, 'metric': metric, 'blocks': len(seeds),
                    'primary': n == 10 and family == 'B' and metric == 'diffusivity',
                    'median_interaction_pp': estimate, 'ci95_low_pp': low, 'ci95_high_pp': high,
                })
                if metric == 'diffusivity':
                    plotting[n, family] = np.column_stack([deltas[t] for t in TIMINGS])

    diagnostics = []
    for n in (10, 50):
        for family in ('A', 'B'):
            for timing in TIMINGS:
                for method in METHODS:
                    group = [r for r in records if (r['n_obs'], r['family'], r['timing'], r['method'])
                             == (n, family, timing, method)]
                    diagnostics.append({
                        'n_obs': n, 'family': family, 'timing': timing, 'method': method,
                        'completed': len(group),
                        'reported_bound_hits': sum(bool(r['at_bound']) for r in group),
                        'smallest_ridge_selections': sum(r.get('selected_ridge') == min(cfg['ridges']) for r in group),
                        'largest_ridge_selections': sum(r.get('selected_ridge') == max(cfg['ridges']) for r in group),
                        'non_completed_budget_traces': sum(
                            r.get('final_training_trace', {}).get('stage') != 'completed_budget'
                            for r in group if method == 'pinn_1000'),
                        'median_job_seconds': float(np.median([r['job_elapsed_s'] for r in group])),
                        'total_job_seconds': float(sum(r['job_elapsed_s'] for r in group)),
                    })
    out.mkdir()
    for name, rows in (('absolute_errors', absolute), ('paired_differences', differences),
                       ('timing_interactions', interactions), ('paired_blocks', blocks),
                       ('diagnostics', diagnostics)):
        save_csv(out / f'{name}.csv', rows)
    primary = next(r for r in interactions if r['primary'])
    lines = ['# Controlled timing evaluation', '',
             '720 completed fits; 30 independent blocks per observation count, shared across families and schedules.',
             f"Verified {len(manifest['output_sha256'])} indexed artifact hashes.", '',
             '## Prespecified primary result', '',
             'N=10, family B; diffusivity error. D = PINN error minus CV12 error.',
             'I = D(mixed) minus D(late), calculated within each block before taking the median.',
             f"Median I: {primary['median_interaction_pp']:.4f} percentage points; "
             f"95% percentile bootstrap interval [{primary['ci95_low_pp']:.4f}, {primary['ci95_high_pp']:.4f}].", '',
             'A negative interaction means the PINN-relative-to-CV12 error is smaller under mixed observations; '
             'a positive interaction means it is larger. This does not by itself identify which method has lower absolute error.', '',
             '## Median diffusivity errors (%)', '',
             '| N | Family | Schedule | CV12 | PINN |', '|---|---|---|---:|---:|']
    for n in (10, 50):
        for family in ('A', 'B'):
            for timing in TIMINGS:
                vals = {r['method']: r['median_error_percent'] for r in absolute
                        if (r['n_obs'], r['family'], r['timing'], r['metric']) == (n, family, timing, 'diffusivity')}
                lines.append(f"| {n} | {family} | {timing} | {vals['cv12']:.3f} | {vals['pinn_1000']:.3f} |")
    lines += ['', '## Interpretation limits', '',
              'All secondary intervals are pointwise, not simultaneous. Bootstrap intervals with 30 blocks '
              'may be sensitive to skewness and boundary estimates. Do not infer significance from a comparison of marginal medians.',
              'The schedules change delay and temporal spread together. Fixed absolute noise means late-time signal-to-noise ratios differ.',
              'The result concerns the frozen procedures, one PINN start, and two synthetic initial fields. '
              'It does not demonstrate restored unconstrained identifiability or universal method superiority.',
              'Job runtimes include saved diagnostics and scoring; distinguish them from fitting-only costs.', '']
    (out / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, ((n, family), values) in zip(axes.flat, plotting.items()):
        ax.plot(range(3), values.T, color='steelblue', alpha=.18, linewidth=.7)
        ax.plot(range(3), np.median(values, axis=0), 'o-', color='black', label='Median')
        ax.axhline(0, color='gray', linewidth=.8)
        ax.set_title(f'N={n}, initial field {family}')
        ax.set_xticks(range(3), TIMINGS)
        ax.set_ylabel('PINN − CV12 error (percentage points)')
        ax.legend()
    fig.suptitle('Paired diffusivity-error differences; each blue line is one block')
    fig.tight_layout()
    fig.savefig(out / 'paired_diffusivity.png', dpi=200)
    fig.savefig(out / 'paired_diffusivity.svg')
    plt.close(fig)
    provenance = {'input_manifest_sha256': digest(source / 'manifest.json'),
                  'analysis_script_sha256': digest(Path(__file__)),
                  'analysis_plan': cfg['analysis_plan'], 'numpy_version': np.__version__,
                  'output_sha256': {p.name: digest(p) for p in out.iterdir() if p.is_file()}}
    (out / 'analysis_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print('\nTIMING EVALUATION ANALYSIS')
    print('\n'.join(lines[2:11]))
    print(f'Analysis saved to: {out}')
    print('Open summary.md for the error table; CSV files contain all paired results.')


if __name__ == '__main__':
    main()
