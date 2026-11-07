"""Plan, freeze, or run the separate controlled-timing evaluation."""

import argparse
import hashlib
import inspect
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import torch
from threadpoolctl import threadpool_info, threadpool_limits

import pilot_bounded_pinn as pilot
from broader_common import make_dataset, write_json
from broader_provenance import (
    canonical_hash, check_hashes, clean_revision, git, sha256, source_hashes,
)
from pinnaudit.spectral_inverse import fit_spectral
from run_broader_study import run_jobs, save_npz
from run_timing_study import ROOT, build_jobs, load_config
from timing_common import generate_timing_designs

CONFIG = ROOT / 'configs/inverse/timing_evaluation.json'
FREEZE = ROOT / 'docs/TIMING_EVALUATION_FREEZE.json'
OUT = ROOT / 'results/timing_evaluation_v1'


def configuration():
    development = load_config(ROOT / 'configs/inverse/timing_development.json')
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    changed = {'study_id', 'phase', 'designs', 'ridges', 'min_complete_pairs'}
    for key, value in development.items():
        if key not in changed and cfg.get(key) != value:
            raise ValueError(f'Unexpected change to fitting settings: {key}')
    expected = {
        'study_id': 'joint_heat_timing_evaluation_v1', 'phase': 'evaluation',
        'designs': {'10': list(range(906000, 906030)),
                    '50': list(range(907000, 907030))},
        'ridges': [1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2],
        'min_complete_pairs': 30,
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f'Evaluation setting differs from declared plan: {key}')
    if cfg['analysis_plan']['primary_stratum'] != {'n_obs': 10, 'family': 'B'}:
        raise ValueError('Primary stratum changed.')
    original = json.loads((ROOT / 'configs/inverse/broader_comparison.json').read_text())
    used = {876543, 876544}
    for prior in (development, original):
        used.update(s for group in prior['designs'].values() for s in group)
    proposed = [s for group in cfg['designs'].values() for s in group]
    if len(set(proposed)) != 60 or used.intersection(proposed):
        raise ValueError('Evaluation seeds overlap prior listed data seeds.')
    jobs = build_jobs(cfg)
    if len(jobs) != 720 or len({j['case_id'] for j in jobs}) != 360:
        raise ValueError('Incorrect evaluation schedule.')
    return cfg, jobs


def sources():
    result = source_hashes(ROOT, CONFIG)
    for name in (
        'scripts/run_timing_evaluation.py', 'scripts/run_timing_study.py',
        'scripts/timing_common.py', 'configs/inverse/timing_development.json',
        'configs/inverse/broader_comparison.json', 'docs/TIMING_PROTOCOL.md',
        'docs/TIMING_DEVELOPMENT_DECISIONS.md',
    ):
        result[name] = sha256(ROOT / name)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan', action='store_true')
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.resume and not args.run:
        parser.error('--resume requires --run')
    cfg, jobs = configuration()
    if Path(inspect.getsourcefile(fit_spectral)).resolve() != (
        ROOT / 'src/pinnaudit/spectral_inverse.py'
    ).resolve():
        raise RuntimeError('Imported package is from another checkout.')
    if args.plan:
        print('Evaluation plan verified: 60 blocks, 360 datasets, 720 jobs.')
        print('CV12: 360; PINN-1000: 360; neural seed: 101.')
        print('Original fitting settings retained except expanded ridge grid.')
        print('Evaluation seed separation and unique job identities verified.')
        print('No evaluation observations generated; no training performed.')
        return

    revision = clean_revision(ROOT)
    frozen_sources = sources()
    environment = {
        'python': platform.python_version(), 'numpy': np.__version__,
        'scipy': scipy.__version__, 'torch': str(torch.__version__),
        'platform': platform.platform(),
    }
    if args.freeze:
        if FREEZE.exists() or OUT.exists():
            raise RuntimeError('Freeze or evaluation output already exists; preserve it.')
        # Check committed content, allowing Windows checkout newline conversion.
        for name in frozen_sources:
            committed = git(ROOT, 'show', f'HEAD:{name}').replace(b'\r\n', b'\n')
            local = (ROOT / name).read_bytes().replace(b'\r\n', b'\n')
            if committed != local:
                raise RuntimeError(f'Uncommitted source contents: {name}')
        write_json(FREEZE, {
            'classification': 'prospective_local_freeze_not_formal_preregistration',
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'implementation_commit': revision, 'source_sha256': frozen_sources,
            'config': cfg, 'environment': environment,
            'schedule_sha256': canonical_hash(jobs),
        })
        print('Freeze written. Commit docs/TIMING_EVALUATION_FREEZE.json before running.')
        print('No evaluation observations generated.')
        return

    freeze = json.loads(FREEZE.read_text(encoding='utf-8'))
    committed_freeze = json.loads(git(ROOT, 'show', 'HEAD:docs/TIMING_EVALUATION_FREEZE.json'))
    if freeze != committed_freeze:
        raise RuntimeError('Freeze must be committed unchanged.')
    git(ROOT, 'merge-base', '--is-ancestor', freeze['implementation_commit'], revision)
    if (freeze['source_sha256'] != frozen_sources or freeze['config'] != cfg
            or freeze['environment'] != environment
            or freeze['schedule_sha256'] != canonical_hash(jobs)):
        raise RuntimeError('Frozen code, configuration, environment or schedule changed.')
    check_hashes(ROOT, frozen_sources)
    identity = canonical_hash(freeze)
    if OUT.exists() and not args.resume:
        raise FileExistsError('Output exists. Use --run --resume; do not delete it.')
    if args.resume and not OUT.is_dir():
        raise FileNotFoundError('No evaluation output to resume.')
    OUT.mkdir(exist_ok=True)
    lock = OUT / '.run.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    try:
        manifest_path = OUT / 'manifest.json'
        if args.resume:
            manifest = json.loads(manifest_path.read_text())
            if manifest['run_identity'] != identity:
                raise RuntimeError('Resume identity mismatch.')
        else:
            manifest = {
                'run_identity': identity, 'freeze': freeze,
                'source_sha256': frozen_sources, 'data_sha256': {},
                'expected_jobs': 720, 'expected_cases': 360,
                'started_utc': datetime.now(timezone.utc).isoformat(),
                'environment': environment, 'status': 'preparing_data',
            }
            write_json(OUT / 'config.json', cfg)
            write_json(OUT / 'schedule.json', jobs)
            write_json(manifest_path, manifest)
        for n in (10, 50):
            for seed in cfg['designs'][str(n)]:
                designs = generate_timing_designs(n, seed, cfg['fold_seed_offset'])
                for family, modes in cfg['families'].items():
                    for timing, design in designs.items():
                        name = f'data/n{n}_seed{seed}_{family}_{timing}.npz'
                        path = OUT / name
                        if path.exists():
                            if manifest['data_sha256'].get(name) != sha256(path):
                                raise RuntimeError(f'Unindexed or modified data: {name}')
                            continue
                        if name in manifest['data_sha256']:
                            raise RuntimeError(f'Indexed data disappeared: {name}')
                        obs = make_dataset(design, cfg['alpha_true_generation_and_scoring_only'],
                                           modes, cfg['noise_sigma_absolute'])
                        obs.update(base_time_draw=design['base_time_draw'],
                                   mixed_early_mask=design['mixed_early_mask'])
                        save_npz(path, **obs)
                        manifest['data_sha256'][name] = sha256(path)
                        write_json(manifest_path, manifest)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with threadpool_limits(limits=1):
            manifest.update(status='running', threadpools=threadpool_info())
            write_json(manifest_path, manifest)
            records = run_jobs(ROOT, OUT, cfg, [], jobs, manifest, pilot)
        failures = sum(r['status'] != 'completed' for r in records)
        manifest.update(status='complete_with_failures' if failures else 'complete',
                        failed_jobs=failures, completed_utc=datetime.now(timezone.utc).isoformat())
        manifest['output_sha256'] = {
            p.relative_to(OUT).as_posix(): sha256(p) for p in OUT.rglob('*')
            if p.is_file() and p.name not in ('manifest.json', '.run.lock')
        }
        write_json(manifest_path, manifest)
        print(f'Evaluation finished: {len(records)} jobs; unsuccessful: {failures}.')
        print('Preserve all results, including failed or interrupted jobs.')
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
