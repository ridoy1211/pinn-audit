"""Fit saved timing development data using the reviewed broader-study routines."""

import argparse
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
from broader_common import write_json
from broader_provenance import canonical_hash, check_hashes, sha256, source_hashes
from pinnaudit.spectral_inverse import fit_spectral
from run_broader_study import run_jobs
from run_timing_study import ROOT, build_jobs, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--first-pair', action='store_true')
    group.add_argument('--all', action='store_true')
    args = parser.parse_args()
    out = ROOT / 'results/timing_development_v1'
    config_path = ROOT / 'configs/inverse/timing_development.json'
    cfg = load_config(config_path)
    jobs = build_jobs(cfg)
    data_manifest_path = out / 'data_manifest.json'
    data = json.loads(data_manifest_path.read_text(encoding='utf-8'))
    if data['config'] != cfg or data['jobs'] != jobs:
        raise RuntimeError('Saved data configuration or schedule differs from current plan.')
    check_hashes(ROOT, data['source_sha256'])
    check_hashes(out, data['data_sha256'])
    if Path(inspect.getsourcefile(fit_spectral)).resolve() != (
        ROOT / 'src/pinnaudit/spectral_inverse.py'
    ).resolve():
        raise RuntimeError('Imported pinnaudit is from another checkout.')

    sources = source_hashes(ROOT, config_path)
    sources.update(data['source_sha256'])
    sources['scripts/fit_timing_development.py'] = sha256(Path(__file__))
    environment = {
        'python': platform.python_version(),
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'torch': str(torch.__version__),
        'platform': platform.platform(),
    }
    identity = canonical_hash({
        'config': cfg, 'sources': sources, 'environment': environment,
        'data_manifest_sha256': sha256(data_manifest_path),
    })
    selected = jobs[:2] if args.first_pair else jobs
    manifest_path = out / 'manifest.json'
    lock = out / '.run.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    manifest = None
    try:
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding='utf-8'))
            if existing['run_identity'] != identity:
                raise RuntimeError('Resume refused: code, inputs, or environment changed.')
            manifest = existing
        else:
            if (out / 'jobs').exists() or (out / 'pde_points.npz').exists():
                raise RuntimeError('Existing fitting artifacts have no run manifest.')
            manifest = {
                'run_identity': identity,
                'mode': 'development_only',
                'started_utc': datetime.now(timezone.utc).isoformat(),
                'expected_jobs': len(jobs),
                'expected_cases': 36,
                'source_sha256': sources,
                'data_sha256': data['data_sha256'],
                'environment': environment,
            }

        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with threadpool_limits(limits=1):
            manifest.update(
                status='running', threadpools=threadpool_info(),
                torch_threads=1, dtype='float64', device='cpu',
            )
            write_json(manifest_path, manifest)
            # Reuse original fitting, scoring, checkpoint and failure handling.
            # Its loader passes only x, t, y and fold_labels to fitting routines.
            records = run_jobs(ROOT, out, cfg, [], selected, manifest, pilot)

        failures = sum(record['status'] != 'completed' for record in records)
        manifest.update(
            status=('development_finished' if args.all else 'first_pair_finished'),
            selected_jobs=len(selected), failed_selected_jobs=failures,
            last_finished_utc=datetime.now(timezone.utc).isoformat(),
        )
        write_json(manifest_path, manifest)
        print('\nDevelopment run summary:')
        for record in records:
            error = record.get('relative_alpha_error')
            score = 'unavailable' if error is None else f'{100 * error:.6g}%'
            print(f"{record['job_id']}: {record['status']}; alpha error={score}")
        print(f'Finished selected jobs: {len(records)}; unsuccessful: {failures}')
        print('Results are development evidence, not final evaluation evidence.')
        if failures:
            raise SystemExit(1)
    except BaseException as exc:
        if manifest is not None and manifest.get('status') == 'running':
            manifest.update(status='stopped', error=repr(exc))
            write_json(manifest_path, manifest)
        raise
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
