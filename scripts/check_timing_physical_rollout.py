"""Independent sine-series heat propagation of saved PINN initial fields."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.fft import dst
from threadpoolctl import threadpool_limits
import pilot_bounded_pinn as pilot
from broader_common import write_json

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coefficients(function, intervals, modes):
    x = np.arange(1, intervals) / intervals
    return (dst(function(x), type=1) / intervals)[:modes]


def propagate(a, alpha, x, t):
    k = np.pi * np.arange(1, len(a)+1)
    return (np.exp(-alpha * t[:, None] * k**2) * a) @ np.sin(k[:, None] * x)


def norm(field, grid):
    return float(np.sqrt(np.trapezoid(np.trapezoid(field**2, grid, axis=-1), grid)))


def relative(difference, reference, grid):
    denominator = norm(reference, grid)
    if denominator <= 0:
        raise ValueError('Zero reference field norm.')
    return norm(difference, grid) / denominator


def self_check():
    x = np.linspace(0, 1, 81)
    a = coefficients(lambda z: np.sin(np.pi*z)+.5*np.sin(3*np.pi*z), 1024, 128)
    exact = (np.exp(-.05*np.pi**2*x[:,None])*np.sin(np.pi*x)[None,:]
             + .5*np.exp(-.05*(3*np.pi)**2*x[:,None])*np.sin(3*np.pi*x)[None,:])
    np.testing.assert_allclose(propagate(a,.05,x,x), exact, atol=1e-12, rtol=1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pilot', action='store_true')
    group.add_argument('--all', action='store_true')
    args = parser.parse_args()
    self_check()
    print('Independent heat-propagation analytic test passed.', flush=True)
    source = ROOT / 'results/timing_evaluation_v1'
    out = ROOT / ('results/timing_rollout_pilot_v1' if args.pilot else 'results/timing_rollout_v1')
    if out.exists():
        raise FileExistsError(f'Preserve previous outputs: {out}')
    manifest = json.loads((source/'manifest.json').read_text())
    for name,h in manifest['output_sha256'].items():
        if sha(source/name) != h:
            raise RuntimeError(f'Modified input artifact: {name}')
    records = json.loads((source/'records.json').read_text())
    jobs = [r for r in records if r['method']=='pinn_1000']
    if len(jobs)!=360:
        raise RuntimeError('Expected 360 saved PINNs.')
    if args.pilot:
        jobs=jobs[:2]  # First early/mixed pair in the existing schedule, not selected by error.
    out.mkdir()
    write_json(out/'protocol.json', {
        'classification':'post_evaluation_physical_consistency_diagnostic',
        'initial_field':'saved_PINN_u(x,0)', 'diffusivity':'saved_PINN_alpha',
        'forward_solution':'independent_sine_series_heat_semigroup_zero_Dirichlet',
        'coarse':{'quadrature_intervals':2048,'modes':256},
        'fine':{'quadrature_intervals':4096,'modes':512},
        'scoring_grid':[401,401], 'scoring_refinement_comparison':[201,201],
        'numerical_screen_relative_tolerance':1e-4,
        'note':'Tolerance screens numerical refinement, not physical acceptability. '
               'Small absolute discrepancies require comparison with numerical error. '
               'No synthetic truth is used to construct the forward solution.',
        'source_manifest_sha256':sha(source/'manifest.json'),
        'script_sha256':sha(Path(__file__)),
    })
    torch.set_num_threads(1)
    rows=[]
    with threadpool_limits(limits=1):
        for number,r in enumerate(jobs,1):
            folder=source/r['artifact_path']
            saved=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
            model=pilot.BoundedHeatField(saved['settings'])
            model.load_state_dict(saved['model_state'])
            model.eval()
            alpha=float(model.alpha().detach())
            np.testing.assert_allclose(alpha,r['alpha_hat'],rtol=1e-12,atol=1e-14)
            with np.load(source/f"data/{r['case_id']}.npz") as obs:
                prediction=pilot.predict_paired(model,obs['x'],obs['t'])
            with np.load(folder/'predictions.npz') as previous:
                np.testing.assert_allclose(prediction,previous['data_predictions'],rtol=1e-10,atol=1e-12)
            initial=lambda x: pilot.predict_paired(model,x,np.zeros_like(x))
            a0=coefficients(initial,2048,256)
            a1=coefficients(initial,4096,512)
            grid=np.linspace(0,1,401)
            xx,tt=np.meshgrid(grid,grid)
            network=pilot.predict_paired(model,xx.ravel(),tt.ravel()).reshape(401,401)
            coarse=propagate(a0,alpha,grid,grid)
            fine=propagate(a1,alpha,grid,grid)
            mismatch=relative(network-fine,fine,grid)
            refinement=relative(fine-coarse,fine,grid)
            coarse_score=relative((network-fine)[::2,::2],fine[::2,::2],grid[::2])
            f0=initial(grid)
            initial_norm=float(np.sqrt(np.trapezoid(f0**2,grid)))
            reconstruction=float(np.sqrt(np.trapezoid((fine[0]-f0)**2,grid)))/initial_norm
            row={k:r[k] for k in ('case_id','n_obs','family','timing','data_seed')}
            row.update(alpha_hat=alpha,rollout_network_relative_l2=mismatch,
                       forward_refinement_relative_l2=refinement,
                       initial_projection_relative_l2=reconstruction,
                       scoring_grid_change_absolute=abs(mismatch-coarse_score),
                       refinement_screen_pass=bool(refinement<1e-4 and reconstruction<1e-4
                                                   and abs(mismatch-coarse_score)<1e-4),
                       max_abs_network_rollout_difference=float(np.max(np.abs(network-fine))))
            np.savez_compressed(out/f"{r['case_id']}.npz",grid=grid,network=network,
                                rollout=fine,coarse_coefficients=a0,fine_coefficients=a1)
            rows.append(row)
            write_json(out/'records.json',rows)
            print(f"{number}/{len(jobs)} {r['case_id']}: mismatch={100*mismatch:.5g}%; "
                  f"forward refinement={100*refinement:.3g}%; screen={row['refinement_screen_pass']}",flush=True)
    with (out/'summary.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(out/'provenance.json',{'output_sha256':{
        p.name:sha(p) for p in out.iterdir() if p.is_file()}})
    print('PHYSICAL ROLLOUT CHECK COMPLETE')
    print(f"Models checked: {len(rows)}; refinement screens failed: "
          f"{sum(not r['refinement_screen_pass'] for r in rows)}")
    print(f'Output: {out}')


if __name__=='__main__':
    main()
