"""Refine every failed rollout screen without changing the original results."""

import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits
import pilot_bounded_pinn as pilot
from broader_common import write_json
from check_timing_physical_rollout import ROOT, sha, coefficients, propagate, relative, self_check


def main():
    source = ROOT / 'results/timing_evaluation_v1'
    previous = ROOT / 'results/timing_rollout_v1'
    out = ROOT / 'results/timing_rollout_refinement_v1'
    if out.exists():
        raise FileExistsError(f'Preserve previous refinement: {out}')
    provenance = json.loads((previous / 'provenance.json').read_text())
    for name, expected in provenance['output_sha256'].items():
        if sha(previous / name) != expected:
            raise RuntimeError(f'Changed rollout artifact: {name}')
    manifest = json.loads((source / 'manifest.json').read_text())
    for name, expected in manifest['output_sha256'].items():
        if sha(source / name) != expected:
            raise RuntimeError(f'Changed evaluation artifact: {name}')
    failed = [r for r in json.loads((previous/'records.json').read_text())
              if not r['refinement_screen_pass']]
    if not failed:
        print('No failed screens to refine.')
        return
    self_check()
    out.mkdir()
    write_json(out/'protocol.json', {
        'selection':'all_failed_numerical_screens_from_original_rollout',
        'levels':[[4096,512],[8192,1024],[16384,2048]],
        'scoring_grid':801,'comparison_grid':401,'tolerance':1e-4,
        'script_sha256':sha(Path(__file__)),
        'original_rollout_provenance_sha256':sha(previous/'provenance.json'),
        'note':'Post-evaluation numerical refinement; original results retained.',
    })
    torch.set_num_threads(1)
    rows=[]
    with threadpool_limits(limits=1):
        for row in failed:
            case=row['case_id']
            folder=source/'jobs'/case/'pinn_1000_train101'
            saved=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
            model=pilot.BoundedHeatField(saved['settings'])
            model.load_state_dict(saved['model_state']); model.eval()
            alpha=float(model.alpha().detach())
            np.testing.assert_allclose(alpha,row['alpha_hat'],rtol=1e-12,atol=1e-14)
            initial=lambda x:pilot.predict_paired(model,x,np.zeros_like(x))
            grid=np.linspace(0,1,801)
            xx,tt=np.meshgrid(grid,grid)
            network=pilot.predict_paired(model,xx.ravel(),tt.ravel()).reshape(801,801)
            fields=[]; amplitudes=[]
            for intervals,modes in ((4096,512),(8192,1024),(16384,2048)):
                a=coefficients(initial,intervals,modes); amplitudes.append(a)
                fields.append(propagate(a,alpha,grid,grid))
                print(f'{case}: completed {modes}-mode forward solution',flush=True)
            fine=fields[-1]
            mismatch=relative(network-fine,fine,grid)
            refinements=[relative(fields[i]-fields[i-1],fields[i],grid) for i in (1,2)]
            coarse_score=relative((network-fine)[::2,::2],fine[::2,::2],grid[::2])
            f0=initial(grid)
            projection=float(np.sqrt(np.trapezoid((fine[0]-f0)**2,grid)/np.trapezoid(f0**2,grid)))
            result={**row,'original_mismatch':row['rollout_network_relative_l2'],
                'rollout_network_relative_l2':mismatch,
                'forward_refinement_relative_l2':refinements[-1],
                'previous_forward_refinement_relative_l2':refinements[0],
                'initial_projection_relative_l2':projection,
                'scoring_grid_change_absolute':abs(mismatch-coarse_score),
                'max_abs_network_rollout_difference':float(np.max(np.abs(network-fine))),
                'refinement_screen_pass':bool(refinements[-1]<1e-4 and projection<1e-4
                                              and abs(mismatch-coarse_score)<1e-4)}
            rows.append(result)
            np.savez_compressed(out/f'{case}.npz',grid=grid,network=network,rollout=fine,
                                coefficients_1024=amplitudes[1],coefficients_2048=amplitudes[2])
            write_json(out/'records.json',rows)
            print(json.dumps(result,indent=2),flush=True)
    write_json(out/'provenance.json',{'output_sha256':{
        p.name:sha(p) for p in out.iterdir() if p.is_file()}})
    print('REFINEMENT COMPLETE')
    print(f"Models: {len(rows)}; failed screens: {sum(not r['refinement_screen_pass'] for r in rows)}")


if __name__=='__main__':
    main()
