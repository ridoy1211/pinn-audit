"""Run the eight-dataset off-basis development pilot; not final evaluation."""

import argparse
import hashlib
import inspect
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
import torch
from threadpoolctl import threadpool_limits, threadpool_info

import offbasis_truth as truth
import pilot_bounded_pinn as pilot
from broader_common import write_json
from broader_provenance import check_hashes, source_hashes
from pinnaudit.spectral_inverse import fit_spectral
from run_broader_study import classical_job, neural_job, predict_grid, save_npz
from timing_common import generate_timing_designs

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/inverse/offbasis_development.json'
OUT=ROOT/'results/offbasis_development_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config():
    cfg=json.loads(CONFIG.read_text())
    base=json.loads((ROOT/'configs/inverse/timing_evaluation.json').read_text())
    changes={'study_id','phase','designs','min_complete_pairs','families','analysis_plan','timing_schedules'}
    for k,v in base.items():
        if k not in changes and cfg.get(k)!=v:
            raise ValueError(f'Unexpected fitting change: {k}')
    if cfg['phase']!='development' or cfg['designs']!={'10':[1006000,1006001],'50':[1007000,1007001]}:
        raise ValueError('Unexpected development design.')
    if cfg['timing_schedules']!={k:base['timing_schedules'][k] for k in ('mixed','late')}:
        raise ValueError('Timing windows changed.')
    definition=cfg['truth']
    if (definition['center'],definition['width'],definition['spectral_modes'],definition['quadrature_points'])!=(.35,.04,256,1024):
        raise ValueError('Unexpected truth definition.')
    if definition['normalization']!=truth.NORMALIZATION or definition['peak_location']!=truth.PEAK_LOCATION:
        raise ValueError('Truth normalization changed.')
    if sha(Path(truth.__file__))!=definition['verified_generator_sha256']:
        raise ValueError('Verified truth source changed.')
    verified=ROOT/'results/offbasis_truth_verification_v1'
    if sha(verified/'provenance.json')!=definition['verification_provenance_sha256']:
        raise ValueError('Truth verification provenance changed.')
    provenance=json.loads((verified/'provenance.json').read_text())
    check_hashes(verified,provenance['output_sha256'])
    if not json.loads((verified/'verification.json').read_text())['passed']:
        raise ValueError('Truth verification failed.')
    return cfg


def score(predict,alpha,obs,cfg,folder,refine):
    started=time.perf_counter()
    def at_grid(size):
        g,field=predict_grid(predict,size)
        target=truth.grid(g,g,cfg['alpha_true_generation_and_scoring_only'])
        initial=truth.initial(g)
        integrated=lambda v:np.trapezoid(np.trapezoid(v,g,axis=1),g)
        scores={
            'field_relative_l2':float(np.sqrt(integrated((field-target)**2)/integrated(target**2))),
            'ic_relative_l2':float(np.sqrt(np.trapezoid((field[0]-initial)**2,g)/np.trapezoid(initial**2,g))),
        }
        return g,field,scores
    g,field,scores=at_grid(cfg['grid_points'])
    data=predict(obs['x'],obs['t'])
    scores.update(relative_alpha_error=abs(alpha-cfg['alpha_true_generation_and_scoring_only'])/cfg['alpha_true_generation_and_scoring_only'],
                  data_mse=float(np.mean((data-obs['y'])**2)))
    arrays={'data_predictions':data}
    if refine:
        _,_,fine=at_grid(cfg['refinement_grid_points'])
        scores['refinement_scores']=fine
        arrays.update(x=g,t=g,u=field)
    save_npz(folder/'predictions.npz',**arrays)
    scores['evaluation_time_s']=time.perf_counter()-started
    return scores


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan',action='store_true')
    mode.add_argument('--run',action='store_true')
    args=parser.parse_args()
    cfg=load_config()
    jobs=[]
    for n in (10,50):
        for seed in cfg['designs'][str(n)]:
            for timing in ('mixed','late'):
                for method,training_seed in (('cv12',None),('pinn_1000',101)):
                    case=f'n{n}_seed{seed}_C_{timing}'
                    jobs.append(dict(case_id=case,n_obs=n,data_seed=seed,family='C',timing=timing,
                                     method=method,training_seed=training_seed))
    assert len(jobs)==16 and len({j['case_id'] for j in jobs})==8
    if args.plan:
        print('Off-basis development plan verified: 8 datasets, 16 fits.')
        print('Verified localized truth; unchanged CV12 and PINN-1000 procedures.')
        print('Direct initial-field scoring; mixed/late pairing; separate PDE points.')
        print('No observations generated or fitting performed.')
        return
    if OUT.exists():
        raise FileExistsError(f'Preserve existing output: {OUT}')
    if Path(inspect.getsourcefile(fit_spectral)).resolve()!=(ROOT/'src/pinnaudit/spectral_inverse.py').resolve():
        raise RuntimeError('Wrong imported checkout.')
    sources=source_hashes(ROOT,CONFIG)
    for name in ('scripts/offbasis_truth.py','scripts/run_offbasis_development.py',
                 'scripts/timing_common.py','docs/OFFBASIS_PROTOCOL.md','configs/inverse/timing_evaluation.json'):
        sources[name]=sha(ROOT/name)
    OUT.mkdir()
    manifest={'phase':'development','config':cfg,'source_sha256':sources,'data_sha256':{},
              'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
              'torch':str(torch.__version__),'expected_jobs':16,'status':'running'}
    write_json(OUT/'config.json',cfg);write_json(OUT/'schedule.json',jobs)
    write_json(OUT/'manifest.json',manifest)
    for n in (10,50):
        for seed in cfg['designs'][str(n)]:
            designs=generate_timing_designs(n,seed,cfg['fold_seed_offset'])
            for timing in ('mixed','late'):
                d=designs[timing]
                clean=truth.paired(d['x'],d['t'],cfg['alpha_true_generation_and_scoring_only'])
                sigma=cfg['noise_sigma_absolute']
                arrays={**d,'clean':clean,'y':clean+sigma*d['unit_noise'],'sigma':sigma,
                        'clean_signal_rms':float(np.sqrt(np.mean(clean**2))),
                        'signal_rms_over_noise_sigma':float(np.sqrt(np.mean(clean**2))/sigma)}
                name=f'data/n{n}_seed{seed}_C_{timing}.npz'
                save_npz(OUT/name,**arrays)
                manifest['data_sha256'][name]=sha(OUT/name)
    collocation=np.random.default_rng(cfg['collocation_seed']).uniform(size=(cfg['collocation_points'],2))
    residual_points=np.random.default_rng(cfg['residual_eval_seed']).uniform(size=(cfg['residual_eval_points'],2))
    save_npz(OUT/'pde_points.npz',training=collocation,evaluation=residual_points)
    manifest['pde_points_sha256']=sha(OUT/'pde_points.npz')
    write_json(OUT/'manifest.json',manifest)
    records=[]
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1):
        manifest['threadpools']=threadpool_info()
        for number,job in enumerate(jobs,1):
            check_hashes(ROOT,sources)
            name=f"data/{job['case_id']}.npz"
            if sha(OUT/name)!=manifest['data_sha256'][name]:
                raise RuntimeError('Observation file changed.')
            with np.load(OUT/name,allow_pickle=False) as d:
                obs={k:d[k].copy() for k in ('x','t','y','fold_labels')}
                snr=float(d['signal_rms_over_noise_sigma'])
            suffix=job['method']+('_train101' if job['training_seed'] is not None else '')
            folder=OUT/'jobs'/job['case_id']/suffix
            row={**job,'signal_rms_over_noise_sigma':snr,'artifact_path':folder.relative_to(OUT).as_posix()}
            print(f"Job {number}/16: {job['case_id']} / {suffix}",flush=True)
            started=time.perf_counter()
            try:
                if job['training_seed'] is None:
                    predict,alpha,details=classical_job(obs,cfg,job,folder)
                else:
                    predict,alpha,details=neural_job(obs,cfg,job,folder,collocation,residual_points,pilot)
                refined=job['data_seed']==cfg['designs'][str(job['n_obs'])][0]
                scores=score(predict,alpha,obs,cfg,folder,refined)
                row.update(status='completed',alpha_hat=alpha,**details,**scores)
            except Exception as exc:
                folder.mkdir(parents=True,exist_ok=True)
                row.update(status='failed',error=repr(exc))
                print(f'Failure retained: {exc}',flush=True)
            row['job_elapsed_s']=time.perf_counter()-started
            row['artifact_sha256']={p.name:sha(p) for p in folder.iterdir() if p.is_file()}
            write_json(folder/'job.json',row);records.append(row)
            write_json(OUT/'records.json',records)
    check_hashes(ROOT,sources)
    failed=sum(r['status']!='completed' for r in records)
    manifest.update(status='complete_with_failures' if failed else 'complete',failed_jobs=failed,
        output_sha256={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*')
                       if p.is_file() and p.name!='manifest.json'})
    write_json(OUT/'manifest.json',manifest)
    print('OFF-BASIS DEVELOPMENT COMPLETE')
    for r in records:
        error=100*r['relative_alpha_error'] if r['status']=='completed' else None
        print(f"{r['case_id']} {r['method']}: {r['status']}; alpha error (%)={error}")
    print(f'Completed: {16-failed}/16; unsuccessful: {failed}')
    print('Development evidence only; evaluation has not started.')


if __name__=='__main__':
    main()
