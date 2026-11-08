"""Prespecified secondary paired analysis of the off-basis evaluation."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save_csv(p,rows):
    with p.open('w',newline='',encoding='utf-8') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def interval(v,indices):
    v=np.asarray(v)
    lo,hi=np.quantile(np.median(v[indices],axis=1),[.025,.975])
    return float(np.median(v)),float(lo),float(hi)


def main():
    source=ROOT/'results/offbasis_evaluation_v1'
    out=ROOT/'results/offbasis_analysis_v1'
    if out.exists():
        raise FileExistsError(f'Preserve existing analysis: {out}')
    manifest=json.loads((source/'manifest.json').read_text())
    print('Verifying evaluation artifacts...',flush=True)
    if manifest['status']!='complete':
        raise RuntimeError('This analysis expects the reported failure-free completed run.')
    for name,h in manifest['output_sha256'].items():
        if sha(source/name)!=h: raise RuntimeError(f'Changed artifact: {name}')
    cfg=json.loads((source/'config.json').read_text())
    if cfg!=manifest['freeze']['config']: raise RuntimeError('Frozen configuration mismatch.')
    records=json.loads((source/'records.json').read_text())
    jobs=json.loads((source/'schedule.json').read_text())
    byid={r['job_id']:r for r in records}
    if len(records)!=240 or len(byid)!=240 or len(jobs)!=240:
        raise RuntimeError('Expected 240 unique jobs.')
    for job in jobs:
        r=byid[job['job_id']]
        if r['status']!='completed' or any(r[k]!=v for k,v in job.items()):
            raise RuntimeError('Incomplete or mismatched job.')
        np.testing.assert_allclose(r['relative_alpha_error'],abs(r['alpha_hat']-.05)/.05,rtol=1e-12)
    lookup={(r['n_obs'],r['data_seed'],r['timing'],r['method']):r for r in records}
    rng=np.random.default_rng(cfg['analysis_plan']['bootstrap_seed'])
    absolute=[];paired=[];interactions=[];blocks=[];diagnostics=[]
    metrics={'diffusivity':'relative_alpha_error','initial_field':'ic_relative_l2','full_field':'field_relative_l2'}
    for n in (10,50):
        seeds=cfg['designs'][str(n)]
        assert len(seeds)==30
        indices=rng.integers(0,30,size=(cfg['analysis_plan']['bootstrap_reps'],30))
        for metric,key in metrics.items():
            deltas={}
            for timing in ('mixed','late'):
                values={}
                for method in ('cv12','pinn_1000'):
                    v=np.array([100*lookup[n,s,timing,method][key] for s in seeds])
                    if not np.isfinite(v).all(): raise RuntimeError('Nonfinite scores.')
                    values[method]=v
                    absolute.append(dict(n_obs=n,timing=timing,metric=metric,method=method,blocks=30,
                        median_error_percent=float(np.median(v)),q25_percent=float(np.quantile(v,.25)),q75_percent=float(np.quantile(v,.75))))
                d=values['pinn_1000']-values['cv12'];deltas[timing]=d
                estimate,lo,hi=interval(d,indices)
                paired.append(dict(n_obs=n,timing=timing,metric=metric,blocks=30,
                    median_difference_pp=estimate,ci95_low_pp=lo,ci95_high_pp=hi,
                    pinn_lower_error_blocks=int(np.sum(d<0))))
                for i,s in enumerate(seeds):
                    blocks.append(dict(n_obs=n,seed=s,timing=timing,metric=metric,
                        cv12_error_percent=float(values['cv12'][i]),pinn_error_percent=float(values['pinn_1000'][i]),difference_pp=float(d[i])))
            estimate,lo,hi=interval(deltas['mixed']-deltas['late'],indices)
            interactions.append(dict(n_obs=n,metric=metric,blocks=30,median_interaction_pp=estimate,ci95_low_pp=lo,ci95_high_pp=hi))
        for timing in ('mixed','late'):
            for method in ('cv12','pinn_1000'):
                group=[lookup[n,s,timing,method] for s in seeds]
                differences=[100*abs(r[k]-v) for r in group for k,v in r.get('refinement_scores',{}).items()]
                diagnostics.append(dict(n_obs=n,timing=timing,method=method,completed=len(group),
                    bound_hits=sum(bool(r['at_bound']) for r in group),
                    smallest_ridge_selections=sum(r.get('selected_ridge')==min(cfg['ridges']) for r in group),
                    largest_ridge_selections=sum(r.get('selected_ridge')==max(cfg['ridges']) for r in group),
                    non_completed_budget_traces=sum(r.get('final_training_trace',{}).get('stage')!='completed_budget' for r in group if method=='pinn_1000'),
                    median_signal_rms_over_noise_sigma=float(np.median([r['signal_rms_over_noise_sigma'] for r in group])),
                    max_checked_grid_error_change_pp=max(differences) if differences else None,
                    total_job_seconds=sum(r['job_elapsed_s'] for r in group)))
    out.mkdir()
    for name,rows in [('absolute_errors',absolute),('paired_differences',paired),('timing_interactions',interactions),('paired_blocks',blocks),('diagnostics',diagnostics)]:
        save_csv(out/f'{name}.csv',rows)
    lines=['# Off-basis robustness evaluation','',
        '240 completed fits; 120 datasets; 30 independent paired blocks per observation count.',
        'All comparisons are secondary. Intervals are 95% percentile whole-block bootstrap intervals (10,000 resamples), pointwise rather than simultaneous.','',
        '## Median absolute errors (%)','',
        '| N | Timing | Outcome | CV12 | PINN |','|---|---|---|---:|---:|']
    for n in (10,50):
        for t in ('mixed','late'):
            for metric in metrics:
                v={r['method']:r['median_error_percent'] for r in absolute if (r['n_obs'],r['timing'],r['metric'])==(n,t,metric)}
                lines.append(f"| {n} | {t} | {metric} | {v['cv12']:.4f} | {v['pinn_1000']:.4f} |")
    lines+=['','## Paired diffusivity contrasts','',
            'D = PINN error minus CV12 error. Negative values favour the PINN.','',
            '| N | Timing | Median D (pp) | 95% interval | PINN lower error |','|---|---|---:|---|---:|']
    for r in paired:
        if r['metric']=='diffusivity':
            lines.append(f"| {r['n_obs']} | {r['timing']} | {r['median_difference_pp']:.4f} | [{r['ci95_low_pp']:.4f}, {r['ci95_high_pp']:.4f}] | {r['pinn_lower_error_blocks']}/30 |")
    lines+=['','## Timing interaction','', 'I = D(mixed) - D(late), computed within blocks before taking the median.','']
    for r in interactions:
        if r['metric']=='diffusivity':
            lines.append(f"N={r['n_obs']}: {r['median_interaction_pp']:.4f} pp; interval [{r['ci95_low_pp']:.4f}, {r['ci95_high_pp']:.4f}].")
    lines+=['','## Limits','',
        'This is one localized initial field and two observation schedules. It does not establish universal robustness.',
        'The twelve-mode initial projection error is about 17.86%; initial-field errors must be interpreted with that representation restriction.',
        'The new field has a different initial L2 energy from A/B despite fixed absolute noise. Changes across truth families cannot be attributed solely to basis mismatch.',
        'Do not infer paired effect sizes from differences of marginal medians. This study does not replace the original primary timing test.',
        'Inspect diagnostics.csv before final interpretation. Reported bound flags use the existing tolerances; zero flags do not exclude near-bound estimates.',
        'No settings have been selected using these evaluation results.','']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    provenance={'input_manifest_sha256':sha(source/'manifest.json'),'script_sha256':sha(Path(__file__)),
                'analysis_plan':cfg['analysis_plan'],'output_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print('OFF-BASIS ANALYSIS COMPLETE')
    print(f'Output: {out}')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
