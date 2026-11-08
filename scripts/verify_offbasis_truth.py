"""Verify off-basis truth before observations: spectral vs finite differences."""

import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags
from threadpoolctl import threadpool_limits
import offbasis_truth as truth

ROOT=Path(__file__).resolve().parents[1]
TIMES=np.array([.001,.005,.01,.02,.05,.1,.2,.4,.8,1.0])


def fd_solution(intervals, times, initial, rtol=1e-9, atol=1e-11):
    x=np.arange(1,intervals)/intervals
    size=intervals-1
    operator=.05*intervals**2*diags(
        [np.ones(size-1),-2*np.ones(size),np.ones(size-1)],[-1,0,1],format='csc')
    sol=solve_ivp(lambda t,y:operator@y,(0,float(max(times))),initial(x),
                  method='BDF',jac=operator,t_eval=times,rtol=rtol,atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return x,sol.y.T


def main():
    out=ROOT/'results/offbasis_truth_verification_v1'
    if out.exists():
        raise FileExistsError(f'Preserve existing verification: {out}')
    out.mkdir(parents=True)
    diagnostics={}
    with threadpool_limits(limits=1):
        # Temporal integration checked against the exact discrete-operator solution.
        sine=lambda x:np.sin(np.pi*x)+.5*np.sin(3*np.pi*x)
        x,u=fd_solution(128,TIMES,sine)
        k=np.array([1,3]); eigen=-4*.05*128**2*np.sin(k*np.pi/(2*128))**2
        exact=(np.exp(TIMES[:,None]*eigen)*np.array([1,.5]))@np.sin(k[:,None]*np.pi*x)
        temporal_test=float(np.max(np.abs(u-exact)))
        diagnostics['analytic_discrete_sine_test_max_abs']=temporal_test
        if temporal_test>1e-7:
            raise RuntimeError('Finite-difference time integrator failed analytic test.')
        print(f'Analytic discrete sine test: {temporal_test:.3g}',flush=True)

        x=np.linspace(0,1,1025)
        c=truth.grid(x,TIMES,modes=128,quadrature_points=512)
        f=truth.grid(x,TIMES,modes=256,quadrature_points=1024)
        finer=truth.grid(x,TIMES,modes=512,quadrature_points=2048)
        spectral_error=float(np.max(np.abs(f-c)))
        finer_error=float(np.max(np.abs(finer-f)))
        diagnostics.update(spectral_refinement_max_abs=spectral_error,
                           further_spectral_refinement_max_abs=finer_error)
        dense=np.linspace(0,1,8193)
        actual=truth.initial(dense)
        k=np.arange(1,13)*np.pi
        projected=truth.coefficients()[:12]@np.sin(k[:,None]*dense)
        norm=float(np.sqrt(np.trapezoid(actual**2,dense)))
        projection=float(np.sqrt(np.trapezoid((actual-projected)**2,dense)))/norm
        a=truth.coefficients()
        reconstructed=a@np.sin(np.arange(1,257)[:,None]*np.pi*dense)
        initial_error=float(np.max(np.abs(reconstructed-actual)))
        diagnostics.update(initial_l2_norm=norm,twelve_mode_projection_relative_l2=projection,
                           spectral_initial_reconstruction_max_abs=initial_error,
                           peak_location=truth.PEAK_LOCATION,normalization=truth.NORMALIZATION)
        np.testing.assert_allclose(truth.initial(np.array([0.,1.])),0,atol=0)
        np.testing.assert_allclose(truth.initial(truth.PEAK_LOCATION),1,rtol=1e-14)
        xp=np.array([.13,.35,.71]);tp=np.array([.01,.2,.8])
        np.testing.assert_allclose(truth.paired(xp,tp),np.diag(truth.grid(xp,tp)),atol=1e-13)
        errors=[]
        for intervals in (256,512,1024):
            xf,uf=fd_solution(intervals,TIMES,truth.initial)
            target=truth.grid(xf,TIMES)
            error=float(np.max(np.abs(uf-target)))
            errors.append(error)
            np.savez_compressed(out/f'fd_{intervals}.npz',x=xf,t=TIMES,fd=uf,spectral=target)
            print(f'FD {intervals} intervals: max absolute difference {error:.6g}',flush=True)
        xt,tight=fd_solution(1024,TIMES,truth.initial,rtol=1e-11,atol=1e-13)
        integration_change=float(np.max(np.abs(tight-uf)))
        order=[float(np.log2(errors[i]/errors[i+1])) for i in (0,1)]
        diagnostics.update(fd_intervals=[256,512,1024],fd_max_abs_errors=errors,
                           observed_spatial_orders=order,time_tolerance_refinement_max_abs=integration_change)
        passed=bool(errors[-1]<1e-4 and all(errors[i+1]<errors[i] for i in (0,1))
                    and spectral_error<1e-8 and finer_error<1e-8 and initial_error<1e-8
                    and integration_change<1e-7 and all(1.5<v<2.5 for v in order))
        diagnostics['passed']=passed
        np.savez_compressed(out/'initial_field.npz',x=dense,initial=actual,projection12=projected,
                            coefficients256=a)
    (out/'verification.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    provenance={'truth_source_sha256':sha(Path(truth.__file__)),
                'verification_script_sha256':sha(Path(__file__)),
                'output_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()},
                'note':'Numerical verification on stated grids/times, not a global error bound.'}
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print('OFF-BASIS TRUTH VERIFICATION COMPLETE')
    print(json.dumps(diagnostics,indent=2))
    print('No observations generated; no inverse fits run.')
    if not passed:
        raise SystemExit('Verification targets not met; investigate before data generation.')


if __name__=='__main__':
    main()
