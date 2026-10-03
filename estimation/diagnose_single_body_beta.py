"""Check source/transit influence in an exported single-body beta fit.

All comparisons are linearized at the saved best fit with its rejection mask
held fixed. They are diagnostics, not replacement nonlinear estimation results.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.stats import chi2


def solve(normal, rhs):
    scale = np.sqrt(np.diag(normal))
    balanced = normal / np.outer(scale, scale)
    covariance = np.linalg.inv(balanced) / np.outer(scale, scale)
    return covariance @ rhs, covariance


def diagnose(directory):
    fit = np.load(directory/'fit.npz')
    obs = np.load(directory/'observations.npz')
    metadata = json.loads((directory/'observation_metadata.json').read_text())
    h = fit['normalized_design_matrix']
    w = sparse.load_npz(directory/'fit_weights.npz')
    best = int(fit['best_iteration'])
    rows = np.flatnonzero(fit['active_flags_per_iteration'][:,best])
    assert len(rows) == len(h)
    residuals = fit['best_residuals'].reshape(-1)
    np.testing.assert_allclose(residuals,fit['residual_history'][rows,best])
    current = fit['best_parameters'].reshape(-1)
    factors = fit['normalization'].reshape(-1)
    prior = fit['prior_inverse_covariance']/np.outer(factors,factors)
    prior_rhs = prior @ ((fit['parameter_history'][:,0]-current)*factors)
    normal = h.T @ (w @ h)+prior
    delta,cov = solve(normal,h.T@(w@residuals)+prior_rhs)
    beta = list(fit['parameter_names']).index('beta')
    np.testing.assert_allclose(np.sqrt(cov[beta,beta])/abs(factors[beta]),fit['formal_errors'][beta],rtol=2e-5)
    set_ids=obs['all_set_ids'][rows]
    groups={}
    gaia_ids=[]
    for set_id in np.unique(set_ids):
        info=metadata[str(set_id)]
        receiver=next((end['body'] for role,end in info['link_ends'].items() if 'receiver' in role.lower()),'')
        kind=info['observable_type']
        group='Gaia' if receiver=='Gaia' else ('radar' if 'range' in kind.lower() or 'doppler' in kind.lower() else 'MPC')
        groups.setdefault(group,[]).append(set_id)
        if group=='Gaia': gaia_ids.append(set_id)
    cases=[(key,np.isin(set_ids,ids)) for key,ids in groups.items()]+[(f'Gaia transit set {sid}',set_ids==sid) for sid in gaia_ids]
    radar_rows=np.flatnonzero(np.isin(set_ids,groups.get('radar',[])))
    for index in radar_rows:
        mask=np.zeros(len(h),dtype=bool);mask[index]=True
        cases.append((f'Radar scalar row {int(rows[index])}',mask))
    result=[]
    for label,omit in cases:
        keep=~omit
        hk=h[keep];wk=w[keep][:,keep]
        try: update,subcov=solve(hk.T@(wk@hk)+prior,hk.T@(wk@residuals[keep])+prior_rhs)
        except np.linalg.LinAlgError: continue
        value=float(current[beta]+update[beta]/factors[beta]);sigma=float(np.sqrt(subcov[beta,beta])/abs(factors[beta]))
        row=dict(omitted=label,scalar_count=int(omit.sum()),linearized_beta=value,sigma_beta=sigma,beta_pull=(value-1)/sigma)
        if label.startswith(('Gaia transit','Radar scalar')):
            hg=h[omit];prediction=residuals[omit]-hg@update
            predictive_cov=np.linalg.inv(w[omit][:,omit].toarray())+hg@subcov@hg.T
            statistic=float(prediction@np.linalg.solve(predictive_cov,prediction))
            row.update(predictive_chi2=statistic,predictive_dof=int(omit.sum()),predictive_p_value=float(chi2.sf(statistic,omit.sum())))
        result.append(row)
    output=dict(method=__doc__,best_beta=float(current[beta]),full_linearized_beta=float(current[beta]+delta[beta]/factors[beta]),formal_beta_sigma=float(fit['formal_errors'][beta]),comparisons=result)
    (directory/'beta_influence.json').write_text(json.dumps(output,indent=2)+'\n')
    for row in result: print(row)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    diagnose(parser.parse_args().directory)
