"""Measure linearized beta influence when each body's data are omitted.

The best-fit rejection mask is held fixed. Initial-state and A2 parameters of
the omitted body are removed, and the remaining nuisance parameters are solved
again. These diagnostics are not replacement nonlinear estimation results.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import sparse


def solve(normal, rhs):
    scales = np.sqrt(np.diag(normal))
    balanced = normal / np.outer(scales, scales)
    covariance = np.linalg.inv(balanced) / np.outer(scales, scales)
    return covariance @ rhs, covariance


def profile(directory):
    fit = np.load(directory/'fit.npz')
    obs = np.load(directory/'observations.npz')
    summary = json.loads((directory/'summary.json').read_text())
    metadata = json.loads((directory/'observation_metadata.json').read_text())
    targets = set(summary['targets'])
    if len(targets) < 2:
        raise ValueError('Joint influence analysis requires at least two bodies')
    best = int(fit['best_iteration'])
    rows = np.flatnonzero(fit['active_flags_per_iteration'][:,best])
    h = fit['normalized_design_matrix']
    weights = sparse.load_npz(directory/'fit_weights.npz')
    residual = fit['best_residuals'].reshape(-1)
    names = np.asarray(fit['parameter_names'])
    factors = fit['normalization'].reshape(-1)
    current = fit['best_parameters'].reshape(-1)
    prior = fit['prior_inverse_covariance']/np.outer(factors,factors)
    reference_delta = (fit['parameter_history'][:,0]-current)*factors
    assert len(rows) == len(h)
    np.testing.assert_allclose(residual,fit['residual_history'][rows,best])
    beta = list(names).index('beta')
    correction,covariance = solve(h.T@(weights@h)+prior,
        h.T@(weights@residual)+prior@reference_delta)
    np.testing.assert_allclose(np.sqrt(covariance[beta,beta])/abs(factors[beta]),
        fit['formal_errors'][beta],rtol=2e-5)
    body_by_set = {}
    for set_id in np.unique(obs['all_set_ids'][rows]):
        bodies = {end['body'] for end in metadata[str(set_id)]['link_ends'].values()} & targets
        if len(bodies) != 1:
            raise ValueError(f'Expected one estimated body in observation set {set_id}')
        body_by_set[set_id] = bodies.pop()
    row_bodies = np.asarray([body_by_set[sid] for sid in obs['all_set_ids'][rows]])
    results = []
    for target in summary['targets']:
        keep_rows = row_bodies != target
        keep_columns = np.asarray([not name.startswith(target+':') for name in names])
        local_h = h[keep_rows][:,keep_columns]
        local_w = weights[keep_rows][:,keep_rows]
        local_prior = prior[np.ix_(keep_columns,keep_columns)]
        delta,cov = solve(local_h.T@(local_w@local_h)+local_prior,
            local_h.T@(local_w@residual[keep_rows])+local_prior@reference_delta[keep_columns])
        local_beta = list(names[keep_columns]).index('beta')
        value = float(current[beta]+delta[local_beta]/factors[beta])
        sigma = float(np.sqrt(cov[local_beta,local_beta])/abs(factors[beta]))
        results.append(dict(omitted_body=target,scalar_count=int((~keep_rows).sum()),
            removed_parameter_count=int((~keep_columns).sum()),linearized_beta=value,
            sigma_beta=sigma,pull_from_one=(value-1)/sigma))
    result = dict(method=__doc__,best_beta=float(current[beta]),
        full_linearized_beta=float(current[beta]+correction[beta]/factors[beta]),
        formal_beta_sigma=float(fit['formal_errors'][beta]),omissions=results)
    (directory/'beta_body_influence.json').write_text(json.dumps(result,indent=2)+'\n')
    for row in results:
        print(f"Without {row['omitted_body']:>6}: beta = {row['linearized_beta']:.10g} +/- {row['sigma_beta']:.6g}; pull = {row['pull_from_one']:.3g}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    profile(parser.parse_args().directory)
