"""Validate saved estimation exports without refitting or querying services."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import sparse


def validate(directory):
    summary = json.loads((directory / 'summary.json').read_text())
    if summary['status'] == 'no_data':
        return
    assert summary['status'] == 'completed', directory
    assert summary.get('saved_state_iterations', 0) == 0
    fit = np.load(directory / 'fit.npz')
    obs = np.load(directory / 'observations.npz')
    eph = np.load(directory / 'ephemeris.npz')
    orbit = np.load(directory / 'orbit_diagnostics.npz')
    count = len(fit['parameter_names'])
    assert fit['covariance'].shape == (count, count)
    assert np.all(np.isfinite(fit['covariance']))
    assert np.all(np.diag(fit['covariance']) > 0)
    np.testing.assert_allclose(fit['formal_errors'], np.sqrt(np.diag(fit['covariance'])))
    beta = list(fit['parameter_names']).index('beta')
    np.testing.assert_allclose(summary['beta_sigma'], fit['formal_errors'][beta])
    np.testing.assert_allclose(fit['last_evaluated_parameters'],
        fit['parameter_history'][:, int(fit['last_evaluated_iteration'])])
    np.testing.assert_allclose(obs['active_residuals'],
        fit['residual_history'][obs['active_rows'], -1])
    np.testing.assert_allclose(obs['active_observations'],
        obs['all_observations'][obs['active_rows']])
    assert np.count_nonzero(~obs['rejected_mask']) == len(obs['active_rows'])
    assert sparse.load_npz(directory / 'observation_weights.npz').shape == (len(obs['all_observations']),)*2
    assert len(eph['state_epochs']) == len(eph['integration_states'])
    assert eph['comparison_epochs'][0] >= eph['integration_epochs'][0] + 10*summary.get("maximum_step",36*3600)
    assert eph['comparison_epochs'][-1] <= eph['integration_epochs'][-1] - 10*summary.get("maximum_step",36*3600)
    # Check rotation and epochwise covariance normalization independently.
    for i in range(len(orbit['targets'])):
        h = orbit['horizons_states'][:, i]
        radial = h[:, :3] / np.linalg.norm(h[:, :3], axis=1)[:, None]
        cross = np.cross(h[:, :3], h[:, 3:])
        cross /= np.linalg.norm(cross, axis=1)[:, None]
        along = np.cross(cross, radial)
        rotation = np.stack((radial, along, cross), axis=1)
        delta = np.einsum('nij,nj->ni', rotation,
            orbit['estimated_states'][:, i, :3] - h[:, :3])
        cov = rotation @ orbit['propagated_covariances'][:, i, :3, :3] @ rotation.transpose(0, 2, 1)
        sigma = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))
        np.testing.assert_allclose(orbit['differences_rsw'][:, i], delta, rtol=1e-8, atol=1e-5)
        np.testing.assert_allclose(orbit['formal_errors_rsw'][:, i], sigma, rtol=1e-8)
        np.testing.assert_allclose(orbit['normalized_differences_rsw'][:, i], delta/sigma, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(orbit['normalized_rms_rsw'][i], np.sqrt(np.mean((delta/sigma)**2, axis=0)), rtol=1e-8)
    print(f'Validated {directory.name}: {count} parameters, {len(obs["active_rows"])} active scalar observations')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    parser.add_argument('--completed-only', action='store_true')
    args = parser.parse_args()
    for directory in sorted((args.campaign / 'targets').iterdir()):
        if directory.is_dir():
            if args.completed_only and (not (directory/'summary.json').exists() or json.loads((directory/'summary.json').read_text())['status'] != 'completed'):
                continue
            validate(directory)
    for directory in sorted(args.campaign.glob('joint_*')):
        if (directory / 'summary.json').exists():
            if args.completed_only and json.loads((directory/'summary.json').read_text())['status'] != 'completed':
                continue
            validate(directory)
