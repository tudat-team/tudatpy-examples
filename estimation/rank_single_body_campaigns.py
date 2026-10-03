"""Combine completed single-body campaigns and select the best beta uncertainties."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def combine(campaigns, output, count):
    rows, absent, seen = [], [], set()
    settings = None
    for campaign in campaigns:
        manifest = json.loads((campaign/'manifest.json').read_text())
        signature = tuple(manifest[key] for key in ('baseline_sha256', 'start', 'end', 'iterations')) + (json.dumps(manifest.get('numerical_overrides', {}),sort_keys=True), json.dumps(manifest.get('diagnostic_overrides', {}),sort_keys=True), json.dumps(manifest.get('kernel_overrides', {}),sort_keys=True), manifest.get('force_model_diagnostic', ''), bool(manifest.get('fixed_beta_diagnostic', False)))
        if settings is None:
            settings = signature
        elif settings != signature:
            raise ValueError('Campaign scientific settings differ')
        for target in manifest['targets']:
            if target in seen:
                raise ValueError(f'Duplicate target {target}')
            seen.add(target)
            directory = campaign/'targets'/target
            summary = json.loads((directory/'summary.json').read_text())
            if summary['status'] == 'no_data':
                absent.append(target)
                continue
            if summary['status'] != 'completed':
                raise ValueError(f'Unfinished target {target}')
            fit = np.load(directory/'fit.npz')
            index = list(fit['parameter_names']).index('beta')
            sigma = float(np.sqrt(fit['covariance'][index, index]))
            if not np.isfinite(sigma) or sigma <= 0:
                raise ValueError(f'Invalid beta uncertainty for {target}')
            np.testing.assert_allclose(sigma, summary['beta_sigma'], rtol=1e-12)
            rows.append(dict(target=target, beta=summary['beta'], beta_sigma=sigma,
                iterations=summary['iterations'], best_iteration=summary['best_iteration'],
                reached_iteration_limit=summary['iterations'] >= summary['iteration_limit'],
                last_position_correction_m=summary['last_position_corrections_m'][target],
                weighted_rank=summary['weighted_rank'], parameter_count=summary['parameter_count'],
                active_scalars=summary['active_scalars'], result_directory=str(directory.resolve())))
    rows.sort(key=lambda row: (row['beta_sigma'], int(row['target'])))
    if len(rows) < count:
        raise ValueError(f'Only {len(rows)} completed fits for selection of {count}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['rank']+list(rows[0]))
        writer.writeheader()
        for i, row in enumerate(rows, 1):
            writer.writerow(dict(rank=i, **row))
    output.with_suffix('.json').write_text(json.dumps(dict(
        targets=[row['target'] for row in rows[:count]], ranking=rows,
        criterion='ascending returned formal beta uncertainty', no_data=absent,
        campaigns=[str(path.resolve()) for path in campaigns],
        scientific_settings=dict(zip(('baseline_sha256','start','end','iterations','numerical_overrides','diagnostic_overrides','kernel_overrides','force_model_diagnostic','fixed_beta_diagnostic'),settings))),indent=2)+'\n')
    for i,row in enumerate(rows[:count],1):
        print(f"{i:2d}. {row['target']:>6}: sigma_beta = {row['beta_sigma']:.8g}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaigns', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True, help='CSV/JSON output stem')
    parser.add_argument('--count', type=int, default=25)
    args = parser.parse_args()
    combine(args.campaigns, args.output, args.count)
