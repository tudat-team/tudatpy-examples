"""Independent asteroid fits, beta ranking, and a joint fit of the best 20.

The companion multi-body example supplies all scientific settings. This runner
freezes a byte-identical copy of it per campaign and never modifies that file.
Run with the tudatpy-dev Python interpreter. Use --help for smoke/resume options.
Residuals/parameters are retained per iteration; states are exported only from
final propagated ephemerides. All numerical exports load without pickle.
"""
import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

RUNNER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

TARGETS = (
    '3200 1566 66146 137924 437844 138127 480883 468468 364136 33342 '
    '85989 85953 2100 99907 2062 153201 524522 2340 162004 276033 '
    '413260 242191 96590 369986 5786 152742 247517 345705 363505 66400 '
    '394130 465402 399457 431760 677579 612162 374158 455426 504181 '
    '386454 438116 267223 467372 40267 331471 136874 164201 105140 '
    '137925 533671'
).split()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def save_arrays(path, **arrays):
    import numpy as np
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    tmp.replace(path)


def retry_request(function, *args, **kwargs):
    import requests
    for attempt in range(6):
        try:
            return function(*args, **kwargs)
        except (requests.RequestException, TimeoutError) as exc:
            if attempt == 5:
                raise
            delay = min(60, 5*2**attempt)
            print(f'Request failed ({exc}); retrying in {delay} s', flush=True)
            time.sleep(delay)


def baseline(manifest, targets):
    spec = importlib.util.spec_from_file_location('campaign_baseline', manifest['snapshot'])
    od = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(od)
    od.TARGETS = list(targets)
    od.ACTIVE_TARGETS = list(targets)
    od.OBSERVATION_START = dt.datetime.fromisoformat(manifest['start'])
    od.OBSERVATION_END = dt.datetime.fromisoformat(manifest['end'])
    od.NUMBER_OF_ESTIMATION_ITERATIONS = manifest['iterations']
    od.RUN_ONLY_LAST_SETUP = True
    od.GAIA_ARCHIVE_PATHS = {
        target: Path(manifest['data_directory']) / f'gaia_{target}_fpr.parquet'
        for target in targets
    }
    # New catalog omissions can be excluded without changing the frozen example.
    original = od.BatchMPC
    excluded = set(manifest.get('extra_excluded_stations', []))
    class CampaignBatchMPC(original):
        def get_observations(self, *args, **kwargs):
            return retry_request(super().get_observations, *args, **kwargs)

        def filter(self, *args, **kwargs):
            kwargs['observatories_exclude'] = sorted(
                excluded | set(kwargs.get('observatories_exclude', [])))
            return super().filter(*args, **kwargs)
    od.BatchMPC = CampaignBatchMPC
    original_horizons = od.HorizonsQuery
    class RetryingHorizonsQuery(original_horizons):
        def cartesian(self, *args, **kwargs):
            return retry_request(super().cartesian, *args, **kwargs)
    od.HorizonsQuery = RetryingHorizonsQuery
    original_radar = od.JPLRadarQuery
    class RetryingRadarQuery(original_radar):
        def to_radar_data(self, *args, **kwargs):
            return retry_request(super().to_radar_data, *args, **kwargs)
    od.JPLRadarQuery = RetryingRadarQuery
    original_gaia = od.load_gaia_astrometry
    od.load_gaia_astrometry = lambda *args, **kwargs: retry_request(original_gaia, *args, **kwargs)
    return od


def parameter_labels(od):
    names, units = [], []
    for target in od.ACTIVE_TARGETS:
        names.extend(f'{target}:{axis}' for axis in ('x', 'y', 'z', 'vx', 'vy', 'vz'))
        units.extend(['m'] * 3 + ['m/s'] * 3)
    if od.ESTIMATE_YARKOVSKY:
        names.extend(f'{target}:A2' for target in od.ACTIVE_TARGETS)
        units.extend(['m/s^2'] * len(od.ACTIVE_TARGETS))
    if od.ESTIMATE_BETA:
        names.append('beta'); units.append('dimensionless')
    if od.ESTIMATE_SUN_J2:
        names.append('Sun:C20'); units.append('dimensionless')
    return names, units


def export_fit(od, output, dataset, estimator, destination, label, first, last, epoch):
    import numpy as np
    from scipy import sparse
    if output.exception_during_inversion or output.exception_during_propagation:
        raise RuntimeError('Estimator reported inversion/propagation failure')
    covariance = np.asarray(output.covariance)
    if not np.all(np.isfinite(covariance)) or not np.all(np.diag(covariance) > 0):
        raise RuntimeError('Invalid parameter covariance')
    if len(output.simulation_results_per_iteration):
        raise RuntimeError('Unexpected saved per-iteration propagation results')
    parameters = np.asarray(output.parameter_history)
    names, units = parameter_labels(od)
    if covariance.shape != (len(names), len(names)):
        raise RuntimeError('Unexpected parameter layout')
    indices = od.global_parameter_indices()
    beta_index = indices['beta']
    grid = od.integration_epochs(estimator)
    margin = 10 * od.INTEGRATOR_MAXIMUM_STEP
    if grid[-1] - grid[0] <= 2 * margin:
        raise RuntimeError('Observation span too short for trimmed diagnostics')
    epochs = np.linspace(grid[0] + margin, grid[-1] - margin, 300)
    propagated = od.estimation_analysis.propagate_covariance(
        covariance, estimator.state_transition_interface, list(epochs))
    states = np.array([
        [od.propagated_ephemeris(estimator, target).cartesian_state(float(t)) for target in od.ACTIVE_TARGETS]
        for t in grid[1:-1]])
    comparison_states = np.array([
        [od.propagated_ephemeris(estimator, target).cartesian_state(float(t)) for target in od.ACTIVE_TARGETS]
        for t in epochs])
    # All 6x6 body blocks, including position/velocity correlations at each epoch.
    body_covariances = np.array([
        [np.asarray(propagated[t])[6*i:6*i+6, 6*i:6*i+6] for i in range(len(od.ACTIVE_TARGETS))]
        for t in epochs])
    prior = np.zeros_like(covariance)
    if od.ESTIMATE_SUN_J2:
        prior[indices['C20'],indices['C20']] = 1/(od.SUN_J2_PRIOR_SIGMA/np.sqrt(5))**2
    save_arrays(destination / 'fit.npz', parameter_names=names, parameter_units=units,
        parameter_history=parameters, last_evaluated_parameters=od.last_iteration_parameters(output),
        covariance=covariance, formal_errors=np.sqrt(np.diag(covariance)),
        correlation=covariance/np.sqrt(np.outer(np.diag(covariance), np.diag(covariance))),
        residual_history=output.residual_history, best_iteration=output.best_iteration,
        last_evaluated_iteration=np.asarray(output.residual_history).shape[1]-1,
        normalized_design_matrix=output.normalized_design_matrix,
        normalization=output.normalization_terms, estimation_epoch=epoch,
        prior_inverse_covariance=prior)
    sparse.save_npz(destination / 'fit_weights.npz', sparse.csc_matrix(output.weight_matrix))
    active = od.observation_scalar_data(dataset)
    active['observable_types'] = np.array([str(v) for v in active['observable_types']])
    all_vector = dataset.observation_vector_data(include_rejected=True)
    active_vector = dataset.observation_vector_data(include_rejected=False)
    all_ids = np.asarray(all_vector.observation_ids)
    all_components = np.asarray(all_vector.scalar_component_ids)
    active_rows = np.array([all_vector.vector_row(oid, component)
        for oid, component in dataset.get_scalar_components(od.observations.observation_query.active, ordering='estimation')])
    save_arrays(destination / 'observations.npz', **active,
        active_residuals=od.last_iteration_residuals(output, dataset),
        active_observations=active_vector.observation_vector,
        all_observations=all_vector.observation_vector, all_times=np.array([float(t) for t in all_vector.times]),
        all_observation_ids=all_ids, all_components=all_components,
        all_set_ids=all_vector.set_ids, active_rows=active_rows,
        rejected_mask=~np.isin(np.arange(len(all_ids)), active_rows))
    sparse.save_npz(destination / 'observation_weights.npz', sparse.csc_matrix(all_vector.sparse_weight_matrix))
    metadata = dataset.get_data(fields=('metadata',), ordering='estimation')['metadata']
    write_json(destination / 'observation_metadata.json', {
        str(key): {'observable_type': str(value['observable_type']),
            'link_ends': {str(role): {'body': end.body_name, 'reference_point': end.reference_point}
                for role, end in value['link_definition'].link_ends.items()}}
        for key, value in metadata.items()})
    for target in od.ACTIVE_TARGETS:
        gaia = od.gaia_residual_data(output, dataset, target)
        if gaia is not None:
            save_arrays(destination / f'gaia_{target}.npz',
                **{key: value for key, value in gaia.items() if key != 'table'},
                **{key: gaia['table'][key].to_numpy() for key in gaia['table'].columns})
    save_arrays(destination / 'ephemeris.npz', targets=od.ACTIVE_TARGETS,
        integration_epochs=grid, state_epochs=grid[1:-1], integration_states=states,
        comparison_epochs=epochs, comparison_states=comparison_states,
        propagated_covariances=body_covariances)
    final_updates = {target: float(np.linalg.norm(parameters[6*i:6*i+3,-1]-parameters[6*i:6*i+3,-2]))
        for i, target in enumerate(od.ACTIVE_TARGETS)}
    h = np.asarray(output.normalized_design_matrix)
    factors = np.asarray(output.normalization_terms).reshape(-1)
    normal = h.T @ (output.weight_matrix @ h) + prior/np.outer(factors,factors)
    scales = np.sqrt(np.diag(normal))
    balanced = normal/np.outer(scales,scales)
    rank = int(np.linalg.matrix_rank(balanced))
    condition = float(np.linalg.cond(balanced))
    print(f'Weighted system including prior: rank {rank}/{len(names)}, balanced condition {condition:.6g}',flush=True)
    summary = dict(status='fit_saved', targets=od.ACTIVE_TARGETS, setup=label,
        runner_sha256=RUNNER_SHA256,
        saved_state_iterations=len(output.simulation_results_per_iteration),
        weighted_rank=rank,parameter_count=len(names),balanced_condition=condition,
        beta=float(od.last_iteration_parameters(output)[beta_index]),
        beta_sigma=float(np.sqrt(covariance[beta_index,beta_index])),
        best_iteration=int(output.best_iteration), iterations=int(output.residual_history.shape[1]),
        iteration_limit=od.NUMBER_OF_ESTIMATION_ITERATIONS,
        observation_bounds=[first,last], estimation_epoch=epoch,
        active_scalars=int(len(active_rows)), rejected_scalars=int(len(all_ids)-len(active_rows)),
        last_position_corrections_m=final_updates,
        baseline_sha256=digest(manifest_path(destination)['snapshot']),
        covariance_iteration='returned estimator covariance; orbit/residuals use last evaluated iteration')
    write_json(destination / 'summary.json', summary)
    print(f"Saved fit: beta = {summary['beta']:.12g}, sigma = {summary['beta_sigma']:.8g}", flush=True)
    od.print_residual_summary(label, output, dataset)
    for target in od.ACTIVE_TARGETS:
        od.print_residual_summary(label, output, dataset, target)
    if od.ESTIMATE_YARKOVSKY: od.print_yarkovsky_result(output)
    od.print_beta_result(output)
    if od.ESTIMATE_SUN_J2: od.print_solar_j2_result(output)
    od.print_beta_correlations(output)


def manifest_path(destination):
    directory = Path(destination).resolve()
    while not (directory / 'manifest.json').exists():
        if directory == directory.parent: raise FileNotFoundError('Campaign manifest not found')
        directory = directory.parent
    return read_json(directory / 'manifest.json')


def finish_diagnostics(destination, manifest):
    import numpy as np
    od = baseline(manifest, read_json(destination / 'summary.json')['targets'])
    saved = np.load(destination / 'ephemeris.npz')
    epochs = saved['comparison_epochs']
    orbit = dict(epochs=epochs, targets=saved['targets'], estimated_states=saved['comparison_states'],
                 propagated_covariances=saved['propagated_covariances'])
    reference, differences, errors = [], [], []
    for i, target in enumerate(saved['targets']):
        cache = destination / f'horizons_{target}.npz'
        if cache.exists():
            states = np.load(cache)['states']
        else:
            print(f'Querying Horizons at {len(epochs)} diagnostic epochs for {target}...', flush=True)
            states = od.HorizonsQuery(query_id=f'{target};', location=od.HORIZONS_ORIGIN,
                epoch_list=list(epochs), extended_query=True).cartesian(frame_orientation=od.FRAME_ORIENTATION)[:,1:]
            save_arrays(cache, epochs=epochs, states=states)
        rotations = np.array([od.inertial_to_rsw_rotation_matrix(state) for state in states])
        diff = np.einsum('nij,nj->ni', rotations, saved['comparison_states'][:,i,:3]-states[:,:3])
        p = saved['propagated_covariances'][:,i,:3,:3]
        cov_rsw = rotations @ p @ rotations.transpose(0,2,1)
        sigma = np.sqrt(np.clip(np.diagonal(cov_rsw,axis1=1,axis2=2),0,None))
        if not np.all(np.isfinite(sigma) & (sigma > 0)): raise RuntimeError('Invalid propagated RSW formal errors')
        reference.append(states); differences.append(diff); errors.append(sigma)
        print(f'{target} orbit minus Horizons:')
        for j, axis in enumerate('RSW'):
            print(f'  {axis}: RMS = {np.sqrt(np.mean(diff[:,j]**2)):.6g} m, normalized RMS = {np.sqrt(np.mean((diff[:,j]/sigma[:,j])**2)):.6g}')
    orbit.update(horizons_states=np.stack(reference,axis=1),
        differences_rsw=np.stack(differences,axis=1), formal_errors_rsw=np.stack(errors,axis=1))
    orbit['normalized_differences_rsw'] = orbit['differences_rsw']/orbit['formal_errors_rsw']
    orbit['rms_rsw'] = np.sqrt(np.mean(orbit['differences_rsw']**2,axis=0))
    orbit['normalized_rms_rsw'] = np.sqrt(np.mean(orbit['normalized_differences_rsw']**2,axis=0))
    save_arrays(destination / 'orbit_diagnostics.npz', **orbit)
    plot_saved(destination)
    summary = read_json(destination / 'summary.json')
    summary['status'] = 'completed'
    summary['orbit_rms_m'] = orbit['rms_rsw'].tolist()
    summary['orbit_normalized_rms'] = orbit['normalized_rms_rsw'].tolist()
    write_json(destination / 'summary.json', summary)
    print(f'Completed successfully. Results: {destination}', flush=True)


def plot_saved(destination):
    """Recreate figures entirely from saved arrays, without Tudat or network."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    eph = np.load(destination / 'ephemeris.npz')
    year = lambda t: 2000 + np.asarray(t)/(365.25*86400)
    fig, ax = plt.subplots()
    ax.plot(year(eph['integration_epochs'][:-1]),np.diff(eph['integration_epochs'])/3600)
    ax.set(xlabel='Year (TDB)',ylabel='Time step [h]')
    fig.savefig(destination / 'time_steps.pdf'); plt.close(fig)
    obs = np.load(destination / 'observations.npz')
    with PdfPages(destination / 'residuals.pdf') as pdf:
        groups = sorted(set(zip(obs['observable_types'],obs['space_astrometry'],obs['components'],obs['targets'])))
        for kind, space, component, target in groups:
            mask = (obs['observable_types']==kind)&(obs['space_astrometry']==space)&(obs['components']==component)&(obs['targets']==target)
            scale = 180/np.pi*3600 if 'angular' in kind.lower() else 1
            unit = 'arcsec' if scale != 1 else ('Hz' if 'doppler' in kind.lower() else 'm')
            fig, axes = plt.subplots(2,1,sharex=True)
            axes[0].scatter(year(obs['times'][mask]),obs['active_residuals'][mask]*scale,s=2)
            axes[1].scatter(year(obs['times'][mask]),obs['active_residuals'][mask]*np.sqrt(obs['weights'][mask]),s=2)
            axes[0].set(title=f'{target}: {kind}, component {component}, space={space}',ylabel=unit)
            axes[1].set(xlabel='Year (TDB)',ylabel='Normalized residual')
            pdf.savefig(fig); plt.close(fig)
        for archive in sorted(destination.glob('gaia_*.npz')):
            data = np.load(archive)
            fig, axes = plt.subplots(2,2,sharex=True)
            for j, name in enumerate(('AL','AC')):
                axes[0,j].scatter(year(data['epoch']),data['scan_residuals'][:,j]*180/np.pi*3600*1000,s=2)
                axes[1,j].scatter(year(data['epoch']),data['scan_residuals'][:,j]/data['scan_sigmas'][:,j],s=2)
                axes[0,j].set(title=f'{archive.stem}: {name}',ylabel='mas')
                axes[1,j].set(xlabel='Year (TDB)',ylabel='Normalized residual')
            pdf.savefig(fig); plt.close(fig)
    orbit = np.load(destination / 'orbit_diagnostics.npz')
    with PdfPages(destination / 'orbit_differences.pdf') as pdf:
        for i,target in enumerate(orbit['targets']):
            fig, axes = plt.subplots(3,2,sharex=True,figsize=(10,8))
            for j,axis in enumerate('RSW'):
                axes[j,0].plot(year(orbit['epochs']),orbit['differences_rsw'][:,i,j])
                axes[j,1].plot(year(orbit['epochs']),orbit['normalized_differences_rsw'][:,i,j])
                axes[j,0].set(ylabel=f'{axis} [m]')
                axes[j,1].set(ylabel=f'{axis} / sigma')
            fig.suptitle(str(target)); axes[2,0].set_xlabel('Year (TDB)'); axes[2,1].set_xlabel('Year (TDB)')
            pdf.savefig(fig); plt.close(fig)


def worker(args):
    manifest = read_json(args.campaign / 'manifest.json')
    destination = args.destination
    destination.mkdir(parents=True,exist_ok=True)
    if (destination / 'summary.json').exists():
        status = read_json(destination / 'summary.json')['status']
        if status in ('completed','no_data'): return
        if status == 'fit_saved':
            finish_diagnostics(destination,manifest); return
    od = baseline(manifest,args.targets)
    print(f"Independent/joint estimation: {', '.join(args.targets)}; {manifest['start']} to {manifest['end']}",flush=True)
    od.spice.load_standard_kernels()
    try:
        setups,first,last,epoch = od.load_tracking_data()
    except RuntimeError as exc:
        if str(exc) != 'No target has observations in the selected interval.': raise
        write_json(destination / 'summary.json',dict(status='no_data',targets=args.targets)); return
    label = next(reversed(setups))
    output,dataset,estimator = od.perform_estimation(*setups[label],first,last,epoch)
    export_fit(od,output,dataset,estimator,destination,label,first,last,epoch)
    finish_diagnostics(destination,manifest)


def rank(campaign, manifest):
    rows = []
    for target in manifest['targets']:
        file = campaign / 'targets' / target / 'summary.json'
        if not file.exists(): raise RuntimeError(f'Missing outcome for {target}')
        summary = read_json(file)
        if summary['status'] not in ('completed','no_data'): raise RuntimeError(f'Unfinished outcome for {target}')
        if summary['status'] == 'completed':
            rows.append(dict(target=target,beta=summary['beta'],beta_sigma=summary['beta_sigma'],
                iterations=summary['iterations'],best_iteration=summary['best_iteration'],
                reached_iteration_limit=summary['iterations']>=summary['iteration_limit'],
                last_position_correction_m=summary['last_position_corrections_m'][target],
                active_scalars=summary['active_scalars'],result_directory=str(file.parent)))
    rows.sort(key=lambda row:(row['beta_sigma'],int(row['target'])))
    with (campaign / 'beta_ranking.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=['rank']+list(rows[0]) if rows else ['rank','target'])
        writer.writeheader()
        for i,row in enumerate(rows,1): writer.writerow(dict(rank=i,**row))
    selected = [row['target'] for row in rows[:20]]
    write_json(campaign / 'selected_top20.json',dict(targets=selected,criterion='ascending returned formal beta uncertainty',ranking=rows))
    print('Beta uncertainty ranking:',flush=True)
    for i,row in enumerate(rows,1): print(f"{i:2d}. {row['target']:>6}: {row['beta_sigma']:.8g}",flush=True)
    return selected


def spawn_worker(args,targets,destination,attempt):
    env = dict(os.environ,MPLBACKEND='Agg',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    path = destination / 'terminal.log'
    destination.mkdir(parents=True,exist_ok=True)
    log = path.open('a' if path.exists() else 'w')
    command = [sys.executable,'-u',str(Path(__file__).resolve()),'--mode','worker',
        '--campaign',str(args.campaign),'--destination',str(destination),'--targets',*targets]
    process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
    log.close()
    print(f"Started {', '.join(targets)} (PID {process.pid}); output: {path}",flush=True)
    return process,path


def run_queue(args,manifest,jobs):
    pending = list(jobs)
    running = {}
    attempts = {}
    while pending or running:
        while pending and len(running)<args.workers:
            targets,destination = pending.pop(0)
            if (destination/'summary.json').exists() and read_json(destination/'summary.json')['status'] in ('completed','no_data'):
                continue
            attempt = attempts.get(str(destination),0)+1
            attempts[str(destination)] = attempt
            process,path = spawn_worker(args,targets,destination,attempt)
            running[process.pid] = process,targets,destination,path
        for pid,(process,targets,destination,path) in list(running.items()):
            result = process.poll()
            if result is None: continue
            del running[pid]
            if result==0:
                print(f"Finished {', '.join(targets)}: {read_json(destination/'summary.json')['status']}",flush=True)
                (destination/'failure.json').unlink(missing_ok=True)
            else:
                tail = path.read_text(errors='replace')[-12000:]
                # Errors identifying unavailable optical stations are excluded on retry.
                matches = re.findall(r'(?:station|observatory)[^\n]*?\b([A-Z][0-9]{2}|[0-9]{3})\b',tail,re.I)
                if matches:
                    manifest['extra_excluded_stations'] = sorted(set(manifest.get('extra_excluded_stations',[]))|set(matches))
                    write_json(args.campaign/'manifest.json',manifest)
                print(f"Failed {', '.join(targets)} (exit {result}); {path}\n{tail[-2500:]}",flush=True)
                if attempts[str(destination)] >= 3:
                    # Preserve running jobs; finish others before reporting unresolved failures.
                    write_json(destination/'failure.json',dict(exit_code=result,log=str(path)))
                else: pending.append((targets,destination))
        time.sleep(3)
    failed = [str(destination) for _,destination in jobs if not (destination/'summary.json').exists()
        or read_json(destination/'summary.json')['status'] not in ('completed','no_data')]
    if failed: raise RuntimeError('Unresolved jobs: '+', '.join(failed))


def campaign(args):
    args.campaign.mkdir(parents=True,exist_ok=True)
    manifest_file = args.campaign/'manifest.json'
    if manifest_file.exists():
        manifest = read_json(manifest_file)
    else:
        source = args.baseline.resolve()
        snapshot = args.campaign/'source'/source.name
        snapshot.parent.mkdir(exist_ok=True)
        shutil.copy2(source,snapshot)
        shutil.copy2(Path(__file__),snapshot.parent/Path(__file__).name)
        spec = importlib.util.spec_from_file_location('settings',source)
        od = importlib.util.module_from_spec(spec); spec.loader.exec_module(od)
        manifest = dict(schema=1,targets=args.targets or TARGETS,snapshot=str(snapshot.resolve()),
            original_source=str(source),baseline_sha256=digest(source),data_directory=str(source.parent),
            start=args.start or '1980-01-01T00:00:00',end=args.end or od.OBSERVATION_END.isoformat(),
            iterations=args.iterations or od.NUMBER_OF_ESTIMATION_ITERATIONS,
            extra_excluded_stations=[],python=sys.executable,runner_sha256=digest(__file__))
        write_json(manifest_file,manifest)
    if digest(manifest['original_source']) != manifest['baseline_sha256']:
        raise RuntimeError('Protected joint source has changed; refusing mixed campaign')
    jobs = [([target],args.campaign/'targets'/target) for target in manifest['targets']]
    run_queue(args,manifest,jobs)
    selected = rank(args.campaign,manifest)
    if not args.skip_joint:
        if len(selected)<20 and len(manifest['targets'])>=20: raise RuntimeError('Fewer than 20 valid fits')
        destination = args.campaign/'joint_top20'
        args.workers = 1
        destination.mkdir(parents=True,exist_ok=True)
        (destination/'terminal.log').touch(exist_ok=True)
        if shutil.which('gedit'):
            subprocess.Popen(['gedit',str(destination/'terminal.log')],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        run_queue(args,manifest,[(selected,destination)])
        if shutil.which('gedit'):
            subprocess.Popen(['gedit',str(destination/'terminal.log')],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    if digest(manifest['original_source']) != manifest['baseline_sha256']:
        raise RuntimeError('Protected joint source changed during campaign')
    print(f'Campaign complete: {args.campaign}',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['campaign','worker','rank','plot'],default='campaign')
    parser.add_argument('--campaign',type=Path,default=Path(__file__).with_name('single_body_beta_campaign'))
    parser.add_argument('--baseline',type=Path,default=Path(__file__).with_name('mpc_radar_gaia_estimation_multi_body.py'))
    parser.add_argument('--destination',type=Path)
    parser.add_argument('--targets',nargs='+')
    parser.add_argument('--workers',type=int,choices=range(1,9),default=5)
    parser.add_argument('--start',help='Override observation cutoff (ISO UTC; default 1980-01-01)')
    parser.add_argument('--end',help='Override end (default companion script setting)')
    parser.add_argument('--iterations',type=int,help='Override only for smoke tests')
    parser.add_argument('--skip-joint',action='store_true',help='Validate independent workers without joint fit')
    args = parser.parse_args()
    args.campaign = args.campaign.resolve()
    if args.destination: args.destination = args.destination.resolve()
    if args.mode=='worker': worker(args)
    elif args.mode=='plot': plot_saved(args.destination)
    elif args.mode=='rank': rank(args.campaign,read_json(args.campaign/'manifest.json'))
    else: campaign(args)


if __name__=='__main__':
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'): stream.reconfigure(line_buffering=True)
    try: main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
