"""Joint MPC/radar/Gaia asteroid-state and (16) Psyche GM estimation.

Reuse mpc_radar_gaia_estimation.py's environment, observation pipelines,
weights, prefit rejection, propagation and estimator settings. Supply up to
ten MPC numbers; Psyche is placed first. The encounter set is configurable,
not selected by this script. Examples:

    python psyche_mass_estimation.py --bodies 16 48542
    python psyche_mass_estimation.py --bodies 16 48542 67125
    python psyche_mass_estimation.py --bodies 673 --fixed-psyche-gm

The last invocation is the single-body compatibility check. All observation
types use the default 1900-01-01 cutoff; --observation-start overrides it.
48542 and 67125 are validation examples from Farnocchia et al. (2024), Table 1
(doi:10.3847/1538-3881/ad50ca); they are not a fixed encounter selection.
Outputs include a PDF with each body's existing residual/orbit diagnostics,
joint covariance/correlation arrays, state corrections and Psyche GM results.
"""

import argparse
import datetime
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

import mpc_radar_gaia_estimation as od


ESTIMATED_BODIES = []  # Set MPC numbers here or supply --bodies.
MAX_ESTIMATED_BODIES = 10
OBSERVATION_DATA = {}  # Per-body overrides, e.g. {"16": {"gaia": False}}.
PHOTOCENTER_RADII = {"67125": 2500.0}  # User-supplied Wikipedia estimate; SBDB for other bodies [m].
PROPAGATION_PRINT_INTERVAL = od.constants.JULIAN_YEAR


def normalize_bodies(body_numbers, estimate_psyche_gm=True):
    """Keep the requested ordering, with Psyche first, and validate the limit."""
    selected = [str(int(body)) for body in body_numbers]
    if not selected or len(selected) > MAX_ESTIMATED_BODIES:
        raise ValueError("Select one to ten asteroid MPC numbers with --bodies or ESTIMATED_BODIES.")
    if len(set(selected)) != len(selected) or any(int(body) <= 0 for body in selected):
        raise ValueError("Estimated bodies must be distinct positive MPC numbers.")
    if estimate_psyche_gm and ("16" not in selected or len(selected) < 2):
        raise ValueError("Mass estimation requires 16 Psyche and at least one additional asteroid.")
    return (["16"] + [body for body in selected if body != "16"]) if "16" in selected else selected


def load_body_data(body, overrides=None, mpc_only=False):
    """Use the existing per-source conversion; absence of one source is allowed."""
    options = dict(OBSERVATION_DATA.get(body, {}))
    options.update(overrides or {})
    optical, optical_extra, radar, radar_extra = od.load_tracking_data(body)
    if not options.get("mpc", True):
        optical, optical_extra = [], []
    if mpc_only or not options.get("radar", True):
        radar, radar_extra = [], []
    gaia = None
    if not mpc_only and options.get("gaia", True):
        gaia = od.load_gaia_astrometry(body, options.get("gaia_archive"))
    if not optical and not radar and gaia is None:
        raise RuntimeError(f"No selected observations for asteroid {body}.")
    return {"mpc": optical, "mpc_extra": optical_extra,
            "radar": radar, "radar_extra": radar_extra, "gaia": gaia}


def merge_gaia(body_data):
    """One geocentric Gaia ephemeris and unchanged transit blocks for all targets."""
    tables = [data["gaia"].table for data in body_data.values() if data["gaia"] is not None]
    return od.GaiaAstrometry(pd.concat(tables, ignore_index=True)) if tables else None


def load_reference_states(selected, body_data):
    """Use the existing interval buffers, common reference epoch and Horizons initialization."""
    tracks = [track for data in body_data.values() for key in ("mpc", "radar") for track in data[key]]
    bounds = list(od.observation_epoch_bounds(tracks)) if tracks else []
    bounds.extend(float(epoch) for data in body_data.values() if data["gaia"] is not None
                  for epoch in (data["gaia"].table.epoch.min(), data["gaia"].table.epoch.max()))
    first_epoch, final_epoch = min(bounds) - od.PROPAGATION_BUFFER, max(bounds) + od.PROPAGATION_BUFFER
    initial_epoch = max(0.0, 0.5 * (min(bounds) + max(bounds)))
    histories, initial_states = {}, []
    for body in selected:
        print(f"Loading Horizons reference ephemeris for {body}...", flush=True)
        states = od.HorizonsQuery(
            query_id=f"{body};", location=od.HORIZONS_ORIGIN,
            epoch_start=float(first_epoch), epoch_end=float(final_epoch),
            epoch_step="1d", extended_query=True,
        ).cartesian(frame_orientation=od.FRAME_ORIENTATION)
        histories[body] = dict(zip(states[:, 0], states[:, 1:]))
        ephemeris = od.environment_setup.create_body_ephemeris(
            od.environment_setup.ephemeris.tabulated(
                histories[body], od.PROPAGATION_CENTRAL_BODY, od.FRAME_ORIENTATION), body)
        initial_states.extend(ephemeris.cartesian_state(float(initial_epoch)))
    return initial_epoch, np.asarray(initial_states), first_epoch, final_epoch, histories


def estimation_setups(body_data):
    """Retain the existing comparison setups when each selected body has data."""
    setups = {}
    if all(data["mpc"] for data in body_data.values()):
        setups["MPC astrometry"] = (
            [t for data in body_data.values() for t in data["mpc"]],
            [s for data in body_data.values() for s in data["mpc_extra"]], None)
    tracks = [t for data in body_data.values() for key in ("mpc", "radar") for t in data[key]]
    extras = [s for data in body_data.values() for key in ("mpc_extra", "radar_extra") for s in data[key]]
    any_radar = any(data["radar"] for data in body_data.values())
    if any_radar and all(data["mpc"] or data["radar"] for data in body_data.values()):
        setups["MPC astrometry and JPL radar"] = (tracks, extras, None)
    gaia = merge_gaia(body_data)
    if gaia is not None:
        label = "MPC astrometry, JPL radar and Gaia" if any_radar else "MPC astrometry and Gaia"
        setups[label] = (tracks, extras, gaia)
    if not setups:
        raise RuntimeError("No estimation setup supplies observations for every selected body.")
    return setups


def dataset_for_body(dataset, body):
    """Keep complete observation sets and covariance blocks for one observed target."""
    q = od.observations.observation_query
    condition = None
    for set_id, metadata in dataset.get_data(fields=("metadata",))["metadata"].items():
        link = metadata["link_definition"]
        if any(end.body_name == body for end in link.link_ends.values()):
            selected = q.set_id == set_id
            condition = selected if condition is None else condition | selected
    if condition is None:
        raise RuntimeError(f"The joint dataset has no observations of {body}.")
    return dataset.create_new_and_keep(condition)


def output_for_body(output, dataset, body_dataset):
    """Select residual rows by preserved event/component IDs, including rejected rows."""
    indices = {component: index for index, component in enumerate(
        dataset.get_scalar_components(ordering="estimation"))}
    rows = [indices[component] for component in body_dataset.get_scalar_components(ordering="estimation")]
    return SimpleNamespace(
        residual_history=np.asarray(output.residual_history)[rows],
        active_flags_per_iteration=np.asarray(output.active_flags_per_iteration)[rows],
    )


def scalar_observation_metadata(dataset, selected):
    """Save body/type/event labels in the same full row order as residual_history."""
    data = dataset.get_data(fields=("times", "observation_ids", "set_ids", "metadata", "scalar_components"),
                            ordering="estimation")
    events = dict(zip(data["observation_ids"], zip(data["times"], data["set_ids"])))
    observed_bodies = {}
    for set_id, metadata in data["metadata"].items():
        targets = {end.body_name for end in metadata["link_definition"].link_ends.values()} & set(selected)
        if len(targets) != 1:
            raise RuntimeError(f"Observation set {set_id} has an ambiguous estimated target: {targets}.")
        observed_bodies[set_id] = targets.pop()
    scalar_events = [events[event] for event, component in data["scalar_components"]]
    return {"scalar_times": np.asarray([float(time) for time, set_id in scalar_events]),
            "scalar_bodies": np.asarray([observed_bodies[set_id] for time, set_id in scalar_events]),
            "scalar_observable_types": np.asarray([str(data["metadata"][set_id]["observable_type"])
                                                   for time, set_id in scalar_events]),
            "scalar_receivers": np.asarray([data["metadata"][set_id]["link_definition"].link_ends[
                                             od.observable_models_setup.links.receiver].body_name
                                             for time, set_id in scalar_events]),
            "scalar_components": np.asarray(data["scalar_components"])}


def parameter_labels(parameter_set):
    labels = [""] * parameter_set.parameter_set_size
    for start, size, description in od.parameter_layout(parameter_set):
        for offset in range(size):
            component = ("x", "y", "z", "vx", "vy", "vz")[offset] if size == 6 else str(offset)
            labels[start + offset] = f"{description}: {component}" if size > 1 else description
    return labels


def report_joint_result(label, result, selected, initial_states, estimate_psyche_gm):
    output, dataset, estimator, gaia, setup = result
    values = od.last_iteration_parameters(output)
    covariance = np.asarray(output.covariance)
    sigma = np.sqrt(np.clip(np.diag(covariance), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = covariance / np.outer(sigma, sigma)
    summary = {"bodies": selected, "state_corrections": {}, "residual_statistics": {},
               "parameter_labels": parameter_labels(setup["parameters"]),
               "last_iteration": output.residual_history.shape[1] - 1,
               "best_iteration": int(output.best_iteration)}
    for index, body in enumerate(selected):
        correction = values[6 * index:6 * index + 6] - initial_states[6 * index:6 * index + 6]
        summary["state_corrections"][body] = correction.tolist()
        print(f"{body} initial-state correction [m, m/s]: {correction}", flush=True)
        view = dataset_for_body(dataset, body)
        view_output = output_for_body(output, dataset, view)
        od.print_residual_summary(f"{label}: {body}", view_output, view)
        scalar = od.observation_scalar_data(view)
        residuals = od.last_iteration_residuals(view_output)
        statistics = {}
        for observable in dict.fromkeys(scalar["observable_types"]):
            mask = scalar["observable_types"] == observable
            statistics[str(observable)] = {"scalar_count": int(mask.sum()),
                                           "RMS": float(np.sqrt(np.mean(residuals[mask]**2)))}
        if gaia is not None and int(body) in gaia.mpc_numbers_in_table:
            body_gaia = od.GaiaAstrometry(gaia.table[gaia.table.number_mp == int(body)].copy())
            projected = od.gaia_residual_data(view_output, view, body_gaia, body)
            mas_per_radian = 180 / np.pi * 3600 * 1000
            statistics["Gaia AL/AC"] = {
                "CCD_count": len(body_gaia.table),
                "RMS_mas": (np.sqrt(np.mean(projected["scan_residuals"]**2, axis=0)) * mas_per_radian).tolist(),
                "normalized_RMS": np.sqrt(np.mean((projected["scan_residuals"] / projected["scan_sigmas"])**2, axis=0)).tolist(),
            }
            mpc = scalar["station_labels"] != "Gaia"
            mpc &= scalar["observable_types"] == od.observations.angular_position_type
            if np.any(mpc):
                statistics["MPC angular position"] = {"scalar_count": int(mpc.sum()),
                                                       "RMS_rad": float(np.sqrt(np.mean(residuals[mpc]**2)))}
        summary["residual_statistics"][body] = statistics
    if estimate_psyche_gm:
        gm_index = 6 * len(selected)
        summary["psyche_GM"] = {"value_m3_s2": float(values[gm_index]),
            "formal_sigma_m3_s2": float(sigma[gm_index]),
            "mass_kg": float(values[gm_index] / od.constants.GRAVITATIONAL_CONSTANT),
            "formal_mass_sigma_kg": float(sigma[gm_index] / od.constants.GRAVITATIONAL_CONSTANT),
            "correlations": correlation[gm_index].tolist()}
        print(f"Psyche GM = {values[gm_index]:.12g} +/- {sigma[gm_index]:.6g} m^3/s^2 (formal 1 sigma)", flush=True)
        print(f"Psyche mass = {summary['psyche_GM']['mass_kg']:.12g} kg", flush=True)
        for index, body in enumerate(selected):
            print(f"  corr(GM_Psyche, {body} state) = {correlation[gm_index, 6*index:6*index+6]}", flush=True)
    if od.ESTIMATE_YARKOVSKY:
        for body in selected:
            identifier = od.parameters_setup.yarkovsky_parameter(body, "Sun").parameter_identifier
            index = setup["parameters"].indices_for_parameter_type(identifier)[0][0]
            print(f"Yarkovsky {body}: {values[index]:.12g} +/- {sigma[index]:.6g} m/s^2", flush=True)
    return summary, covariance, correlation


def save_figures(pdf):
    for number in od.plt.get_fignums():
        figure = od.plt.figure(number)
        pdf.savefig(figure)
        od.plt.close(figure)


def plot_joint_diagnostics(label, result, selected, prefix, estimate_psyche_gm, pdf):
    """Retain each body's original residual, orbit difference and formal-error figures."""
    output, dataset, estimator, gaia, setup = result
    history = od.last_state_history(output)
    epochs = np.array(sorted(history))
    epochs = epochs[np.unique(np.linspace(0, len(epochs)-1, min(300, len(epochs)), dtype=int))]
    covariance_history = od.estimation_analysis.propagate_covariance(
        output.covariance, estimator.state_transition_interface, list(epochs))
    slug = "gaia" if gaia is not None else "radar" if "radar" in label else "mpc"
    for index, body in enumerate(selected):
        view = dataset_for_body(dataset, body)
        view_output = output_for_body(output, dataset, view)
        body_gaia = None
        if gaia is not None and int(body) in gaia.mpc_numbers_in_table:
            body_gaia = od.GaiaAstrometry(gaia.table[gaia.table.number_mp == int(body)].copy())
        od.plot_residuals(f"{label}: {body}", view_output, view, body_gaia, body)
        if body_gaia is not None:
            od.plot_gaia_residuals(label, view_output, view, body_gaia, body,
                prefix.with_name(f"{prefix.name}_{slug}_{body}_gaia_residuals.csv"))
        print(f"Querying Horizons at {len(epochs)} plot epochs for {body}...", flush=True)
        reference = od.HorizonsQuery(query_id=f"{body};", location=od.HORIZONS_ORIGIN,
            epoch_list=list(epochs), extended_query=True).cartesian(frame_orientation=od.FRAME_ORIENTATION)[:, 1:]
        od.plot_orbit_difference(label, output, estimator, epochs, reference,
            slice(6*index, 6*index+6), body, covariance_history)
        save_figures(pdf)
    if estimate_psyche_gm:
        covariance = np.asarray(output.covariance)
        sigma = np.sqrt(np.clip(np.diag(covariance), 0, None))
        correlation = covariance / np.outer(sigma, sigma)
        figure, axis = od.plt.subplots(figsize=(11.7, 7.5))
        image = axis.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
        ticks = [6*i+2.5 for i in range(len(selected))] + [6*len(selected)]
        axis.set_xticks(ticks, [f"{body} state" for body in selected] + ["Psyche GM"], rotation=45)
        axis.set_yticks(ticks, [f"{body} state" for body in selected] + ["Psyche GM"])
        figure.colorbar(image, ax=axis, label="Parameter correlation")
        axis.set_title(f"{label}: joint parameter correlations")
        figure.tight_layout()
        save_figures(pdf)


def run(selected, output_prefix, estimate_psyche_gm=True, overrides=None, mpc_only=False):
    selected = normalize_bodies(selected, estimate_psyche_gm)
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    od.spice.load_standard_kernels()
    print(f"Estimated bodies: {selected}; global frame {od.GLOBAL_FRAME_ORIGIN}/{od.FRAME_ORIENTATION}", flush=True)
    print(f"Observation interval: {od.OBSERVATION_START} through {od.OBSERVATION_END}", flush=True)
    print("Propagation status is printed every Julian year of integration time.", flush=True)
    data = {body: load_body_data(body, (overrides or {}).get(body), mpc_only) for body in selected}
    radii = dict(PHOTOCENTER_RADII)
    for body in selected:
        if data[body]["gaia"] is not None:
            radius = radii.get(body, radii.get(int(body), od.PHOTOCENTER_RADIUS if body == od.TARGET else None))
            radii[body] = od.sbdb_photocenter_radius(body) if radius is None else radius
            if not np.isfinite(radii[body]) or radii[body] <= 0:
                raise ValueError(f"The photocenter radius for {body} must be positive [m].")
            print(f"{body} Gaia photocenter radius: {radii[body]:.12g} m", flush=True)
    epoch, initial_states, first, final, references = load_reference_states(selected, data)
    summaries, results = {}, {}
    with od.PdfPages(prefix.with_suffix(".pdf")) as pdf:
        for label, (tracking, extras, gaia) in estimation_setups(data).items():
            print(f"\nRunning joint {label}...", flush=True)
            result = od.perform_estimation(tracking, extras, epoch, initial_states.copy(), first, final,
                od.ESTIMATE_YARKOVSKY, gaia, references, selected, estimate_psyche_gm,
                radii, PROPAGATION_PRINT_INTERVAL, return_setup=True)
            summary, covariance, correlation = report_joint_result(label, result, selected, initial_states, estimate_psyche_gm)
            summaries[label], results[label] = summary, result
            output = result[0]
            history = od.last_state_history(output)
            times = np.array(sorted(history))
            slug = "gaia" if gaia is not None else "radar" if "radar" in label else "mpc"
            np.savez_compressed(prefix.with_name(f"{prefix.name}_{slug}.npz"),
                bodies=np.asarray(selected), times=times, states=np.array([history[t] for t in times]),
                parameter_history=np.asarray(output.parameter_history), residual_history=np.asarray(output.residual_history),
                active_flags=np.asarray(output.active_flags_per_iteration), covariance=covariance, correlation=correlation,
                parameter_labels=np.asarray(summary["parameter_labels"]), initial_epoch=epoch,
                initial_states=initial_states, last_parameters=od.last_iteration_parameters(output),
                **scalar_observation_metadata(result[1], selected))
            plot_joint_diagnostics(label, result, selected, prefix, estimate_psyche_gm, pdf)
            prefix.with_suffix(".json").write_text(json.dumps(summaries, indent=2))
    if "MPC astrometry" in results and "MPC astrometry and JPL radar" in results:
        for index, body in enumerate(selected):
            od.print_orbit_difference_rsw(results["MPC astrometry"][0],
                results["MPC astrometry and JPL radar"][0], epoch, slice(6*index, 6*index+6), body)
    print(f"Saved diagnostics to {prefix.with_suffix('.pdf')}", flush=True)
    return results


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bodies", nargs="+", default=ESTIMATED_BODIES, help="MPC numbers, including 16 Psyche; at most ten.")
    parser.add_argument("--fixed-psyche-gm", action="store_true", help="Estimate states only; permits single-body compatibility checks.")
    parser.add_argument("--gaia-archive", action="append", default=[], metavar="BODY=FILE", help="Local FPR archive for one target; repeat as needed.")
    parser.add_argument("--photocenter-radius", action="append", default=[], metavar="BODY=METRES", help="Spherical Gaia photocenter radius where SBDB lacks a diameter; repeat as needed.")
    parser.add_argument("--mpc-only", action="store_true", help="Use MPC only, e.g. for incremental validation.")
    parser.add_argument("--observation-start", type=datetime.datetime.fromisoformat, default=od.OBSERVATION_START)
    parser.add_argument("--observation-end", type=datetime.datetime.fromisoformat, default=od.OBSERVATION_END)
    parser.add_argument("--output-prefix", type=Path, default=Path(__file__).with_suffix(""))
    args = parser.parse_args()
    try:
        selected = normalize_bodies(args.bodies, not args.fixed_psyche_gm)
        if args.observation_start >= args.observation_end:
            raise ValueError("Observation start must precede observation end.")
        overrides = {}
        for entry in args.gaia_archive:
            body, path = entry.split("=", 1)
            body = str(int(body))
            if body not in selected or not Path(path).is_file():
                raise ValueError(f"Invalid Gaia archive override: {entry}")
            overrides[body] = {"gaia_archive": Path(path)}
        for entry in args.photocenter_radius:
            body, radius = entry.split("=", 1)
            body, radius = str(int(body)), float(radius)
            if body not in selected or not np.isfinite(radius) or radius <= 0:
                raise ValueError(f"Invalid photocenter radius override: {entry}")
            PHOTOCENTER_RADII[body] = radius
    except ValueError as error:
        parser.error(str(error))
    od.OBSERVATION_START, od.OBSERVATION_END = args.observation_start, args.observation_end
    run(selected, args.output_prefix, not args.fixed_psyche_gm, overrides, args.mpc_only)


if __name__ == "__main__":
    main()
