"""
# MPC, JPL radar and Gaia astrometry estimation

Copyright (c) 2010-2026, Delft University of Technology. All rights reserved.
This file is part of Tudat. Redistribution and use in source and binary forms,
with or without modification, are permitted exclusively under the terms of the
Modified BSD license. See https://tudat.tudelft.nl/LICENSE.

This example estimates the state of the small body selected by TARGET from MPC optical astrometry,
then from MPC and Gaia astrometry together. JPL radar is included when
available. It is a copy of ``mpc_and_radar_estimation.py`` with Gaia added.
The example follows the current data path directly:

1. MPC, Gaia and available JPL records are converted to ``TrackingData``.
2. Their supplementary data are applied to the system of bodies.
3. The tracking data are converted to an ``ObservationDataset``.
4. Orbit determinations are run without and with Gaia.

Gaia's full transit covariance includes random RA/Dec correlations and
systematic errors shared by the CCD observations in each transit. Relativistic light deflection and a
spherical photocenter correction are evaluated during dataset conversion.

Gaia along-scan and cross-scan plots use the published scan position angle and
RA multiplied by cos(dec). Cross-scan is the perpendicular tangent-plane
projection; aberration makes Gaia's actual AC direction slightly nonorthogonal
to AL, as described in the FPR data model. Normalized plots use marginal
uncertainties, while the fit retains the full correlated weights.
Residual and orbit diagnostics use the last evaluated iteration. Formal errors
reuse the estimator's returned covariance, effectively unchanged between iterations.

MPC and Horizons queries require an internet connection. Gaia uses
``gaia_<TARGET>_fpr.parquet`` next to this script, configured by
``GAIA_ARCHIVE_PATH``. If the file is missing, it queries the AIP mirror and saves the
full response there for subsequent runs. The Gaia run is omitted if no Gaia data are
available for TARGET. Radar is added when available.
"""

import datetime
from collections import Counter
from functools import lru_cache
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from tudatpy import constants
from tudatpy.astro.frame_conversion import inertial_to_rsw_rotation_matrix
from tudatpy.astro import time_representation
from tudatpy.data_input.environment_data import spice
from tudatpy.data_input.environment_data.horizons import HorizonsQuery
from tudatpy.data_input.environment_data.sbdb import SBDBquery
from tudatpy.data_input.tracking_data import get_tracking_data_epoch_bounds
from tudatpy.data_input.tracking_data.gaia import load_gaia_astrometry
from tudatpy.data_input.tracking_data.jpl_radar import JPLRadarQuery
from tudatpy.data_input.tracking_data.mpc import BatchMPC
from tudatpy.data_input.tracking_data.optical_utilities import optical_table_to_tracking_data
from tudatpy.data_input.tracking_data.radar_utilities import (
    radar_data_to_tracking_data,
)
from tudatpy.dynamics import (
    environment_setup, parameters_setup, propagation_setup,
)
from tudatpy.estimation import estimation_analysis, observable_models_setup, observations
from tudatpy.estimation.observations_setup import observations_simulation_settings
from tudatpy.math import interpolators


TARGET = "99942"
HORIZONS_TARGET = f"{TARGET};"
# Light-time geometry uses barycentric positions at emission and reception.
GLOBAL_FRAME_ORIGIN = "SSB"
PROPAGATION_CENTRAL_BODY = "Sun"
HORIZONS_ORIGIN = "500@10"
FRAME_ORIENTATION = "J2000"

OBSERVATION_START = datetime.datetime(1900, 1, 1)
OBSERVATION_END = datetime.datetime(2026, 7, 1)
PROPAGATION_BUFFER = 2.0 * 31.0 * constants.JULIAN_DAY
ASTEROID_EPHEMERIS_STEP = 14.0 * constants.JULIAN_DAY
ASTEROID_EPHEMERIS_BUFFER = constants.JULIAN_YEAR
ASTEROID_EPHEMERIS_INTERPOLATION_POINTS = 10
INTEGRATOR_STEP = 6.0 * 3600.0
NUMBER_OF_ESTIMATION_ITERATIONS = 6
ESTIMATE_YARKOVSKY = True

# Read the target's FPR archive, or query AIP and save it for subsequent runs.
GAIA_ARCHIVE_PATH = Path(__file__).with_name(f"gaia_{TARGET}_fpr.parquet")
# Radius [m]: half the JPL SBDB diameter unless specified here.
PHOTOCENTER_RADIUS = None

# The 21 most massive main-belt asteroids in the SiMDA data set distributed
# with the Tudat example. Their ephemerides and gravitational parameters are
# read from the standard SPICE asteroid kernels. States are tabulated every
# 14 days, with one extra year on each side of the propagation interval,
# using 10-point Lagrange interpolation.
ASTEROID_PERTURBERS = [
    (1, "Ceres"),
    (4, "Vesta"),
    (2, "Pallas"),
    (10, "Hygiea"),
    (704, "Interamnia"),
    (15, "Eunomia"),
    (511, "Davida"),
    (3, "Juno"),
    (52, "Europa"),
    (16, "Psyche"),
    (65, "Cybele"),
    (87, "Sylvia"),
    (31, "Euphrosyne"),
    (7, "Iris"),
    (29, "Amphitrite"),
    (6, "Hebe"),
    (532, "Herculina"),
    (451, "Patientia"),
    (107, "Camilla"),
    (536, "Merapi"),
    (324, "Bamberga"),
]


#####################################################################
#################   RETRIEVE/PROCESS DATA    ########################
#####################################################################


def load_tracking_data():
    """Load observations and assemble comparison fits with a common TDB interval."""

    # Load MPC data
    batch = BatchMPC()
    batch.get_observations([TARGET], use_mpc80_format=True)
    batch.filter(
        epoch_start=OBSERVATION_START,
        epoch_end=OBSERVATION_END,
        observatories_exclude=["C57", "258", "704", "S04"],
    )
    if batch.table.empty:
        raise RuntimeError("The selected interval contains no MPC astrometry.")

    print(f"Loaded {len(batch.table)} MPC optical observations.", flush=True)
    tracking_data, supplementary_data = optical_table_to_tracking_data(
        batch.table,
        add_weights=True,
        add_star_catalog_corrections=True,
        add_ancillary_data=True,
    )
    setups = {"MPC astrometry": (tracking_data, supplementary_data)}

    # Load JPL radar data
    radar_query = JPLRadarQuery(TARGET, timeout=60.0)
    radar_table = radar_query.to_radar_data(
        target_body=TARGET,
        epoch_start=OBSERVATION_START,
        epoch_end=OBSERVATION_END,
        target_point="C",
    )
    if radar_table.empty:
        print("No JPL radar observations found.")
    else:
        radar_tracking_data, radar_supplementary_data = radar_data_to_tracking_data(
            radar_table
        )
        print(f"Loaded {len(radar_table)} JPL center-of-mass radar observations.")
        tracking_data = tracking_data + radar_tracking_data
        supplementary_data = supplementary_data + radar_supplementary_data
        setups["MPC astrometry and JPL radar"] = (tracking_data, supplementary_data)

    # Load Gaia data
    gaia_astrometry = load_gaia_astrometry(
        target=int(TARGET),
        archive_path=GAIA_ARCHIVE_PATH,
        epoch_start=OBSERVATION_START,
        epoch_end=OBSERVATION_END,
        time_scale=time_representation.utc_scale,
    )
    if gaia_astrometry is not None:
        radius = sbdb_photocenter_radius(TARGET) if PHOTOCENTER_RADIUS is None else PHOTOCENTER_RADIUS
        gaia_tracking_data, gaia_supplementary_data = gaia_astrometry.to_tracking_data(
            light_deflection_bodies=("Sun", "Jupiter"),
            photocenter_body_dimensions={int(TARGET): radius},
        )
        tracking_data = tracking_data + gaia_tracking_data
        supplementary_data = supplementary_data + gaia_supplementary_data
        label = "MPC astrometry and Gaia" if radar_table.empty else "MPC astrometry, JPL radar and Gaia"
        setups[label] = (tracking_data, supplementary_data)
    else:
        print("No Gaia data available; skipping the Gaia-inclusive run.", flush=True)

    # Get data bounds and return full tracking data and supplementary data
    first_epoch, last_epoch = get_tracking_data_epoch_bounds(tracking_data)
    return setups, float(first_epoch), float(last_epoch)


def sbdb_photocenter_radius(target):
    """Read the same SBDB diameter even when Astroquery drops its unit on asymmetric errors."""
    query = SBDBquery(str(target))
    try:
        return 0.5 * query.diameter
    except ValueError:
        from astroquery.jplsbdb import SBDB
        from astropy import units as u
        parameters = SBDB.query_async(str(target), phys=True).json().get("phys_par", [])
        diameter = next((p for p in parameters if p["name"] == "diameter"), None)
        if diameter is None or diameter["value"] is None or not diameter["units"]:
            raise ValueError(f"SBDB has no diameter for {target}; supply its photocenter radius [m].") from None
        return 0.5 * (float(diameter["value"]) * u.Unit(diameter["units"])).to_value(u.m)


#####################################################################
#################   CREATE ENVIRONMENT   ############################
#####################################################################


def asteroid_ephemeris_settings(number, first_epoch, final_epoch):
    """Tabulate a perturber from SPICE or a cached Horizons reference."""

    start = float(first_epoch) - ASTEROID_EPHEMERIS_BUFFER
    end = float(final_epoch) + ASTEROID_EPHEMERIS_BUFFER
    interpolation = interpolators.lagrange_interpolation(ASTEROID_EPHEMERIS_INTERPOLATION_POINTS)
    return environment_setup.ephemeris.interpolated_spice(
            start, end, ASTEROID_EPHEMERIS_STEP, GLOBAL_FRAME_ORIGIN, FRAME_ORIENTATION,
            interpolation, spice.asteroid_spice_id(number))


@lru_cache(maxsize=10)
def load_reference_state_history(target, first_epoch, final_epoch):
    """Reuse each target's Horizons reference across the comparison fits."""
    states = HorizonsQuery(
        query_id=f"{target};", location=HORIZONS_ORIGIN,
        epoch_start=float(first_epoch), epoch_end=float(final_epoch),
        epoch_step="1d", extended_query=True,
    ).cartesian(frame_orientation=FRAME_ORIENTATION)
    return dict(zip(states[:, 0], states[:, 1:]))



def create_bodies(first_epoch, final_epoch):
    """Create the estimation environment and Earth observing stations."""
    estimated_bodies = [TARGET]

    body_names = [
        "Sun",
        "Mercury",
        "Venus",
        "Earth",
        "Moon",
        "Mars",
        "Jupiter",
        "Saturn",
        "Uranus",
        "Neptune",
    ]
    body_settings = environment_setup.get_default_body_settings(
        body_names,
        GLOBAL_FRAME_ORIGIN,
        FRAME_ORIENTATION,
    )

    for body_name in ["Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]:
        barycenter_name = f"{body_name} Barycenter"
        settings = body_settings.get(body_name)
        settings.ephemeris_settings = environment_setup.ephemeris.direct_spice(
            GLOBAL_FRAME_ORIGIN,
            FRAME_ORIENTATION,
            barycenter_name,
        )
        settings.gravity_field_settings = (
            environment_setup.gravity_field.central_spice(barycenter_name)
        )

    # Radar residuals are sensitive to station positions and Earth rotation.
    earth_settings = body_settings.get("Earth")
    # Convert the Earth gravity model's TT-compatible GM to TDB (IERS §1.2).
    l_b, l_g = 1.550519768e-8, 6.969290134e-10
    earth_settings.gravity_field_settings.gravitational_parameter *= (1.0 - l_b) / (1.0 - l_g)
    earth_settings.shape_settings = environment_setup.shape.oblate_spherical(
        6378137.0,
        1.0 / 298.257223563,
    )
    earth_settings.rotation_model_settings = (
        environment_setup.rotation_model.gcrs_to_itrs(
            environment_setup.rotation_model.iau_2006,
            FRAME_ORIENTATION,
        )
    )
    earth_settings.gravity_field_settings.associated_reference_frame = "ITRS"
    earth_settings.ground_station_settings = (
        environment_setup.ground_station.optical_telescope_stations()
    )

    for body in estimated_bodies:
        body_settings.add_empty_settings(body)
        body_settings.get(body).ephemeris_settings = environment_setup.ephemeris.tabulated(
                load_reference_state_history(body, first_epoch, final_epoch),
                PROPAGATION_CENTRAL_BODY, FRAME_ORIENTATION )
    for number, _ in ASTEROID_PERTURBERS:
        body_name = str(number)
        selected = body_name in estimated_bodies
        spice_id = spice.asteroid_spice_id(number)
        if not selected:
            body_settings.add_empty_settings(body_name)
        settings = body_settings.get(body_name)
        if not selected:
            settings.ephemeris_settings = asteroid_ephemeris_settings(
                number, first_epoch, final_epoch,
            )
        if spice.check_body_property_in_kernel_pool(spice_id, "GM"):
            settings.gravity_field_settings = environment_setup.gravity_field.central_spice(spice_id)
        else:
            settings.gravity_field_settings = environment_setup.gravity_field.central(
                constants.GRAVITATIONAL_CONSTANT * SIMDA_FALLBACK_MASSES[number],
            )

    return environment_setup.create_system_of_bodies(body_settings)


#####################################################################
#################   ESTIMATION MODEL SETTINGS  ######################
#####################################################################


def acceleration_settings(estimate_yarkovsky, estimated_bodies=None):
    """Return the force model used for the configured target."""
    estimated_bodies = [TARGET] if estimated_bodies is None else list(estimated_bodies)
    sun_accelerations = [
        propagation_setup.acceleration.point_mass_gravity(),
        propagation_setup.acceleration.relativistic_correction(
            use_schwarzschild=True
        ),
    ]
    if estimate_yarkovsky:
        sun_accelerations.append(
            propagation_setup.acceleration.yarkovsky(0.0)
        )

    target_accelerations = {
        "Sun": sun_accelerations,
        "Mercury": [propagation_setup.acceleration.point_mass_gravity()],
        "Venus": [propagation_setup.acceleration.point_mass_gravity()],
        "Earth": [propagation_setup.acceleration.point_mass_gravity()],
        "Moon": [propagation_setup.acceleration.point_mass_gravity()],
        "Mars": [propagation_setup.acceleration.point_mass_gravity()],
        "Jupiter": [propagation_setup.acceleration.point_mass_gravity()],
        "Saturn": [propagation_setup.acceleration.point_mass_gravity()],
        "Uranus": [propagation_setup.acceleration.point_mass_gravity()],
        "Neptune": [propagation_setup.acceleration.point_mass_gravity()],
    }
    models = {}
    for target in estimated_bodies:
        models[target] = dict(target_accelerations)
        models[target].update({
            str(number): [
                propagation_setup.acceleration.point_mass_gravity()
            ]
            for number, _ in ASTEROID_PERTURBERS if str(number) != target
        })
    return models


def observation_model_settings(observation_dataset):
    """Create observation models for every link present in the dataset."""
    settings = []
    corrections = [
        observable_models_setup.light_time_corrections.first_order_relativistic_light_time_correction(
            ["Sun"]
        )
    ]

    added_links = set()
    for metadata in observation_dataset.observation_set_metadata:
        observable_type = metadata.observable_type
        link_id = metadata.link_definition_id
        if (observable_type, link_id) in added_links:
            continue
        link = observation_dataset.link_definition(link_id)

        if observable_type == observable_models_setup.model_settings.angular_position_type:
            settings.append(
                observable_models_setup.model_settings.angular_position(
                    link,
                    corrections,
                )
            )
        elif observable_type == observable_models_setup.model_settings.n_way_range_type:
            settings.append(
                observable_models_setup.model_settings.n_way_range(
                    link,
                    corrections,
                    time_scale_for_observable=time_representation.utc_scale,
                )
            )
        elif observable_type == observable_models_setup.model_settings.doppler_measured_frequency_type:
            settings.append(
                observable_models_setup.model_settings.doppler_measured_frequency(
                    link,
                    corrections,
                )
            )
        else:
            continue
        added_links.add((observable_type, link_id))
    return settings

#####################################################################
#################   CORE ESTIMATION FUNCTION   ######################
#####################################################################


def perform_estimation(
    tracking_data,
    supplementary_data,
    observation_start_epoch,
    observation_end_epoch,
):
    """Estimate state of target asteroid."""

    # Use the observation-interval midpoint or J2000, whichever is later.
    estimation_epoch = max(0.0, 0.5 * (observation_start_epoch + observation_end_epoch))

    # Extend both ends by the configured propagation buffer.
    first_epoch = observation_start_epoch - PROPAGATION_BUFFER
    final_epoch = observation_end_epoch + PROPAGATION_BUFFER

    # Create bodies
    bodies = create_bodies(first_epoch, final_epoch)
    initial_state = bodies.get(TARGET).ephemeris.cartesian_state(float(estimation_epoch))

    # This installs transmitter-frequency histories, identifies the passive
    # radar reflector, and creates the space-telescope bodies and ephemerides.
    observations.set_tracking_supplementary_data_in_bodies(
        bodies,
        supplementary_data,
    )
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies,
        acceleration_settings(ESTIMATE_YARKOVSKY),
        [TARGET],
        [PROPAGATION_CENTRAL_BODY],
    )
    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
        time_representation.Time(INTEGRATOR_STEP),
        propagation_setup.integrator.CoefficientSets.rkf_45,
    )
    termination_settings = propagation_setup.propagator.non_sequential_termination(
        propagation_setup.propagator.time_termination(final_epoch),
        propagation_setup.propagator.time_termination(first_epoch),
    )
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies=[PROPAGATION_CENTRAL_BODY],
        acceleration_models=acceleration_models,
        bodies_to_integrate=[TARGET],
        initial_states=initial_state,
        initial_time=estimation_epoch,
        integrator_settings=integrator_settings,
        termination_settings=termination_settings,
    )

    print("Creating and populating observation dataset.", flush=True)
    observation_dataset = observations.create_observation_dataset_from_tracking_data(
        tracking_data, bodies, apply_corrections=True)

    print("Initializing the estimator and propagating variational equations.", flush=True)
    parameter_settings = parameters_setup.initial_states(propagator_settings, bodies)
    if ESTIMATE_YARKOVSKY:
        parameter_settings.append(parameters_setup.yarkovsky_parameter(TARGET, "Sun"))
    parameters_to_estimate = parameters_setup.create_parameter_set(
        parameter_settings,
        bodies,
        propagator_settings,
    )
    estimator = estimation_analysis.Estimator(
        bodies=bodies,
        estimated_parameters=parameters_to_estimate,
        observation_settings=observation_model_settings(observation_dataset),
        propagator_settings=propagator_settings,
        integrate_on_creation=True,
    )
    estimation_input = estimation_analysis.EstimationInput(
        observation_dataset=observation_dataset,
        outlier_rejection_settings=estimation_analysis.carpino_outlier_rejection_settings(
            chi2_rejection_threshold=5.0,
            chi2_recovery_threshold=4.0,
            maximum_rejected_fraction=1.0,
            first_iteration_with_rejection=0,
        ),
        convergence_checker=estimation_analysis.estimation_convergence_checker(
            maximum_iterations=NUMBER_OF_ESTIMATION_ITERATIONS,
        ),
    )
    estimation_input.define_estimation_settings(
        reintegrate_variational_equations=False,
        reintegrate_equations_on_first_iteration=False,
        print_output_to_terminal=True,
        save_residuals_and_parameters_per_iteration=True,
        save_state_history_per_iteration=True,
    )

    print("Starting estimation; iteration output follows.", flush=True)
    output = estimator.perform_estimation(estimation_input)
    print(
        f"Using last evaluated iteration {output.residual_history.shape[1] - 1} "
        f"for diagnostics (best: {output.best_iteration}; zero-based indices).",
        flush=True,
    )
    return output, observation_dataset, estimator


#####################################################################
#################   OUTPUT SUMMARY   ################################
#####################################################################


def last_iteration_residuals(output, observation_dataset):
    """Select last-iteration residuals for observations that remain unrejected."""
    residuals = np.asarray(output.residual_history)[:, -1]
    vector_data = observation_dataset.observation_vector_data(include_rejected=True)
    indices = [
        vector_data.vector_row(observation_id, component)
        for observation_id, component in observation_dataset.get_scalar_components(
            observations.observation_query.active, ordering="estimation",
        )
    ]
    return residuals[indices]


def last_iteration_parameters(output):
    """Select the parameters used to evaluate the last residuals and orbit."""
    # The final parameter-history column can contain a correction that has
    # never been propagated. Match the residual column instead of using -1.
    last_iteration = np.asarray(output.residual_history).shape[1] - 1
    return np.asarray(output.parameter_history)[:, last_iteration]


def print_yarkovsky_result(output):
    """Print the estimated A2 value and its formal uncertainty."""
    a2_si = float(last_iteration_parameters(output)[-1])
    sigma_si = float(np.sqrt(np.asarray(output.covariance)[-1, -1]))
    si_to_au_per_day_squared = constants.JULIAN_DAY**2 / constants.ASTRONOMICAL_UNIT
    print(
        "  Estimated Yarkovsky A2: "
        f"({a2_si:.9g} ± {sigma_si:.3g}) m/s² = "
        f"({a2_si * si_to_au_per_day_squared:.9g} ± "
        f"{sigma_si * si_to_au_per_day_squared:.3g}) au/day²"
    )


def print_residual_summary(label, output, observation_dataset):
    """Print last-iteration residual RMS values for each observable family."""
    residuals = last_iteration_residuals(output, observation_dataset)
    observable_units = {
        observable_models_setup.model_settings.angular_position_type: "rad",
        observable_models_setup.model_settings.n_way_range_type: "m",
        observable_models_setup.model_settings.doppler_measured_frequency_type: "Hz",
    }
    print(f"\n{label}")
    data = observation_scalar_data(observation_dataset)
    for observable_type in dict.fromkeys(data["observable_types"]):
        values = residuals[data["observable_types"] == observable_type]
        rms = np.sqrt(np.mean(values**2))
        unit = observable_units.get(observable_type, "")
        print(f"  {observable_type}: {len(values)} scalar residuals, RMS = {rms:.6g} {unit}")


def last_state_history(output):
    """Return the propagated state history from the last evaluated iteration."""
    return output.simulation_results_per_iteration[
        -1
    ].dynamics_results.state_history_float


def print_orbit_difference_rsw(reference_output, comparison_output, estimation_epoch,
                               state_slice=None, target=None):
    """Print the radar solution minus the optical solution in the optical RSW frame."""
    reference_history = last_state_history(reference_output)
    comparison_history = last_state_history(comparison_output)
    state_slice = slice(0, 6) if state_slice is None else state_slice
    target = TARGET if target is None else str(target)

    common_epochs = sorted(set(reference_history).intersection(comparison_history))
    if not common_epochs:
        raise RuntimeError("The estimated state histories have no common epochs.")

    epoch = min(common_epochs, key=lambda value: abs(value - estimation_epoch))
    if not np.isclose(epoch, estimation_epoch, rtol=0.0, atol=1.0e-6):
        raise RuntimeError("The estimation epoch is absent from the state histories.")

    def position_difference_rsw(current_epoch):
        reference_state = np.asarray(reference_history[current_epoch])[state_slice]
        comparison_state = np.asarray(comparison_history[current_epoch])[state_slice]
        return inertial_to_rsw_rotation_matrix(reference_state) @ (
            comparison_state[:3] - reference_state[:3]
        )

    difference_at_estimation_epoch = position_difference_rsw(epoch)
    differences = np.array(
        [position_difference_rsw(current_epoch) for current_epoch in common_epochs]
    )
    rms_difference = np.sqrt(np.mean(differences**2, axis=0))

    print(
        f"\n{target} orbit difference "
        "(MPC + radar minus MPC-only; MPC-only heliocentric RSW frame)"
    )
    print(
        "  At estimation epoch [km]: "
        f"R = {difference_at_estimation_epoch[0] / 1000.0:.6g}, "
        f"S = {difference_at_estimation_epoch[1] / 1000.0:.6g}, "
        f"W = {difference_at_estimation_epoch[2] / 1000.0:.6g}"
    )
    print(
        f"  RMS over {len(common_epochs)} state-history epochs [km]: "
        f"R = {rms_difference[0] / 1000.0:.6g}, "
        f"S = {rms_difference[1] / 1000.0:.6g}, "
        f"W = {rms_difference[2] / 1000.0:.6g}"
    )

#####################################################################
#################   PLOTTING   ######################################
#####################################################################


def observation_scalar_data(observation_dataset):
    """Align dataset metadata with the scalar order used by the estimator."""
    data = observation_dataset.get_data(
        observations.observation_query.active,
        fields=(
            "times", "observation_ids", "set_ids", "metadata", "scalar_components", "weight_diagonal"
        ),
        ordering="estimation",
    )
    event_indices = {
        observation_id: index for index, observation_id in enumerate(data["observation_ids"])
    }
    scalar_events = np.array(
        [event_indices[observation_id] for observation_id, _ in data["scalar_components"]],
        dtype=int,
    )
    scalar_set_ids = [data["set_ids"][index] for index in scalar_events]

    link_data = {}
    for set_id, metadata in data["metadata"].items():
        link_ends = metadata["link_definition"].link_ends
        receiver = link_ends[observable_models_setup.links.receiver]
        stations = [
            link_ends[role].reference_point
            for role in (
                observable_models_setup.links.transmitter,
                observable_models_setup.links.receiver,
            )
            if role in link_ends
            and link_ends[role].body_name == "Earth"
            and link_ends[role].reference_point
        ]
        link_data[set_id] = (
            " → ".join(dict.fromkeys(stations)) or receiver.body_name,
            receiver.body_name != "Earth",
        )
    return {
        "times": np.array([float(epoch) for epoch in data["times"]])[scalar_events],
        "weights": np.asarray(data["weight_diagonal"]).reshape(-1),
        "observable_types": np.array(
            [data["metadata"][set_id]["observable_type"] for set_id in scalar_set_ids],
            dtype=object,
        ),
        "components": np.array([component for _, component in data["scalar_components"]]),
        "station_labels": np.array([link_data[set_id][0] for set_id in scalar_set_ids]),
        "space_astrometry": np.array([link_data[set_id][1] for set_id in scalar_set_ids]),
    }


def add_grouped_scatter(ax, times, values, station_labels):
    """Plot values for the ten busiest stations and group the remainder."""
    label_counts = Counter(station_labels.tolist())
    top_labels = [
        label
        for label, _ in label_counts.most_common(10)
    ]
    other_mask = ~np.isin(station_labels, top_labels)
    if np.any(other_mask):
        ax.scatter(
            times[other_mask],
            values[other_mask],
            s=9,
            alpha=0.55,
            color="lightgrey",
            label=f"Other ({int(np.count_nonzero(other_mask))})",
        )

    colors = plt.get_cmap("tab10")
    for index, station_label in enumerate(top_labels):
        mask = station_labels == station_label
        ax.scatter(
            times[mask],
            values[mask],
            s=11,
            alpha=0.8,
            color=colors(index),
            label=f"{station_label} ({int(np.count_nonzero(mask))})",
        )


def project_gaia_residuals(residuals, covariance, declinations, scan_angles):
    """Project unstarred RA/Dec into AL and perpendicular tangent-plane AC.

    FPR position_angle_scan is zero towards North and pi/2 towards increasing
    RA. AL = sin(theta) * dRA*cos(dec) + cos(theta) * dDec;
    AC = cos(theta) * dRA*cos(dec) - sin(theta) * dDec.
    The same transformation rotates the marginal covariance.

    Scan-angle convention and aberrated AC caveat:
    https://gea.esac.esa.int/archive/documentation/FPR/chap_datamodel/sec_dm_focused_product_release/ssec_dm_sso_observation.html
    """
    sine, cosine = np.sin(scan_angles), np.cos(scan_angles)
    cos_dec = np.cos(declinations)
    rotation = np.empty((len(residuals), 2, 2))
    rotation[:, 0, 0] = sine * cos_dec
    rotation[:, 0, 1] = cosine
    rotation[:, 1, 0] = cosine * cos_dec
    rotation[:, 1, 1] = -sine
    projected = np.einsum("nij,nj->ni", rotation, residuals)
    rotated_covariance = rotation @ covariance @ rotation.transpose(0, 2, 1)
    sigmas = np.sqrt(np.diagonal(rotated_covariance, axis1=1, axis2=2))
    if not np.all(np.isfinite(projected)) or not np.all(np.isfinite(sigmas) & (sigmas > 0)):
        raise RuntimeError("Invalid Gaia scan residuals or marginal uncertainties.")
    return projected, sigmas


def gaia_residual_data(output, observation_dataset, target=None):
    """Get Gaia residuals, covariance and scan angles using stable observation IDs."""
    target = TARGET if target is None else str(target)
    query = observations.observation_query
    condition = (
        query.active
        & (query.receiver == observations.LinkEndId("Gaia", ""))
        & (query.transmitter == observations.LinkEndId(target, ""))
    )
    data = observation_dataset.get_data(
        condition, fields=("times", "observations", "observation_ids"), ordering="estimation",
    )
    observation_ids = data["observation_ids"]
    if not observation_ids:
        return None
    vector_data = observation_dataset.observation_vector_data(include_rejected=False)
    scalar_indices = np.array([
        [vector_data.vector_row(observation_id, 0), vector_data.vector_row(observation_id, 1)]
        for observation_id in observation_ids
    ])
    residuals = last_iteration_residuals(output, observation_dataset)[scalar_indices]
    covariance = np.array([
        vector_data.inverse_weight_matrix_for_observation(observation_id)
        for observation_id in observation_ids
    ])
    scan_angles = observation_dataset.get_numerical_observation_metadata(
        "along_scan_angle", observation_ids,
    )
    angles = np.asarray(data["observations"])
    scan_residuals, scan_sigmas = project_gaia_residuals(
        residuals, covariance, angles[:, 1], scan_angles,
    )
    table = pd.DataFrame({
        "observation_id": observation_ids,
        "epoch": [float(epoch) for epoch in data["times"]],
        "ra": angles[:, 0], "dec": angles[:, 1], "along_scan_angle": scan_angles,
    })
    return {
        "table": table, "scalar_indices": scalar_indices, "residuals": residuals,
        "covariance": covariance, "scan_residuals": scan_residuals, "scan_sigmas": scan_sigmas,
    }


def plot_gaia_residuals(setup_label, output, observation_dataset,
                        target=None, csv_path=None):
    """Plot actual Gaia CCD AL/AC residuals in mas and marginal sigma units."""
    target = TARGET if target is None else str(target)
    data = gaia_residual_data(output, observation_dataset, target)
    if data is None:
        return
    mas_per_radian = 180.0 / np.pi * 3600.0 * 1000.0
    years = 2000.0 + data["table"]["epoch"].to_numpy() / constants.JULIAN_YEAR
    figure, axes = plt.subplots(2, 2, figsize=(11.7, 8.5), sharex=True)
    figure.suptitle(f"{setup_label}: {target}, {len(years)} Gaia CCD observations")
    table = data["table"].copy()
    for component, name in enumerate(("Along-scan (AL)", "Cross-scan (AC)")):
        values = data["scan_residuals"][:, component] * mas_per_radian
        sigmas = data["scan_sigmas"][:, component] * mas_per_radian
        normalized = values / sigmas
        axes[component, 0].scatter(years, values, s=9, alpha=0.7)
        axes[component, 1].scatter(years, normalized, s=9, alpha=0.7)
        for axis in axes[component]:
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.grid(alpha=0.3)
        axes[component, 0].set_ylabel(f"{name} residual [mas]")
        axes[component, 1].set_ylabel(f"{name} residual / marginal σ")
        axes[component, 0].set_title(f"RMS = {np.sqrt(np.mean(values**2)):.4g} mas")
        axes[component, 1].set_title(f"Normalized RMS = {np.sqrt(np.mean(normalized**2)):.4g}")
        prefix = "al" if component == 0 else "ac"
        table[f"{prefix}_residual_mas"] = values
        table[f"{prefix}_sigma_mas"] = sigmas
        table[f"{prefix}_normalized_residual"] = normalized
        print(
            f"  Gaia {name}: {len(values)} CCD residuals, RMS {np.sqrt(np.mean(values**2)):.6g} mas, "
            f"normalized RMS {np.sqrt(np.mean(normalized**2)):.6g}."
        )
    for axis in axes[-1]:
        axis.set_xlabel("Year")
    figure.tight_layout()
    table.to_csv(csv_path or Path(__file__).with_name(Path(__file__).stem + "_4_gaia_residuals.csv"), index=False)


def plot_residuals(setup_label, output, observation_dataset, target=None):
    """Plot ground, space and radar residuals in separate four-panel figures."""
    residuals = last_iteration_residuals(output, observation_dataset)
    data = observation_scalar_data(observation_dataset)
    normalized_residuals = residuals * np.sqrt(data["weights"])
    years = 2000.0 + data["times"] / constants.JULIAN_YEAR
    station_labels = data["station_labels"]
    angular_type = observable_models_setup.model_settings.angular_position_type
    range_type = observable_models_setup.model_settings.n_way_range_type
    doppler_type = observable_models_setup.model_settings.doppler_measured_frequency_type
    arcseconds_per_radian = 180.0 / np.pi * 3600.0
    mpc_astrometry = (data["observable_types"] == angular_type) & (station_labels != "Gaia")

    for category in ("MPC ground astrometry", "MPC space astrometry", "Radar"):
        if category == "Radar":
            selected = (data["observable_types"] == range_type) | (data["observable_types"] == doppler_type)
            components = [
                (range_type, 0, "Range", 1.0, "m"),
                (doppler_type, 0, "Doppler", 1.0, "Hz"),
            ]
        else:
            if category == "MPC ground astrometry":
                selected = mpc_astrometry & ~data["space_astrometry"]
            else:
                selected = mpc_astrometry & data["space_astrometry"]
            components = [
                (angular_type, 0, "Right ascension", arcseconds_per_radian, "arcsec"),
                (angular_type, 1, "Declination", arcseconds_per_radian, "arcsec"),
            ]
        if not np.any(selected):
            continue

        figure, axes = plt.subplots(2, 2, figsize=(11.7, 8.5), sharex=True)
        figure.suptitle(f"{setup_label}: {category}")
        for row, (observable_type, component, name, scale, unit) in enumerate(components):
            indices = np.flatnonzero(
                selected & (data["observable_types"] == observable_type)
                & (data["components"] == component)
            )
            values = residuals[indices] * scale
            normalized = normalized_residuals[indices]
            for column in (0, 1):
                axis = axes[row, column]
                plotted_values = values if column == 0 else normalized
                if indices.size:
                    add_grouped_scatter(axis, years[indices], plotted_values, station_labels[indices])
                    rms = np.sqrt(np.mean(plotted_values**2))
                    axis.set_title(f"RMS = {rms:.4g} {unit}" if column == 0 else f"Normalized RMS = {rms:.4g}")
                    if column == 0:
                        axis.legend(loc="best", fontsize=7)
                else:
                    axis.set_title(f"No {name.lower()} observations")
                axis.axhline(0.0, color="black", linewidth=0.8)
                axis.grid(alpha=0.3)
                axis.set_ylabel(f"{name} residual [{unit}]" if column == 0 else f"{name} residual / 1σ uncertainty")
                if column == 1 and category == "Radar":
                    axis.axhline(3.0, color="black", linestyle="--", linewidth=0.8)
                    axis.axhline(-3.0, color="black", linestyle="--", linewidth=0.8)
        for axis in axes[-1]:
            axis.set_xlabel("Year")
        figure.tight_layout()


def plot_orbit_difference(setup_label, output, estimator, epochs, horizons_states,
                          state_slice=None, target=None, propagated_covariances=None):
    """Plot R/S/W orbit differences and differences divided by formal uncertainty."""
    state_history = last_state_history(output)
    # Reuse the returned covariance: its iteration-to-iteration change is
    # negligible here. The orbit and residuals use the last iteration above.
    state_slice = slice(0, 6) if state_slice is None else state_slice
    target = TARGET if target is None else str(target)
    position_slice = slice(state_slice.start, state_slice.start + 3)
    if propagated_covariances is None:
        propagated_covariances = estimation_analysis.propagate_covariance(
            output.covariance,
            estimator.state_transition_interface,
            list(epochs),
        )

    differences_rsw = np.empty((len(epochs), 3))
    formal_errors_rsw = np.empty_like(differences_rsw)
    for index, (epoch, horizons_state) in enumerate(zip(epochs, horizons_states)):
        rotation = inertial_to_rsw_rotation_matrix(horizons_state)
        differences_rsw[index] = rotation @ (
            np.asarray(state_history[epoch])[position_slice] - horizons_state[:3]
        )
        position_covariance = np.asarray(propagated_covariances[epoch])[position_slice, position_slice]
        covariance_rsw = rotation @ position_covariance @ rotation.T
        formal_errors_rsw[index] = np.sqrt(
            np.clip(np.diag(covariance_rsw), 0.0, None)
        )

    differences_rsw /= 1000.0
    formal_errors_rsw /= 1000.0
    years = 2000.0 + np.asarray(epochs) / (365.25 * constants.JULIAN_DAY)

    figure, axes = plt.subplots(3, 2, figsize=(11.7, 10.5), sharex=True)
    for component, name in enumerate("RSW"):
        difference = differences_rsw[:, component]
        sigma = formal_errors_rsw[:, component]
        normalized = np.divide(
            difference, sigma, out=np.full_like(difference, np.nan), where=sigma > 0.0,
        )
        axes[component, 0].plot(years, difference, color="tab:blue")
        axes[component, 1].plot(years, normalized, color="tab:orange")
        axes[component, 0].set_ylabel(f"{name} [km]")
        axes[component, 1].set_ylabel(f"{name} / σ{name}")
        for axis in axes[component]:
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.grid(alpha=0.3)
    axes[0, 0].set_title("Estimate − Horizons")
    axes[0, 1].set_title("Orbit difference / formal uncertainty")
    for axis in axes[-1]:
        axis.set_xlabel("Year")
    figure.suptitle(f"{setup_label}\n{target} orbit difference with respect to JPL Horizons")
    figure.tight_layout()


def plot_results(results):
    """Display each estimation setup's diagnostics as Matplotlib figures."""
    histories = [last_state_history(result[0]) for result in results.values()]
    common_epochs = sorted(set.intersection(*(set(history) for history in histories)))
    if not common_epochs:
        raise RuntimeError("The estimated state histories have no common plotting epochs.")
    number_of_epochs = min(300, len(common_epochs))
    epoch_indices = np.unique(
        np.linspace(0, len(common_epochs) - 1, number_of_epochs, dtype=int)
    )
    plot_epochs = np.array([common_epochs[index] for index in epoch_indices])
    print(f"Querying Horizons at {len(plot_epochs)} epochs for orbit-difference plots...", flush=True)
    horizons_states = HorizonsQuery(
        query_id=HORIZONS_TARGET,
        location=HORIZONS_ORIGIN,
        epoch_list=list(plot_epochs),
        extended_query=True,
    ).cartesian(frame_orientation=FRAME_ORIENTATION)[:, 1:]

    for setup_label, (output, observation_dataset, estimator) in results.items():
        print(f"Creating residual and orbit-difference figures: {setup_label}...", flush=True)
        plot_residuals(setup_label, output, observation_dataset)
        plot_gaia_residuals(setup_label, output, observation_dataset)
        plot_orbit_difference(
            setup_label, output, estimator, plot_epochs, horizons_states
        )
    if plt.get_backend().lower() == "agg":
        plot_file = Path(__file__).with_name(f"{Path(__file__).stem}_4.pdf")
        with PdfPages(plot_file) as pdf:
            for number in plt.get_fignums():
                pdf.savefig(plt.figure(number))
        plt.close("all")
        print(f"Saved diagnostics to {plot_file}")
    else:
        print("Displaying figures; close the plot windows to finish.", flush=True)
        plt.show()

#####################################################################
#################   MAIN LOOP  ######################################
#####################################################################

def main():
    """Compare MPC and MPC/Gaia estimation, including any available radar."""
    print(f"Starting estimation of {TARGET}", flush=True)

    # Loading spice kernels
    spice.load_standard_kernels()

    # Load data, categorized in MPC, MPC + radar and MPC + radar + Gaia
    setups, observation_start_epoch, observation_end_epoch = load_tracking_data()

    # Run estimation for the different data categories
    results = {}
    for label, (tracking_data, supplementary_data) in setups.items():
        print(f"\nRunning estimation for: {label}...", flush=True)
        output, observation_dataset, estimator = perform_estimation(
            tracking_data,
            supplementary_data,
            observation_start_epoch,
            observation_end_epoch,
        )

        # Print summary of results to terminal
        print_residual_summary(label, output, observation_dataset)
        if ESTIMATE_YARKOVSKY:
            print_yarkovsky_result(output)

        # Save results
        results[label] = (output, observation_dataset, estimator)

    # Plot estimation results
    plot_results(results)


if __name__ == "__main__":
    # Keep Python progress visible in terminals, IDE consoles and redirected logs.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(line_buffering=True)
    main()
