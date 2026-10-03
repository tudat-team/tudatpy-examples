"""
# Joint MPC, JPL radar and Gaia astrometry estimation for multiple asteroids

Copyright (c) 2010-2026, Delft University of Technology. All rights reserved.
This file is part of Tudat. Redistribution and use in source and binary forms,
with or without modification, are permitted exclusively under the terms of the
Modified BSD license. See https://tudat.tudelft.nl/LICENSE.

This is a multi-body copy of ``mpc_radar_gaia_estimation.py``. Targets with
observations in the selected interval are propagated and estimated together,
using the same 51 asteroid perturbers.
ESTIMATE_YARKOVSKY adds a separate A2 for each target. ESTIMATE_BETA adds one
shared PPN beta; PPN gamma remains fixed at 1. RUN_ONLY_LAST_SETUP selects only
the final available data combination. ESTIMATE_SUN_J2 estimates the Sun's
normalized C20 using a simple-from-SPICE rotation model referenced to J2000.
Its J2 prior is anchored to the nominal value at the start of estimation.

The longest observation span after applying OBSERVATION_START/END defines the
shared estimation epoch and nominal interval. The interval is extended if
necessary to include all retained observations, then PROPAGATION_BUFFER is added
at both ends. Every asteroid uses these same propagation times.

MPC, Gaia and available JPL radar are converted to TrackingData, supplementary
data are installed in the environment, and an ObservationDataset is built.
Gaia retains its full transit covariance and the existing light-deflection and
photocenter corrections. Each target uses its own Gaia archive and radius.

Diagnostics use the last evaluated iteration and the returned covariance.
Only residual and parameter histories are saved per iteration. Orbit states
are read from the propagated bodies' ephemerides after estimation.
Residual RMS values are printed pooled and per target, one physical and one
normalized value for each panel of the original residual figures. Gaia AL/AC diagnostics
use the original marginal-uncertainty normalization. Orbit differences against
Horizons are reported for each target: R/S/W RMS in metres and RMS after dividing
each sample by its propagated formal uncertainty. They use 300 shared epochs,
the ephemerides' interpolation, and exclude 10 * INTEGRATOR_MAXIMUM_STEP at each
integration boundary. Only the time-step figure is produced.

MPC, Horizons and missing Gaia archives require an internet connection.
"""

import datetime
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
from tudatpy.math import interpolators


TARGETS = [
    "3200", "1566", "66146", "137924", "437844", "138127", "480883", "468468",
    "364136", "33342", "85989", "85953", "2100", "99907", "2062", "153201",
    "524522", "2340", "162004", "276033", "413260", "242191", "96590",
    "369986", "5786", "152742", "247517", "345705", "363505", "66400",
    "394130", "465402", "399457", "431760", "677579", "612162", "374158", "455426",
    "504181", "386454", "438116", "267223", "467372", "40267", "331471", "136874",
    "164201", "105140", "137925", "533671"
]
# Populated after filtering observations; keep the requested list available for reruns.
ACTIVE_TARGETS = list(TARGETS)
# Light-time geometry uses barycentric positions at emission and reception.
GLOBAL_FRAME_ORIGIN = "SSB"
PROPAGATION_CENTRAL_BODY = "Sun"
HORIZONS_ORIGIN = "500@10"
FRAME_ORIENTATION = "J2000"

# User-defined UTC observation limits; edit either bound to change the interval.
OBSERVATION_START = datetime.datetime(1980, 1, 1)
OBSERVATION_END = datetime.datetime(2026, 9, 1)
PROPAGATION_BUFFER = 2.0 * 31.0 * constants.JULIAN_DAY
ASTEROID_EPHEMERIS_STEP = 14.0 * constants.JULIAN_DAY
ASTEROID_EPHEMERIS_BUFFER = constants.JULIAN_YEAR
ASTEROID_EPHEMERIS_INTERPOLATION_POINTS = 10
INTEGRATOR_MAXIMUM_STEP = 36.0 * 3600.0
INTEGRATOR_TOLERANCE = 1.0e-14
NUMBER_OF_ESTIMATION_ITERATIONS = 15
ESTIMATE_YARKOVSKY = True
ESTIMATE_BETA = True
ESTIMATE_SUN_J2 = True
# Dimensionless J2 and its 1-sigma prior width; normalized C20 = -J2 / sqrt(5).
SUN_INITIAL_J2 = 2.2e-7
SUN_J2_PRIOR_SIGMA = 0.2e-7
SUN_INITIAL_C20 = -SUN_INITIAL_J2 / np.sqrt(5.0)
RUN_ONLY_LAST_SETUP = True

# Read each target's FPR archive, or query AIP and save it for subsequent runs.
GAIA_ARCHIVE_PATHS = {
    target: Path(__file__).with_name(f"gaia_{target}_fpr.parquet") for target in TARGETS
}

# The 51 most massive asteroid perturbers in the bundled SiMDA mass ranking.
# Use SPICE ephemerides and gravitational parameters where available;
# otherwise use Horizons ephemerides and the SiMDA masses below.
# States are tabulated every
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
    # (532, "Herculina"),
    # (451, "Patientia"),
    # (107, "Camilla"),
    # (536, "Merapi"),
    # (324, "Bamberga"),
    # (88, "Thisbe"),
    # (409, "Aspasia"),
    # (624, "Hektor"),
    # (702, "Alauda"),
    # (259, "Aletheia"),
    # (19, "Fortuna"),
    # (13, "Egeria"),
    # (354, "Eleonora"),
    # (9, "Metis"),
    # (334, "Chicago"),
    # (375, "Ursula"),
    # (22, "Kalliope"),
    # (165, "Loreley"),
    # (420, "Bertholda"),
    # (154, "Bertha"),
    # (48, "Doris"),
    # (423, "Diotima"),
    # (139, "Juewa"),
    # (1686, "De Sitter"),
    # (386, "Siegena"),
    # (96, "Aegle"),
    # (89, "Julia"),
    # (241, "Germania"),
    # (130, "Elektra"),
    # (45, "Eugenia"),
    # (33, "Polyhymnia"),
    # (41, "Daphne"),
    # (372, "Palma"),
    # (69, "Hesperia"),
    # (117, "Lomia"),
]

# SiMDA_240512.csv masses [kg] for bodies absent from the standard SPICE kernels.
SIMDA_FALLBACK_MASSES = {624: 9.82e18, 1686: 6.76e18, 33: 6.2e18}


#####################################################################
#################   RETRIEVE/PROCESS DATA    ########################
#####################################################################


def load_tracking_data_for_target(target):
    """Load observations and assemble comparison fits with a common TDB interval."""

    print(f"Loading observations for {target}...", flush=True)

    # Load MPC data
    batch = BatchMPC()
    batch.get_observations([target], use_mpc80_format=True)
    batch.filter(
        epoch_start=OBSERVATION_START,
        epoch_end=OBSERVATION_END,
        observatories_exclude=["C57", "C51", "786", "258", "704", "S04", "V55", "X71", "U91"],
    )
    tracking_data, supplementary_data, setups = [], [], {}
    print(f"Loaded {len(batch.table)} MPC optical observations.", flush=True)
    if not batch.table.empty:
        tracking_data, supplementary_data = optical_table_to_tracking_data(
            batch.table,
            add_weights=True,
            add_star_catalog_corrections=True,
            add_ancillary_data=True,
        )
        setups["MPC astrometry"] = (tracking_data, supplementary_data)

    # Load JPL radar data
    radar_query = JPLRadarQuery(target, timeout=60.0)
    radar_table = radar_query.to_radar_data(
        target_body=target,
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
        target=int(target),
        archive_path=GAIA_ARCHIVE_PATHS[target],
        epoch_start=OBSERVATION_START,
        epoch_end=OBSERVATION_END,
        time_scale=time_representation.utc_scale,
    )
    if gaia_astrometry is not None:
        radius = sbdb_photocenter_radius(target)
        gaia_tracking_data, gaia_supplementary_data = gaia_astrometry.to_tracking_data(
            light_deflection_bodies=("Sun", "Jupiter"),
            photocenter_body_dimensions={int(target): radius},
        )
        tracking_data = tracking_data + gaia_tracking_data
        supplementary_data = supplementary_data + gaia_supplementary_data
        label = "MPC astrometry and Gaia" if radar_table.empty else "MPC astrometry, JPL radar and Gaia"
        setups[label] = (tracking_data, supplementary_data)
    else:
        print(f"No Gaia data for {target}; retaining available MPC and radar data.", flush=True)

    # A target without MPC may still have usable radar or Gaia observations.
    if not tracking_data:
        print(f"Removing {target}: no observations in the selected interval.", flush=True)
        return {}, None, None
    first_epoch, last_epoch = get_tracking_data_epoch_bounds(tracking_data)
    return setups, float(first_epoch), float(last_epoch)


def load_tracking_data():
    """Combine each target's data and select a common propagation interval."""
    global ACTIVE_TARGETS
    if not TARGETS or len(TARGETS) != len(set(TARGETS)):
        raise ValueError("TARGETS must contain distinct asteroid identifiers.")
    target_setups, epoch_bounds = {}, {}
    for target in TARGETS:
        setups, first_epoch, last_epoch = load_tracking_data_for_target(target)
        if setups:
            target_setups[target] = setups
            epoch_bounds[target] = (first_epoch, last_epoch)
    ACTIVE_TARGETS = list(target_setups)
    if not ACTIVE_TARGETS:
        raise RuntimeError("No target has observations in the selected interval.")
    print(f"Propagating and estimating: {', '.join(ACTIVE_TARGETS)}", flush=True)

    def combine(selected_setups):
        return (
            [track for tracks, _ in selected_setups for track in tracks],
            [extra for _, extras in selected_setups for extra in extras],
        )

    mpc_label = "MPC astrometry"
    radar_label = "MPC astrometry and JPL radar"
    setups = {}
    any_radar = any(radar_label in data for data in target_setups.values())
    any_gaia = any(any("Gaia" in label for label in data) for data in target_setups.values())
    stages = [(mpc_label, lambda label: label == mpc_label)]
    if any_radar:
        stages.append((radar_label, lambda label: "Gaia" not in label))
    if any_gaia:
        label = "MPC astrometry, JPL radar and Gaia" if any_radar else "MPC astrometry and Gaia"
        stages.append((label, lambda label: True))
    for label, include in stages:
        selected = [
            [value for name, value in data.items() if include(name)]
            for data in target_setups.values()
        ]
        # Do not create a comparison fit with an unobserved body's free parameters.
        if all(selected):
            setups[label] = combine([available[-1] for available in selected])
        else:
            print(f"Skipping {label}: some retained targets have no data in this setup.", flush=True)

    longest_span_target = max(
        ACTIVE_TARGETS, key=lambda target: epoch_bounds[target][1] - epoch_bounds[target][0]
    )
    first_epoch, last_epoch = epoch_bounds[longest_span_target]
    estimation_epoch = max(0.0, 0.5 * (first_epoch + last_epoch))
    # Keep the longest-span reference epoch while covering every retained observation.
    first_epoch = min(first_epoch, *(bounds[0] for bounds in epoch_bounds.values()))
    last_epoch = max(last_epoch, *(bounds[1] for bounds in epoch_bounds.values()))
    print(
        f"Common estimation epoch from {longest_span_target}'s longest observation span: "
        f"{estimation_epoch:.6f} s TDB since J2000.\n"
        f"Shared observation bounds: {first_epoch:.6f} to {last_epoch:.6f} s TDB; "
        f"propagation buffer: {PROPAGATION_BUFFER / constants.JULIAN_DAY:g} days per end.",
        flush=True,
    )
    return setups, first_epoch, last_epoch, estimation_epoch


def sbdb_photocenter_radius(target):
    """Use half the SBDB diameter, or a 100 m radius if no diameter is available."""
    query = SBDBquery(str(target))
    try:
        return 0.5 * query.diameter
    except ValueError:
        # Astroquery can drop the diameter unit when its errors are asymmetric.
        from astroquery.jplsbdb import SBDB
        from astropy import units as u
        parameters = SBDB.query_async(str(target), phys=True).json().get("phys_par", [])
        diameter = next((p for p in parameters if p["name"] == "diameter"), None)
        if diameter is not None and diameter["value"] is not None and diameter["units"]:
            return 0.5 * (float(diameter["value"]) * u.Unit(diameter["units"])).to_value(u.m)
        radius = 100.0
        print(f"No SBDB diameter for {target}; using fallback photocenter radius {radius:g} m.", flush=True)
        return radius


#####################################################################
#################   CREATE ENVIRONMENT   ############################
#####################################################################


def asteroid_ephemeris_settings(number, first_epoch, final_epoch):
    """Tabulate a perturber from SPICE or a cached Horizons reference."""

    start = float(first_epoch) - ASTEROID_EPHEMERIS_BUFFER
    end = float(final_epoch) + ASTEROID_EPHEMERIS_BUFFER
    interpolation = interpolators.lagrange_interpolation(ASTEROID_EPHEMERIS_INTERPOLATION_POINTS)
    if number in SIMDA_FALLBACK_MASSES:
        print(f"Loading Horizons ephemeris for asteroid {number}...", flush=True)
        history = load_reference_state_history(
            number, start, end,
            epoch_step=f"{ASTEROID_EPHEMERIS_STEP / constants.JULIAN_DAY:g}d",
        )
        return environment_setup.ephemeris.tabulated_from_existing(
            environment_setup.ephemeris.tabulated(history, PROPAGATION_CENTRAL_BODY, FRAME_ORIENTATION),
            min(history), max(history), ASTEROID_EPHEMERIS_STEP, interpolation,
        )
    return environment_setup.ephemeris.interpolated_spice(
            start, end, ASTEROID_EPHEMERIS_STEP, GLOBAL_FRAME_ORIGIN, FRAME_ORIENTATION,
            interpolation, spice.asteroid_spice_id(number))


@lru_cache(maxsize=10)
def load_reference_state_history(target, first_epoch, final_epoch, epoch_step="1d"):
    """Reuse each target's Horizons reference across the comparison fits."""
    states = HorizonsQuery(
        query_id=f"{target};", location=HORIZONS_ORIGIN,
        epoch_start=float(first_epoch), epoch_end=float(final_epoch),
        epoch_step=epoch_step, extended_query=True,
    ).cartesian(frame_orientation=FRAME_ORIENTATION)
    return dict(zip(states[:, 0], states[:, 1:]))


def set_solar_gravity_and_rotation(body_settings):
    """Use a degree-2 solar field in the simple SPICE frame at t = 0 TDB."""
    sun = body_settings.get("Sun")
    sun.rotation_model_settings = environment_setup.rotation_model.simple_from_spice(
        base_frame=FRAME_ORIENTATION,
        target_frame="IAU_Sun_Simple",
        target_frame_spice="IAU_SUN",
        initial_time=0.0,
    )
    cosine = np.zeros((3, 3))
    cosine[0, 0] = 1.0
    cosine[2, 0] = SUN_INITIAL_C20
    sun.gravity_field_settings = environment_setup.gravity_field.spherical_harmonic(
        gravitational_parameter=spice.get_body_gravitational_parameter("Sun"),
        reference_radius=spice.get_average_radius("Sun"),
        normalized_cosine_coefficients=cosine,
        normalized_sine_coefficients=np.zeros((3, 3)),
        associated_reference_frame="IAU_Sun_Simple",
    )


def create_bodies(first_epoch, final_epoch):
    """Create the estimation environment and Earth observing stations."""
    estimated_bodies = ACTIVE_TARGETS

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
    set_solar_gravity_and_rotation(body_settings)

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
    estimated_bodies = ACTIVE_TARGETS if estimated_bodies is None else list(estimated_bodies)
    sun_accelerations = [
        propagation_setup.acceleration.spherical_harmonic_gravity(2, 0),
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


def solar_j2_inverse_apriori_covariance(parameters_to_estimate):
    """Constrain normalized solar C20, leaving all other parameters unconstrained."""
    size = len(parameters_to_estimate.parameter_vector)
    inverse_covariance = np.zeros((size, size))
    if ESTIMATE_SUN_J2:
        sigma_c20 = SUN_J2_PRIOR_SIGMA / np.sqrt(5.0)
        if not np.isfinite(sigma_c20) or sigma_c20 <= 0.0:
            raise ValueError("SUN_J2_PRIOR_SIGMA must be finite and positive.")
        index = global_parameter_indices()["C20"]
        inverse_covariance[index, index] = 1.0 / sigma_c20**2
    return inverse_covariance


def perform_estimation(
    tracking_data,
    supplementary_data,
    observation_start_epoch,
    observation_end_epoch,
    estimation_epoch,
):
    """Jointly estimate all asteroid states at the common estimation epoch."""

    # Extend both ends by the configured propagation buffer.
    first_epoch = observation_start_epoch - PROPAGATION_BUFFER
    final_epoch = observation_end_epoch + PROPAGATION_BUFFER

    # Create bodies
    bodies = create_bodies(first_epoch, final_epoch)
    initial_state = np.concatenate([
        bodies.get(target).ephemeris.cartesian_state(float(estimation_epoch))
        for target in ACTIVE_TARGETS
    ])

    # This installs transmitter-frequency histories, identifies the passive
    # radar reflector, and creates the space-telescope bodies and ephemerides.
    observations.set_tracking_supplementary_data_in_bodies(
        bodies,
        supplementary_data,
    )
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies,
        acceleration_settings(ESTIMATE_YARKOVSKY),
        ACTIVE_TARGETS,
        [PROPAGATION_CENTRAL_BODY] * len(ACTIVE_TARGETS),
    )
    # Apply tolerances to position and velocity blocks, including when
    # the state is propagated together with the variational equations.
    step_size_control_settings = (
        propagation_setup.integrator.step_size_control_custom_blockwise_scalar_tolerance(
            block_indices_function=propagation_setup.integrator.standard_cartesian_state_element_blocks,
            relative_error_tolerance=INTEGRATOR_TOLERANCE,
            absolute_error_tolerance=INTEGRATOR_TOLERANCE,
        )
    )
    step_size_validation_settings = propagation_setup.integrator.step_size_validation(
        minimum_step=1.0e-3,
        maximum_step=INTEGRATOR_MAXIMUM_STEP,
    )
    integrator_settings = propagation_setup.integrator.runge_kutta_variable_step(
        initial_time_step=time_representation.Time(0.1 * INTEGRATOR_MAXIMUM_STEP),
        coefficient_set=propagation_setup.integrator.CoefficientSets.rkf_78,
        step_size_control_settings=step_size_control_settings,
        step_size_validation_settings=step_size_validation_settings,
    )
    termination_settings = propagation_setup.propagator.non_sequential_termination(
        propagation_setup.propagator.time_termination(final_epoch),
        propagation_setup.propagator.time_termination(first_epoch),
    )
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies=[PROPAGATION_CENTRAL_BODY] * len(ACTIVE_TARGETS),
        acceleration_models=acceleration_models,
        bodies_to_integrate=ACTIVE_TARGETS,
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
        parameter_settings.extend(
            parameters_setup.yarkovsky_parameter(target, "Sun") for target in ACTIVE_TARGETS
        )
    if ESTIMATE_BETA:
        parameter_settings.append(parameters_setup.ppn_parameter_beta())
    if ESTIMATE_SUN_J2:
        parameter_settings.append(
            parameters_setup.spherical_harmonics_c_coefficients_block("Sun", [(2, 0)])
        )
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
        inverse_apriori_covariance=solar_j2_inverse_apriori_covariance(parameters_to_estimate),
        # Anchor the prior to the first iteration's parameters across all updates.
        apply_apriori_parameter_deviation=True,
        outlier_rejection_settings=estimation_analysis.carpino_outlier_rejection_settings(
            chi2_rejection_threshold=6.0,
            chi2_recovery_threshold=3.0,
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
        save_state_history_per_iteration=False,
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
    """Print each asteroid's A2 and formal uncertainty in parameter-set order."""
    values = last_iteration_parameters(output)
    covariance = np.asarray(output.covariance)
    si_to_au_per_day_squared = constants.JULIAN_DAY**2 / constants.ASTRONOMICAL_UNIT
    # The 6N states precede scalar A2/beta parameters, then the C20 vector block.
    for body_index, target in enumerate(ACTIVE_TARGETS):
        parameter_index = 6 * len(ACTIVE_TARGETS) + body_index
        a2_si = float(values[parameter_index])
        sigma_si = float(np.sqrt(covariance[parameter_index, parameter_index]))
        print(
            f"  {target} Yarkovsky A2: ({a2_si:.9g} ± {sigma_si:.3g}) m/s² = "
            f"({a2_si * si_to_au_per_day_squared:.9g} ± "
            f"{sigma_si * si_to_au_per_day_squared:.3g}) au/day²"
        )


def global_parameter_indices():
    """Indices of optional global parameters after the states and scalar A2s."""
    index = 6 * len(ACTIVE_TARGETS) + (len(ACTIVE_TARGETS) if ESTIMATE_YARKOVSKY else 0)
    indices = {}
    if ESTIMATE_BETA:
        indices["beta"] = index
        index += 1
    if ESTIMATE_SUN_J2:
        indices["C20"] = index
    return indices


def print_beta_result(output):
    """Print the estimated dimensionless PPN beta and its formal uncertainty."""
    index = global_parameter_indices()["beta"]
    beta = float(last_iteration_parameters(output)[index])
    sigma = float(np.sqrt(np.asarray(output.covariance)[index, index]))
    print(f"  Estimated PPN beta: {beta:.12g} ± {sigma:.3g}")


def print_solar_j2_result(output):
    """Report the estimated normalized C20 and its equivalent unnormalized J2."""
    index = global_parameter_indices()["C20"]
    c20 = float(last_iteration_parameters(output)[index])
    sigma = float(np.sqrt(np.asarray(output.covariance)[index, index]))
    print(f"  Sun normalized C20: {c20:.12g} ± {sigma:.6g} (dimensionless)")
    print(f"  Sun J2 = -sqrt(5) * C20: {-np.sqrt(5.0) * c20:.12g} ± {np.sqrt(5.0) * sigma:.6g}")
    print(f"  Solar J2 prior: {SUN_INITIAL_J2:.9g} ± {SUN_J2_PRIOR_SIGMA:.6g} (1 sigma), anchored to the initial value.")
    print(f"  Solar reference radius: {spice.get_average_radius('Sun'):.9g} m; simple-from-SPICE rotation at t = 0 TDB.")


def print_beta_correlations(output):
    """Use the same covariance as the formal errors; J2 reverses the C20 sign."""
    indices = global_parameter_indices()
    if "beta" not in indices:
        return
    covariance = np.asarray(output.covariance)
    beta_index = indices["beta"]

    def correlation(index):
        return covariance[beta_index, index] / np.sqrt(
            covariance[beta_index, beta_index] * covariance[index, index]
        )

    if "C20" in indices:
        rho = correlation(indices["C20"])
        print(f"  Correlation(beta, Sun normalized C20): {rho:.9g}")
        print(f"  Correlation(beta, Sun J2): {-rho:.9g}")
    if ESTIMATE_YARKOVSKY:
        for body_index, target in enumerate(ACTIVE_TARGETS):
            rho = correlation(6 * len(ACTIVE_TARGETS) + body_index)
            print(f"  Correlation(beta, {target} A2): {rho:.9g}")


def print_residual_summary(label, output, observation_dataset, target=None):
    """Print physical/normalized RMS pooled over all targets or for one target."""
    residuals = last_iteration_residuals(output, observation_dataset)
    data = observation_scalar_data(observation_dataset)
    normalized_residuals = residuals * np.sqrt(data["weights"])
    angular_type = observable_models_setup.model_settings.angular_position_type
    range_type = observable_models_setup.model_settings.n_way_range_type
    doppler_type = observable_models_setup.model_settings.doppler_measured_frequency_type
    arcseconds_per_radian = 180.0 / np.pi * 3600.0
    mpc_astrometry = (data["observable_types"] == angular_type) & (data["station_labels"] != "Gaia")
    scope = f"pooled over all {len(ACTIVE_TARGETS)} asteroids" if target is None else f"for asteroid {target}"
    print(f"\n{label}: residual RMS {scope}")
    target_mask = np.ones(len(residuals), dtype=bool) if target is None else data["targets"] == str(target)

    def print_rms(name, values, normalized, unit):
        if not len(values):
            print(f"  {name}: N/A (no active observations)")
            return
        print(
            f"  {name}: N = {len(values)}, RMS = {np.sqrt(np.mean(values**2)):.6g} {unit}, "
            f"normalized RMS = {np.sqrt(np.mean(normalized**2)):.6g}"
        )

    for category in ("MPC ground astrometry", "MPC space astrometry", "Radar"):
        if category == "Radar":
            selected = (data["observable_types"] == range_type) | (data["observable_types"] == doppler_type)
            components = [(range_type, 0, "Range", 1.0, "m"), (doppler_type, 0, "Doppler", 1.0, "Hz")]
        else:
            selected = mpc_astrometry & (
                ~data["space_astrometry"] if category == "MPC ground astrometry"
                else data["space_astrometry"]
            )
            components = [
                (angular_type, 0, "Right ascension", arcseconds_per_radian, "arcsec"),
                (angular_type, 1, "Declination", arcseconds_per_radian, "arcsec"),
            ]
        for observable_type, component, name, scale, unit in components:
            selected_components = (
                target_mask & selected & (data["observable_types"] == observable_type)
                & (data["components"] == component)
            )
            print_rms(
                f"{category} / {name}", residuals[selected_components] * scale,
                normalized_residuals[selected_components], unit,
            )

    gaia = gaia_residual_data(output, observation_dataset, target)
    for component, name in enumerate(("Along-scan (AL)", "Cross-scan (AC)")):
        values = np.array([]) if gaia is None else gaia["scan_residuals"][:, component]
        normalized = values if gaia is None else values / gaia["scan_sigmas"][:, component]
        print_rms(f"Gaia / {name}", values * arcseconds_per_radian * 1000.0, normalized, "mas")


def propagated_ephemeris(estimator, target):
    """Read the body's ephemeris, which contains the last propagated orbit."""
    bodies = estimator.variational_solver.dynamics_simulator.bodies
    return bodies.get(target).ephemeris


def integration_epochs(estimator):
    """Read accepted step epochs from the final propagation's timing records."""
    simulator = estimator.variational_solver.dynamics_simulator
    # The bodies share one integration grid. Timing records retain its epochs
    # without requiring saved state histories or access to an ephemeris interpolator.
    return np.array(sorted(simulator.cumulative_computation_time_history), dtype=float)


#####################################################################
#################   DIAGNOSTICS   ###################################
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
            next((end.body_name for end in link_ends.values() if end.body_name in ACTIVE_TARGETS), ""),
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
        "targets": np.array([link_data[set_id][2] for set_id in scalar_set_ids]),
    }


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
    query = observations.observation_query
    condition = (
        query.active
        & (query.receiver == observations.LinkEndId("Gaia", ""))
    )
    if target is not None:
        condition = condition & (query.transmitter == observations.LinkEndId(str(target), ""))
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


def print_orbit_difference(setup_label, output, estimator, epochs, horizons_states,
                          state_slice, target, propagated_covariances=None):
    """Print R/S/W RMS in metres and RMS normalized by propagated formal errors."""
    ephemeris = propagated_ephemeris(estimator, target)
    integrated_epochs = integration_epochs(estimator)
    # Avoid interpolation near the integration boundaries.
    epoch_margin = 10.0 * INTEGRATOR_MAXIMUM_STEP
    epochs = np.asarray(epochs, dtype=float)
    retained = (
        (epochs >= integrated_epochs[0] + epoch_margin)
        & (epochs <= integrated_epochs[-1] - epoch_margin)
    )
    epochs = epochs[retained]
    horizons_states = np.asarray(horizons_states)[retained]
    if not len(epochs):
        raise RuntimeError(
            "No orbit-comparison epochs remain after trimming the integration boundaries."
        )
    # Reuse the returned covariance: its iteration-to-iteration change is
    # negligible here. The orbit and residuals use the last iteration above.
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
            np.asarray(ephemeris.cartesian_state(float(epoch)))[:3]
            - horizons_state[:3]
        )
        position_covariance = np.asarray(propagated_covariances[epoch])[position_slice, position_slice]
        covariance_rsw = rotation @ position_covariance @ rotation.T
        formal_errors_rsw[index] = np.sqrt(
            np.clip(np.diag(covariance_rsw), 0.0, None)
        )

    normalized = np.divide(
        differences_rsw, formal_errors_rsw, out=np.full_like(differences_rsw, np.nan),
        where=formal_errors_rsw > 0.0,
    )
    rms = np.sqrt(np.mean(differences_rsw**2, axis=0))
    normalized_rms = np.sqrt(np.mean(normalized**2, axis=0))
    print(f"  {target} orbit minus Horizons ({len(epochs)} epochs):")
    for component, name in enumerate("RSW"):
        print(
            f"    {name}: RMS = {rms[component]:.6g} m, "
            f"normalized RMS = {normalized_rms[component]:.6g}"
        )
    return rms, normalized_rms


def plot_time_steps(setup_label, estimator):
    """Plot accepted time-step magnitudes from the last evaluated iteration."""
    # Sorting gives positive step sizes for both propagation directions.
    epochs = integration_epochs(estimator)
    time_steps = np.diff(epochs)
    midpoint_epochs = 0.5 * (epochs[:-1] + epochs[1:])
    years = 2000.0 + midpoint_epochs / constants.JULIAN_YEAR

    figure, axis = plt.subplots(figsize=(11.7, 5.0))
    axis.plot(years, time_steps / 3600.0, color="tab:blue")
    axis.set_xlabel("Year")
    axis.set_ylabel("Time step [h]")
    axis.set_title(f"{setup_label}\nIntegration time steps (last evaluated iteration)")
    axis.grid(alpha=0.3)
    figure.tight_layout()


def report_results(results):
    """Print final residual/orbit diagnostics and show only the time-step figure."""
    last_setup_label = next(reversed(results))
    plot_time_steps(last_setup_label, results[last_setup_label][2])
    epoch_grids = [integration_epochs(result[2]) for result in results.values()]
    # Adaptive-step histories have different epochs. Compare interpolated
    # orbits over their shared interval, excluding both integration edges.
    epoch_margin = 10.0 * INTEGRATOR_MAXIMUM_STEP
    first_epoch = max(epochs[0] for epochs in epoch_grids) + epoch_margin
    final_epoch = min(epochs[-1] for epochs in epoch_grids) - epoch_margin
    if first_epoch >= final_epoch:
        raise RuntimeError(
            "The estimated state histories have no common plotting interval "
            "after trimming the integration boundaries."
        )
    plot_epochs = np.linspace(first_epoch, final_epoch, 300)
    horizons_states = {}
    for target in ACTIVE_TARGETS:
        print(f"Querying Horizons at {len(plot_epochs)} diagnostic epochs for {target}...", flush=True)
        horizons_states[target] = HorizonsQuery(
            query_id=f"{target};",
            location=HORIZONS_ORIGIN,
            epoch_list=list(plot_epochs),
            extended_query=True,
        ).cartesian(frame_orientation=FRAME_ORIENTATION)[:, 1:]

    for setup_label, (output, observation_dataset, estimator) in results.items():
        print_residual_summary(setup_label, output, observation_dataset)
        for target in ACTIVE_TARGETS:
            print_residual_summary(setup_label, output, observation_dataset, target)
        if ESTIMATE_YARKOVSKY:
            print_yarkovsky_result(output)
        if ESTIMATE_BETA:
            print_beta_result(output)
        if ESTIMATE_SUN_J2:
            print_solar_j2_result(output)
        print_beta_correlations(output)
        propagated_covariances = estimation_analysis.propagate_covariance(
            output.covariance, estimator.state_transition_interface, list(plot_epochs),
        )
        for body_index, target in enumerate(ACTIVE_TARGETS):
            print_orbit_difference(
                setup_label, output, estimator, plot_epochs, horizons_states[target],
                slice(6 * body_index, 6 * body_index + 6), target, propagated_covariances,
            )
    if plt.get_backend().lower() == "agg":
        plot_file = Path(__file__).with_name(f"{Path(__file__).stem}_time_steps.pdf")
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
    print(f"Starting joint estimation of {', '.join(TARGETS)}", flush=True)
   
    # Loading spice kernels
    spice.load_standard_kernels()
    
    # Load data, categorized in MPC, MPC + radar and MPC + radar + Gaia
    setups, observation_start_epoch, observation_end_epoch, estimation_epoch = load_tracking_data()
    if RUN_ONLY_LAST_SETUP:
        last_setup_label = next(reversed(setups))
        setups = {last_setup_label: setups[last_setup_label]}

    # Run estimation for the different data categories
    results = {}
    for label, (tracking_data, supplementary_data) in setups.items():
        print(f"\nRunning estimation for: {label}...", flush=True)
        output, observation_dataset, estimator = perform_estimation(
            tracking_data,
            supplementary_data,
            observation_start_epoch,
            observation_end_epoch,
            estimation_epoch,
        )
        
        # Save results
        results[label] = (output, observation_dataset, estimator)

    # Report all final diagnostics, with only the time-step figure.
    report_results(results)


if __name__ == "__main__":
    # Keep Python progress visible in terminals, IDE consoles and redirected logs.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(line_buffering=True)
    main()
