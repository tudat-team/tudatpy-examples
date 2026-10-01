"""
# MPC, JPL radar and Gaia astrometry estimation

Copyright (c) 2010-2026, Delft University of Technology. All rights reserved.
This file is part of Tudat. Redistribution and use in source and binary forms,
with or without modification, are permitted exclusively under the terms of the
Modified BSD license. See https://tudat.tudelft.nl/LICENSE.

This example estimates the state of 673 Edda from MPC optical astrometry,
then from MPC and Gaia astrometry together. JPL radar is included when
available. It is a copy of ``mpc_and_radar_estimation.py`` with Gaia added.
The example follows the current data path directly:

1. MPC, Gaia and available JPL records are converted to ``TrackingData``.
2. Their supplementary data are applied to the system of bodies.
3. The tracking data are converted to an ``ObservationDataset``.
4. Orbit determinations are run without and with Gaia.

Gaia's full transit covariance includes random RA/Dec correlations and
systematic errors shared by the CCD observations in each transit. Relativistic light deflection and a
spherical photocenter correction are applied before creating the dataset.

Gaia along-scan and cross-scan plots use the published scan position angle and
RA multiplied by cos(dec). Cross-scan is the perpendicular tangent-plane
projection; aberration makes Gaia's actual AC direction slightly nonorthogonal
to AL, as described in the FPR data model. Normalized plots use marginal
uncertainties, while the fit retains the full correlated weights.

The queries require an internet connection. Use ``--gaia-archive FILE.parquet``
to load a local Gaia FPR archive instead of querying ESA. This copy uses Edda to
demonstrate MPC/Gaia fusion: the Gaia FPR and DR3 queries for Bennu (101955)
returned zero observations. The Gaia run requires nonempty Gaia data for
TARGET and stops if none are available. Radar is added when available.
"""

import argparse
import datetime
from collections import Counter
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from tudatpy import constants
from tudatpy.astro.frame_conversion import inertial_to_rsw_rotation_matrix
from tudatpy.astro import time_representation
from tudatpy.data_input.environment_data import spice
from tudatpy.data_input.environment_data.horizons import HorizonsQuery
from tudatpy.data_input.environment_data.sbdb import SBDBquery
from tudatpy.data_input.tracking_data.gaia import GaiaAstrometry
from tudatpy.data_input.tracking_data.jpl_radar import JPLRadarQuery
from tudatpy.data_input.tracking_data.mpc import BatchMPC
from tudatpy.data_input.tracking_data.optical_utilities import (
    SPACECRAFT_POSITION_COLUMNS,
    optical_table_to_tracking_data,
)
from tudatpy.data_input.tracking_data.radar_utilities import (
    radar_data_to_tracking_data,
)
from tudatpy.dynamics import (
    environment_setup, parameters_setup, propagation_setup, simulator,
)
from tudatpy.estimation import estimation_analysis, observable_models_setup, observations


TARGET = "673"
HORIZONS_TARGET = f"{TARGET};"
FRAME_ORIGIN = "Sun"
HORIZONS_ORIGIN = "500@10"
FRAME_ORIENTATION = "J2000"

OBSERVATION_START = datetime.datetime(2014, 1, 1)
OBSERVATION_END = datetime.datetime(2020, 7, 1)
PROPAGATION_BUFFER = 2.0 * 31.0 * constants.JULIAN_DAY
INTEGRATOR_STEP = 12.0 * 3600.0
NUMBER_OF_ESTIMATION_ITERATIONS = 4
ESTIMATE_YARKOVSKY = False
NUMBER_OF_COLORED_OBSERVATORIES = 10
NUMBER_OF_HORIZONS_PLOT_EPOCHS = 300

YARKOVSKY_INITIAL_A2 = 0.0

# Optional FPR parquet archive and spherical radius [m]. Without an override,
# use half the diameter reported by JPL SBDB for the configured TARGET.
GAIA_ARCHIVE_PATH = None
PHOTOCENTER_RADIUS = None

# The 21 most massive main-belt asteroids in the SiMDA data set distributed
# with the Tudat example. Their ephemerides and gravitational parameters are
# read from the standard SPICE asteroid kernels.
ASTEROID_PERTURBERS = [
    (1, "Ceres"),
    (4, "Vesta"),
    (2, "Pallas"),
    (10, "Hygiea"),
    (704, "Interamnia"),
    # (15, "Eunomia"),
    # (511, "Davida"),
    # (3, "Juno"),
    # (52, "Europa"),
    # (16, "Psyche"),
    # (65, "Cybele"),
    # (87, "Sylvia"),
    # (31, "Euphrosyne"),
    # (7, "Iris"),
    # (29, "Amphitrite"),
    # (6, "Hebe"),
    # (532, "Herculina"),
    # (451, "Patientia"),
    # (107, "Camilla"),
    # (536, "Merapi"),
    # (324, "Bamberga"),
]


def asteroid_body_name(number, name):
    """Return the Tudat environment name used for an asteroid perturber."""
    return f"{number} {name}"


def asteroid_spice_id(number):
    """Return the NAIF name used by the standard 300-asteroid SPICE kernel."""
    return f"200{number:04d}"


def load_tracking_data():
    """Load and convert the MPC astrometry and JPL radar observations."""
    batch = BatchMPC()
    batch.get_observations([TARGET], use_mpc80_format=True)
    batch.filter(
        epoch_start=OBSERVATION_START,
        epoch_end=OBSERVATION_END,
        # Gaia CCDs are loaded separately with their full transit covariance;
        # exclude Gaia's MPC observatory code to avoid counting them twice.
        observatories_exclude=["C57", "258"],
    )
    if batch.table.empty:
        raise RuntimeError("The selected interval contains no MPC astrometry.")

    space_columns = set(SPACECRAFT_POSITION_COLUMNS)
    number_of_space_observations = (
        int(batch.table[list(space_columns)].notna().all(axis=1).sum())
        if space_columns.issubset(batch.table.columns)
        else 0
    )
    print(
        f"Loaded {len(batch.table)} MPC optical observations, "
        f"including {number_of_space_observations} space-based observations."
    )

    optical_tracking_data, optical_supplementary_data = optical_table_to_tracking_data(
        batch.table,
        add_weights=True,
        add_star_catalog_corrections=True,
        add_ancillary_data=True,
    )

    radar_query = JPLRadarQuery(TARGET, timeout=60.0)
    if isinstance(radar_query._content, str):
        radar_query.__dict__["_content"] = json.loads(radar_query._content)
    radar_table = radar_query.to_radar_data(
        target_body=TARGET,
        epoch_start=OBSERVATION_START,
        epoch_end=OBSERVATION_END,
        target_point="C",
    )
    if radar_table.empty:
        radar_tracking_data, radar_supplementary_data = [], []
        print("No JPL radar observations found; skipping the radar-inclusive run.")
    else:
        radar_tracking_data, radar_supplementary_data = radar_data_to_tracking_data(
            radar_table
        )
        print(f"Loaded {len(radar_table)} JPL center-of-mass radar observations.")

    return (
        optical_tracking_data,
        optical_supplementary_data,
        radar_tracking_data,
        radar_supplementary_data,
    )


def observation_epoch_bounds(tracking_data):
    """Return the earliest/latest UTC tracking-data epochs converted to TDB."""
    epoch_bounds_utc = (
        min(min(data.epochs) for data in tracking_data),
        max(max(data.epochs) for data in tracking_data),
    )
    converter = time_representation.default_time_scale_converter()
    return tuple(
        converter.convert_time(
            input_scale=time_representation.utc_scale,
            output_scale=time_representation.tdb_scale,
            input_value=float(epoch),
        )
        for epoch in epoch_bounds_utc
    )


def load_gaia_astrometry():
    """Require actual Gaia CCD observations for the configured target."""
    if GAIA_ARCHIVE_PATH is None:
        gaia = GaiaAstrometry.load_from_astroquery(int(TARGET))
    else:
        gaia = GaiaAstrometry.load_from_local_archive(GAIA_ARCHIVE_PATH, int(TARGET))
    converter = time_representation.default_time_scale_converter()
    bounds = [
        converter.convert_time(
            input_scale=time_representation.utc_scale,
            output_scale=time_representation.tdb_scale,
            input_value=time_representation.DateTime.from_python_datetime(value).epoch(),
        )
        for value in (OBSERVATION_START, OBSERVATION_END)
    ]
    gaia.apply_filters(epoch_start=bounds[0], epoch_end=bounds[1])
    if gaia.table.empty or set(gaia.mpc_numbers_in_table) != {int(TARGET)}:
        raise RuntimeError("Nonempty Gaia astrometry for TARGET is required.")
    print(
        f"Loaded {len(gaia.table)} Gaia CCD observations in "
        f"{gaia.table.transit_id.nunique()} transits for {TARGET}."
    )
    return gaia


def create_bodies(gaia_astrometry=None):
    """Create the estimation environment and Earth observing stations."""
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
        FRAME_ORIGIN,
        FRAME_ORIENTATION,
    )

    for body_name in ["Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]:
        barycenter_name = f"{body_name} Barycenter"
        settings = body_settings.get(body_name)
        settings.ephemeris_settings = environment_setup.ephemeris.direct_spice(
            FRAME_ORIGIN,
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

    body_settings.add_empty_settings(TARGET)
    if gaia_astrometry is not None:
        body_settings.add_empty_settings("Gaia")
        body_settings.get("Gaia").ephemeris_settings = (
            gaia_astrometry.get_gaia_ephemeris_settings()
        )

    for number, name in ASTEROID_PERTURBERS:
        body_name = asteroid_body_name(number, name)
        spice_id = asteroid_spice_id(number)
        body_settings.add_empty_settings(body_name)
        settings = body_settings.get(body_name)
        settings.ephemeris_settings = environment_setup.ephemeris.direct_spice(
            FRAME_ORIGIN,
            FRAME_ORIENTATION,
            spice_id,
        )
        settings.gravity_field_settings = environment_setup.gravity_field.central_spice(
            spice_id
        )

    return environment_setup.create_system_of_bodies(body_settings)


def acceleration_settings(estimate_yarkovsky):
    """Return the force model used for the configured target."""
    sun_accelerations = [
        propagation_setup.acceleration.point_mass_gravity(),
        propagation_setup.acceleration.relativistic_correction(
            use_schwarzschild=True
        ),
    ]
    if estimate_yarkovsky:
        sun_accelerations.append(
            propagation_setup.acceleration.yarkovsky(YARKOVSKY_INITIAL_A2)
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
    target_accelerations.update(
        {
            asteroid_body_name(number, name): [
                propagation_setup.acceleration.point_mass_gravity()
            ]
            for number, name in ASTEROID_PERTURBERS
            if str(number) != str(TARGET)
        }
    )
    return {TARGET: target_accelerations}


def observation_model_settings(observation_dataset):
    """Create observation models for every link present in the dataset."""
    settings = []
    corrections = [
        observable_models_setup.light_time_corrections.first_order_relativistic_light_time_correction(
            ["Sun"]
        )
    ]

    model_factories = {
        observable_models_setup.model_settings.angular_position_type: (
            lambda link: observable_models_setup.model_settings.angular_position(
                link,
                bias_settings=None,
            )
        ),
        observable_models_setup.model_settings.n_way_range_type: (
            lambda link: observable_models_setup.model_settings.n_way_range(
                link,
                corrections,
                bias_settings=None,
                # Radar delays are station-clock measurements and must be
                # modelled in UTC rather than with the default TDB time scale.
                time_scale_for_observable=time_representation.utc_scale,
            )
        ),
        observable_models_setup.model_settings.doppler_measured_frequency_type: (
            lambda link: observable_models_setup.model_settings.doppler_measured_frequency(
                link,
                corrections,
                bias_settings=None,
            )
        ),
    }

    for observable_type, factory in model_factories.items():
        link_ids = sorted(
            {
                metadata.link_definition_id
                for metadata in observation_dataset.observation_set_metadata
                if metadata.observable_type == observable_type
            }
        )
        settings.extend(
            factory(observation_dataset.link_definition(link_id)) for link_id in link_ids
        )
    return settings


def perform_estimation(
    tracking_data,
    supplementary_data,
    initial_epoch,
    initial_state,
    first_epoch,
    final_epoch,
    estimate_yarkovsky,
    gaia_astrometry=None,
):
    """Estimate the target state for one data setup."""
    bodies = create_bodies(gaia_astrometry)

    # This installs transmitter-frequency histories, identifies the passive
    # radar reflector, and creates the space-telescope bodies and ephemerides.
    observations.set_tracking_supplementary_data_in_bodies(
        bodies,
        supplementary_data,
    )
    for data in supplementary_data:
        state_data = data.translational_state_supplementary_data
        if not state_data.state_history:
            continue
        # Supplementary receiver states have double-precision epoch keys.
        # Round queries to that same precision: an extended-precision UTC/TDB
        # conversion can otherwise fall a fraction of an ulp below the first
        # key and spuriously report out-of-range interpolation (e.g. C51).
        ephemeris = bodies.get(data.body_name).ephemeris
        bodies.get(data.body_name).ephemeris = environment_setup.create_body_ephemeris(
            environment_setup.ephemeris.custom_ephemeris(
                ephemeris.cartesian_state,
                state_data.frame_origin,
                state_data.frame_orientation,
            ),
            data.body_name,
        )
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies,
        acceleration_settings(estimate_yarkovsky),
        [TARGET],
        [FRAME_ORIGIN],
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
        central_bodies=[FRAME_ORIGIN],
        acceleration_models=acceleration_models,
        bodies_to_integrate=[TARGET],
        initial_states=initial_state,
        initial_time=initial_epoch,
        integrator_settings=integrator_settings,
        termination_settings=termination_settings,
    )

    corrected_gaia = None
    if gaia_astrometry is not None:
        if gaia_astrometry.table.empty:
            raise RuntimeError("Cannot run a Gaia-inclusive fit with zero Gaia observations.")
        # Populate the target ephemeris for the observation corrections using
        # the same nominal dynamics and initial state as the estimation.
        environment_setup.add_empty_tabulated_ephemeris(bodies, TARGET, FRAME_ORIGIN)
        propagator_settings.processing_settings.set_integrated_result = True
        simulator.create_dynamics_simulator(bodies, propagator_settings)
        radius = PHOTOCENTER_RADIUS
        if radius is None:
            radius = 0.5 * SBDBquery(TARGET).diameter
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("The photocenter correction requires a positive radius [m].")
        corrected_gaia = gaia_astrometry.copy()
        corrected_gaia.apply_corrections(
            bodies,
            light_deflection_bodies=("Sun", "Jupiter"),
            photocenter_body_dimensions={int(TARGET): float(radius)},
        )
        gaia_tracking_data, _ = corrected_gaia.to_tracking_data()
        tracking_data = list(tracking_data) + gaia_tracking_data
        print(f"Gaia photocenter correction: spherical radius {radius:.6g} m.")

    observation_dataset = observations.create_observation_dataset_from_tracking_data(
        tracking_data, bodies, apply_corrections=True,
    )
    if corrected_gaia is not None:
        scalar_data = observation_scalar_data(observation_dataset)
        scalar_count = np.count_nonzero(scalar_data["station_labels"] == "Gaia")
        if scalar_count != 2 * len(corrected_gaia.table) or scalar_count == 0:
            raise RuntimeError("The estimation dataset lost Gaia CCD observations.")
        print(f"Estimation dataset contains {scalar_count // 2} Gaia CCD observations.")

    parameter_settings = parameters_setup.initial_states(propagator_settings, bodies)
    if estimate_yarkovsky:
        parameter_settings.append(
            parameters_setup.yarkovsky_parameter(TARGET, "Sun")
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
        convergence_checker=estimation_analysis.estimation_convergence_checker(
            maximum_iterations=NUMBER_OF_ESTIMATION_ITERATIONS,
        ),
    )
    estimation_input.define_estimation_settings(
        reintegrate_variational_equations=True,
        print_output_to_terminal=True,
        save_state_history_per_iteration=True,
        # Retain the finite warning threshold used by the MPC/radar example.
        limit_condition_number_for_warning=1.0e10,
    )
    output = estimator.perform_estimation(estimation_input)
    return output, observation_dataset, estimator, corrected_gaia


def print_yarkovsky_result(output):
    """Print the estimated A2 value and its formal uncertainty."""
    a2_si = float(np.asarray(output.final_parameters)[-1])
    sigma_si = float(np.sqrt(np.asarray(output.covariance)[-1, -1]))
    si_to_au_per_day_squared = constants.JULIAN_DAY**2 / constants.ASTRONOMICAL_UNIT
    print(
        "  Estimated Yarkovsky A2: "
        f"({a2_si:.9g} ± {sigma_si:.3g}) m/s² = "
        f"({a2_si * si_to_au_per_day_squared:.9g} ± "
        f"{sigma_si * si_to_au_per_day_squared:.3g}) au/day²"
    )


def print_residual_summary(label, output, observation_dataset):
    """Print final residual RMS values for each observable family."""
    residuals = np.asarray(output.final_residuals)
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


def best_state_history(output):
    """Return the propagated state history from the estimation's best iteration."""
    return output.simulation_results_per_iteration[
        output.best_iteration
    ].dynamics_results.state_history_float


def print_orbit_difference_rsw(reference_output, comparison_output, estimation_epoch):
    """Print the radar solution minus the optical solution in the optical RSW frame."""
    reference_history = best_state_history(reference_output)
    comparison_history = best_state_history(comparison_output)

    common_epochs = sorted(set(reference_history).intersection(comparison_history))
    if not common_epochs:
        raise RuntimeError("The estimated state histories have no common epochs.")

    epoch = min(common_epochs, key=lambda value: abs(value - estimation_epoch))
    if not np.isclose(epoch, estimation_epoch, rtol=0.0, atol=1.0e-6):
        raise RuntimeError("The estimation epoch is absent from the state histories.")

    def position_difference_rsw(current_epoch):
        reference_state = np.asarray(reference_history[current_epoch])
        comparison_state = np.asarray(comparison_history[current_epoch])
        return inertial_to_rsw_rotation_matrix(reference_state) @ (
            comparison_state[:3] - reference_state[:3]
        )

    difference_at_estimation_epoch = position_difference_rsw(epoch)
    differences = np.array(
        [position_difference_rsw(current_epoch) for current_epoch in common_epochs]
    )
    rms_difference = np.sqrt(np.mean(differences**2, axis=0))

    print(
        f"\n{TARGET} orbit difference "
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
        for label, _ in label_counts.most_common(NUMBER_OF_COLORED_OBSERVATORIES)
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


def gaia_marginal_covariance(table):
    """Return each CCD's random covariance plus its shared transit covariance."""
    covariance = np.zeros((len(table), 2, 2))
    for kind in ("random", "systematic"):
        columns = [f"ra_error_{kind}", f"dec_error_{kind}", f"ra_dec_correlation_{kind}"]
        values = table[columns]
        if kind == "systematic":
            # Match GaiaAstrometry.to_tracking_data: one systematic block,
            # taken from the first CCD, is shared by the entire transit.
            values = table.groupby("transit_id", sort=False)[columns].transform("first")
        sigma_ra, sigma_dec, correlation = values.to_numpy().T
        covariance[:, 0, 0] += sigma_ra**2
        covariance[:, 1, 1] += sigma_dec**2
        covariance[:, 0, 1] += sigma_ra * sigma_dec * correlation
    covariance[:, 1, 0] = covariance[:, 0, 1]
    return covariance


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


def gaia_residual_data(output, observation_dataset, gaia_astrometry):
    """Match Gaia CCD epochs and event/component IDs to estimation residuals."""
    table = gaia_astrometry.table.sort_values("epoch").reset_index(drop=True)
    epochs = table["epoch"].to_numpy()
    if not len(epochs) or np.any(np.diff(epochs) <= 0):
        raise RuntimeError("Gaia residual plotting requires nonempty, unique CCD epochs.")
    data = observation_dataset.get_data(
        observations.observation_query.active,
        fields=("times", "observation_ids", "set_ids", "metadata", "scalar_components"),
        ordering="estimation",
    )
    event_rows = {}
    for epoch, observation_id, set_id in zip(data["times"], data["observation_ids"], data["set_ids"]):
        metadata = data["metadata"][set_id]
        link = metadata["link_definition"].link_ends
        if link[observable_models_setup.links.receiver].body_name != "Gaia":
            continue
        if link[observable_models_setup.links.transmitter].body_name != TARGET:
            raise RuntimeError("The Gaia dataset contains a different target.")
        epoch = float(epoch)
        insertion = np.searchsorted(epochs, epoch)
        candidates = [i for i in (insertion - 1, insertion) if 0 <= i < len(epochs)]
        row = min(candidates, key=lambda i: abs(epochs[i] - epoch))
        if abs(epochs[row] - epoch) > 1.0e-6:
            raise RuntimeError("A Gaia residual epoch does not match the Gaia catalogue.")
        event_rows[observation_id] = row
    if len(event_rows) != len(table) or len(set(event_rows.values())) != len(table):
        raise RuntimeError("Gaia CCD catalogue and active dataset observations differ.")
    scalar_indices = np.full((len(table), 2), -1, dtype=int)
    for index, (observation_id, component) in enumerate(data["scalar_components"]):
        if observation_id in event_rows:
            scalar_indices[event_rows[observation_id], component] = index
    if np.any(scalar_indices < 0):
        raise RuntimeError("A Gaia CCD residual is missing an angular component.")
    residuals = np.asarray(output.final_residuals).reshape(-1)[scalar_indices]
    covariance = gaia_marginal_covariance(table)
    scan_residuals, scan_sigmas = project_gaia_residuals(
        residuals, covariance, table["dec"].to_numpy(), table["position_angle_scan"].to_numpy(),
    )
    return {
        "table": table, "scalar_indices": scalar_indices, "residuals": residuals,
        "covariance": covariance, "scan_residuals": scan_residuals, "scan_sigmas": scan_sigmas,
    }


def plot_gaia_residuals(setup_label, output, observation_dataset, gaia_astrometry):
    """Plot actual Gaia CCD AL/AC residuals in mas and marginal sigma units."""
    data = gaia_residual_data(output, observation_dataset, gaia_astrometry)
    mas_per_radian = 180.0 / np.pi * 3600.0 * 1000.0
    years = 2000.0 + data["table"]["epoch"].to_numpy() / constants.JULIAN_YEAR
    figure, axes = plt.subplots(2, 2, figsize=(11.7, 8.5), sharex=True)
    figure.suptitle(f"{setup_label}: {TARGET}, {len(years)} Gaia CCD observations")
    table = data["table"].copy()
    for component, name in enumerate(("Along-scan (AL)", "Cross-scan (AC)")):
        values = data["scan_residuals"][:, component] * mas_per_radian
        sigmas = data["scan_sigmas"][:, component] * mas_per_radian
        normalized = values / sigmas
        axes[component, 0].errorbar(years, values, yerr=sigmas, fmt=".", markersize=4, alpha=0.55)
        axes[component, 1].scatter(years, normalized, s=9, alpha=0.7)
        for axis in axes[component]:
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.grid(alpha=0.3)
        for threshold in (-3.0, 3.0):
            axes[component, 1].axhline(threshold, color="black", linestyle="--", linewidth=0.8)
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
    table.to_csv(Path(__file__).with_name(Path(__file__).stem + "_gaia_residuals.csv"), index=False)


def plot_residuals(setup_label, output, observation_dataset, gaia_astrometry=None):
    """Plot residuals by observable, separating ground and space astrometry."""
    residuals = np.asarray(output.final_residuals).reshape(-1)
    data = observation_scalar_data(observation_dataset)
    normalized_residuals = residuals * np.sqrt(data["weights"])
    if gaia_astrometry is not None:
        gaia_data = gaia_residual_data(output, observation_dataset, gaia_astrometry)
        sigmas = np.sqrt(np.diagonal(gaia_data["covariance"], axis1=1, axis2=2))
        normalized_residuals[gaia_data["scalar_indices"]] = gaia_data["residuals"] / sigmas
    years = 2000.0 + data["times"] / (365.25 * constants.JULIAN_DAY)
    station_labels = data["station_labels"]
    space_astrometry = data["space_astrometry"]

    angular_type = observable_models_setup.model_settings.angular_position_type
    arcseconds_per_radian = 180.0 / np.pi * 3600.0
    component_settings = {
        angular_type: [
            ("Right ascension", arcseconds_per_radian, "arcsec"),
            ("Declination", arcseconds_per_radian, "arcsec"),
        ],
        observable_models_setup.model_settings.n_way_range_type: [("Range", 1.0, "m")],
        observable_models_setup.model_settings.doppler_measured_frequency_type: [
            ("Doppler", 1.0, "Hz")
        ],
    }

    for observable_type, components in component_settings.items():
        for offset, (component_name, scale, unit) in enumerate(components):
            indices = np.flatnonzero(
                (data["observable_types"] == observable_type) & (data["components"] == offset)
            )
            is_space = (observable_type == angular_type) & space_astrometry[indices]
            ground_indices = indices[~is_space]
            groups = [
                (ground_indices, False, ""),
                (ground_indices, True, "ground-based" if is_space.any() else ""),
            ]
            if is_space.any():
                groups.append((indices[is_space], True, "space-based"))

            for plot_indices, normalized, subset in groups:
                if not plot_indices.size:
                    continue
                values = (
                    normalized_residuals[plot_indices]
                    if normalized else residuals[plot_indices] * scale
                )
                figure, axis = plt.subplots(figsize=(11.7, 7.5))
                add_grouped_scatter(
                    axis, years[plot_indices], values, station_labels[plot_indices]
                )
                axis.axhline(0.0, color="black", linewidth=0.8)
                if normalized:
                    axis.axhline(3.0, color="black", linestyle="--", linewidth=0.8)
                    axis.axhline(-3.0, color="black", linestyle="--", linewidth=0.8)
                axis.grid(alpha=0.3)
                axis.set_xlabel("Year")
                axis.set_ylabel(
                    "Residual / 1σ uncertainty"
                    if normalized
                    else f"Post-fit residual [{unit}]"
                )
                qualifier = f"{subset} " if subset else ""
                kind = "normalized post-fit" if normalized else "post-fit"
                rms_unit = "" if normalized else f" {unit}"
                axis.set_title(
                    f"{setup_label}\n{component_name} {qualifier}{kind} residuals "
                    f"(RMS = {np.sqrt(np.mean(values**2)):.6g}{rms_unit})"
                )
                axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
                figure.tight_layout()


def plot_orbit_difference(setup_label, output, estimator, epochs, horizons_states):
    """Plot the RSW orbit difference with ±3σ bands, and 1σ formal errors alone."""
    state_history = best_state_history(output)
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
            np.asarray(state_history[epoch])[:3] - horizons_state[:3]
        )
        position_covariance = np.asarray(propagated_covariances[epoch])[:3, :3]
        covariance_rsw = rotation @ position_covariance @ rotation.T
        formal_errors_rsw[index] = np.sqrt(
            np.clip(np.diag(covariance_rsw), 0.0, None)
        )

    differences_rsw /= 1000.0
    formal_errors_rsw /= 1000.0
    years = 2000.0 + np.asarray(epochs) / (365.25 * constants.JULIAN_DAY)

    for formal_only in (False, True):
        figure, axes = plt.subplots(3, 1, figsize=(11.7, 10.5), sharex=True)
        for component, (axis, component_name) in enumerate(zip(axes, "RSW")):
            sigma = formal_errors_rsw[:, component]
            if formal_only:
                axis.plot(years, sigma, color="tab:orange", label=f"σ{component_name}")
                axis.set_ylim(bottom=0.0)
            else:
                difference = differences_rsw[:, component]
                axis.plot(years, difference, color="tab:blue", label="Estimate − Horizons")
                axis.fill_between(
                    years, difference - 3.0 * sigma, difference + 3.0 * sigma,
                    color="tab:blue", alpha=0.22, label="±3σ formal error",
                )
                axis.axhline(0.0, color="black", linewidth=0.8)
            axis.grid(alpha=0.3)
            axis.set_ylabel(f"{'σ' if formal_only else ''}{component_name} [km]")
            axis.legend(loc="best")
        axes[-1].set_xlabel("Year")
        title = (
            "Propagated formal position errors in heliocentric RSW"
            if formal_only else f"{TARGET} orbit difference with respect to JPL Horizons"
        )
        figure.suptitle(f"{setup_label}\n{title}")
        figure.tight_layout()


def plot_diagnostics(results):
    """Display each estimation setup's diagnostics as Matplotlib figures."""
    histories = [best_state_history(result[0]) for result in results.values()]
    common_epochs = sorted(set.intersection(*(set(history) for history in histories)))
    if not common_epochs:
        raise RuntimeError("The estimated state histories have no common plotting epochs.")
    number_of_epochs = min(NUMBER_OF_HORIZONS_PLOT_EPOCHS, len(common_epochs))
    epoch_indices = np.unique(
        np.linspace(0, len(common_epochs) - 1, number_of_epochs, dtype=int)
    )
    plot_epochs = np.array([common_epochs[index] for index in epoch_indices])
    horizons_states = HorizonsQuery(
        query_id=HORIZONS_TARGET,
        location=HORIZONS_ORIGIN,
        epoch_list=list(plot_epochs),
        extended_query=True,
    ).cartesian(frame_orientation=FRAME_ORIENTATION)[:, 1:]

    for setup_label, (output, observation_dataset, estimator, gaia_astrometry) in results.items():
        plot_residuals(setup_label, output, observation_dataset, gaia_astrometry)
        if gaia_astrometry is not None:
            plot_gaia_residuals(setup_label, output, observation_dataset, gaia_astrometry)
        plot_orbit_difference(
            setup_label, output, estimator, plot_epochs, horizons_states
        )
    if plt.get_backend().lower() == "agg":
        plot_file = Path(__file__).with_suffix(".pdf")
        with PdfPages(plot_file) as pdf:
            for number in plt.get_fignums():
                pdf.savefig(plt.figure(number))
        plt.close("all")
        print(f"Saved diagnostics to {plot_file}")
    else:
        plt.show()

#####################################################################
#################   MAIN LOOP  ######################################
#####################################################################

def main(estimate_yarkovsky):
    """Compare MPC and MPC/Gaia estimation, including any available radar."""
    spice.load_standard_kernels()
    # Fail before running any baseline fits if no actual Gaia data exist.
    gaia_astrometry = load_gaia_astrometry()
    print(
        "Estimation series: "
        + ("estimating Yarkovsky A2" if estimate_yarkovsky else "Yarkovsky disabled")
    )
    print(
        "Asteroid point-mass perturbers: "
        + ", ".join(
            f"{number} {name}"
            for number, name in ASTEROID_PERTURBERS
            if str(number) != str(TARGET)
        )
    )
    (
        optical_tracking_data,
        optical_supplementary_data,
        radar_tracking_data,
        radar_supplementary_data,
    ) = load_tracking_data()

    observation_start_epoch, observation_end_epoch = observation_epoch_bounds(
        optical_tracking_data + radar_tracking_data
    )
    # GaiaAstrometry epochs are already TDB; do not convert them from UTC again.
    observation_start_epoch = min(observation_start_epoch, gaia_astrometry.table.epoch.min())
    observation_end_epoch = max(observation_end_epoch, gaia_astrometry.table.epoch.max())
    initial_epoch = max(
        0.0,
        0.5 * (observation_start_epoch + observation_end_epoch),
    )
    first_epoch = observation_start_epoch - PROPAGATION_BUFFER
    final_epoch = observation_end_epoch + PROPAGATION_BUFFER

    initial_state = HorizonsQuery(
        query_id=HORIZONS_TARGET,
        location=HORIZONS_ORIGIN,
        epoch_list=[float(initial_epoch)],
        extended_query=True,
    ).cartesian(frame_orientation=FRAME_ORIENTATION)[0, 1:]

    setups = {
        "MPC astrometry": (
            optical_tracking_data,
            optical_supplementary_data,
            None,
        ),
    }
    if radar_tracking_data:
        setups["MPC astrometry and JPL radar"] = (
            optical_tracking_data + radar_tracking_data,
            optical_supplementary_data + radar_supplementary_data,
            None,
        )
    gaia_label = (
        "MPC astrometry, JPL radar and Gaia"
        if radar_tracking_data else "MPC astrometry and Gaia"
    )
    setups[gaia_label] = (
        optical_tracking_data + radar_tracking_data,
        optical_supplementary_data + radar_supplementary_data,
        gaia_astrometry,
    )

    results = {}
    for label, (tracking_data, supplementary_data, gaia) in setups.items():
        output, observation_dataset, estimator, corrected_gaia = perform_estimation(
            tracking_data,
            supplementary_data,
            initial_epoch,
            initial_state,
            first_epoch,
            final_epoch,
            estimate_yarkovsky,
            gaia,
        )
        print_residual_summary(label, output, observation_dataset)
        if estimate_yarkovsky:
            print_yarkovsky_result(output)
        results[label] = (output, observation_dataset, estimator, corrected_gaia)

    if "MPC astrometry and JPL radar" in results:
        print_orbit_difference_rsw(
            results["MPC astrometry"][0],
            results["MPC astrometry and JPL radar"][0],
            initial_epoch,
        )
    plot_diagnostics(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit MPC and Gaia astrometry of 673 Edda.")
    parser.add_argument(
        "--gaia-archive", type=Path, default=GAIA_ARCHIVE_PATH,
        help="Use a local Gaia FPR parquet archive instead of querying the ESA archive.",
    )
    GAIA_ARCHIVE_PATH = parser.parse_args().gaia_archive
    main(ESTIMATE_YARKOVSKY)
