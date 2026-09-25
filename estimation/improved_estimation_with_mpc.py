"""
# MPC space astrometry and JPL radar estimation

Copyright (c) 2010-2026, Delft University of Technology. All rights reserved.
This file is part of Tudat. Redistribution and use in source and binary forms,
with or without modification, are permitted exclusively under the terms of the
Modified BSD license. See https://tudat.tudelft.nl/LICENSE.

This example estimates the state of a configured minor planet twice: first from
MPC optical astrometry, and then from the same astrometry with JPL radar delay
and Doppler observations added.

It is an updated, compact version of the earlier
``improved_estimation_with_mpc.py`` application.  The observation-query and
residual-filtering machinery used by the old PR #905 version has deliberately
been left out.  The example follows the current data path directly:

1. MPC and JPL records are converted to ``TrackingData``.
2. Their supplementary data are applied to the system of bodies.
3. The tracking data are converted to an ``ObservationCollection``.
4. One orbit determination is run without radar and one with radar.

The MPC and JPL queries require an internet connection.
"""

import datetime
from collections import Counter
import json

import numpy as np
import matplotlib.pyplot as plt

from tudatpy import constants
from tudatpy.astro.frame_conversion import inertial_to_rsw_rotation_matrix
from tudatpy.astro import time_representation
from tudatpy.data_input.environment_data import spice
from tudatpy.data_input.environment_data.horizons import HorizonsQuery
from tudatpy.data_input.tracking_data.jpl_radar import JPLRadarQuery
from tudatpy.data_input.tracking_data.mpc import BatchMPC
from tudatpy.data_input.tracking_data.optical_utilities import (
    SPACECRAFT_POSITION_COLUMNS,
    optical_table_to_tracking_data,
)
from tudatpy.data_input.tracking_data.radar_utilities import (
    radar_data_to_tracking_data,
)
from tudatpy.dynamics import environment_setup, parameters_setup, propagation_setup
from tudatpy.estimation import estimation_analysis, observable_models_setup, observations


TARGET = "101955"
HORIZONS_TARGET = f"{TARGET};"
FRAME_ORIGIN = "Sun"
HORIZONS_ORIGIN = "500@10"
FRAME_ORIENTATION = "J2000"

OBSERVATION_START = datetime.datetime(1990, 1, 1)
OBSERVATION_END = datetime.datetime(2026, 7, 1)
PROPAGATION_BUFFER = 2.0 * 31.0 * constants.JULIAN_DAY
INTEGRATOR_STEP = 12.0 * 3600.0
NUMBER_OF_ESTIMATION_ITERATIONS = 4
ESTIMATE_YARKOVSKY = True
NUMBER_OF_COLORED_OBSERVATORIES = 10
NUMBER_OF_HORIZONS_PLOT_EPOCHS = 300

YARKOVSKY_INITIAL_A2 = 0.0

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
        observatories_exclude=["C57"],
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


def create_bodies():
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


def observation_model_settings(observation_collection):
    """Create observation models for every link present in the collection."""
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
        links = observation_collection.get_link_definitions_for_observables(
            observable_type=observable_type
        )
        settings.extend(factory(link) for link in links)
    return settings


def perform_estimation(
    tracking_data,
    supplementary_data,
    initial_epoch,
    initial_state,
    first_epoch,
    final_epoch,
    estimate_yarkovsky,
):
    """Estimate the target state for one data setup."""
    bodies = create_bodies()

    # This installs transmitter-frequency histories, identifies the passive
    # radar reflector, and creates the space-telescope bodies and ephemerides.
    observations.set_tracking_supplementary_data_in_bodies(
        bodies,
        supplementary_data,
    )
    observation_collection = (
        observations.create_observation_collection_from_tracking_data(
            tracking_data,
            bodies,
            apply_corrections=True,
        )
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
        observation_settings=observation_model_settings(observation_collection),
        propagator_settings=propagator_settings,
        integrate_on_creation=True,
    )
    estimation_input = estimation_analysis.EstimationInput(
        observations_and_times=observation_collection,
        convergence_checker=estimation_analysis.estimation_convergence_checker(
            maximum_iterations=NUMBER_OF_ESTIMATION_ITERATIONS,
        ),
    )
    estimation_input.define_estimation_settings(
        reintegrate_variational_equations=True,
        print_output_to_terminal=True,
        save_state_history_per_iteration=True,
    )
    output = estimator.perform_estimation(estimation_input)
    return output, observation_collection, estimator


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


def print_residual_summary(label, output, observation_collection):
    """Print final residual RMS values for each observable family."""
    residuals = np.asarray(output.final_residuals)
    observable_units = {
        observable_models_setup.model_settings.angular_position_type: "rad",
        observable_models_setup.model_settings.n_way_range_type: "m",
        observable_models_setup.model_settings.doppler_measured_frequency_type: "Hz",
    }
    print(f"\n{label}")
    for observable_type, (start, size) in (
        observation_collection.observable_type_start_index_and_size.items()
    ):
        values = residuals[start : start + size]
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


def observation_labels_and_space_mask(observation_collection):
    """Return observer labels and a space-receiver mask for scalar observations."""
    link_data = {}
    for link_id, link_ends in observation_collection.link_definition_ids.items():
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
        link_data[link_id] = (
            " → ".join(dict.fromkeys(stations)) or receiver.body_name,
            receiver.body_name != "Earth",
        )
    rows = np.array(
        [link_data[int(link_id)] for link_id in observation_collection.concatenated_link_definition_ids],
        dtype=object,
    )
    return rows[:, 0], rows[:, 1].astype(bool)


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


def plot_residuals(setup_label, output, observation_collection):
    """Plot residuals by observable, separating ground and space astrometry."""
    residuals = np.asarray(output.final_residuals).reshape(-1)
    weights = np.asarray(observation_collection.concatenated_weights).reshape(-1)
    normalized_residuals = residuals * np.sqrt(weights)
    times = np.asarray(observation_collection.concatenated_times, dtype=float)
    years = 2000.0 + times / (365.25 * constants.JULIAN_DAY)
    station_labels, space_astrometry = observation_labels_and_space_mask(
        observation_collection
    )

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

    observable_slices = observation_collection.observable_type_start_index_and_size
    for observable_type, components in component_settings.items():
        if observable_type not in observable_slices:
            continue
        start, size = observable_slices[observable_type]
        for offset, (component_name, scale, unit) in enumerate(components):
            indices = np.arange(start + offset, start + size, len(components))
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

    for setup_label, (output, observation_collection, estimator) in results.items():
        plot_residuals(setup_label, output, observation_collection)
        plot_orbit_difference(
            setup_label, output, estimator, plot_epochs, horizons_states
        )
    plt.show()

#####################################################################
#################   MAIN LOOP  ######################################
#####################################################################

def main(estimate_yarkovsky):
    """Run astrometry-only and, when available, astrometry-plus-radar estimation."""
    spice.load_standard_kernels()
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
        ),
    }
    if radar_tracking_data:
        setups["MPC astrometry and JPL radar"] = (
            optical_tracking_data + radar_tracking_data,
            optical_supplementary_data + radar_supplementary_data,
        )

    results = {}
    for label, (tracking_data, supplementary_data) in setups.items():
        output, observation_collection, estimator = perform_estimation(
            tracking_data,
            supplementary_data,
            initial_epoch,
            initial_state,
            first_epoch,
            final_epoch,
            estimate_yarkovsky,
        )
        print_residual_summary(label, output, observation_collection)
        if estimate_yarkovsky:
            print_yarkovsky_result(output)
        results[label] = (output, observation_collection, estimator)

    if "MPC astrometry and JPL radar" in results:
        print_orbit_difference_rsw(
            results["MPC astrometry"][0],
            results["MPC astrometry and JPL radar"][0],
            initial_epoch,
        )
    plot_diagnostics(results)


if __name__ == "__main__":
    main(ESTIMATE_YARKOVSKY)
