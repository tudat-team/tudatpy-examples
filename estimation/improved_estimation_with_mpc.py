"""
# MPC space astrometry and JPL radar estimation

This example estimates the state of a configured minor planet twice: first from
MPC optical astrometry, and then from the same astrometry with JPL radar delay
and Doppler observations added (if any are available).

The example consists of the following steps:

1. MPC and JPL data are loaded converted to ``TrackingData`` objects.
2. Their supplementary data are applied to the environment
3. The tracking data are converted to an ``ObservationCollection``.
4. One orbit determination is run without radar and one with radar.

The MPC and JPL queries require an internet connection. Note that this script does not use any outlier rejection
"""

import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from tudatpy import constants
from tudatpy.astro.frame_conversion import inertial_to_rsw_rotation_matrix
from tudatpy.astro import time_representation
from tudatpy.data_input.environment_data import spice
from tudatpy.data_input.environment_data.horizons import HorizonsQuery
from tudatpy.data_input.tracking_data import TrackingData, TrackingSupplementaryData
from tudatpy.data_input.tracking_data.jpl_radar import read_jpl_radar_data
from tudatpy.data_input.tracking_data.mpc import BatchMPC
from tudatpy.dynamics import environment, environment_setup, parameters_setup, propagation_setup
from tudatpy.estimation import estimation_analysis, observable_models_setup, observations
from tudatpy.estimation.observable_models_setup import model_settings


"""
Target body number, time interval over which to extract data, and choice of estimating Yarkovsky parameter
"""
TARGET = "101955"
OBSERVATION_START = datetime.datetime(1990, 1, 1)
OBSERVATION_END = datetime.datetime(2024, 1, 1)
ESTIMATE_YARKOVSKY = True

"""
Definitions of frames that are used in propagation and observation modelling
"""
GLOBAL_FRAME_ORIGIN = "SSB"
GLOBAL_FRAME_ORIENTATION = "J2000"
PROPAGATION_CENTRAL_BODY = "Sun"

"""
Definitions of frames used in JPL Horizons
"""
HORIZONS_TARGET = f"{TARGET};"
HORIZONS_ORIGIN = "500@10"

"""
Numerical model settings
"""
PROPAGATION_BUFFER = 2.0 * 31.0 * constants.JULIAN_DAY
INTEGRATOR_STEP = 12.0 * 3600.0

"""
Plotting settings
"""
NUMBER_OF_COLORED_OBSERVATORIES = 10
NUMBER_OF_HORIZONS_PLOT_EPOCHS = 300
ASTROMETRY_SETUP_NAME = "MPC astrometry"
RADAR_SETUP_NAME = "MPC astrometry and JPL radar"

"""
List of names and numbers of asteroids used in analysis as perturbers
"""
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


"""
In the function below, the environment is created. This defines the properties of the bodies used in the simulations.
These bodies include the most massive asteroids, whose perturbations are taken into account. For the planets, we do not
use the masses/ephemerides of the planets and moons separately (with the exception of the Earth and its Moon), instead,
we place the planet at the center of mass of its own system, and give it the total system mass
"""


def create_bodies() -> environment.SystemOfBodies:
    """Create the estimation environment and Earth observing stations.

    Returns
    -------
    environment.SystemOfBodies
        Bodies used in the estimation, including the target and asteroid perturbers.
    """
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
        GLOBAL_FRAME_ORIENTATION,
    )

    # Move each planet from Mars onwards to the barycenter of its own planetary system, and give it the
    # full system's mass
    for body_name in ["Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]:
        barycenter_name = f"{body_name} Barycenter"
        settings = body_settings.get(body_name)
        settings.ephemeris_settings = environment_setup.ephemeris.direct_spice(
            GLOBAL_FRAME_ORIGIN,
            GLOBAL_FRAME_ORIENTATION,
            barycenter_name,
        )
        settings.gravity_field_settings = (
            environment_setup.gravity_field.central_spice(barycenter_name)
        )

    # Modify Earth settings to high-quality shape and rotation model
    earth_settings = body_settings.get("Earth")
    earth_settings.shape_settings = environment_setup.shape.oblate_spherical(
        6378137.0,
        1.0 / 298.257223563,
    )
    earth_settings.rotation_model_settings = (
        environment_setup.rotation_model.gcrs_to_itrs(
            environment_setup.rotation_model.iau_2006,
            GLOBAL_FRAME_ORIENTATION,
        )
    )
    earth_settings.gravity_field_settings.associated_reference_frame = "ITRS"
    
    # Add optical telescopes and radar stations to Earth
    earth_settings.ground_station_settings = (
        environment_setup.ground_station.optical_telescope_stations()
    )
    
    # Add estimated asteroid to environment
    body_settings.add_empty_settings(TARGET)

    # Add each asteroid perturber to the environment with mass and ephemeris from spice
    for number, name in ASTEROID_PERTURBERS:
        body_name = f"{number} {name}"
        spice_id = f"200{number:04d}"  # NAIF identifier used by the standard 300-asteroid SPICE kernel
        body_settings.add_empty_settings(body_name)
        settings = body_settings.get(body_name)
        settings.ephemeris_settings = environment_setup.ephemeris.direct_spice(
            GLOBAL_FRAME_ORIGIN,
            GLOBAL_FRAME_ORIENTATION,
            spice_id,
        )
        settings.gravity_field_settings = environment_setup.gravity_field.central_spice(
            spice_id
        )

    return environment_setup.create_system_of_bodies(body_settings)


"""
Below we define the set of accelerations acting on our asteroid. We include the planets', Earth's moon's,
and the selected asteroid perturbers' point mass attractions, as well as the Sun's relativistic correction and,
if selected, the Yarkovsky acceleration.
"""

def acceleration_settings() -> dict[str, dict[str, list[propagation_setup.acceleration.AccelerationSettings]]]:
    """Return the force model of the target.

    Returns
    -------
    dict[str, dict[str, list[AccelerationSettings]]]
        Acceleration settings per exerting body, for the target body.
    """
    sun_accelerations = [
        propagation_setup.acceleration.point_mass_gravity(),
        propagation_setup.acceleration.relativistic_correction(
            use_schwarzschild=True
        ),
    ]
    if ESTIMATE_YARKOVSKY:
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
    target_accelerations.update(
        {
            f"{number} {name}": [
                propagation_setup.acceleration.point_mass_gravity()
            ]
            for number, name in ASTEROID_PERTURBERS
            if str(number) != TARGET
        }
    )
    return {TARGET: target_accelerations}

"""
The available data is loaded from teh MPC and JPL (for radar data). The data and any associated data
such as radar transmission frequencies is returned to be use in the estimation
"""


def load_tracking_data() -> tuple[
    list[TrackingData],
    list[TrackingSupplementaryData],
    list[TrackingData],
    list[TrackingSupplementaryData],
]:
    """Retrieve the MPC astrometry and JPL radar observations of the target.

    Returns
    -------
    tuple[list[TrackingData], list[TrackingSupplementaryData], list[TrackingData], list[TrackingSupplementaryData]]
        Optical tracking data, optical supplementary data, radar tracking data
        and radar supplementary data. The radar lists are empty if JPL has no
        radar observations in the selected interval.
    """
    batch = BatchMPC()
    batch.get_observations([TARGET], use_mpc80_format=True)
    # MPC 80-column records also contain radar observations. Listing only the
    # optical types removes these, as the radar data are retrieved from JPL below.
    batch.filter(
        observation_types=batch.table["note2"].unique().tolist(),
        epoch_start=OBSERVATION_START,
        epoch_end=OBSERVATION_END,
        observatories_exclude=["C57"],
    )
    if batch.table.empty:
        raise RuntimeError("The selected interval contains no MPC astrometry.")
    print(f"Loaded {len(batch.table)} MPC optical observations.")

    optical_tracking_data, optical_supplementary_data = batch.to_tracking_dataset(
        add_weights=True,
        add_star_catalog_corrections=True,
        add_ancillary_data=True,
    )

    radar_tracking_data, radar_supplementary_data = read_jpl_radar_data(
        TARGET,
        epoch_start=OBSERVATION_START,
        epoch_end=OBSERVATION_END,
        timeout=60.0,
    )
    if not radar_tracking_data:
        print("No JPL radar observations found; skipping the radar-inclusive run.")

    return (
        optical_tracking_data,
        optical_supplementary_data,
        radar_tracking_data,
        radar_supplementary_data,
    )


"""
From the data loaded and stored in the ObservationCollection object (observed data), we can
define the settings that are needed to create the observation models, from which the computed data
are generated during the estimation.
"""


def observation_model_settings(
    observation_collection: observations.ObservationCollection,
) -> list[model_settings.ObservationModelSettings]:
    """Create observation models for every link present in the collection.

    Parameters
    ----------
    observation_collection : observations.ObservationCollection
        Observations for which models are created.

    Returns
    -------
    list[ObservationModelSettings]
        Settings of one observation model per link and observable.
    """
    
    
    # Retrieve sets of link ends per observation type
    links = observation_collection.get_link_definitions_for_observables
    
    # Use a simple relativistic correction to the light time for each observation model
    corrections = [
        observable_models_setup.light_time_corrections.first_order_relativistic_light_time_correction(
            ["Sun"]
        )
    ]

    # Create astrometry (angular position) model settings
    settings = [
        model_settings.angular_position(link)
        for link in links(model_settings.angular_position_type)
    ]

    # Create radar range model settings, where the time interval converted to range is to be in UTC
    settings += [
        model_settings.n_way_range(
            link,
            corrections,
            time_scale_for_observable=time_representation.utc_scale,
        )
        for link in links(model_settings.n_way_range_type)
    ]
    
    # Create radar Doppler model settings
    settings += [
        model_settings.doppler_measured_frequency(link, corrections)
        for link in links(model_settings.doppler_measured_frequency_type)
    ]
    return settings


"""
Now, we can put everything together, and fit the data to the dynamical model estimating the
parameters we want. The output is the converged values of the parameters with the associated residuals
and parameter covariance
"""


def perform_estimation(
    tracking_data: list[TrackingData],
    supplementary_data: list[TrackingSupplementaryData],
    initial_epoch: float,
    initial_state: np.ndarray,
    first_epoch: float,
    final_epoch: float,
) -> tuple[
    estimation_analysis.EstimationOutput,
    observations.ObservationCollection,
    estimation_analysis.Estimator,
]:
    """Estimate the target state for one data setup.

    Parameters
    ----------
    tracking_data : list[TrackingData]
        Observations to fit.
    supplementary_data : list[TrackingSupplementaryData]
        Supplementary data belonging to the observations.
    initial_epoch : float
        Epoch of the estimated state [s since J2000, TDB].
    initial_state : numpy.ndarray
        A priori Cartesian state of the target at the initial epoch [m, m/s].
    first_epoch : float
        Start of the propagation interval [s since J2000, TDB].
    final_epoch : float
        End of the propagation interval [s since J2000, TDB].

    Returns
    -------
    tuple[EstimationOutput, ObservationCollection, Estimator]
        Estimation output, observations used and estimator.
    """
    
    # Create environment
    bodies = create_bodies()

    # Create observation_collection object from tracking data; update environment
    # with relevant information from tracking data source (radar transmitter frequencies, space observatory ephemerides, etc.)
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

    # Create settings for dynamics model
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies,
        acceleration_settings(),
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
        initial_time=initial_epoch,
        integrator_settings=integrator_settings,
        termination_settings=termination_settings,
    )

    # Define parameters to estimate: initial asteroid state and (if requested) Yarkovsky parameter
    parameter_settings = parameters_setup.initial_states(propagator_settings, bodies)
    if ESTIMATE_YARKOVSKY:
        parameter_settings.append(
            parameters_setup.yarkovsky_parameter(TARGET, "Sun")
        )
    parameters_to_estimate = parameters_setup.create_parameter_set(
        parameter_settings,
        bodies,
        propagator_settings,
    )

    # Create estimator and perform estimation
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
            maximum_iterations=5,
        ),
    )
    estimation_input.define_estimation_settings(
        reintegrate_variational_equations=True,
        print_output_to_terminal=True,
        save_state_history_per_iteration=True,
    )
    output = estimator.perform_estimation(estimation_input)
    return output, observation_collection, estimator


"""
The code below generates the terminal output and figures. The residuals are plotted both
as their values, for right ascension and declination in arcseconds, for radar (if any) in meters and Hz
Residuals are also plotted normalized by their uncertainty (as taken from the weight matrix)
The post-fit orbit difference w.r.t. Horizons is also shown, along with the formal uncertainty from our estimation 
"""


def observer_labels(observation_collection: observations.ObservationCollection) -> np.ndarray:
    """Return the name of the receiving observer of each scalar observation.

    Parameters
    ----------
    observation_collection : observations.ObservationCollection
        Observations used in the estimation.

    Returns
    -------
    numpy.ndarray
        Station name, or body name for a space telescope, per scalar observation.
    """
    labels = {}
    for link_id, link_ends in observation_collection.link_definition_ids.items():
        receiver = link_ends[observable_models_setup.links.receiver]
        labels[link_id] = receiver.reference_point or receiver.body_name
    return np.array(
        [labels[int(link_id)] for link_id in observation_collection.concatenated_link_definition_ids]
    )


def add_grouped_scatter(
    axis: Axes, years: np.ndarray, values: np.ndarray, labels: np.ndarray
) -> None:
    """Plot values per observer for the busiest observers, grouping the remainder.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Axis to plot on.
    years : numpy.ndarray
        Observation times [yr].
    values : numpy.ndarray
        Value per observation.
    labels : numpy.ndarray
        Observer name per observation.
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    top_labels = unique_labels[np.argsort(counts)[::-1][:NUMBER_OF_COLORED_OBSERVATORIES]]
    other = ~np.isin(labels, top_labels)
    if other.any():
        axis.scatter(years[other], values[other], s=9, color="lightgrey", label="Other")
    for label in top_labels:
        axis.scatter(years[labels == label], values[labels == label], s=11, label=label)


def plot_residuals(
    setup_label: str,
    output: estimation_analysis.EstimationOutput,
    observation_collection: observations.ObservationCollection,
) -> None:
    """Plot the post-fit and normalized residuals of each observable component.

    Parameters
    ----------
    setup_label : str
        Name of the data setup.
    output : estimation_analysis.EstimationOutput
        Estimation output.
    observation_collection : observations.ObservationCollection
        Observations used in the estimation.
    """
    residuals = np.asarray(output.final_residuals).reshape(-1)
    weights = np.asarray(observation_collection.concatenated_weights).reshape(-1)
    normalized_residuals = residuals * np.sqrt(weights)
    times = np.asarray(observation_collection.concatenated_times, dtype=float)
    years = 2000.0 + times / constants.JULIAN_YEAR
    labels = observer_labels(observation_collection)

    arcseconds_per_radian = 180.0 / np.pi * 3600.0
    component_settings = {
        model_settings.angular_position_type: [
            ("Right ascension", arcseconds_per_radian, "arcsec"),
            ("Declination", arcseconds_per_radian, "arcsec"),
        ],
        model_settings.n_way_range_type: [("Range", 1.0, "m")],
        model_settings.doppler_measured_frequency_type: [("Doppler", 1.0, "Hz")],
    }

    for observable_type, (start, size) in (
        observation_collection.observable_type_start_index_and_size.items()
    ):
        components = component_settings[observable_type]
        for offset, (component_name, scale, unit) in enumerate(components):
            indices = np.arange(start + offset, start + size, len(components))
            scaled_residuals = residuals[indices] * scale
            figure, axes = plt.subplots(2, 1, figsize=(11.7, 7.5), sharex=True)
            for axis, values, y_label in (
                (axes[0], scaled_residuals, f"Post-fit residual [{unit}]"),
                (axes[1], normalized_residuals[indices], "Residual / 1σ uncertainty"),
            ):
                add_grouped_scatter(axis, years[indices], values, labels[indices])
                axis.axhline(0.0, color="black", linewidth=0.8)
                axis.grid(alpha=0.3)
                axis.set_ylabel(y_label)
            axes[1].axhline(3.0, color="black", linestyle="--", linewidth=0.8)
            axes[1].axhline(-3.0, color="black", linestyle="--", linewidth=0.8)
            axes[1].set_xlabel("Year")
            axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
            figure.suptitle(
                f"{setup_label}\n{component_name} post-fit residuals "
                f"(RMS = {np.sqrt(np.mean(scaled_residuals**2)):.6g} {unit})"
            )
            figure.tight_layout()


def best_state_history(
    output: estimation_analysis.EstimationOutput,
) -> dict[float, np.ndarray]:
    """Return the propagated state history of the best estimation iteration.

    Parameters
    ----------
    output : estimation_analysis.EstimationOutput
        Estimation output.

    Returns
    -------
    dict[float, numpy.ndarray]
        Cartesian state [m, m/s] per epoch [s since J2000, TDB].
    """
    return output.simulation_results_per_iteration[
        output.best_iteration
    ].dynamics_results.state_history_float


def plot_orbit_difference(
    setup_label: str,
    output: estimation_analysis.EstimationOutput,
    estimator: estimation_analysis.Estimator,
    epochs: np.ndarray,
    horizons_states: np.ndarray,
) -> None:
    """Plot the RSW difference with Horizons with ±3σ bands, and the 1σ formal errors.

    Parameters
    ----------
    setup_label : str
        Name of the data setup.
    output : estimation_analysis.EstimationOutput
        Estimation output.
    estimator : estimation_analysis.Estimator
        Estimator that produced the output.
    epochs : numpy.ndarray
        Plot epochs [s since J2000, TDB].
    horizons_states : numpy.ndarray
        Horizons Cartesian state per plot epoch [m, m/s].
    """
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
        position_covariance = np.asarray(propagated_covariances[epoch])[:3, :3]
        differences_rsw[index] = rotation @ (
            np.asarray(state_history[epoch])[:3] - horizons_state[:3]
        )
        formal_errors_rsw[index] = np.sqrt(
            np.diag(rotation @ position_covariance @ rotation.T)
        )
    differences_rsw /= 1000.0
    formal_errors_rsw /= 1000.0
    years = 2000.0 + epochs / constants.JULIAN_YEAR

    figure, axes = plt.subplots(3, 2, figsize=(14.0, 10.5), sharex=True)
    for component, component_name in enumerate("RSW"):
        difference = differences_rsw[:, component]
        sigma = formal_errors_rsw[:, component]
        axes[component, 0].plot(years, difference, color="tab:blue", label="Estimate − Horizons")
        axes[component, 0].fill_between(
            years, difference - 3.0 * sigma, difference + 3.0 * sigma,
            color="tab:blue", alpha=0.22, label="±3σ formal error",
        )
        axes[component, 0].set_ylabel(f"{component_name} [km]")
        axes[component, 1].plot(years, sigma, color="tab:orange", label="1σ formal error")
        axes[component, 1].set_ylabel(f"σ{component_name} [km]")
    for axis in axes.flat:
        axis.grid(alpha=0.3)
        axis.legend(loc="best")
    for axis in axes[-1]:
        axis.set_xlabel("Year")
    figure.suptitle(
        f"{setup_label}\n{TARGET} orbit difference with respect to JPL Horizons, heliocentric RSW"
    )
    figure.tight_layout()


def print_orbit_difference_rsw(
    reference_output: estimation_analysis.EstimationOutput,
    comparison_output: estimation_analysis.EstimationOutput,
    estimation_epoch: float,
) -> None:
    """Print the radar solution minus the optical solution in the optical RSW frame.

    Parameters
    ----------
    reference_output : estimation_analysis.EstimationOutput
        Estimation output of the optical-only setup.
    comparison_output : estimation_analysis.EstimationOutput
        Estimation output of the setup including radar.
    estimation_epoch : float
        Epoch of the estimated state [s since J2000, TDB].
    """
    reference_history = best_state_history(reference_output)
    comparison_history = best_state_history(comparison_output)

    common_epochs = sorted(set(reference_history).intersection(comparison_history))
    if not common_epochs:
        raise RuntimeError("The estimated state histories have no common epochs.")

    epoch = min(common_epochs, key=lambda value: abs(value - estimation_epoch))
    if not np.isclose(epoch, estimation_epoch, rtol=0.0, atol=1.0e-6):
        raise RuntimeError("The estimation epoch is absent from the state histories.")

    differences = np.array(
        [
            inertial_to_rsw_rotation_matrix(np.asarray(reference_history[current_epoch]))
            @ (
                np.asarray(comparison_history[current_epoch])[:3]
                - np.asarray(reference_history[current_epoch])[:3]
            )
            for current_epoch in common_epochs
        ]
    )
    difference_at_estimation_epoch = differences[common_epochs.index(epoch)]
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


def plot_diagnostics(
    results: dict[
        str,
        tuple[
            estimation_analysis.EstimationOutput,
            observations.ObservationCollection,
            estimation_analysis.Estimator,
        ],
    ],
    estimation_epoch: float,
) -> None:
    """Show the figures of each data setup, and print the effect of adding radar.

    Parameters
    ----------
    results : dict[str, tuple[EstimationOutput, ObservationCollection, Estimator]]
        Estimation output, observations and estimator per data setup.
    estimation_epoch : float
        Epoch of the estimated state [s since J2000, TDB].
    """
    if RADAR_SETUP_NAME in results:
        print_orbit_difference_rsw(
            results[ASTROMETRY_SETUP_NAME][0],
            results[RADAR_SETUP_NAME][0],
            estimation_epoch,
        )

    histories = [best_state_history(result[0]) for result in results.values()]
    common_epochs = np.array(sorted(set.intersection(*(set(history) for history in histories))))
    number_of_epochs = min(NUMBER_OF_HORIZONS_PLOT_EPOCHS, len(common_epochs))
    plot_epochs = common_epochs[np.linspace(0, len(common_epochs) - 1, number_of_epochs, dtype=int)]
    horizons_states = HorizonsQuery(
        query_id=HORIZONS_TARGET,
        location=HORIZONS_ORIGIN,
        epoch_list=list(plot_epochs),
        extended_query=True,
    ).cartesian(frame_orientation=GLOBAL_FRAME_ORIENTATION)[:, 1:]

    for setup_label, (output, observation_collection, estimator) in results.items():
        plot_residuals(setup_label, output, observation_collection)
        plot_orbit_difference(setup_label, output, estimator, plot_epochs, horizons_states)
    plt.show()



def print_residual_summary(
    label: str,
    output: estimation_analysis.EstimationOutput,
    observation_collection: observations.ObservationCollection,
) -> None:
    """Print the final residual RMS for each observable type.

    Parameters
    ----------
    label : str
        Name of the data setup.
    output : estimation_analysis.EstimationOutput
        Estimation output.
    observation_collection : observations.ObservationCollection
        Observations used in the estimation.
    """
    residuals = np.asarray(output.final_residuals)
    observable_units = {
        model_settings.angular_position_type: "rad",
        model_settings.n_way_range_type: "m",
        model_settings.doppler_measured_frequency_type: "Hz",
    }
    print(f"\n{label}")
    for observable_type, (start, size) in (
        observation_collection.observable_type_start_index_and_size.items()
    ):
        values = residuals[start : start + size]
        rms = np.sqrt(np.mean(values**2))
        unit = observable_units.get(observable_type, "")
        print(f"  {observable_type}: {len(values)} scalar residuals, RMS = {rms:.6g} {unit}")


def print_yarkovsky_result(output: estimation_analysis.EstimationOutput) -> None:
    """Print the estimated Yarkovsky A2 value and its formal uncertainty.

    Parameters
    ----------
    output : estimation_analysis.EstimationOutput
        Estimation output, with A2 as the last estimated parameter.
    """
    a2_si = float(np.asarray(output.final_parameters)[-1])
    sigma_si = float(np.sqrt(np.asarray(output.covariance)[-1, -1]))
    si_to_au_per_day_squared = constants.JULIAN_DAY**2 / constants.ASTRONOMICAL_UNIT
    print(
        "  Estimated Yarkovsky A2: "
        f"({a2_si:.9g} ± {sigma_si:.3g}) m/s² = "
        f"({a2_si * si_to_au_per_day_squared:.9g} ± "
        f"{sigma_si * si_to_au_per_day_squared:.3g}) au/day²"
    )


"""
Call the functionality defined above to run the orbit estimation for the requested object
"""


if __name__ == "__main__":
    spice.load_standard_kernels()
    
    print(
        "Estimation series: "
        + ("estimating Yarkovsky A2" if ESTIMATE_YARKOVSKY else "Yarkovsky disabled")
    )
    print(
        "Asteroid point-mass perturbers: "
        + ", ".join(
            f"{number} {name}"
            for number, name in ASTEROID_PERTURBERS
            if str(number) != TARGET
        )
    )
    
    # Load the tracking data
    ( optical_tracking_data,optical_supplementary_data,
        radar_tracking_data, radar_supplementary_data ) = load_tracking_data()

    # Define propagation interval from the observation times, placing the initial epoch in the
    # middle. The observation times are in UTC: the difference with TDB is negligible here.
    all_tracking_data = optical_tracking_data + radar_tracking_data
    observation_start_epoch = float(min(min(data.epochs) for data in all_tracking_data))
    observation_end_epoch = float(max(max(data.epochs) for data in all_tracking_data))
    initial_epoch = 0.5 * (observation_start_epoch + observation_end_epoch)
    first_epoch = observation_start_epoch - PROPAGATION_BUFFER
    final_epoch = observation_end_epoch + PROPAGATION_BUFFER

    # Define two estimation setups: astrometry only and (if any) astrometry + radar
    setups = {
        ASTROMETRY_SETUP_NAME: (
            optical_tracking_data,
            optical_supplementary_data,
        ),
    }
    if radar_tracking_data:
        setups[RADAR_SETUP_NAME] = (
            optical_tracking_data + radar_tracking_data,
            optical_supplementary_data + radar_supplementary_data,
        )


    # Get initial guess of asteroid state from Horizons 
    initial_state = HorizonsQuery(
        query_id=HORIZONS_TARGET,
        location=HORIZONS_ORIGIN,
        epoch_list=[float(initial_epoch)],
        extended_query=True,
    ).cartesian(frame_orientation=GLOBAL_FRAME_ORIENTATION)[0, 1:]

    # Iterate over cases (with/without radar) and perform estimation and save results
    results = {}
    for label, (tracking_data, supplementary_data) in setups.items():
        output, observation_collection, estimator = perform_estimation(
            tracking_data,
            supplementary_data,
            initial_epoch,
            initial_state,
            first_epoch,
            final_epoch,
        )
        print_residual_summary(label, output, observation_collection)
        if ESTIMATE_YARKOVSKY:
            print_yarkovsky_result(output)
        results[label] = (output, observation_collection, estimator)

    # Plot estimation results
    plot_diagnostics(results, initial_epoch)
