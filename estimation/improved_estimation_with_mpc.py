"""
# Bennu orbit estimation from MPC astrometry and radar data

This example estimates the orbit (and Yarkovsky acceleration) of a minor planet, here (101955) Bennu, from MPC optical
astrometry and JPL radar delay and Doppler observations. All observations are stored in a single
``ObservationDataset``, which is used to:

1. inspect the data, using *queries* that select observations per observable type, observatory and time window;
2. inspect (and modify) the observation weights;
3. create the datasets of the different estimation setups (with and without radar);
4. reject outliers during the estimation, using the scheme of Carpino et al. (2003) or a simple threshold;
5. compute post-fit residual statistics per observatory;
6. perform a covariance analysis showing the contribution of each radar apparition.

The example consists of the following steps:

1. MPC and JPL data are loaded and converted to ``TrackingData`` objects.
2. Their supplementary data (e.g. radar transmitter frequencies) are applied to the environment.
3. The tracking data are converted to an ``ObservationDataset``, which is inspected with queries.
4. The orbit is estimated from the optical data only, and from the optical and radar data together, without outlier
   rejection.
5. The optical and radar data are fitted again with automatic outlier rejection, using two different schemes.
6. The residuals, the rejected observations, the formal uncertainties and the difference with the JPL Horizons orbit
   are analysed.

The MPC, JPL and Horizons queries require an internet connection.

References

* Carpino, M., Milani, A., Chesley, S. R. (2003). Error statistics of asteroid optical astrometric observations.
  Icarus 166, 248-270.
* Chesley, S. R., et al. (2014). Orbit and bulk density of the OSIRIS-REx target asteroid (101955) Bennu. Icarus 235,
  5-22.
* Eggl, S., Farnocchia, D., Chamberlin, A. B., Chesley, S. R. (2020). Star catalog position and proper motion
  corrections in asteroid astrometry II: The Gaia era. Icarus 339, 113596.
* Vereš, P., Farnocchia, D., Chesley, S. R., Chamberlin, A. B. (2017). Statistical analysis of astrometric errors for
  the most productive asteroid surveys. Icarus 296, 139-149.
"""

import datetime

import numpy as np
import matplotlib.pyplot as plt

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
from tudatpy.estimation.observable_models_setup import links, model_settings
from tudatpy.estimation.observations import observation_query


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
EARTH_MOON_EPHEMERIS_STEP = 6.0 * 3600.0
PLANET_EPHEMERIS_STEP = constants.JULIAN_DAY
ASTEROID_EPHEMERIS_STEP = 5.0 * constants.JULIAN_DAY

"""
Outlier rejection settings. The scheme of Carpino et al. (2003) rejects an observation when the chi-squared value of its
residual (the squared residual normalized by its covariance, which accounts for the fit itself) exceeds a threshold, and
recovers it when the value drops below a slightly lower threshold. The thresholds of 9 and 8 (i.e. about 3 sigma) are
the defaults; Chesley et al. (2014) compare several thresholds for Bennu, and reject 7 of their 569 optical observations
with this one. The simple scheme instead rejects an
observation when the absolute value of a residual component exceeds a fixed value per observable type, without
accounting for the observation uncertainty.
"""
CARPINO_CHI2_REJECTION = 9.0
CARPINO_CHI2_RECOVERY = 8.0
SIMPLE_REJECTION_THRESHOLDS = {
    model_settings.angular_position_type: 2.0 / 3600.0 * np.pi / 180.0,  # 2 arcseconds [rad]
    model_settings.n_way_range_type: 3.0e3,  # [m] (round-trip range, i.e. 10 microseconds of delay)
    model_settings.doppler_measured_frequency_type: 5.0,  # [Hz]
}
SIGMA_SCREENING_LIMIT = 3.0

"""
Plotting settings
"""
NUMBER_OF_COLORED_OBSERVATORIES = 10
NUMBER_OF_HORIZONS_PLOT_EPOCHS = 300
ASTROMETRY_SETUP_NAME = "MPC astrometry"
RADAR_SETUP_NAME = "MPC astrometry and JPL radar"
CARPINO_SETUP_NAME = "MPC astrometry and JPL radar, Carpino et al. (2003) outlier rejection"
SIMPLE_SETUP_NAME = "MPC astrometry and JPL radar, simple outlier rejection"
PLOTTED_SETUPS = [ASTROMETRY_SETUP_NAME, CARPINO_SETUP_NAME]
ARCSECONDS_PER_RADIAN = 180.0 / np.pi * 3600.0

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
Conditions (queries) selecting the optical and the radar observations in an observation dataset. A condition is
created by comparing an attribute of ``observation_query`` to a value; conditions are combined with ``&`` (and),
``|`` (or) and ``~`` (not).
"""
OPTICAL = observation_query.observable_type == model_settings.angular_position_type
RADAR = observation_query.observable_type != model_settings.angular_position_type
OBSERVABLE_UNITS = {
    model_settings.angular_position_type: ("Angular position", ARCSECONDS_PER_RADIAN, "arcsec"),
    model_settings.n_way_range_type: ("Range", 1.0, "m"),
    model_settings.doppler_measured_frequency_type: ("Doppler", 1.0, "Hz"),
}


def epoch(year: int, month: int = 1, day: int = 1) -> float:
    """Return the epoch of a calendar date [s since J2000]."""
    return time_representation.DateTime(year, month, day).to_epoch()


"""
In the function below, the environment is created. This defines the properties of the bodies used in the simulations.
These bodies include the most massive asteroids, whose perturbations are taken into account. For the planets, we do not
use the masses/ephemerides of the planets and moons separately (with the exception of the Earth and its Moon), instead,
we place the planet at the center of mass of its own system, and give it the total system mass
"""


def create_bodies(first_epoch: float, final_epoch: float) -> environment.SystemOfBodies:
    """Create the estimation environment and Earth observing stations.

    Parameters
    ----------
    first_epoch : float
        Start of the propagation interval [s since J2000, TDB].
    final_epoch : float
        End of the propagation interval [s since J2000, TDB].

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

    # Use ephemerides tabulated from spice, since reading the spice kernels during the propagation is very slow.
    # Move each planet from Mars onwards to the barycenter of its own planetary system, and give it the
    # full system's mass
    for body_name in body_names:
        is_planetary_system = body_name in ["Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
        spice_name = f"{body_name} Barycenter" if is_planetary_system else body_name
        time_step = EARTH_MOON_EPHEMERIS_STEP if body_name in ["Earth", "Moon"] else PLANET_EPHEMERIS_STEP
        settings = body_settings.get(body_name)
        settings.ephemeris_settings = environment_setup.ephemeris.interpolated_spice(
            first_epoch - 10.0 * time_step,
            final_epoch + 10.0 * time_step,
            time_step,
            GLOBAL_FRAME_ORIGIN,
            GLOBAL_FRAME_ORIENTATION,
            body_name_to_use=spice_name,
        )
        if is_planetary_system:
            settings.gravity_field_settings = (
                environment_setup.gravity_field.central_spice(spice_name)
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
        settings.ephemeris_settings = environment_setup.ephemeris.interpolated_spice(
            first_epoch - 10.0 * ASTEROID_EPHEMERIS_STEP,
            final_epoch + 10.0 * ASTEROID_EPHEMERIS_STEP,
            ASTEROID_EPHEMERIS_STEP,
            GLOBAL_FRAME_ORIGIN,
            GLOBAL_FRAME_ORIENTATION,
            body_name_to_use=spice_id,
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
The available data is loaded from the MPC and JPL (for radar data). The optical astrometry is weighted with the scheme
of Vereš et al. (2017), in which the uncertainty depends on the observatory, the star catalog and the observation
epoch, and is inflated by sqrt(N/4) when an observatory reports N > 4 observations in one night. The star catalog
biases are removed with the corrections of Eggl et al. (2020). The radar weights follow from the uncertainties in the
JPL radar database. The data, and any associated data such as radar transmission frequencies, are returned to be used
in the estimation
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
        print("No JPL radar observations found; skipping the radar-inclusive runs.")

    return (
        optical_tracking_data,
        optical_supplementary_data,
        radar_tracking_data,
        radar_supplementary_data,
    )


"""
## Inspecting the observation dataset

The ``ObservationDataset`` groups the observations in observation sets: here one set per observatory for the optical
data, and one set per transmitter, receiver and observable for the radar data. The properties of each set (observable
type, link ends, ...) are given by ``get_metadata``. Most other functions of the dataset act on the observations
selected by a condition. Below, the conditions select the observations of a single observatory (``receiver``),
of an observable type, and within a time window.
"""


def receivers(observation_dataset: observations.ObservationDataset, condition) -> list[links.LinkEndId]:
    """Return the receiving stations (or space telescopes) of the selected observations, each once.

    Parameters
    ----------
    observation_dataset : observations.ObservationDataset
        Observations.
    condition : observations.ObservationSelectionCondition
        Condition selecting the observations.

    Returns
    -------
    list[LinkEndId]
        Receiver link end ids, sorted by decreasing number of selected observations.
    """
    number_of_observations = {}
    for metadata in observation_dataset.get_metadata(condition).values():
        receiver = metadata["link_definition"].link_end_id(links.receiver)
        number_of_observations[receiver] = len(
            observation_dataset.get_observation_ids(condition & (observation_query.receiver == receiver))
        )
    return sorted(number_of_observations, key=number_of_observations.get, reverse=True)


RADAR_STATION_NAMES = {"JPL:-1": "Arecibo", "JPL:-14": "Goldstone DSS-14"}


def receiver_name(receiver: links.LinkEndId) -> str:
    """Return the station name (MPC code for optical observatories), or body name for a space telescope."""
    name = receiver.reference_point or receiver.body_name
    return RADAR_STATION_NAMES.get(name, name)


def radar_years(observation_dataset: observations.ObservationDataset) -> list[int]:
    """Return the calendar years in which radar observations were made."""
    times = np.asarray(observation_dataset.get_times(RADAR), dtype=float)
    return sorted({time_representation.DateTime.from_epoch(time).year for time in times})


def print_dataset_overview(observation_dataset: observations.ObservationDataset) -> None:
    """Print the number of observations per observable type, radar apparition and observatory, with their weights.

    Parameters
    ----------
    observation_dataset : observations.ObservationDataset
        Observations to describe.
    """
    print(
        f"\nObservation dataset: {observation_dataset.number_of_observations} observations in "
        f"{observation_dataset.number_of_observation_sets} observation sets"
    )
    for observable_type, (name, scale, unit) in OBSERVABLE_UNITS.items():
        condition = observation_query.observable_type == observable_type
        weights = observation_dataset.get_weight_diagonal(condition)
        if len(weights) == 0:
            continue
        sigmas = scale / np.sqrt(weights)
        print(
            f"  {name}: {len(observation_dataset.get_times(condition))} observations, "
            f"1-sigma uncertainty from the weights {np.min(sigmas):.3g} to {np.max(sigmas):.3g} {unit}"
        )

    # Radar observations per year, selected with a time window
    for year in radar_years(observation_dataset):
        apparition = RADAR & observation_query.time.between(epoch(year), epoch(year + 1))
        print(f"  Radar observations in {year}: {len(observation_dataset.get_times(apparition))}")

    # Observations and (median) weight per optical observatory
    print("\n  Observatory   Observations   Median 1-sigma uncertainty from the weights [arcsec]")
    for receiver in receivers(observation_dataset, OPTICAL)[:NUMBER_OF_COLORED_OBSERVATORIES]:
        condition = OPTICAL & (observation_query.receiver == receiver)
        sigmas = ARCSECONDS_PER_RADIAN / np.sqrt(observation_dataset.get_weight_diagonal(condition))
        print(
            f"  {receiver_name(receiver):>11}   {len(observation_dataset.get_times(condition)):>12}   "
            f"{np.median(sigmas):.3g}"
        )


"""
From the data stored in the ObservationDataset object (observed data), we can define the settings that are needed to
create the observation models, from which the computed data are generated during the estimation.
"""


def observation_model_settings(
    observation_dataset: observations.ObservationDataset,
) -> list[model_settings.ObservationModelSettings]:
    """Create observation models for every link present in the dataset.

    Parameters
    ----------
    observation_dataset : observations.ObservationDataset
        Observations for which models are created.

    Returns
    -------
    list[ObservationModelSettings]
        Settings of one observation model per link and observable.
    """
    # Retrieve sets of link ends per observation type
    links_for_type = observation_dataset.link_definitions_for_observable

    # Use a simple relativistic correction to the light time for each observation model
    corrections = [
        observable_models_setup.light_time_corrections.first_order_relativistic_light_time_correction(
            ["Sun"]
        )
    ]

    # Create astrometry (angular position) model settings
    settings = [
        model_settings.angular_position(link)
        for link in links_for_type(model_settings.angular_position_type)
    ]

    # Create radar range model settings, where the time interval converted to range is to be in UTC
    settings += [
        model_settings.n_way_range(
            link,
            corrections,
            time_scale_for_observable=time_representation.utc_scale,
        )
        for link in links_for_type(model_settings.n_way_range_type)
    ]

    # Create radar Doppler model settings
    settings += [
        model_settings.doppler_measured_frequency(link, corrections)
        for link in links_for_type(model_settings.doppler_measured_frequency_type)
    ]
    return settings


"""
Now, we can put everything together, and fit the data to the dynamical model estimating the parameters we want. The
output is the converged values of the parameters with the associated residuals and parameter covariance. The
post-fit residuals are also stored in the observation dataset and, if outlier rejection settings are provided, the
observations that are rejected are marked as such in the dataset.
"""


def perform_estimation(
    observation_dataset: observations.ObservationDataset,
    supplementary_data: list[TrackingSupplementaryData],
    initial_epoch: float,
    initial_state: np.ndarray,
    first_epoch: float,
    final_epoch: float,
    outlier_rejection_settings: estimation_analysis.OutlierRejectionSettings | None = None,
) -> tuple[estimation_analysis.EstimationOutput, estimation_analysis.Estimator]:
    """Estimate the target state for one data setup.

    Parameters
    ----------
    observation_dataset : observations.ObservationDataset
        Observations to fit. Their residuals (and rejection status) are updated by the estimation.
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
    outlier_rejection_settings : estimation_analysis.OutlierRejectionSettings | None, default None
        Settings of the outlier rejection; no outliers are rejected if None.

    Returns
    -------
    tuple[EstimationOutput, Estimator]
        Estimation output and estimator.
    """

    # Create environment, and update it with relevant information from the tracking data source (radar transmitter
    # frequencies, space observatory ephemerides, etc.)
    bodies = create_bodies(first_epoch, final_epoch)
    observations.set_tracking_supplementary_data_in_bodies(
        bodies,
        supplementary_data,
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
        observation_settings=observation_model_settings(observation_dataset),
        propagator_settings=propagator_settings,
        integrate_on_creation=True,
    )
    estimation_input = estimation_analysis.EstimationInput(
        observation_dataset=observation_dataset,
        convergence_checker=estimation_analysis.estimation_convergence_checker(
            maximum_iterations=5,
        ),
        outlier_rejection_settings=outlier_rejection_settings,
    )
    estimation_input.define_estimation_settings(
        reintegrate_variational_equations=True,
        print_output_to_terminal=True,
        save_state_history_per_iteration=True,
    )
    output = estimator.perform_estimation(estimation_input)
    return output, estimator


"""
## Post-fit residual statistics per observatory

After the estimation, the residuals of the best iteration are stored in the dataset. Using a condition per
observatory, we compute the number of used and rejected observations, the mean (bias) and RMS of the residuals, and
the RMS of the normalized residuals (residual divided by its uncertainty from the weights), which should be close to 1
if the weights are realistic (Vereš et al., 2017).
"""


def print_observatory_statistics(label: str, observation_dataset: observations.ObservationDataset) -> None:
    """Print the post-fit residual statistics of the optical observatories with most observations.

    Parameters
    ----------
    label : str
        Name of the data setup.
    observation_dataset : observations.ObservationDataset
        Observations used in the estimation, with post-fit residuals.
    """
    print(f"\n{label}: post-fit optical residuals per observatory")
    print(
        "  Observatory   Used   Rejected   Mean RA*cos(Dec), Dec [arcsec]   RMS RA*cos(Dec), Dec [arcsec]   "
        "Normalized RMS [-]"
    )
    for receiver in receivers(observation_dataset, OPTICAL)[:NUMBER_OF_COLORED_OBSERVATORIES]:
        station = OPTICAL & (observation_query.receiver == receiver)
        used = station & observation_query.active
        data = observation_dataset.get_data(used, fields=["observations", "residuals", "weight_diagonal"])
        residuals = np.array(data["residuals"]) * ARCSECONDS_PER_RADIAN
        # Right ascension residuals are multiplied by cos(Dec), to obtain the residual on the sky
        residuals[:, 0] *= np.cos(np.array(data["observations"])[:, 1])
        normalized = np.array(data["residuals"]).reshape(-1) * np.sqrt(data["weight_diagonal"])
        number_rejected = len(observation_dataset.get_observation_ids(station & observation_query.rejected))
        print(
            f"  {receiver_name(receiver):>11}   {len(residuals):>4}   {number_rejected:>8}   "
            f"{np.mean(residuals[:, 0]):>14.3f}, {np.mean(residuals[:, 1]):<14.3f}   "
            f"{np.sqrt(np.mean(residuals[:, 0] ** 2)):>13.3f}, {np.sqrt(np.mean(residuals[:, 1] ** 2)):<14.3f}   "
            f"{np.sqrt(np.mean(normalized ** 2)):.3f}"
        )


def print_residual_summary(
    label: str,
    output: estimation_analysis.EstimationOutput,
    observation_dataset: observations.ObservationDataset,
) -> None:
    """Print the final residual RMS for each observable type, and the number of rejected observations.

    Parameters
    ----------
    label : str
        Name of the data setup.
    output : estimation_analysis.EstimationOutput
        Estimation output.
    observation_dataset : observations.ObservationDataset
        Observations used in the estimation, with post-fit residuals.
    """
    print(f"\n{label}")
    for observable_type, (name, scale, unit) in OBSERVABLE_UNITS.items():
        selected = observation_query.observable_type == observable_type
        residuals = np.concatenate(
            observation_dataset.get_residuals(selected & observation_query.active) or [np.zeros(0)]
        )
        if len(residuals) == 0:
            continue
        number_rejected = len(observation_dataset.get_observation_ids(selected & observation_query.rejected))
        print(
            f"  {name}: {len(residuals)} scalar residuals used, {number_rejected} observations rejected, "
            f"RMS = {np.sqrt(np.mean((residuals * scale) ** 2)):.6g} {unit}"
        )
    if ESTIMATE_YARKOVSKY:
        print_yarkovsky_result(output)


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
## Comparing the outlier rejection schemes

The observation ids are the same in all datasets that were created from the same original dataset, so that the sets
of rejected observations can be compared directly. We also compare with a simple post-fit screening, which rejects
the observations whose residual exceeds 3 times the uncertainty from the weights in the fit without outlier rejection.
Contrary to the scheme of Carpino et al. (2003), this screening is a single step, which does not account for the
effect of each observation on the fit, and cannot recover observations.
"""


def compare_rejections(results: dict) -> None:
    """Print the number of rejected observations per scheme, and their overlap.

    Parameters
    ----------
    results : dict[str, tuple[EstimationOutput, ObservationDataset, Estimator]]
        Estimation output, observations and estimator per data setup.
    """
    screened_dataset = results[RADAR_SETUP_NAME][1].create_new_and_keep(observations.ObservationSelectionCondition.all())
    screened_dataset.reject_observations(
        observations.ObservationSelectionCondition.all(),
        sigma_limit=SIGMA_SCREENING_LIMIT,
        reason=f"{SIGMA_SCREENING_LIMIT:g}-sigma post-fit screening",
    )
    rejected = {
        f"{SIGMA_SCREENING_LIMIT:g}-sigma screening": screened_dataset,
        "Carpino et al. (2003)": results[CARPINO_SETUP_NAME][1],
        "Simple threshold": results[SIMPLE_SETUP_NAME][1],
    }
    rejected = {
        name: set(dataset.get_observation_ids(observation_query.rejected)) for name, dataset in rejected.items()
    }
    print("\nRejected observations (optical and radar)")
    for name, ids in rejected.items():
        print(f"  {name}: {len(ids)}")
    names = list(rejected)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            print(f"  Rejected by both '{first}' and '{second}': {len(rejected[first] & rejected[second])}")


"""
## Covariance analysis

The formal uncertainty of the estimated parameters depends on which observations are used. Since conditions can be
combined, it is easy to create datasets with subsets of the observations, and compute the covariance for each of them
with the estimator of the full setup. Here, we show how the formal uncertainty of the Yarkovsky parameter decreases
when the radar apparitions are added one by one, as discussed by Chesley et al. (2014) for the 2011 Arecibo
observations.
"""


def print_covariance_analysis(
    observation_dataset: observations.ObservationDataset, estimator: estimation_analysis.Estimator
) -> None:
    """Print the formal uncertainty of the Yarkovsky parameter for subsets of the observations.

    Parameters
    ----------
    observation_dataset : observations.ObservationDataset
        Observations used in the estimation (rejected observations are not used here either).
    estimator : estimation_analysis.Estimator
        Estimator of the setup, containing the observation models of all observations.
    """
    subsets = {"Optical only": OPTICAL}
    for year in radar_years(observation_dataset):
        subsets[f"Optical and radar up to {year}"] = OPTICAL | (RADAR & (observation_query.time < epoch(year + 1)))
    subsets["Radar only"] = RADAR

    si_to_au_per_day_squared = constants.JULIAN_DAY**2 / constants.ASTRONOMICAL_UNIT
    print("\nCovariance analysis: formal 1-sigma uncertainty for subsets of the observations")
    print("  Observations                          Position [km]   Yarkovsky A2 [au/day²]")
    for name, condition in subsets.items():
        covariance_input = estimation_analysis.CovarianceAnalysisInput(
            observation_dataset.create_new_and_keep(condition)
        )
        covariance_input.define_covariance_settings(
            reintegrate_variational_equations=False, print_output_to_terminal=False
        )
        covariance = np.asarray(estimator.compute_covariance(covariance_input).covariance)
        position_sigma = np.sqrt(np.trace(covariance[:3, :3])) / 1000.0
        a2_sigma = np.sqrt(covariance[-1, -1]) * si_to_au_per_day_squared if ESTIMATE_YARKOVSKY else np.nan
        print(f"  {name:<36}  {position_sigma:>13.4g}   {a2_sigma:.3g}")


"""
## Observatory-specific weights

The normalized post-fit RMS per observatory is well below 1 for most observatories: the weights of Vereš et al. (2017)
are deliberately conservative, since they are based on the upper bound of the residual RMS of each observatory, and are
reduced for observatories that report more than four observations per night, to account for the correlation of the
errors within a night. Older weighting schemes instead derived the weight of each observatory from the RMS of its own
residuals (see Vereš et al., 2017). As an illustration of setting weights per observatory, we rescale the weights of
each observatory with at least ``MINIMUM_OBSERVATIONS_FOR_OBSERVATORY_WEIGHT`` used observations, such that its
normalized post-fit RMS becomes 1, and compute the formal uncertainties with these weights. Since this ignores the
correlation of the errors within a night, the resulting formal uncertainties are optimistic.
"""

MINIMUM_OBSERVATIONS_FOR_OBSERVATORY_WEIGHT = 10


def print_covariance_with_observatory_weights(
    observation_dataset: observations.ObservationDataset, estimator: estimation_analysis.Estimator
) -> None:
    """Print the formal uncertainties obtained with weights rescaled per optical observatory.

    Parameters
    ----------
    observation_dataset : observations.ObservationDataset
        Observations used in the estimation, with post-fit residuals. This dataset is not modified.
    estimator : estimation_analysis.Estimator
        Estimator of the setup, containing the observation models of all observations.
    """
    reweighted_dataset = observation_dataset.create_new_and_keep(observations.ObservationSelectionCondition.all())
    print("\nObservatory-specific weights: factor by which the 1-sigma uncertainty is multiplied [-]")
    for receiver in receivers(observation_dataset, OPTICAL):
        station = OPTICAL & (observation_query.receiver == receiver)
        used = station & observation_query.active
        residuals = np.concatenate(observation_dataset.get_residuals(used) or [np.zeros(0)])
        if len(residuals) < 2 * MINIMUM_OBSERVATIONS_FOR_OBSERVATORY_WEIGHT:
            continue
        normalized_rms = np.sqrt(np.mean(residuals**2 * observation_dataset.get_weight_diagonal(used)))
        print(f"  {receiver_name(receiver):>11}: {normalized_rms:.3f}")
        # The optical observations of one observatory form one observation set, whose weights are rescaled
        for set_id in reweighted_dataset.get_metadata(station):
            weights = np.concatenate(reweighted_dataset.weights_for_set(set_id))
            reweighted_dataset.set_weight_vector_for_set(set_id, weights / normalized_rms**2)

    si_to_au_per_day_squared = constants.JULIAN_DAY**2 / constants.ASTRONOMICAL_UNIT
    print("  Weights                                Position [km]   Yarkovsky A2 [au/day²]")
    for name, dataset in {"Vereš et al. (2017)": observation_dataset, "Observatory-specific": reweighted_dataset}.items():
        covariance_input = estimation_analysis.CovarianceAnalysisInput(dataset)
        covariance_input.define_covariance_settings(
            reintegrate_variational_equations=False, print_output_to_terminal=False
        )
        covariance = np.asarray(estimator.compute_covariance(covariance_input).covariance)
        position_sigma = np.sqrt(np.trace(covariance[:3, :3])) / 1000.0
        a2_sigma = np.sqrt(covariance[-1, -1]) * si_to_au_per_day_squared if ESTIMATE_YARKOVSKY else np.nan
        print(f"  {name:<37}  {position_sigma:>13.4g}   {a2_sigma:.3g}")


"""
The code below generates the figures. The residuals are plotted both as their values, for right ascension and
declination in arcseconds, for radar (if any) in meters and Hz, and normalized by their uncertainty (as taken from the
weights). Rejected observations are shown with black crosses. The post-fit orbit difference w.r.t. Horizons is also
shown, along with the formal uncertainty from our estimation. The figures are made for the setups in
``PLOTTED_SETUPS``; the RMS of the difference with Horizons is printed for all setups.
"""


def plot_residuals(setup_label: str, observation_dataset: observations.ObservationDataset) -> None:
    """Plot the post-fit and normalized residuals of each observable component, per observatory.

    Parameters
    ----------
    setup_label : str
        Name of the data setup.
    observation_dataset : observations.ObservationDataset
        Observations used in the estimation, with post-fit residuals.
    """
    component_names = {
        model_settings.angular_position_type: ["Right ascension", "Declination"],
        model_settings.n_way_range_type: ["Range"],
        model_settings.doppler_measured_frequency_type: ["Doppler"],
    }
    for observable_type, (_, scale, unit) in OBSERVABLE_UNITS.items():
        selected = observation_query.observable_type == observable_type
        stations = receivers(observation_dataset, selected)
        if not stations:
            continue
        colored = stations[:NUMBER_OF_COLORED_OBSERVATORIES]
        groups = [(receiver_name(receiver), selected & (observation_query.receiver == receiver), {}) for receiver in colored]
        if len(stations) > len(colored):
            other = selected
            for receiver in colored:
                other = other & (observation_query.receiver != receiver)
            groups.insert(0, ("Other", other, {"color": "lightgrey"}))

        for component, component_name in enumerate(component_names[observable_type]):
            figure, axes = plt.subplots(2, 1, figsize=(11.7, 7.5), sharex=True)
            groups_with_status = [(name, condition & observation_query.active, style) for name, condition, style in groups]
            groups_with_status.append(
                ("Rejected", selected & observation_query.rejected, {"color": "black", "marker": "x"})
            )
            for name, condition, style in groups_with_status:
                data = observation_dataset.get_data(condition, fields=["times", "residuals", "weight_diagonal"])
                if len(data["times"]) == 0:
                    continue
                years = 2000.0 + np.array([float(time) for time in data["times"]]) / constants.JULIAN_YEAR
                residuals = np.array(data["residuals"])[:, component]
                sigmas = 1.0 / np.sqrt(np.reshape(data["weight_diagonal"], (len(years), -1))[:, component])
                axes[0].scatter(years, residuals * scale, s=11, label=name, **style)
                axes[1].scatter(years, residuals / sigmas, s=11, **style)
            for axis, y_label in zip(axes, [f"Post-fit residual [{unit}]", "Residual / 1σ uncertainty"]):
                axis.axhline(0.0, color="black", linewidth=0.8)
                axis.grid(alpha=0.3)
                axis.set_ylabel(y_label)
            axes[1].axhline(3.0, color="black", linestyle="--", linewidth=0.8)
            axes[1].axhline(-3.0, color="black", linestyle="--", linewidth=0.8)
            axes[1].set_xlabel("Year")
            axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
            used = np.concatenate(observation_dataset.get_residuals(selected & observation_query.active))
            rms = np.sqrt(np.mean(np.reshape(used, (-1, len(component_names[observable_type])))[:, component] ** 2))
            figure.suptitle(f"{setup_label}\n{component_name} post-fit residuals (RMS = {rms * scale:.6g} {unit})")
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
    plot: bool = True,
) -> None:
    """Print the RMS of the RSW difference with Horizons and (optionally) plot it, with ±3σ bands and 1σ formal errors.

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
    plot : bool, default True
        Whether to plot the difference.
    """
    state_history = best_state_history(output)
    propagated_covariances = estimation_analysis.propagate_covariance(
        output.covariance,
        estimator.state_transition_interface,
        list(epochs),
    )

    differences_rsw = np.empty((len(epochs), 3))
    formal_errors_rsw = np.empty_like(differences_rsw)
    for index, (epoch_value, horizons_state) in enumerate(zip(epochs, horizons_states)):
        rotation = inertial_to_rsw_rotation_matrix(horizons_state)
        position_covariance = np.asarray(propagated_covariances[epoch_value])[:3, :3]
        differences_rsw[index] = rotation @ (
            np.asarray(state_history[epoch_value])[:3] - horizons_state[:3]
        )
        formal_errors_rsw[index] = np.sqrt(
            np.diag(rotation @ position_covariance @ rotation.T)
        )
    differences_rsw /= 1000.0
    formal_errors_rsw /= 1000.0
    print(
        f"\n{setup_label}\n  RMS difference with Horizons over {len(epochs)} epochs [km]: "
        + ", ".join(f"{name} = {value:.4g}" for name, value in zip("RSW", np.sqrt(np.mean(differences_rsw**2, axis=0))))
    )
    if not plot:
        return
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

    epoch_value = min(common_epochs, key=lambda value: abs(value - estimation_epoch))
    if not np.isclose(epoch_value, estimation_epoch, rtol=0.0, atol=1.0e-6):
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
    difference_at_estimation_epoch = differences[common_epochs.index(epoch_value)]
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
            observations.ObservationDataset,
            estimation_analysis.Estimator,
        ],
    ],
    estimation_epoch: float,
) -> None:
    """Show the figures of each data setup, and print the effect of adding radar.

    Parameters
    ----------
    results : dict[str, tuple[EstimationOutput, ObservationDataset, Estimator]]
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

    for setup_label, (output, observation_dataset, estimator) in results.items():
        plot = setup_label in PLOTTED_SETUPS
        if plot:
            plot_residuals(setup_label, observation_dataset)
        plot_orbit_difference(setup_label, output, estimator, plot_epochs, horizons_states, plot)
    plt.show()


"""
Call the functionality defined above to run the orbit estimation for the requested object.

For Bennu, the optical data alone give only a weak (about 3 sigma) detection of the Yarkovsky acceleration. Adding the
radar data, in particular the 2011 Arecibo observations, reduces its formal uncertainty by more than an order of
magnitude, as found by Chesley et al. (2014). Their estimate, A_T = (-4.618 ± 0.024) × 10⁻¹⁴ au/day² for a transverse
acceleration proportional to r^-2.25, corresponds to about -4.55 × 10⁻¹⁴ au/day² for the r^-2 dependence used here (same
mean semimajor axis drift). The simplified force model of this example (point masses only, no solar radiation pressure,
no planetary relativistic perturbations) and the different data arc explain part of the difference.
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
    (optical_tracking_data, optical_supplementary_data,
        radar_tracking_data, radar_supplementary_data) = load_tracking_data()
    tracking_data = optical_tracking_data + radar_tracking_data
    supplementary_data = optical_supplementary_data + radar_supplementary_data

    # Define propagation interval from the observation times, placing the initial epoch in the
    # middle. The observation times are in UTC: the difference with TDB is negligible here.
    observation_start_epoch = float(min(min(data.epochs) for data in tracking_data))
    observation_end_epoch = float(max(max(data.epochs) for data in tracking_data))
    initial_epoch = 0.5 * (observation_start_epoch + observation_end_epoch)
    first_epoch = observation_start_epoch - PROPAGATION_BUFFER
    final_epoch = observation_end_epoch + PROPAGATION_BUFFER

    # Create a single observation dataset with all observations. The environment is used to convert the observation
    # times to TDB, to compute the optical weights (which depend on the number of observations per local night) and to
    # apply the star catalog corrections.
    bodies = create_bodies(first_epoch, final_epoch)
    observations.set_tracking_supplementary_data_in_bodies(bodies, supplementary_data)
    observation_dataset = observations.create_observation_dataset_from_tracking_data(
        tracking_data,
        bodies,
        apply_corrections=True,
    )
    print_dataset_overview(observation_dataset)

    # Define the estimation setups: astrometry only, and (if any) astrometry + radar, without and with outlier
    # rejection. Each setup uses its own dataset, since the estimation stores the residuals and the rejected
    # observations in it. A copy of a dataset is created by keeping all of its observations.
    all_observations = observations.ObservationSelectionCondition.all()
    setups = {
        ASTROMETRY_SETUP_NAME: (observation_dataset.create_new_and_keep(OPTICAL), None),
    }
    if radar_tracking_data:
        setups[RADAR_SETUP_NAME] = (observation_dataset.create_new_and_keep(all_observations), None)
        setups[CARPINO_SETUP_NAME] = (
            observation_dataset.create_new_and_keep(all_observations),
            estimation_analysis.carpino_outlier_rejection_settings(
                chi2_rejection_threshold=CARPINO_CHI2_REJECTION,
                chi2_recovery_threshold=CARPINO_CHI2_RECOVERY,
            ),
        )
        setups[SIMPLE_SETUP_NAME] = (
            observation_dataset.create_new_and_keep(all_observations),
            estimation_analysis.simple_outlier_rejection_settings(SIMPLE_REJECTION_THRESHOLDS),
        )

    # Get initial guess of asteroid state from Horizons
    initial_state = HorizonsQuery(
        query_id=HORIZONS_TARGET,
        location=HORIZONS_ORIGIN,
        epoch_list=[float(initial_epoch)],
        extended_query=True,
    ).cartesian(frame_orientation=GLOBAL_FRAME_ORIENTATION)[0, 1:]

    # Iterate over the setups, perform the estimation and save the results
    results = {}
    for label, (setup_dataset, outlier_rejection_settings) in setups.items():
        output, estimator = perform_estimation(
            setup_dataset,
            supplementary_data,
            initial_epoch,
            initial_state,
            first_epoch,
            final_epoch,
            outlier_rejection_settings,
        )
        print_residual_summary(label, output, setup_dataset)
        print_observatory_statistics(label, setup_dataset)
        results[label] = (output, setup_dataset, estimator)

    if radar_tracking_data:
        compare_rejections(results)
        print_covariance_analysis(*results[CARPINO_SETUP_NAME][1:])
        print_covariance_with_observatory_weights(*results[CARPINO_SETUP_NAME][1:])

    # Plot estimation results
    plot_diagnostics(results, initial_epoch)
