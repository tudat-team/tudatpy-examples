import sys
sys.path.insert(0, "/home/dominic/Tudat/tudat-monorepo/tudatpy/cmake-build-release/src")
"""
# Loading and Using Tracking Observations

Copyright (c) 2010-2024, Delft University of Technology. All rights reserved. This file is part of the Tudat. Redistribution and use in source and binary forms, with or without modification, are permitted exclusively under the terms of the Modified BSD license. You should have received a copy of the license with this file. If not, please or visit: http://tudat.tudelft.nl/LICENSE.

## Objectives

With this example, we will explore how to **load tracking observations** into Tudat so that they can be used for estimation purposes. 

We then show how to **simulate the same measurements and reduce the observation residuals** by adding a **more accurate rotation model and relativistic corrections**.

The example uses **range measurements** from the **Mars Reconnaissance Orbiter (MRO)** with a variety of **Deep Space Network (DSN)** ground stations. The data is already corrected so that it represents the **two-way light time** between those ground stations and Mars system barycenter. To simulate the observations, we start from SPICE ephemerides and obtain residuals in the order of a few hundred meters.

## Prerequisites

To run this example, you need [the data file](https://ssd.jpl.nasa.gov/dat/planets/mrorange2006-2013.txt) from the NASA JPL and store it in a subfolder called ``./data``. For your convenience, this file [has been added to the example repository](./data/mrorange2006-2013.txt) already.

"""


"""
## Import statements

Typically - in the most pythonic way - all required modules are imported at the very beginning.

Some standard modules are first loaded: `numpy` and `matplotlib.pyplot`.

Then, the different modules of `tudatpy` that will be used are imported. Most notably, some elements of the `observation`, `estimation` and `estimation_setup` modules will be used and demonstrated within this example.

"""


# General imports
import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy import signal as scipy_signal
except ImportError:
    scipy_signal = None

# Tudatpy imports
from tudatpy import constants
from tudatpy.astro import time_representation
from tudatpy.data_input.tracking_data.generic_text_file import TrackingDataType, read_generic_text_data
from tudatpy.data_input.environment_data import spice
from tudatpy.dynamics import environment, environment_setup
from tudatpy.estimation import observable_models_setup, observations, observations_setup
from tudatpy.estimation.observable_models_setup import links


MARS_BARYCENTER_COLUMNS = [
    "spacecraft_id",
    "dsn_transmitting_station_nr",
    "dsn_receiving_station_nr",
    "year",
    "month_three_letter",
    "day",
    "hour",
    "minute",
    "second",
    "round_trip_light_time_microseconds",
    "time_tag_delay_microseconds",
]

TRACKING_DATASETS = [
    {
        "name": "MRO",
        "file_name": "./data/mrorange2006-2013.txt",
        "columns": MARS_BARYCENTER_COLUMNS,
        "collection_type": "dsn_to_mars_barycenter",
        "spacecraft_name": "Mars",
    },
    {
        "name": "Mars Odyssey",
        "file_name": "./data/odyrange2002-2013.txt",
        "columns": MARS_BARYCENTER_COLUMNS,
        "collection_type": "dsn_to_mars_barycenter",
        "spacecraft_name": "Mars",
    },
    {
        "name": "Mars Global Surveyor",
        "file_name": "./data/mgsrange1999-2006.txt",
        "columns": MARS_BARYCENTER_COLUMNS,
        "collection_type": "dsn_to_mars_barycenter",
        "spacecraft_name": "Mars",
        "thinning_step": 100,
    },
]

ACTIVE_TRACKING_DATASETS = TRACKING_DATASETS
if not ACTIVE_TRACKING_DATASETS:
    raise RuntimeError("At least one tracking data set must be enabled.")

for tracking_dataset in ACTIVE_TRACKING_DATASETS:
    try:
        with open(tracking_dataset["file_name"], "r"):
            pass
    except FileNotFoundError:
        print(
            f"FILE {tracking_dataset['file_name']} NOT FOUND. Download the JPL data file and store it in ./data."
        )
        exit(1)


"""
## Read the observations

To investigate the observation data, we will start by **reading the observations** into a format that is useful for Tudat.

After inspecting the data file, we can see that it contains the following columns:

1) the spacecraft id
2) the DSN stations involved with the measurement
3) a date in UTC format
4) the round-trip light
5) a correction term.

We can use the `read_generic_text_data` function to read this **raw data** and translate it into an **intermediate format** that takes care of appropriate unit conversions for known column identifiers.

The file columns specified here are all known to Tudat, and can be used to **process the observation** (see `tudatpy.data_input.tracking_data.generic_text_file.TrackingDataType` for a complete list of available column types). If a file contains additional columns, they can be specified with any unknown string and the `read_generic_text_data` function will load them in string format without using them further. If needed, these can be accessed as a dictionary through `raw_datafile.raw_datamap`.

"""


def read_range_data_file(tracking_dataset):
    raw_datafile = read_generic_text_data(
        file_names=[tracking_dataset["file_name"]],
        column_types=tracking_dataset["columns"],
        comment_symbol="#",
        value_separators=",:\t ",
    )[0]

    return raw_datafile


"""
### Convert to Observation Collection

We can now specify any required **ancillary settings**; in this case, we use the factory function for **N-way range** observations, where all signals are in the **X frequency band**. Then, all the necessary information is available to create the **observation collection** with "Mars" as main body. 

Recall that the observations were made using the MRO spacecraft, but **already corrected for Mars system barycenter**. You could consider that there might be slightly **difference between Mars itself and the system barycenter**, but since both **Deimos and Phobos are very small** (7 and 8 orders of magnitude less massive than Mars), **this difference is negligible** for the example.

An `ObservationCollection` is the useful type for Tudat to perform all its estimation functionality. You can read up on it in [the documentation](https://docs.tudat.space/en/latest/_src_user_guide/state_estimation/observation_simulation.html#creating-observations). In this case, we obtained that collection from real tracking data, but it is also possible to artificially create such a collection from a simulation or from known ephemerides, which is what we will demonstrate [below](#simulation).

"""


# Create ancillary settings. The two entries define the uplink and downlink
# frequency bands, which are also needed by frequency-dependent corrections.
ancillary_settings = observations_setup.ancillary_settings.n_way_range_ancillary_settings(
    frequency_bands=[
        observations_setup.ancillary_settings.FrequencyBands.x_band,
        observations_setup.ancillary_settings.FrequencyBands.x_band,
    ]
)

def dss_station_name(station_id):
    return f"DSS-{int(station_id)}"


def get_selected_observation_indices(raw_datafile, tracking_dataset):
    number_of_observations = len(raw_datafile.double_datamap[TrackingDataType.n_way_light_time])
    thinning_step = tracking_dataset.get("thinning_step", 1)
    if thinning_step <= 1:
        return range(number_of_observations)
    return range(0, number_of_observations, thinning_step)


def calendar_observation_times_tdb(raw_datafile, selected_indices=None):
    time_scale_converter = time_representation.default_time_scale_converter()
    dsn_positions = environment_setup.ground_station.get_approximate_dsn_ground_station_positions()
    raw_data = raw_datafile.double_datamap

    years = raw_data[TrackingDataType.year]
    months = raw_data[TrackingDataType.month]
    days = raw_data[TrackingDataType.day]
    hours = raw_data[TrackingDataType.hour]
    minutes = raw_data[TrackingDataType.minute]
    seconds = raw_data[TrackingDataType.second]
    receiving_stations = raw_data.get(TrackingDataType.dsn_receiving_station_nr, [None] * len(years))
    time_tag_delays = raw_data.get(TrackingDataType.time_tag_delay, [0.0] * len(years))
    if selected_indices is None:
        selected_indices = range(len(years))

    observation_times = []
    for idx in selected_indices:
        utc_time = time_representation.DateTime(
            int(years[idx]),
            int(months[idx]),
            int(days[idx]),
            int(hours[idx]),
            int(minutes[idx]),
            seconds[idx],
        ).to_epoch()
        if receiving_stations[idx] is None:
            earth_fixed_position = np.zeros(3)
        else:
            earth_fixed_position = dsn_positions[dss_station_name(receiving_stations[idx])]

        observation_times.append(
            time_scale_converter.convert_time(
                time_representation.TimeScales.utc_scale,
                time_representation.TimeScales.tdb_scale,
                utc_time,
                earth_fixed_position,
            ) - time_tag_delays[idx]
        )

    return observation_times


def create_manual_n_way_collection(raw_datafile, link_end_builder, tracking_dataset=None):
    raw_data = raw_datafile.double_datamap
    if tracking_dataset is None:
        selected_indices = range(len(raw_data[TrackingDataType.n_way_light_time]))
    else:
        selected_indices = get_selected_observation_indices(raw_datafile, tracking_dataset)
    selected_indices = list(selected_indices)

    values = np.asarray([raw_data[TrackingDataType.n_way_light_time][idx] for idx in selected_indices]) * constants.SPEED_OF_LIGHT
    times = calendar_observation_times_tdb(raw_datafile, selected_indices)

    grouped_observations = {}
    for selected_index, value, time in zip(selected_indices, values, times):
        link_ends = link_end_builder(raw_data, selected_index)
        link_key = tuple(sorted((str(link_end_type), str(link_end_id)) for link_end_type, link_end_id in link_ends.items()))
        if link_key not in grouped_observations:
            grouped_observations[link_key] = {
                "link_ends": link_ends,
                "observations": [],
                "times": [],
            }
        grouped_observations[link_key]["observations"].append(np.array([value]))
        grouped_observations[link_key]["times"].append(time)

    observation_sets = [
        observations.create_single_observation_set(
            observable_models_setup.model_settings.n_way_range_type,
            grouped_data["link_ends"],
            grouped_data["observations"],
            grouped_data["times"],
            links.receiver,
            ancillary_settings,
        )
        for grouped_data in grouped_observations.values()
    ]
    return observations.ObservationCollection(observation_sets)


def create_mars_barycenter_link_ends(raw_data, idx):
    return {
        links.transmitter: links.body_reference_point_link_end_id(
            "Earth", dss_station_name(raw_data[TrackingDataType.dsn_transmitting_station_nr][idx])
        ),
        links.reflector1: links.body_origin_link_end_id("Mars"),
        links.receiver: links.body_reference_point_link_end_id(
            "Earth", dss_station_name(raw_data[TrackingDataType.dsn_receiving_station_nr][idx])
        ),
    }


def create_range_observation_collection(tracking_dataset):
    raw_data = read_range_data_file(tracking_dataset)

    if tracking_dataset.get("thinning_step", 1) > 1:
        return create_manual_n_way_collection(raw_data, create_mars_barycenter_link_ends, tracking_dataset)
    return observations.create_tracking_txtfile_observation_collection(
        raw_data,
        tracking_dataset["spacecraft_name"],
        ancillary_settings=ancillary_settings,
    )


OBSERVATION_COLLECTION_CACHE = {}


def get_range_observation_collection(tracking_dataset):
    dataset_name = tracking_dataset["name"]
    if dataset_name not in OBSERVATION_COLLECTION_CACHE:
        OBSERVATION_COLLECTION_CACHE[dataset_name] = create_range_observation_collection(tracking_dataset)
    return OBSERVATION_COLLECTION_CACHE[dataset_name]


"""
Now, it is possible to **extract the observation times and values and plot them** for inspection.
The range from Earth to Mars and back oscillates between about 1.2 AU at closest approach and 5 AU when furthest apart. This is certainly within intuitive expectations for a planet at ~1.5 AU semi-major axis.

"""


"""
## Simulation

As we mentioned earlier, within this example we also aim to **mimic the loaded real observations** starting from **SPICE ephemerides**. To achieve this, the first step is to load the standard SPICE kernels into our program.
"""


spice.load_standard_kernels()


"""
## Create Bodies

We then continue to set up the environment by creating the relevant bodies and applying their default body settings. A global frame with origin at Solar System Barycenter (SSB) and J2000 orientation is chosen. For this example, we want to show the influence of adding a more precise rotation model, so a simple utility function is introduced to create the bodies.

Because the observations are N-way radio range observables, Tudat also needs a spacecraft transponder turnaround-ratio model. In this simplified example, the observations have already been corrected to Mars system barycenter, so the retransmitting spacecraft link end is represented by the `"Mars"` body. We therefore assign the default DSN turnaround-ratio function to that body's vehicle systems.

For the solar corona correction, Tudat must also know the transmitting frequency at the ground station. The original reduced text file does not contain ramp information, so we use a representative constant DSN X-band uplink frequency for all DSN stations. This keeps the example self-contained while giving the frequency-dependent corona model the correct order of magnitude.

"""


def create_bodies(use_itrf_rotation_model: bool = False) -> environment.SystemOfBodies:
    # Create default body settings
    bodies_to_create = ["Sun", "Earth", "Mars"]
    global_frame_origin = "SSB"
    global_frame_orientation = "J2000"
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create, global_frame_origin, global_frame_orientation
    )

    # Add ground stations DSN
    body_settings.get("Earth").ground_station_settings = environment_setup.ground_station.dsn_stations()
    bodies = environment_setup.create_system_of_bodies(body_settings)

    # Rotation model
    if use_itrf_rotation_model:
        body_settings.get("Earth").rotation_model_settings = environment_setup.rotation_model.gcrs_to_itrs(
            environment_setup.rotation_model.iau_2006, global_frame_orientation
        )

    # Create system of bodies
    bodies = environment_setup.create_system_of_bodies(body_settings)

    dsn_x_band_uplink_frequency = 7.2e9
    for ground_station in bodies.get_body("Earth").ground_station_list.values():
        ground_station.set_transmitting_frequency_calculator(
            environment.ConstantTransmittingFrequencyCalculator(dsn_x_band_uplink_frequency)
        )

    for body_name in ["Mars"]:
        bodies.get_body(body_name).system_models.set_default_transponder_turnaround_ratio_function()
    return bodies


bodies = create_bodies()


"""
## Create Simulated observations

To simulate observations, we need three main things:

* Observation simulators, defining which observation types need to be simulated and which linkends need to be used.
* Observation simulation settings, defining the times at which to simulate the observations.
* The system of bodies relevant for the simulation

Do check out the [documentation](https://docs.tudat.space/en/latest/_src_user_guide/state_estimation/observation_simulation.html#simulating-the-observations) for a more rigorous explanation of the technicalities.

The system of bodies was already defined above, and all the other required information is in the collection of real observations that were loaded from the data file. For the observation simulation settings, there is a convenience function that extracts the settings from the collection `observation_settings_from_collection`. Creating the simulators is slightly more involved, as we need to specify the correct link definition for every observation - recall that the measurements are made using a variety of ground station.
"""


def create_observation_model_settings(
    source_observation_collection,
    light_time_corrections=None,
    time_scale_for_observable=None,
):
    current_distinct_linkdefs = source_observation_collection.get_link_definitions_for_observables(
        observable_models_setup.model_settings.n_way_range_type
    )
    if hasattr(current_distinct_linkdefs, "values"):
        current_link_definitions = current_distinct_linkdefs.values()
    else:
        current_link_definitions = current_distinct_linkdefs

    observation_settings = []
    for link_definition in current_link_definitions:
        if time_scale_for_observable is None:
            observation_settings.append(
                observable_models_setup.model_settings.n_way_range(
                    link_definition,
                    light_time_corrections or [],
                )
            )
        else:
            observation_settings.append(
                observable_models_setup.model_settings.n_way_range(
                    link_definition,
                    light_time_corrections or [],
                    time_scale_for_observable=time_scale_for_observable,
                )
            )

    return observation_settings


def create_observations(observation_model_settings, bodies, source_observation_collection):
    # Create the observation simulators
    observation_simulators = observations_setup.observations_simulation_settings.create_observation_simulators(observation_model_settings, bodies)

    # Get the simulator settings directly from the real observations
    observation_simulation_settings = observations_setup.observations_simulation_settings.observation_settings_from_collection(source_observation_collection, bodies)

    # Simulate the observations
    simulated_observations = observations.simulate_observations(
        observation_simulation_settings, observation_simulators, bodies
    )

    return simulated_observations

bodies_rotation = create_bodies(use_itrf_rotation_model=True)

"""
## Adding light time corrections

After adjusting the rotation model, it is clear that the simulation systematically underestimates the range. This seems to behave asymptotically as the distance between Mars and Earth increases, but notice that there are no measurements in the most extreme regions. At those times, the Sun is in the way, preventing useful observations. This also indicates that the presence of the Sun influences the travel time of the light.

To account for the **relativistic effects due to the Sun**, we can add a *light time correction* to the settings of the observation model.
Doing this and once more plotting the residuals shows that **the error signal related to the Earth-Mars synodic period is removed**, leaving a residual that oscillates annually in the order **a few hundreds of meters**.

By default, the range observable is represented as an elapsed time in **TDB**. However, the loaded observations are time-tagged by DSN ground equipment, so the measured elapsed light time is an elapsed time on the ground in **UTC**. We therefore add a run case that uses the same relativistic light-time correction, but requests the range observable in UTC.

The signal also passes through the **solar corona**, where plasma causes an additional delay that becomes most relevant close to solar conjunctions. As a final correction case, we add an inverse-power-series solar corona correction using coefficients derived from Mars Express 2011 solar conjunction data. This correction is frequency-dependent, so it uses the X-band uplink and downlink settings defined in the observation ancillary settings.
"""


# Create light time corrections
light_time_correction_list = [
    observable_models_setup.light_time_corrections.first_order_relativistic_light_time_correction(["Sun"])
]

solar_corona_model_name = "Verma MEX 2006"
solar_corona_coefficients = [1.90e12]
solar_corona_positive_exponents = [2.54]

light_time_correction_list_with_corona = light_time_correction_list + [
    observable_models_setup.light_time_corrections.inverse_power_series_solar_corona_light_time_correction(
        solar_corona_coefficients,
        solar_corona_positive_exponents,
    )
]


"""
## Additional Mars Orbiter Range Data Sets

The same JPL archive contains Odyssey and MGS range data in the same
DSN-station-to-Mars-barycenter format as the MRO file. The MGS file is thinned to
one in every 100 data points to keep the example runtime manageable.

The plots below are grouped by model setting rather than by spacecraft: each
figure shows all three orbiters with a different colour per spacecraft.
"""


MULTI_DATASET_RUN_CASES = [
    {
        "name": "Simple geometric range",
        "bodies": bodies,
        "light_time_corrections": [],
        "time_scale_for_observable": None,
    },
    {
        "name": "ITRF Earth rotation model",
        "bodies": bodies_rotation,
        "light_time_corrections": [],
        "time_scale_for_observable": None,
    },
    {
        "name": "ITRF + relativistic light time (TDB observable)",
        "bodies": bodies_rotation,
        "light_time_corrections": light_time_correction_list,
        "time_scale_for_observable": None,
    },
    {
        "name": "ITRF + relativistic light time (UTC observable)",
        "bodies": bodies_rotation,
        "light_time_corrections": light_time_correction_list,
        "time_scale_for_observable": time_representation.TimeScales.utc_scale,
    },
    {
        "name": f"ITRF + relativistic light time + solar corona ({solar_corona_model_name})",
        "bodies": bodies_rotation,
        "light_time_corrections": light_time_correction_list_with_corona,
        "time_scale_for_observable": time_representation.TimeScales.utc_scale,
        "requires_solar_corona_support": True,
    },
]


SUN_RADIUS = 695700.0e3


def distance_from_origin_to_segment(first_point, second_point):
    segment = second_point - first_point
    segment_norm_squared = np.dot(segment, segment)
    if segment_norm_squared == 0.0:
        return np.linalg.norm(first_point)

    segment_fraction = -np.dot(first_point, segment) / segment_norm_squared
    segment_fraction = np.clip(segment_fraction, 0.0, 1.0)
    closest_point = first_point + segment_fraction * segment
    return np.linalg.norm(closest_point)


def compute_signal_path_solar_distances(observation_times):
    solar_distances = []
    for observation_time in observation_times:
        earth_position = spice.get_body_cartesian_state_at_epoch(
            "Earth", "Sun", "J2000", "NONE", observation_time
        )[:3]
        mars_position = spice.get_body_cartesian_state_at_epoch(
            "Mars", "Sun", "J2000", "NONE", observation_time
        )[:3]
        solar_distances.append(distance_from_origin_to_segment(earth_position, mars_position) / SUN_RADIUS)

    return np.asarray(solar_distances)


def simulate_tracking_dataset(tracking_dataset, run_case, print_summary=True):
    current_collection = get_range_observation_collection(tracking_dataset)
    current_observations = np.asarray(current_collection.concatenated_observations)
    current_times = np.asarray(current_collection.concatenated_times)
    current_times_year = current_times / constants.JULIAN_YEAR + 2000

    current_model_settings = create_observation_model_settings(
        current_collection,
        run_case["light_time_corrections"],
        time_scale_for_observable=run_case["time_scale_for_observable"],
    )
    current_simulated_observations = create_observations(
        current_model_settings,
        run_case["bodies"],
        current_collection,
    )
    current_residuals = np.asarray(current_simulated_observations.concatenated_observations) - current_observations

    rms_residual = np.sqrt(np.mean(current_residuals ** 2))
    mean_residual = np.mean(current_residuals)
    median_residual = np.median(current_residuals)

    if print_summary:
        print(
            f"{run_case['name']} | {tracking_dataset['name']}: "
            f"{current_residuals.size} points, RMS={rms_residual:.3f} m, "
            f"mean={mean_residual:.3f} m, median={median_residual:.3f} m"
        )

    return {
        "dataset": tracking_dataset,
        "times_year": current_times_year,
        "solar_distances": compute_signal_path_solar_distances(current_times),
        "residuals": current_residuals,
        "rms": rms_residual,
        "mean": mean_residual,
        "median": median_residual,
    }


mission_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
mission_color_by_name = {
    dataset["name"]: mission_colors[dataset_index % len(mission_colors)]
    for dataset_index, dataset in enumerate(ACTIVE_TRACKING_DATASETS)
}
run_case_results_by_name = {}

for run_case in MULTI_DATASET_RUN_CASES:
    run_case_results = [
        result
        for tracking_dataset in ACTIVE_TRACKING_DATASETS
        if (result := simulate_tracking_dataset(tracking_dataset, run_case)) is not None
    ]
    run_case_results_by_name[run_case["name"]] = run_case_results

    if not run_case_results:
        continue

    plt.figure(figsize=(15, 7))
    for result in run_case_results:
        dataset = result["dataset"]
        plt.plot(
            result["times_year"],
            result["residuals"],
            ".",
            markersize=2,
            color=mission_color_by_name[dataset["name"]],
            label=(
                f"{dataset['name']} "
                f"(RMS={result['rms']:.1f} m, mean={result['mean']:.1f} m, "
                f"median={result['median']:.1f} m)"
            ),
        )

    plt.title(run_case["name"])
    plt.xlabel("Time [year]")
    plt.ylabel("Residual [m]")
    plt.grid(True)
    plt.axhline(0.0, color="k", zorder=0)
    plt.legend(markerscale=4, fontsize="small")
    plt.tight_layout()
    plt.show()


def compute_zero_mean_rms(residuals):
    zero_mean_residuals = residuals - np.mean(residuals)
    return np.sqrt(np.mean(zero_mean_residuals ** 2))


def compute_lomb_scargle_psd(times_year, residuals, number_of_frequencies=2000):
    times_year = np.asarray(times_year)
    residuals = np.asarray(residuals) - np.mean(residuals)
    time_span = np.max(times_year) - np.min(times_year)
    sorted_times = np.sort(times_year)
    time_steps = np.diff(sorted_times)
    median_time_step = np.median(time_steps[time_steps > 0.0])

    minimum_frequency = 1.0 / time_span
    maximum_frequency = min(24.0, 0.5 / median_time_step)
    frequencies = np.linspace(minimum_frequency, maximum_frequency, number_of_frequencies)
    angular_frequencies = 2.0 * np.pi * frequencies

    if scipy_signal is not None:
        power = scipy_signal.lombscargle(
            times_year,
            residuals,
            angular_frequencies,
            normalize=True,
        )
    else:
        power = []
        for angular_frequency in angular_frequencies:
            phase_shift = 0.5 * np.arctan2(
                np.sum(np.sin(2.0 * angular_frequency * times_year)),
                np.sum(np.cos(2.0 * angular_frequency * times_year)),
            ) / angular_frequency
            cosine_terms = np.cos(angular_frequency * (times_year - phase_shift))
            sine_terms = np.sin(angular_frequency * (times_year - phase_shift))
            power.append(
                0.5
                * (
                    np.sum(residuals * cosine_terms) ** 2 / np.sum(cosine_terms ** 2)
                    + np.sum(residuals * sine_terms) ** 2 / np.sum(sine_terms ** 2)
                )
                / np.var(residuals)
                / residuals.size
            )
        power = np.asarray(power)

    return frequencies, power


def log_bin_spectrum(frequencies, power, number_of_bins=80):
    bin_edges = np.logspace(np.log10(frequencies[0]), np.log10(frequencies[-1]), number_of_bins + 1)
    binned_frequencies = []
    binned_power = []

    for lower_edge, upper_edge in zip(bin_edges[:-1], bin_edges[1:]):
        bin_indices = (frequencies >= lower_edge) & (frequencies < upper_edge)
        if not np.any(bin_indices):
            continue
        binned_frequencies.append(np.exp(np.mean(np.log(frequencies[bin_indices]))))
        binned_power.append(np.median(power[bin_indices]))

    return np.asarray(binned_frequencies), np.asarray(binned_power)


final_run_cases = MULTI_DATASET_RUN_CASES[-2:]
fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True, sharey=True)

for axis, run_case in zip(axes, final_run_cases):
    for result in run_case_results_by_name[run_case["name"]]:
        dataset = result["dataset"]
        axis.plot(
            result["solar_distances"],
            result["residuals"],
            ".",
            markersize=2,
            color=mission_color_by_name[dataset["name"]],
            label=dataset["name"],
        )

    axis.set_title(run_case["name"])
    axis.set_ylabel("Residual [m]")
    axis.grid(True)
    axis.axhline(0.0, color="k", zorder=0)
    axis.legend(markerscale=4, fontsize="small")

axes[-1].set_xlabel("Minimum signal-path solar distance [$R_\\odot$]")
fig.suptitle("Residuals as a function of signal-path solar distance")
fig.tight_layout()
plt.show()


corrected_run_case = MULTI_DATASET_RUN_CASES[-1]
corrected_results = run_case_results_by_name[corrected_run_case["name"]]

plt.figure(figsize=(15, 7))
for result in corrected_results:
    dataset = result["dataset"]
    residual_mean = np.mean(result["residuals"])
    zero_mean_residuals = result["residuals"] - residual_mean
    zero_mean_rms = compute_zero_mean_rms(result["residuals"])
    plt.plot(
        result["times_year"],
        zero_mean_residuals,
        ".",
        markersize=2,
        color=mission_color_by_name[dataset["name"]],
        label=(
            f"{dataset['name']} "
            f"(mean={residual_mean:.2f} m, zero-mean RMS={zero_mean_rms:.2f} m)"
        ),
    )

plt.title(f"Mean-subtracted residuals after solar corona correction ({solar_corona_model_name})")
plt.xlabel("Time [year]")
plt.ylabel("Residual minus spacecraft mean [m]")
plt.grid(True)
plt.axhline(0.0, color="k", zorder=0)
plt.legend(markerscale=4, fontsize="small")
plt.tight_layout()
plt.show()


combined_times_year = []
combined_zero_mean_residuals = []
for result in corrected_results:
    post_2000_indices = result["times_year"] >= 2000.0
    if not np.any(post_2000_indices):
        continue

    zero_mean_residuals = result["residuals"] - np.mean(result["residuals"])
    combined_times_year.append(result["times_year"][post_2000_indices])
    combined_zero_mean_residuals.append(zero_mean_residuals[post_2000_indices])

if combined_times_year:
    combined_times_year = np.concatenate(combined_times_year)
    combined_zero_mean_residuals = np.concatenate(combined_zero_mean_residuals)
    chronological_order = np.argsort(combined_times_year)
    combined_times_year = combined_times_year[chronological_order]
    combined_zero_mean_residuals = combined_zero_mean_residuals[chronological_order]

    frequencies, psd_power = compute_lomb_scargle_psd(
        combined_times_year,
        combined_zero_mean_residuals,
    )
    binned_frequencies, binned_psd_power = log_bin_spectrum(frequencies, psd_power)
    binned_asd = np.sqrt(binned_psd_power)

    plt.figure(figsize=(15, 7))
    plt.loglog(
        binned_frequencies,
        binned_asd,
        linewidth=2.0,
        label="Smoothed LS ASD",
    )
    plt.title("Smoothed Lomb-Scargle ASD of combined post-2000 corrected residuals")
    plt.xlabel("Frequency [cycles/year]")
    plt.ylabel("Normalized ASD")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()
