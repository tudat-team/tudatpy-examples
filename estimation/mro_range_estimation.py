# 
# MRO Range Data
"""

Copyright (c) 2010-2022, Delft University of Technology. All rights reserved. This file is part of the Tudat. Redistribution and use in source and binary forms, with or without modification, are permitted exclusively under the terms of the Modified BSD license. You should have received a copy of the license with this file. If not, please or visit: http://tudat.tudelft.nl/LICENSE.


"""

## Context
"""
With this example, we will explore how to load tracking observations into Tudat so that they can be used for estimation purposes. We then show how to simulate the same measurements and reduce the observation residuals by adding a more accurate rotation model and relativistic corrections.

The example uses range measurements from the Mars Reconnaissance Orbiter (MRO) with a variety of Deep Space Network (DSN) ground stations. The data is already corrected so that it represents the two-way light time between those ground stations and Mars system barycenter. To simulate the observations, we start from SPICE ephemerides and obtain residuals in the order of a few hundred meters.

"""

## Prerequisites
"""
To run this example, make sure to download [the data file](https://ssd.jpl.nasa.gov/dat/planets/mrorange2006-2013.txt) from the NASA JPL small-body database and store it in a subfolder called [./data](./data) (or adjust the filename below to the absolute path of your choice). If you are using a terminal, you could get the MRO data by running the follownig commands from the current directory:

```bash
mkdir data
(cd data && curl -O https://ssd.jpl.nasa.gov/dat/planets/mrorange2006-2013.txt)
```

Then set the filename of the tracking data accordingly
"""

# Set the filename of the data file
TRACKING_FNAME = "./data/mrorange2006-2013.txt"

# Check if you have the file in the correct place
try:
    with open(TRACKING_FNAME, "r") as f:
        pass
except FileNotFoundError:
    print(f"FILE {TRACKING_FNAME} NOT FOUND. Make sure you have downloaded the data file and stored in the correct place.")
    exit(1)


## Import statements
"""
Typically - in the most pythonic way - all required modules are imported at the very beginning.

Some standard modules are first loaded: `numpy` and `matplotlib.pyplot`.

Then, the different modules of `tudatpy` that will be used are imported. Most notably, some elements of the `observation`, `estimation` and `estimation_setup` modules will be used and demonstrated within this example.
"""

# General imports
import numpy as np
import matplotlib.pyplot as plt

# Tudatpy imports
from tudatpy import data
from tudatpy import constants
from tudatpy.interface import spice
from tudatpy.numerical_simulation import environment_setup, environment
from tudatpy.numerical_simulation import estimation, estimation_setup
from tudatpy.numerical_simulation.estimation_setup import observation


## Read the observations
"""

To investigate the observation data, we will start by reading the observations into a format that is useful for Tudat. 

After inspecting the data file, we can see that it contains columns related to the spacecraft id, the DSN stations involved with the measurement, a date in UTC format, the round-trip light time and a correction term. We can use the `read_tracking_txt_file` function to read that raw data into an intermediate format that takes care of appropriate unit conversions for known column identifiers

The file columns specified here are all known to Tudat, and can be used to process the observation (see [LINK]() for a complete list of available column types). If a file contains additional columns, they can be specified with any unknown string and the `read_tracking_txt_file` function will load them in string format without using them further. If needed, these can be accessed as a dictionary through `raw_datafile.raw_datamap`.

"""

file_columns = [
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
    "light_time_measurement_delay_microseconds",
]

raw_datafile = data.read_tracking_txt_file(
    file_name=TRACKING_FNAME, column_types=file_columns, comment_symbol="#", value_separators=",:\t "
)


### Convert to Observation Collection
"""

We can then specify any required ancillary settings; in this case we use the factory function for N-way range observations, where all signals are in the X-band frequency range. Then, all the necessary information is available to create the observation collection with "Mars" as main body. Recall that the observations were made using the MRO spacecraft, but already corrected for Mars system barycenter. You could consider that there might be slightly difference between Mars itself and the system barycenter, but since both Deimos and Phobos are very small (7 and 8 orders of magnitude less massive than Mars), this difference is negligible for the example.

An `ObservationCollection` is the useful type for Tudat to perform all its estimation functionality. You can read up on it in [the documentation](https://docs.tudat.space/en/latest/_src_user_guide/state_estimation/observation_simulation.html#creating-observations). In this case, we obtained that collection from real tracking data, but it is also possible to artifically create such a collection from a simiulation or from known ephemerides, which is what we will demonstrate [below](#simulation). 
"""

# Create ancillary settings
ancillary_settings = observation.n_way_range_ancilliary_settings(frequency_bands=[observation.FrequencyBands.x_band])

# Create the observation collection
observations = observation.create_tracking_txtfile_observation_collection(
    raw_datafile, "Mars", ancillary_settings=ancillary_settings
)


# We can then extract the observation times and values and plot them for inspection. Notice that using a line plot results in seemingly erratice behaviour. This is because the concatenated observations are not ordered chronologically but rather per set of link ends (in this case there are measurements from a variety of DSN stations which are grouped together). You could sort the results if necessary but a scatter plot is sufficient.
# 
# The range from Earth to Mars and back oscillates between about 1.2 AU at closest approach and 5 AU when furthest apart. This is certainly within intuitive expectations for a planet at ~1.5 AU semi-major axis.

observation_times = np.array(observations.concatenated_times)
observation_vals = observations.concatenated_observations


plt.figure()
plt.title("Observed two-way range to Mars")
plt.plot(observation_times, observation_vals / constants.ASTRONOMICAL_UNIT, "k--", linewidth=0.3, label="line plot")
plt.plot(observation_times, observation_vals / constants.ASTRONOMICAL_UNIT, ".", label="Scatter plot")
plt.xlabel("Time [s]")
plt.ylabel("Two-way range [AU]")
plt.legend()
plt.grid("on")
plt.show()


# 

linkdef_ids = observations.concatenated_link_definition_ids
distinct_linkdefs = observations.get_link_definitions_for_observables(observation.n_way_range_type)
observation_settings_list = [observation.n_way_range(linkdef) for linkdef in distinct_linkdefs]


## Simulation
"""

We will now aim to mimic the loaded observations by calculating them from SPICE ephemerides.  
> Elaborate

To achieve that, the first step is to load the standard SPICE kernels into our program.
"""

spice.load_standard_kernels()


# We then continue to set up the environment by creating the relavant bodies and applying their default body settings. A global frame with origin at Solar System Barycenter (SSB) and J2000 orientation is chosen.

# Create default body settings for selected celestial bodies
bodies_to_create = ["Sun", "Earth", "Mars"]
global_frame_origin = "SSB"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation
)

# Add ground stations DSN
body_settings.get("Earth").ground_station_settings = environment_setup.ground_station.dsn_stations()
bodies = environment_setup.create_system_of_bodies(body_settings)


## Create Bodies
"""
> TODO: Elaborate
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
    return bodies


## Create Observation Simulator
"""
> From Spice
> Elaborate
"""

def create_simulated_observations(
    bodies, link_definition_list, linkdef_ids, use_relativistic_correction=False, use_corona_correction=False
):

    #  Create light time corrections
    light_time_correction_list = list()

    if use_relativistic_correction:
        light_time_correction_list.append(
            estimation_setup.observation.first_order_relativistic_light_time_correction(["Sun"])
        )

    if use_corona_correction:
        light_time_correction_list.append(
            estimation_setup.observation.inverse_power_series_solar_corona_light_time_correction(
                coefficients=[7.8207e-06], positive_exponents=[2.0], delay_coefficient=40.3, sun_body_name="Sun"
            )
        )

    observation_model_settings = list()
    for linkdef_id in linkdef_ids:
        observation_model_settings.append(
            estimation_setup.observation.n_way_range(link_definition_list[linkdef_id], light_time_correction_list)
        )

    observation_simulators = estimation_setup.create_observation_simulators(observation_model_settings, bodies)
    observation_simulation_settings = estimation_setup.observation.observation_settings_from_collection(observations)

    simulated_observations = estimation.simulate_observations(
        observation_simulation_settings, observation_simulators, bodies
    )
    return simulated_observations


print("Simple Simulation")
bodies = create_bodies(use_itrf_rotation_model=False)
simulated_observations = create_simulated_observations(
    bodies, distinct_linkdefs, linkdef_ids, use_relativistic_correction=False, use_corona_correction=False
)
simulated_observation_vals_simple = simulated_observations.concatenated_observations


## Adding Rotation Model
print("Adding ITRF Rotation Model")
bodies = create_bodies(use_itrf_rotation_model=True)
simulated_observations = create_simulated_observations(
    bodies, distinct_linkdefs, linkdef_ids, use_relativistic_correction=False, use_corona_correction=False
)
simulated_observation_vals_itrf = simulated_observations.concatenated_observations


print("Adding Relativistic Correction")
bodies = create_bodies(use_itrf_rotation_model=True)
simulated_observations = create_simulated_observations(
    bodies, distinct_linkdefs, linkdef_ids, use_relativistic_correction=True, use_corona_correction=False
)
simulated_observation_vals_rel = simulated_observations.concatenated_observations



# # %%
## Adding Corona Correction
"""
FIXME: This is not working
> Elaborate. Don't run separately?
"""

# print("Adding Solar Corona Correction")
# bodies = create_bodies(use_itrf_rotation_model=True)
# for gs in bodies.get("Earth").ground_station_list.values():
#     frequency_interpolator = environment.ConstantFrequencyInterpolator(7.1e9) # 8.4 GHz downlink
#     gs.set_transmitting_frequency_calculator(frequency_interpolator)


# bodies.get("Mars").system_models.set_transponder_turnaround_ratio({(observation.FrequencyBands.s_band, observation.FrequencyBands.x_band): 1.66})

# simulated_observations = create_simulated_observations(
#     bodies, linkdefs, linkdef_ids, use_relativistic_correction=True, use_corona_correction=True
# )
# simulated_observation_vals_corona = simulated_observations.concatenated_observations
"""
"""

## Compare Observations to simulations
"""
> Elaborate

"""

fig, ax = plt.subplots(4, 1, figsize=(10, 10))

# Plot the observed and simulated range
ax[0].plot(
    observation_times / constants.JULIAN_YEAR,
    observation_vals / constants.ASTRONOMICAL_UNIT,
    "<",
    label="Observed Range",
)
ax[0].plot(
    observation_times / constants.JULIAN_YEAR,
    simulated_observation_vals_simple / constants.ASTRONOMICAL_UNIT,
    ".",
    label="Simulated Range",
)
ax[0].set_ylabel("2-way Range [AU]")
ax[0].legend()
ax[0].set_title("Observations")

# Plot the residuals

residuals_simple = observation_vals - simulated_observation_vals_simple
ax[1].set_title("Simple Simulation")
ax[1].scatter(observation_times / constants.JULIAN_YEAR, residuals_simple, marker=".")

residuals_itrf = observation_vals - simulated_observation_vals_itrf
ax[2].set_title("Adding ITRF rotation model")
ax[2].scatter(observation_times / constants.JULIAN_YEAR, residuals_itrf, marker=".")

residuals_rel = observation_vals - simulated_observation_vals_rel
ax[3].set_title("Adding first order relativistic correction")
ax[3].scatter(observation_times / constants.JULIAN_YEAR, residuals_rel, marker=".")


[ax_i.axhline(0, color="k", linestyle="-", zorder=0) for ax_i in ax[1:]]
[ax_i.set_ylabel("Residual [m]") for ax_i in ax[1:]]
ax[-1].set_xlabel("Time [yr]")
fig.align_labels()

plt.grid("on")
plt.tight_layout()

plt.show()


## Estimate Parameters
"""
None
"""

plt.show()

