# %% [markdown]
"""
# MRO Range Data Estimation

Copyright (c) 2010-2022, Delft University of Technology. All rights reserved. This file is part of the Tudat. Redistribution and use in source and binary forms, with or without modification, are permitted exclusively under the terms of the Modified BSD license. You should have received a copy of the license with this file. If not, please or visit: http://tudat.tudelft.nl/LICENSE.
"""

## Context
"""
In this example, we will read from a file containing range measurements to the Mars Reconnaissance Orbiter (MRO) and compare these to simulated range measurements.
"""

## Prerequisites
"""
To run this example, make sure to download [https://ssd.jpl.nasa.gov/dat/planets/mrorange2006-2013.txt](the data file) and store it in a subfolder called [./data](./data) (or adjust the filename below to the absolute path of your choice). You can get the MRO data from

```bash
mkdir data
cd data
wget https://ssd.jpl.nasa.gov/dat/planets/mrorange2006-2013.txt
```

Then set the filename accordingly
"""

TRACKING_FNAME = "./data/mrorange2006-2013.txt"

# %% [markdown]
## Import statements
"""
Typically - in the most pythonic way - all required modules are imported at the very beginning.

Some standard modules are first loaded: `numpy` and `matplotlib.pyplot`.

Then, the different modules of `tudatpy` that will be used are imported. Most notably, the `estimation`, `estimation_setup`, and `observations` modules will be used and demonstrated within this example.
"""

# General imports
import math
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("bmh")
plt.rcParams.update({"axes.facecolor": "#ffffff"})

# Tudatpy imports
from tudatpy import util
from tudatpy import data
from tudatpy import constants
from tudatpy.interface import spice
from tudatpy.numerical_simulation import environment_setup, environment
from tudatpy.numerical_simulation import estimation, estimation_setup, propagation_setup
from tudatpy.numerical_simulation.estimation_setup import observation


# %%
## Orbit Simulation
"""
> TODO: Elaborate
"""

# Load spice kernels
spice.load_standard_kernels()

# %%
## Read the observations
"""
> TODO: Elaborate
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

raw_datafile = data.read_tracking_txt_file(TRACKING_FNAME, file_columns)
double_data = raw_datafile.double_datamap

"""
> Elaborate
"""

ancillary_settings = observation.n_way_range_ancilliary_settings(frequency_bands=[observation.FrequencyBands.x_band])

observations = observation.create_tracking_txtfile_observation_collection(
    raw_datafile, "Mars", ancillary_settings=ancillary_settings
)

times = np.array(observations.concatenated_times)
linkdef_ids = observations.concatenated_link_definition_ids
observation_vals = observations.concatenated_observations

linkdefs = observations.get_link_definitions_for_observables(observation.n_way_range_type)
observation_settings_list = [observation.n_way_range(linkdef) for linkdef in linkdefs]

idx_time_sorted = np.argsort(times)

plt.figure()
plt.title("Observed Two-way Range to Mars")
plt.plot(times, observation_vals, ".")
plt.xlabel("Time [s]")
plt.ylabel("Range [m]")


# %%
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

    # Rotation model
    if use_itrf_rotation_model:
        body_settings.get("Earth").rotation_model_settings = environment_setup.rotation_model.gcrs_to_itrs(
            environment_setup.rotation_model.iau_2006, global_frame_orientation
        )

    # Create system of bodies
    bodies = environment_setup.create_system_of_bodies(body_settings)
    return bodies


# %%
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


# %%
## Simple Simulation
print("Simple Simulation")
bodies = create_bodies(use_itrf_rotation_model=False)
simulated_observations = create_simulated_observations(
    bodies, linkdefs, linkdef_ids, use_relativistic_correction=False, use_corona_correction=False
)
simulated_observation_vals_simple = simulated_observations.concatenated_observations

# %%
## Adding Rotation Model
print("Adding ITRF Rotation Model")
bodies = create_bodies(use_itrf_rotation_model=True)
simulated_observations = create_simulated_observations(
    bodies, linkdefs, linkdef_ids, use_relativistic_correction=False, use_corona_correction=False
)
simulated_observation_vals_itrf = simulated_observations.concatenated_observations


# %%
## Adding Relativistic Correction
print("Adding Relativistic Correction")
bodies = create_bodies(use_itrf_rotation_model=True)
simulated_observations = create_simulated_observations(
    bodies, linkdefs, linkdef_ids, use_relativistic_correction=True, use_corona_correction=False
)
simulated_observation_vals_rel = simulated_observations.concatenated_observations


# # %%
# ## Adding Corona Correction
# FIXME: This is not working
# """
# > Elaborate. Don't run separately?
# """

# print("Adding Solar Corona Correction")
# bodies = create_bodies(use_itrf_rotation_model=True)
# for gs in bodies.get("Earth").ground_station_list.values():
#     frequency_interpolator = environment.ConstantFrequencyInterpolator(7.1e9) # 8.4 GHz downlink
#     gs.set_transmitting_frequency_calculator(frequency_interpolator)


# bodies.get("Mars").system_models.set_transponder_turnaround_ratio({(observation.FrequencyBands.s_band, observation.FrequencyBands.x_band): 1.66})

# simulated_observations = create_simulated_observations(
#     bodies, linkdefs, linkdef_ids, use_relativistic_correction=True, use_corona_correction=True
# )
# # simulated_observation_vals_corona = simulated_observations.concatenated_observations

# %%
## Compare Observations to simulations
"""
> Elaborate
"""

fig, ax = plt.subplots(4, 1, figsize=(10, 10))

# Plot the observed and simulated range
ax[0].plot(
    times / constants.JULIAN_YEAR,
    observation_vals / constants.ASTRONOMICAL_UNIT,
    "<",
    label="Observed Range",
)
ax[0].plot(
    times / constants.JULIAN_YEAR,
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
ax[1].scatter(times / constants.JULIAN_YEAR, residuals_simple, marker=".")

residuals_itrf = observation_vals - simulated_observation_vals_itrf
ax[2].set_title("Adding ITRF rotation model")
ax[2].scatter(times / constants.JULIAN_YEAR, residuals_itrf, marker=".")

residuals_rel = observation_vals - simulated_observation_vals_rel
ax[3].set_title("Adding first order relativistic correction")
ax[3].scatter(times / constants.JULIAN_YEAR, residuals_rel, marker=".")


[ax_i.axhline(0, color="k", linestyle="-", zorder=0) for ax_i in ax[1:]]
[ax_i.set_ylabel("Residual [m]") for ax_i in ax[1:]]
ax[-1].set_xlabel("Time [yr]")
fig.align_labels()
plt.tight_layout()

# %% [markdown]
# ## Estimate Parameters
#


plt.show()
