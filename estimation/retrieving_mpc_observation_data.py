"""
# Retrieving Observations From the Minor Planet Centre
## Objectives
The [Minor Planet Centre](https://www.minorplanetcenter.net/) (MPC) provides positional elements and observation data for minor planets, comets and outer irregular natural satellites of the major planets. Tudat's `BatchMPC` class allows for the retrieval and processing of observational data for these objects. 

This example highlights the complete functionality of the `BatchMPC` class. The [Estimation with MPC](estimation_with_mpc.ipynb) example showcases how to perform an estimation using MPC observations, but **we recommend going through this example first**.



MPC receives and stores observations from observatories across the world. These are optical observations in a **Right Ascension (RA)** and **Declination (DEC)** format which are processed into an **Earth-inertial J2000 format**. Objects are all assigned a unique minor-planet **designation number** (see examples below), while **comets use a distinct designation**. Larger objects are often also given a name (only about 4% have been given a name currently). Similarly, observatories are also assigned a **unique 3-symbol code**.

The following asteroids will be used in the example:

- [433 Eros](https://en.wikipedia.org/wiki/433_Eros) (also the main focus of the [Estimation with MPC](estimation_with_mpc.ipynb) example)
- [238 Hypatia](https://en.wikipedia.org/wiki/238_Hypatia)
- [329 Svea](https://en.wikipedia.org/wiki/329_Svea)

"""

"""
### Import statements
In this example we do not perform an estimation, as such we only need the `BatchMPC` class from `data_input.tracking_data.mpc`, `environment_setup` and `observation` to convert our observations to Tudat and optionally datetime to filter our batch. We will also use the **Tudat Horizons** interface to compare observation output and load the standard `SPICE` kernels.
"""


from tudatpy.data_input.tracking_data.mpc import BatchMPC
from tudatpy.data_input.environment_data import spice
from tudatpy.astro import time_representation
from tudatpy.dynamics import environment, environment_setup
from tudatpy.dynamics import propagation_setup, parameters_setup, simulator
from tudatpy.estimation import observable_models_setup, observable_models, observations_setup, observations, estimation_analysis


from tudatpy.data_input.environment_data.horizons import HorizonsQuery

from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt


def utc_seconds_to_tdb(utc_seconds):
    time_scale_converter = time_representation.default_time_scale_converter()
    return [
        time_scale_converter.convert_time(
            input_scale=time_representation.utc_scale,
            output_scale=time_representation.tdb_scale,
            input_value=float(epoch),
        )
        for epoch in utc_seconds
    ]


# Load spice kernels
spice.load_standard_kernels()


"""
### Retrieval
"""

"""
We initialise a `BatchMPC` object, create a list with the objects we want and use `.get_observations()` to retrieve the observations. `.get_observations()` uses [astroquery](https://astroquery.readthedocs.io/en/latest/mpc/mpc.html) to retrieve data from MPC and requires an internet connection. The observations are cached for faster retrieval in subsequent runs. The `BatchMPC` object removes duplicates if `.get_observations()` is ran twice.

Tudat's estimation tools allow for multiple Objects to be analysed at the same time. `BatchMPC`  can process multiple objects into a single observation collection automatically. For now lets retrieve the observations for Eros and Svea. `BatchMPC`  uses MPC codes for objects and observatories. To get an overview of the batch we can use the `summary()` method. Let's also get some details on some of the observatories that retrieved the data using the `observatories_table()` method.
"""


asteroid_MPC_codes = [433, 329] # Eros and Svea

batch1 = BatchMPC()

batch1.get_observations(asteroid_MPC_codes)

batch1.summary()
print(batch1.observatories_table(only_in_batch=True, only_space_telescopes=False, include_positions=False))
print("Space Telescopes:")
print(batch1.observatories_table(only_in_batch=True, only_space_telescopes=True, include_positions=False))


"""
We can also directly have a look at the the observations themselves. For example, lets take a look at the first and final observations from TESS and WISE. The table property allows for read only access to the observations in pandas dataframe format. 
"""


obs_by_TESS = batch1.table.query("observatory == 'C57'").loc[:, ["number", "epoch_seconds_UTC", "RA", "DEC"]]
obs_by_WISE = batch1.table.query("observatory == 'C51'").loc[:, ["number", "epoch_seconds_UTC", "RA", "DEC"]]

print("Initial and Final Observations by TESS")
print(obs_by_TESS.iloc[[0, -1]] if not obs_by_TESS.empty else "No TESS observations in this batch.")
print("Initial and Final Observations by WISE")
print(obs_by_WISE.iloc[[0, -1]] if not obs_by_WISE.empty else "No WISE observations in this batch.")


"""
### Filtering
"""

"""
From the summary we can see that even the first observations from the 1890s are included. This is not ideal. We might want to exclude some observatories. To fix this we can use the `.filter()` method. Dates can be filtered using the standard seconds since J2000 TDB format or through python's datetime standard library in UTC for simplicity. Additionally, specific bands can be selected and observatories can explicitly be included or excluded. The `.filter()` method alters the original batch in place, an alternative is shown in the Additional Features section.
"""


observatories_to_exclude = ["000", "C59"] # chosen as an example

print(f"Size before filter: {batch1.size}")
batch1.filter(observatories_exclude=observatories_to_exclude, epoch_start=datetime(2018, 1, 1), epoch_end=746013855.0)
print(f"Size after filter: {batch1.size}")

batch1.summary()


"""
### Set up the system of bodies
A **system of bodies** must be created to keep observatories' positions consistent with Earth's shape model and to allow the attachment of these observatories to Earth. For the purposes of this example, we keep it as simple as possible. See the [Estimation with MPC](estimation_with_mpc.ipynb) for a more complete setup and explanation appropriate for estimation. For our bodies, we only use **Earth and the Sun**. We set our origin to `"SSB"`, the solar system barycenter. We use the default body settings from the `SPICE` kernel to initialise the planet and use it to create a system of bodies. This system of bodies is used when converting the loaded tracking data to an observation collection.
"""


bodies_to_create = ["Sun", "Earth"]

# Create default body settings
global_frame_origin = "SSB"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation)
body_settings.get("Earth").ground_station_settings = (
    environment_setup.ground_station.optical_telescope_stations()
)
for body_name in batch1.MPC_objects:
    body_settings.add_empty_settings(str(body_name))

# Create system of bodies
bodies = environment_setup.create_system_of_bodies(body_settings)


"""
### Retrieve Observation Collection

"""

"""
Now that our batch is ready, we can transform it to Tudat tracking-data objects and then create an `ObservationCollection`.

This does the following for us:

1. Creates an empty body for each minor planet with their MPC code as a name.
2. Adds this body to the system of bodies inputted to the method.
3. Retrieves the global position of the terrestrial observatories in the batch and adds these stations to the Tudat environment.
4. Creates link definitions between each unique terrestrial observatory/ minor planet combination in the batch.
5. (Optionally) creates a link definition between each space telescope / minor planet combination in the batch. This requires the corresponding spacecraft bodies to be present in the environment.
6. Creates a `SingleObservationSet` object for each unique link that includes all observations for that link.
7. Returns an `ObservationCollection` object.

If our batch includes space telescopes like WISE and TESS, their Tudat bodies must be available when the observation collection is created. The additional features section shows an example of how to add a spacecraft body.
"""


tracking_data, supplementary_data = batch1.to_tracking_dataset()
observations.set_tracking_supplementary_data_in_bodies(bodies, supplementary_data)
observation_collection = observations.create_observation_collection_from_tracking_data(tracking_data, bodies)


"""
The names of the bodies added to the system of bodies object as well as the dates of the oldest and latest observations can be retrieved from the batch:
"""


epoch_start = min(utc_seconds_to_tdb(batch1.table.epoch_seconds_UTC)) # in seconds since J2000 TDB (Tudat default)
epoch_end = max(utc_seconds_to_tdb(batch1.table.epoch_seconds_UTC))
object_names = batch1.MPC_objects


"""
We can now retrieve the links from the `ObservationCollection` and create settings for these links. This is where link biases would be set, for now we just keep the settings default.
"""


observation_settings_list = list()

link_list = list(
    observation_collection.get_link_definitions_for_observables(
        observable_type=observable_models_setup.model_settings.angular_position_type
    )
)

for link in link_list:
    # add optional bias settings
    observation_settings_list.append(
        observable_models_setup.model_settings.angular_position(link, bias_settings=None)
    )


"""
With the `observation_collection` and `observation_settings_list` ready, we have all the observation inputs we need to perform an estimation.
"""

"""
### Comparing to JPL Horizons Interpolated RA and DEC
The **Horizons Ephemeris API** provides interpolated RA and DEC values for many objects in the solar system. Tudat includes an interface for the JPL Horizons system. Please note that **these are not real observations**, but are instead based on ephemerides. 

As validation, let's compare these interpolated RA and DEC to MPC's values for **329 Svea**:
"""


# Let's simplify by using only 329 Svea and removing observations from space telescopes
target = "329"
target_horizons = target + ";" # ; specifies minor bodies

batch_eros = BatchMPC()
batch_eros.get_observations([target])
batch_eros.filter(
    epoch_start=datetime(2018, 1, 1),
    observatories_exclude=["C51", "C57", "C59"],
)

# Retrieve MPC observation times, RA and DEC
batch_times = utc_seconds_to_tdb(batch_eros.table.epoch_seconds_UTC)
batch_times_utc = batch_eros.table.epoch_seconds_UTC.to_list()
batch_RA = batch_eros.table.RA
batch_DEC = batch_eros.table.DEC

# Create Horizons query, see Horizons Documentation for more info.
hypatia_horizons_query = HorizonsQuery(
    query_id=target_horizons,
    location="500@399",  # geocenter @ Earth
    epoch_list=batch_times,
    extended_query=True,
)

# retrieve JPL observations
jpl_observations = hypatia_horizons_query.interpolated_observations()
jpl_RA = jpl_observations[:, 1]
jpl_DEC = jpl_observations[:, 2]

max_diff_RA = np.abs(jpl_RA - batch_RA).max()
max_diff_DEC = np.abs(jpl_DEC - batch_DEC).max()
print("Maximum difference between Interpolated Horizons data and MPC observations:")
print(f"Right Ascension: {np.round(max_diff_RA, 10)} rad")
print(f"Declination: {np.round(max_diff_DEC, 10)} rad")

# create plot
fig, (ax_ra, ax_dec) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

ax_ra.scatter(batch_times_utc, (jpl_RA - batch_RA), marker="+")
ax_dec.scatter(batch_times_utc, (jpl_DEC - batch_DEC), marker="+")

ax_ra.set_ylabel("Error [rad]")
ax_dec.set_ylabel("Error [rad]")
ax_dec.set_xlabel("Date")

ax_ra.grid()
ax_dec.grid()

ax_ra.set_title("Right Ascension")
ax_dec.set_title("Declination")

plt.show()


"""
That's it! Next, check out the [Estimation with MPC](estimation_with_mpc.ipynb) example to try estimation with the observations we have retrieved here. The remainder of the example discusses additional features of the BatchMPC interface.
"""

"""
## Additional Features
"""

"""
### Using satellite observations.
Space Telescopes in Tudat are treated as bodies instead of stations. To use their observations, their motion should be known to Tudat. A user may for example retrieve their ephemerides from a SPICE kernel or propagate the satellite. The body must then be available in the system of bodies when the observation collection is created. The MPC code for TESS can be obtained using the `observatories_table()` method as used previously. Bellow is an example using a spice kernel.
"""


# Note that we are using the add_empty_settings() method instead of add_empty_body().
# This allows us to add ephemeris settings, 
# which tudat uses to create an ephemeris which is consistent with the rest of the environment.
TESS_code = "-95"
if os.path.exists("tess_20_year_long_predictive.bsp"):
    body_settings.add_empty_settings("TESS")

    # Set up the space telescope's dynamics, TESS orbits earth
    # the spice kernel can be retrieved from: https://archive.stsci.edu/missions/tess/models/
    spice.load_kernel(r"tess_20_year_long_predictive.bsp")
    body_settings.get("TESS").ephemeris_settings =  environment_setup.ephemeris.direct_spice(
         "Earth", global_frame_orientation, TESS_code)

    # Create system of bodies
    bodies = environment_setup.create_system_of_bodies(body_settings)
    tracking_data, supplementary_data = batch1.to_tracking_dataset()
    observations.set_tracking_supplementary_data_in_bodies(bodies, supplementary_data)
    observation_collection = observations.create_observation_collection_from_tracking_data(tracking_data, bodies)
else:
    print("Skipping optional TESS SPICE example because tess_20_year_long_predictive.bsp is not available.")


"""
### Manual retrieval from existing tables
Those with existing filtering or retrieval processes may use the `from_astropy()` and `from_pandas()` methods to import tabular observations before converting the batch to Tudat tracking data. The input must meet the requirements described in the API documentation.
"""


batch2 = BatchMPC()
batch2.get_observations([238])

batch2.summary()


"""
### Combining batches
"""

"""
Batches can be combined using the `+` operator, duplicates are removed.
"""


batch3 = batch2 + batch1
batch3.summary()


"""
### Copying and non in-place filtering
We may want to compare results between batches. In that case it is useful to copy a batch or perform non-destructive filtering:
"""


# Copying existing batches:
import copy
batch1_copy = copy.copy(batch1)
# simpler equivalent:
batch1_copy = batch1.copy()

# normal in-place/destructive filter
batch1_copy.filter(epoch_start=datetime(2023, 1, 1)) # returns None
# non-destructive filter:
batch1_copy2 = batch1.filter(epoch_start=datetime(2023, 1, 1), in_place=False) # returns filtered copy

batch1_copy.summary()
batch1_copy2.summary()


"""
### Plotting observations
The `.plot_observations_sky()` method can be used to view a projection of the observations. Similarly, `.plot_observations_temporal()` shows the declination and right ascension of a batch's bodies over time.
"""


fig = plt.figure()
ax = fig.add_subplot(111, projection="aitoff")
ax.scatter(batch1.table.RA - np.pi, batch1.table.DEC, marker="+")
ax.grid()

fig = plt.figure()
ax = fig.add_subplot(111)
object_329 = batch1.table.query("number == '329' or number == 329")
ax.scatter(object_329.RA, object_329.DEC, marker="+")
ax.set_xlabel("Right ascension [rad]")
ax.set_ylabel("Declination [rad]")

plt.show()



# Similar to the sky plot, specific bodies can be chosen to be plotted with the objects argument
fig, ax = plt.subplots()
ax.scatter(batch1.table.epoch_seconds_UTC, batch1.table.RA, marker="+", label="Right ascension")
ax.scatter(batch1.table.epoch_seconds_UTC, batch1.table.DEC, marker="+", label="Declination")
ax.set_xlabel("Epoch [s since J2000 UTC]")
ax.set_ylabel("Angle [rad]")
ax.legend()

plt.show()


plt.show()
