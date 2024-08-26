#Retrieving observation data from the Minor Planet Centre
"""
Copyright (c) 2010-2023, Delft University of Technology. All rights reserved. This file is part of the Tudat. Redistribution and use in source and binary forms, with or without modification, are permitted exclusively under the terms of the Modified BSD license. You should have received a copy of the license with this file. If not, please or visit: http://tudat.tudelft.nl/LICENSE.

"""

## Context
"""
The [Minor Planet Centre](https://www.minorplanetcenter.net/) (MPC) provides positional elements and observation data for minor planets, comets and outer irregular natural satellites of the major planets. Tudat's BatchMPC class allows for the retrieval and processing of observational data for these objects. This example highlights the complete functionality of the BatchMPC class itself. The [Estimation with MPC](https://docs.tudat.space/en/latest/_src_getting_started/_src_examples/notebooks/estimation/estimation_with_mpc.html) example showcases estimation with MPC observations, but we recommend going through this example first.

MPC receives and stores observations from observatories across the world. These are optical observations in a Right Ascension (RA) and Declination (DEC) format which are processed into an Earth-inertial J2000 format. Objects are all assigned a unique minor-planet designation number (see examples below), comets use a distinct designation. Larger objects are often also given a name (only about 4% have been given a name currently). Similarly, observatories are also assigned a unique 3 symbol code.

The following asteroids will be used in the example:

- [433 Eros](https://en.wikipedia.org/wiki/433_Eros) (also the main focus of the [Estimation with MPC](https://docs.tudat.space/en/latest/_src_getting_started/_src_examples/notebooks/estimation/estimation_with_mpc.html) example)
- [238 Hypatia](https://en.wikipedia.org/wiki/238_Hypatia)
- [329 Svea](https://en.wikipedia.org/wiki/329_Svea)
"""

## Basic Usage
"""
"""

### Import statements
"""
In this example we do not perform an estimation, as such we only need the batchMPC class from data, environment_setup and observation to convert our observations to Tudat and optionally datetime to filter our batch. We will also use the Tudat Horizons interface to compare observation ouput and load the standard SPICE kernels.
"""

from tudatpy.data.mpc import BatchMPC
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation.estimation_setup import observation
from tudatpy.kernel.interface import spice

from tudatpy.data.horizons import HorizonsQuery

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


# Load spice kernels
spice.load_standard_kernels()

### Retrieval
"""
We initialise a `BatchMPC` object, create a list with the objects we want and use `.get_observations()` to retrieve the observations. `.get_observations()` uses [astroquery](https://astroquery.readthedocs.io/en/latest/mpc/mpc.html) to retrieve data from MPC and requires an internet connection. The observations are cached for faster retrieval in subsequent runs. The `BatchMPC` object removes duplicates if `.get_observations()` is ran twice.

Tudat's estimation tools allow for multiple Objects to be analysed at the same time. BatchMPC can process multiple objects into a single observation collection automatically. For now lets retrieve the observations for Eros and Svea. BatchMPC uses MPC codes for objects and observatories. To get an overview of the batch we can use the `summary()` method. Let's also get some details on some of the observatories that retrieved the data using the `observatories_table()` method.
"""
from astroquery.mpc import MPC
import requests
import re
from tudatpy import constants

from astropy.time import Time
from astropy.table import Table, Column
from bs4 import BeautifulSoup
import pandas as pd

MPC.query_object = requests.get("https://www.projectpluto.com/pluto/mpecs/23053a.htm", "JUICE")
observations = MPC.query_object.text #use this if you want the whole html page
# Find the position of the <a name="stations"></a> tag
match = re.search(r'<a name="stations"></a>', observations)
if match:
    # Keep only the part of the HTML before the matched line
    html_cleaned = observations[:match.start()]
    #print(html_cleaned)
match = re.search(r'<a href="https://www.projectpluto.com/mpec_xpl.htm#astrometry"> <b>Astrometry:</b> </a>', html_cleaned)
if match:
    # Keep only the part of the HTML before the matched line
    html_cleaned = html_cleaned[match.end():].lstrip()
observations = [line.strip() for line in html_cleaned.splitlines() if line.strip()]

# Initialize lists to hold the extracted data
numbers = []
epochs = []
RAs = []
DECs = []
bands = []
observatories = []

for i,observation_string in enumerate(observations):
    soup = BeautifulSoup(observation_string, 'html.parser')

    number = i+1
    print(f'observation n. {number}')

    # Extract the observation string (without the band and observatory)
    observation_string = soup.get_text().split(soup.find_all('a')[1].get_text())[1].split(soup.find_all('a')[2].get_text())[0].strip()
    print(observation_string[0:3])
    if observation_string[2] == 'K' or observation_string[1] == 'B':
        print('no can do')
        continue
    # Extract and parse date and time
    if observation_string[0:2] == 'KC' or observation_string[0:2] == '0C' or observation_string[0:2] == '3C':
        year = observation_string[2:6]  # Year (e.g., 2023)
        month = observation_string[7:9]  # Month (e.g., 04)
        date_part, frac_day = observation_string[10:19].split('.')
        print(f'Day:{date_part}, Fraction of Day:{frac_day}')
        numbers.append(number)
    elif observation_string[0] == 'C':
        year = observation_string[1:5]  # Year (e.g., 2023)
        month = observation_string[6:8]  # Month (e.g., 04)
        date_part, frac_day = observation_string[9:18].split('.')
        print(f'Day:{date_part}, Fraction of Day:{frac_day}')
        numbers.append(number)

    # Calculate the time in hours, minutes, seconds
    hours = float("0." + frac_day) * 24
    minutes = (hours % 1) * 60
    seconds = (minutes % 1) * 60
    print(f'Hours:{hours}, Minutes:{minutes}, Seconds:{seconds}')
    # Convert to Julian date
    time_string = f"{date_part} {int(hours):02}:{int(minutes):02}:{round(seconds):02}"
    epoch = f'{year}-{month}-{date_part} {int(hours):02}:{int(minutes):02}:{round(seconds):02}'
    print(f'Epoch:{epoch}')
    dt = datetime.strptime(epoch, "%Y-%m-%d %H:%M:%S")
    dt_jd = Time(dt).jd
    epochs.append(dt_jd)

    # Extract RA and DEC
    if observation_string[0:2] == 'KC' or observation_string[0:2] == '0C' or observation_string[0:2] == '3C':
        ra_dec_str = observation_string[19:45]  # RA and DEC part
        print(f'ra & dec str:{ra_dec_str}')
        ra_part = ra_dec_str[:12].strip()  # Right Ascension
        print(f'ra part: {ra_part}')
        dec_part = ra_dec_str[12:].strip()  # Declination (considering no space if negative)
        print(f'dec part:{dec_part}')

        #Right Ascension
        parts_RA = ra_part.split()
        hours_RA = float(parts_RA[0])
        minutes_RA = float(parts_RA[1]) if len(parts_RA) > 1 else 0
        seconds_RA = float(parts_RA[2]) if len(parts_RA) > 2 else 0
        deg_ra = 15*hours_RA + 0.25*minutes_RA + seconds_RA/240

        #Declination
        parts_DEC = dec_part.split()
        degrees_DEC = float(parts_DEC[0])
        minutes_DEC = float(parts_DEC[1]) if len(parts_DEC) > 1 else 0
        seconds_DEC = float(parts_DEC[2]) if len(parts_DEC) > 2 else 0
        if len(parts_DEC[0]) == 3:
            deg_dec = - (abs(degrees_DEC) + minutes_DEC / 60 + seconds_DEC / 3600)
        else:
            deg_dec = (abs(degrees_DEC) + minutes_DEC/ 60 + seconds_DEC/ 3600)
        print(f'deg_ra {deg_ra}')
        print(f'deg_dec {deg_dec}')
        # Extract Band
        band = observation_string[57:58]

        bands.append(band)

    elif observation_string[0] == 'C':
        ra_dec_str = observation_string[18:45]  # RA and DEC part
        print(f'ra & dec str:{ra_dec_str}')
        ra_part = ra_dec_str[:12].strip()  # Right Ascension
        print(f'ra part: {ra_part}')
        dec_part = ra_dec_str[12:].strip()  # Declination (considering no space if negative)
        print(f'dec part:{dec_part}')

        #Right Ascension
        parts_RA = ra_part.split()
        hours_RA = float(parts_RA[0])
        minutes_RA = float(parts_RA[1]) if len(parts_RA) > 1 else 0
        seconds_RA = float(parts_RA[2]) if len(parts_RA) > 2 else 0
        print(hours_RA, minutes_RA, seconds_RA)
        deg_ra = 15*hours_RA + 0.25*minutes_RA + seconds_RA/240

        #Declination
        parts_DEC = dec_part.split()
        degrees_DEC = float(parts_DEC[0])
        minutes_DEC = float(parts_DEC[1]) if len(parts_DEC) > 1 else 0
        seconds_DEC = float(parts_DEC[2]) if len(parts_DEC) > 2 else 0
        if len(parts_DEC[0]) == 3:
            deg_dec = - (abs(degrees_DEC) + minutes_DEC/ 60 + seconds_DEC/ 3600)
        else:
            deg_dec = (abs(degrees_DEC) + minutes_DEC/ 60 + seconds_DEC/ 3600)
        print(f'deg_ra: {deg_ra}')
        print(f'deg_dec {deg_dec}')
        # Extract Band
        band = observation_string[57:58]

        bands.append(band)

# Extract the observatory code
    observatory = soup.find_all('a')[2].get_text().strip()
    observatories.append(observatory)
    RAs.append(deg_ra)
    DECs.append(deg_dec)
# Create the table
table = Table()
# Add columns to the table
table.add_column(Column(name='number', data=numbers))
table.add_column(Column(name='epoch', data=epochs))
table.add_column(Column(name='RA', data=RAs, unit='deg'))  # RA in degrees
table.add_column(Column(name='DEC', data=DECs, unit='deg'))  # DEC in degrees
table.add_column(Column(name='band', data=bands))
table.add_column(Column(name='observatory', data=observatories))

# Display the table
print(table)

batch1 = BatchMPC()
batch1.from_astropy(table)
batch1.summary()
print(batch1.observatories_table(only_in_batch=True, only_space_telescopes=False, include_positions=False))
print("Space Telescopes:")
print(batch1.observatories_table(only_in_batch=True, only_space_telescopes=True, include_positions=False))

# We can also directly have a look at the the observations themselves, for example, lets take a look at the first and final observations from TESS and WISE. The table property allows for read only access to the observations in pandas dataframe format. 

obs_by_Pan_STARRS = batch1.table.query("observatory == 'F51'").loc[:, ["number", "epochUTC", "RA", "DEC"]].iloc[[0, -1]]
obs_by_Calar_Alto = batch1.table.query("observatory == 'Z84'").loc[:, ["number", "epochUTC", "RA", "DEC"]].iloc[[0, -1]]

print("Initial and Final Observations by Pan_STARRS")
print(obs_by_Pan_STARRS)
print("Initial and Final Observations by Calar Alto-Schmidt")
print(obs_by_Calar_Alto)

### Set up the system of bodies
"""
A system of bodies must be created to keep observatories' positions consistent with Earth's shape model and to allow the attachment of these observatories to Earth. For the purposes of this example, we keep it as simple as possible. See the [Estimation with MPC](https://docs.tudat.space/en/latest/_src_getting_started/_src_examples/notebooks/estimation/estimation_with_mpc.html) for a more complete setup and explanation appropriate for estimation. For our bodies, we only use Earth and the Sun. We set our origin to `"SSB"`, the solar system barycenter. We use the default body settings from the `SPICE` kernel to initialise the planet and use it to create a system of bodies. This system of bodies is used in the `to_tudat()` method.
"""

bodies_to_create = ["Sun", "Earth"]

# Create default body settings
global_frame_origin = "SSB"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation)

# Create system of bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

### Retrieve Observation Collection
"""

"""

# Now that our batch is ready, we can transform it to a Tudat `ObservationCollection` object using the `to_tudat()` method.
# 
# The `.to_tudat()` does the following for us:
# 
# 1. Creates an empty body for each minor planet with their MPC code as a name.
# 2. Adds this body to the system of bodies inputted to the method.
# 3. Retrieves the global position of the terrestrial observatories in the batch and adds these stations to the Tudat environment.
# 4. Creates link definitions between each unique terrestrial observatory/ minor planet combination in the batch.
# 5. (Optionally) creates a link definition between each space telescope / minor planet combination in the batch. This requires an addional input.
# 6. Creates a `SingleObservationSet` object for each unique link that includes all observations for that link.
# 7. Returns an `ObservationCollection` object.
# 7. Returns an `ObservationCollection` object.
#
# If our batch includes space telescopes like WISE and TESS we must either link their Tudat name or exclude them. For now we exclude them by setting `included_satellites` to `None`. The additional features section shows an example of how to link satellites to the `to_tudat()` method. The '.to_tudat()' method does not alter the batch object itself.

observation_collection = batch1.to_tudat(bodies, included_satellites=None, apply_star_catalog_debias = False)

# The names of the bodies added to the system of bodies object as well as the dates of the oldest and latest observations can be retrieved from the batch:


epoch_start = batch1.epoch_start# in seconds since J2000 TDB (Tudat default)
epoch_end = batch1.epoch_end
object_names = batch1.MPC_objects
# We can now retrieve the links from the ObservationCollection we got from `to_tudat()` and we can now create settings for these links. This is where link biases would be set, for now we just keep the settings default.

observation_settings_list = list()

link_list = list(
    observation_collection.get_link_definitions_for_observables(
        observable_type=observation.angular_position_type
    )
)

for link in link_list:
    # add optional bias settings
    observation_settings_list.append(
        observation.angular_position(link, bias_settings=None)
    )

# Retrieve MPC observation times, RA and DEC
batch_times = batch1.table.epochJ2000secondsTDB.to_list()
batch_times_utc = batch1.table.epochUTC.to_list()
batch_RA = batch1.table.RA
batch_DEC = batch1.table.DEC

print(batch_times_utc)

# Create Horizons query, see Horizons Documentation for more info.
JUICE_horizons_query = HorizonsQuery(
    query_id="-28",
    location="500@399",  # geocenter @ Earth
    epoch_list=batch_times,
    extended_query=True,
)

# retrieve JPL observations
jpl_observations = JUICE_horizons_query.interpolated_observations()
jpl_RA = jpl_observations[:, 1]
jpl_DEC = jpl_observations[:,2]

#print('yoo',batch_RA)
print(batch_RA)
print(jpl_RA)
print(batch_DEC)
print(jpl_DEC)
max_diff_RA = np.abs(jpl_RA - batch_RA).max()
max_diff_DEC = np.abs(jpl_DEC - batch_DEC).max()
print("Maximum difference between Interpolated Horizons data and MPC observations:")
print(f"Right Ascension: {np.round(max_diff_RA, 10)} rad")
print(f"Declination: {np.round(max_diff_DEC, 10)} rad")

# create plot
fig, (ax_ra, ax_dec) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

print(f'RA residuals:\n {jpl_RA - batch_RA}')
print(f'DEC residuals:\n + {jpl_DEC-batch_DEC}')

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


### Plotting observations
"""
The `.plot_observations_sky()` method can be used to view a projection of the observations. Similarly, `.plot_observations_temporal()` shows the declination and right ascension of a batch's bodies over time.
"""


import matplotlib.pyplot as plt

# Try some of the other projections: 'hammer', 'mollweide' and 'lambert'
fig = batch1.plot_observations_sky(projection="aitoff")
plt.show()

fig = batch1.plot_observations_temporal()
plt.show()


