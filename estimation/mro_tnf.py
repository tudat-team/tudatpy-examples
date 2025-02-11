# %%================================================================================================
from tudatpy.numerical_simulation import (
    environment_setup,
    estimation_setup,
)  # type:ignore
from tudatpy.numerical_simulation import estimation, environment  # type:ignore
from tudatpy.numerical_simulation.estimation_setup import observation  # type:ignore
from tudatpy.astro import time_conversion
from tudatpy.interface import spice
from tudatpy.data.mission_data_downloader import *
from tudatpy.math import interpolators

from tudatpy.data import processTrk234

import numpy as np

from tudatpy.data.mission_data_downloader import LoadPDS
from load_pds_files import download_url_files_time, download_url_files_time_interval

import matplotlib.pyplot as plt


def get_mro_files(local_path, start_date, end_date):

    # Check if local_path designates an existing directory and creates the directory is not
    if not os.path.isdir(local_path):
        os.mkdir(local_path)

    # Clock file (a single file is necessary)
    print("---------------------------------------------")
    print("Download MRO clock ")
    clock_files = ["mro_sclkscet_00112_65536.tsc"]
    # Define url where clock files can be downloaded for MRO
    url_clock_files = "https://naif.jpl.nasa.gov/pub/naif/pds/data/mro-m-spice-6-v1.0/mrosp_1000/data/sclk/"
    for file in clock_files:
        # Check if the relevant clock file exists locally
        if os.path.exists(local_path + file) == False:
            print("download", local_path + file)
            # If not, download it
            urlretrieve(url_clock_files + file, local_path + file)

    # Print and store all relevant clock file names
    print("relevant clock files")
    for k in range(len(clock_files)):
        clock_files[k] = local_path + clock_files[k]
        print(clock_files[k])

    # Orientation files (multiple orientation kerneks are required, typically covering intervals of a few days)
    # For this MRO example, orientation kernels should be loaded both for the spacecraft and for the MRO antenna specifically.
    print("---------------------------------------------")
    print("Download MRO orientation kernels")
    # Define url where orientation kernels can be downloaded for MRO
    url_orientation_files = "https://naif.jpl.nasa.gov/pub/naif/pds/data/mro-m-spice-6-v1.0/mrosp_1000/data/ck/"
    # Retrieve the names of all spacecraft orientation kernels required to cover the time interval of interest, and download them if they
    # do not exist locally yet.
    orientation_files = download_url_files_time_interval(
        local_path=local_path,
        filename_format="mro_sc_psp_*.bc",
        start_date=start_date,
        end_date=end_date,
        url=url_orientation_files,
        time_interval_format="%y%m%d_%y%m%d",
    )

    # Retrieve the names of all antenna orientation kernels required to cover the time interval of interest, and download them if they
    # do not exist locally yet
    antenna_files = download_url_files_time_interval(
        local_path=local_path,
        filename_format="mro_hga_psp_*.bc",
        start_date=start_date,
        end_date=end_date,
        url=url_orientation_files,
        time_interval_format="%y%m%d_%y%m%d",
    )

    # Print and store all relevant orientation file names (both for the MRO spacecraft and antenna)
    for file in antenna_files:
        orientation_files.append(file)

    print("relevant orientation files")
    for f in orientation_files:
        print(f)

    # Tropospheric corrections (multiple tropospheric correction files are required, typically covering intervals of a few days)
    print("---------------------------------------------")
    print("Download MRO tropospheric corrections files")
    # Define url where tropospheric correction files can be downloaded for MRO
    url_tro_files = "https://pds-geosciences.wustl.edu/mro/mro-m-rss-1-magr-v1/mrors_0xxx/ancillary/tro/"
    # Retrieve the names of all tropospheric correction files required to cover the time interval of interest, and download them if they
    # do not exist locally yet
    tro_files = download_url_files_time_interval(
        local_path=local_path,
        filename_format="mromagr*.tro",
        start_date=start_date,
        end_date=end_date,
        url=url_tro_files,
        time_interval_format="%Y_%j_%Y_%j",
    )

    # Print all relevant tropospheric correction file names
    print("relevant tropospheric corrections files")
    for f in tro_files:
        print(f)

    # Ionospheric corrections (multiple ionospheric correction files are required, typically covering intervals of a few days)
    print("---------------------------------------------")
    print("Download MRO ionospheric corrections files")
    # Define url where ionospheric correction files can be downloaded for MRO
    url_ion_files = "https://pds-geosciences.wustl.edu/mro/mro-m-rss-1-magr-v1/mrors_0xxx/ancillary/ion/"
    # Retrieve the names of all ionospheric correction files required to cover the time interval of interest, and download them if they
    # do not exist locally yet
    ion_files = download_url_files_time_interval(
        local_path=local_path,
        filename_format="mromagr*.ion",
        start_date=start_date,
        end_date=end_date,
        url=url_ion_files,
        time_interval_format="%Y_%j_%Y_%j",
    )

    # Print all relevant ionospheric correction file names
    print("relevant ionospheric corrections files")
    for f in ion_files:
        print(f)

    # ODF files (multiple ODF files are required, typically one per day)
    print("---------------------------------------------")
    print("Download MRO ODF files")
    # Define url where ODF files can be downloaded for MRO
    url_odf = (
        "https://pds-geosciences.wustl.edu/mro/mro-m-rss-1-magr-v1/mrors_0xxx/odf/"
    )
    # Retrieve the names of all existing ODF files within the time interval of interest, and download them if they do not exist locally yet
    odf_files = download_url_files_time(
        local_path=local_path,
        filename_format="mromagr*_\w\w\w\wxmmmv1.odf",
        start_date=start_date,
        end_date=end_date,
        url=url_odf,
        time_format="%Y_%j",
        indices_date_filename=[7],
    )

    # Print the name of all relevant ODF files that have been identified over the time interval of interest
    print("relevant odf files")
    for f in odf_files:
        print(f)

    # MRO trajectory files (multiple files are necessary to cover one entire year, typically each file covers ~ 3-4 months)
    # Note that the file names hard coded below cover calendar year 2012. This should be modified in case the time interval
    # of the example is modified.
    print("---------------------------------------------")
    print("Download MRO trajectory files")
    trajectory_files = [
        "mro_psp21.bsp",
        "mro_psp22.bsp",
        "mro_psp23.bsp",
        "mro_psp24.bsp",
        "mro_psp25.bsp",
    ]
    # Define url where trajectory files can be downloaded for MRO
    url_trajectory_files = "https://naif.jpl.nasa.gov/pub/naif/pds/data/mro-m-spice-6-v1.0/mrosp_1000/data/spk/"
    for file in trajectory_files:
        # Check if the relevant trajectory file exists locally
        if os.path.exists(local_path + file) == False:
            print("download", local_path + file)
            # If not, download it
            urlretrieve(url_trajectory_files + file, local_path + file)

    # Print and store all relevant trajectory file names
    print("relevant trajectory files")
    for k in range(len(trajectory_files)):
        trajectory_files[k] = local_path + trajectory_files[k]
        print(trajectory_files[k])

    # Frames definition file for the MRO spacecraft (only one file is necessary). This is useful for HGA and
    # spacecraft-fixed frames definition.
    print("---------------------------------------------")
    print("Download MRO frames definition file")
    frames_def_file = "mro_v16.tf"
    # Define url where the frames definition file can be downloaded for MRO
    url_frames_def_file = "https://naif.jpl.nasa.gov/pub/naif/pds/data/mro-m-spice-6-v1.0/mrosp_1000/data/fk/"
    # Check if the relevant frames definition file exists locally
    if os.path.exists(local_path + frames_def_file) == False:
        print("download", local_path + frames_def_file)
        # If not, download it
        urlretrieve(url_frames_def_file + frames_def_file, local_path + frames_def_file)

    # Print and store the frames definition file name
    print("relevant MRO frames definition file")
    frames_def_file = local_path + frames_def_file
    print(frames_def_file)

    # Structure file for the MRO spacecraft (only one file is necessary). This is useful to retrieve the antenna
    # position in spacecraft-fixed frame.
    print("---------------------------------------------")
    print("Download MRO structure file")
    structure_file = "mro_struct_v10.bsp"
    # Define url where the MRO structure file can be downloaded
    url_structure_file = "https://naif.jpl.nasa.gov/pub/naif/pds/data/mro-m-spice-6-v1.0/mrosp_1000/data/spk/"
    # Check if the relevant structure file exists locally
    if os.path.exists(local_path + structure_file) == False:
        print("download", local_path + structure_file)
        # If not, download it
        urlretrieve(url_structure_file + structure_file, local_path + structure_file)

    # Print and store the structure file name
    print("relevant MRO structure file")
    structure_file = local_path + structure_file
    print(structure_file)

    # Return filenames lists for clock files, orientation kernels, tropospheric and ionospheric corrections, odf files,
    # trajectory files, MRO reference frames file, and MRO structure file.
    return (
        clock_files,
        orientation_files,
        tro_files,
        ion_files,
        odf_files,
        trajectory_files,
        frames_def_file,
        structure_file,
    )


# %%===========================================================================
# Load data

startTimeDatetime = datetime(2012, 1, 1, 0, 0, 0)
endTimeDatetime = datetime(2012, 1, 9, 0, 0, 0)

# First retrieve the names of all the relevant kernels and data files necessary to cover the specified time interval
(
    clock_files,
    orientation_files,
    tro_files,
    ion_files,
    odf_files,
    trajectory_files,
    frames_def_file,
    structure_file,
) = get_mro_files("mro_kernels/", startTimeDatetime, endTimeDatetime)

dir_path = "/Users/valeriofilice/Workspace/tudat-bundle/tudatpy/examples/estimation/mroDownloads/tnf/"
# dir_path = "."
# file_names = os.listdir(dir_path)
tnfFiles = [
    dir_path + "mromagr2012_001_2220xmmmv1.tnf",
    dir_path + "mromagr2012_005_1255xmmmv1.tnf",
    dir_path + "mromagr2012_004_1550xmmmv1.tnf",
    dir_path + "mromagr2012_008_2200xmmmv1.tnf",
    dir_path + "mromagr2012_007_1640xmmmv1.tnf",
    dir_path + "mromagr2012_003_1407xmmmv1.tnf",
    dir_path + "mromagr2012_002_1426xmmmv1.tnf",
    dir_path + "mromagr2012_006_1355xmmmv1.tnf",
    # fileName,
]

# downloader = LoadPDS()
# kernel_files_mro, radio_science_files_mro, ancillary_files_mro = (
#     downloader.get_mission_files(
#         input_mission="mro",
#         start_date=startTimeDatetime,
#         end_date=endTimeDatetime,
#         custom_output="mroDownloads",
#         load_kernels=False,
#         radio_science_file_type="tnf",
#     )
# )

# tnfFiles = radio_science_files_mro["tnf"]
# orientation_files = kernel_files_mro["ck"]
# clock_files = kernel_files_mro["sclk"]
# tro_files = ancillary_files_mro["tro"]
# ion_files = ancillary_files_mro["ion"]
# frames_def_file = kernel_files_mro["fk"][0]
# trajectory_files = kernel_files_mro["spk"][:-1]
# structure_file = kernel_files_mro["spk"][-1]

spice.load_standard_kernels()

for orientation_file in orientation_files:
    spice.load_kernel(orientation_file)

for clock_file in clock_files:
    spice.load_kernel(clock_file)

spice.load_kernel(frames_def_file)

for trajectory_file in trajectory_files:
    spice.load_kernel(trajectory_file)

spice.load_kernel(structure_file)

dates_to_filter = [
    time_conversion.DateTime(2012, 10, 15, 0, 0, 0.0),
    time_conversion.DateTime(2012, 10, 30, 0, 0, 0),
    time_conversion.DateTime(2012, 11, 6, 0, 0, 0),
    time_conversion.DateTime(2012, 11, 7, 0, 0, 0),
]

# %%===========================================================================
# Create models

# Convert start and end times to Tudat-compatible format
startTime = (
    time_conversion.datetime_to_tudat(startTimeDatetime).epoch().to_float() - 86400.0
)
endTime = (
    time_conversion.datetime_to_tudat(endTimeDatetime).epoch().to_float() + 86400.0
)

# Create default body settings for celestial bodies
bodies_to_create = [
    "Earth",
    "Sun",
    "Mercury",
    "Venus",
    "Mars",
    "Jupiter",
    "Saturn",
    "Moon",
]
global_frame_origin = "SSB"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings_time_limited(
    bodies_to_create,
    startTime,
    endTime,
    global_frame_origin,
    global_frame_orientation,
)

# Modify Earth default settings
body_settings.get("Earth").shape_settings = (
    environment_setup.shape.oblate_spherical_spice()
)
body_settings.get("Earth").rotation_model_settings = (
    environment_setup.rotation_model.gcrs_to_itrs(
        environment_setup.rotation_model.iau_2006,
        global_frame_orientation,
        interpolators.interpolator_generation_settings_float(
            interpolators.cubic_spline_interpolation(),
            startTime,
            endTime,
            3600.0,
        ),
        interpolators.interpolator_generation_settings_float(
            interpolators.cubic_spline_interpolation(),
            startTime,
            endTime,
            3600.0,
        ),
        interpolators.interpolator_generation_settings_float(
            interpolators.cubic_spline_interpolation(),
            startTime,
            endTime,
            60.0,
        ),
    )
)
body_settings.get("Earth").gravity_field_settings.associated_reference_frame = "ITRS"
body_settings.get("Earth").ground_station_settings = (
    environment_setup.ground_station.dsn_stations()
)

# Create empty settings for the MRO spacecraft
spacecraft_name = "MRO"
spacecraft_central_body = "Mars"
body_settings.add_empty_settings(spacecraft_name)

# Retrieve translational ephemeris from SPICE
body_settings.get(spacecraft_name).ephemeris_settings = (
    environment_setup.ephemeris.interpolated_spice(
        startTime,
        endTime,
        10.0,
        spacecraft_central_body,
        global_frame_orientation,
    )
)

# Retrieve rotational ephemeris from SPICE
body_settings.get(spacecraft_name).rotation_model_settings = (
    environment_setup.rotation_model.spice(
        global_frame_orientation, spacecraft_name + "_SPACECRAFT", ""
    )
)

# Create environment
bodies = environment_setup.create_system_of_bodies(body_settings)

# %%===========================================================================
# Load observations and define observation models

observationCollection, rampDf, dopplerDf = (
    processTrk234.create_observation_collection_from_tnf(
        tnfFiles, bodies, spacecraftName="MRO"
    )
)

# Compress Doppler observations from 1.0 s integration time to 60.0 s
compressed_observations = (
    estimation_setup.observation.create_compressed_doppler_collection(
        observationCollection, 60, 10
    )
)

# Set Antenna as reference point
# Define MRO center-of-mass (COM) position w.r.t. the origin of the MRO-fixed reference frame (frame spice ID: MRO_SPACECRAFT)
# This value was taken from Konopliv et al. (2011) doi:10.1016/j.icarus.2010.10.004
# This is necessary to define the position of the antenna w.r.t. the COM, in the MRO-fixed frame (see below)
com_position = np.array([0.0, -1.11, 0.0])

antenna_position_history = dict()

for obs_times in compressed_observations.get_observation_times():
    time = obs_times[0].to_float() - 3600.0
    while time <= obs_times[-1].to_float() + 3600.0:
        state = np.zeros((6, 1))

        # For each observation epoch, retrieve the antenna position (spice ID "-74214") w.r.t. the origin of the MRO-fixed frame (spice ID "-74000")
        state[:3, 0] = spice.get_body_cartesian_position_at_epoch(
            "-74214", "-74000", "MRO_SPACECRAFT", "none", time
        )

        # Translate the antenna position to account for the offset between the origin of the MRO-fixed frame and the COM
        state[:3, 0] = state[:3, 0] - com_position

        # Store antenna position w.r.t. COM in the MRO-fixed frame
        antenna_position_history[time] = state
        time += 60.0

# Create tabulated ephemeris settings from antenna position history
antenna_ephemeris_settings = environment_setup.ephemeris.tabulated(
    antenna_position_history, "-74000", "MRO_SPACECRAFT"
)

# Create tabulated ephemeris for the MRO antenna
antenna_ephemeris = environment_setup.ephemeris.create_ephemeris(
    antenna_ephemeris_settings, "Antenna"
)

# Set the spacecraft's reference point position to that of the antenna (in the MRO-fixed frame)
compressed_observations.set_reference_point(
    bodies, antenna_ephemeris, "Antenna", "MRO", observation.reflector1
)

#  Create light-time corrections list
light_time_correction_list = list()
light_time_correction_list.append(
    estimation_setup.observation.first_order_relativistic_light_time_correction(["Sun"])
)

# Add tropospheric correction
light_time_correction_list.append(
    estimation_setup.observation.dsn_tabulated_tropospheric_light_time_correction(
        tro_files
    )
)

# Add ionospheric correction
spacecraft_name_per_id = dict()
spacecraft_name_per_id[74] = "MRO"
light_time_correction_list.append(
    estimation_setup.observation.dsn_tabulated_ionospheric_light_time_correction(
        ion_files, spacecraft_name_per_id
    )
)

# Create observation model settings for the Doppler observables. This first implies creating the link ends defining all relevant
# tracking links between various ground stations and the MRO spacecraft. The list of light-time corrections defined above is then
# added to each of these link ends.
doppler_link_ends = compressed_observations.link_definitions_per_observable[
    estimation_setup.observation.dsn_n_way_averaged_doppler
]

observation_model_settings = list()
for current_link_definition in doppler_link_ends:
    observation_model_settings.append(
        estimation_setup.observation.dsn_n_way_doppler_averaged(
            current_link_definition, light_time_correction_list
        )
    )

# Create observation simulators.
observation_simulators = estimation_setup.create_observation_simulators(
    observation_model_settings, bodies
)

# Add elevation and SEP angles dependent variables to the compressed observation collection
elevation_angle_settings = observation.elevation_angle_dependent_variable(
    observation.receiver
)
elevation_angle_parser = compressed_observations.add_dependent_variable(
    elevation_angle_settings, bodies
)
sep_angle_settings = observation.avoidance_angle_dependent_variable(
    "Sun", observation.retransmitter, observation.receiver
)
sep_angle_parser = compressed_observations.add_dependent_variable(
    sep_angle_settings, bodies
)

# Compute and set residuals in the compressed observation collection
estimation.compute_residuals_and_dependent_variables(
    compressed_observations, observation_simulators, bodies
)
# Filter out outliers
threshold = 0.1  # residuals > 0.1 Hz

filter_residuals = estimation.observation_filter(
    estimation.residual_filtering, threshold
)
compressed_observations.filter_observations(filter_residuals)

# %%
time = np.array(compressed_observations.concatenated_float_times)
residuals = compressed_observations.get_concatenated_residuals()

plt.rc("font", size=20)

fig, ax = plt.subplots(1, 1, figsize=(15, 10))
ax.scatter((time - np.min(time)) / 86400.0, residuals)
ax.set_xlabel("Time [days since {}]".format(startTimeDatetime))
ax.set_ylabel("Residuals [Hz]")
ax.grid(which="both", linestyle="--", alpha=0.6)

plt.rcdefaults()
