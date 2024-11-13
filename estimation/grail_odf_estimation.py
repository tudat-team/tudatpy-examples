# MARS EXPRESS - Using Different Dynamical Models for the Simulation of Observations and the Estimation
"""
Copyright (c) 2010-2022, Delft University of Technology. All rights reserved. This file is part of the Tudat. Redistribution and use in source and binary forms, with or without modification, are permitted exclusively under the terms of the Modified BSD license. You should have received a copy of the license with this file. If not, please or visit: http://tudat.tudelft.nl/LICENSE.
"""
from turtledemo.forest import start

## Context
"""
"""

import sys
sys.path.insert(0, "/home/mfayolle/Tudat/tudat-bundle/cmake-build-release-2/tudatpy")

# Load required standard modules
import multiprocessing as mp
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# Load required tudatpy modules
from tudatpy import constants
from tudatpy.data import grail_mass_level_0_file_reader
from tudatpy.data import grail_antenna_file_reader
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.astro import time_conversion
from tudatpy.astro import frame_conversion
from tudatpy.astro import element_conversion
from tudatpy.math import interpolators
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation
from tudatpy.numerical_simulation.environment_setup import radiation_pressure
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.numerical_simulation import estimation, estimation_setup
from tudatpy.numerical_simulation import create_dynamics_simulator
from tudatpy.numerical_simulation.estimation_setup import observation
from tudatpy import util

from load_pds_files import download_url_files_time, download_url_files_time_interval
from datetime import datetime, timedelta
from urllib.request import urlretrieve

current_directory = os.getcwd() + '/'


# This function retrieves all relevant files necessary to run the example over the time interval of interest
# (and automatically downloads them if they cannot be found locally). It returns a tuple containing the lists of
# relevant clock files, orientation kernels, tropospheric correction files, ionospheric correction files, manoeuvre file, antenna switch files
# and odf files that should be loaded.
def get_grail_files(local_path, start_date, end_date):

    # Check if local_path designates an existing directory and creates the directory is not
    if not os.path.isdir(local_path):
        os.mkdir(local_path)


    # Clock files
    print('---------------------------------------------')
    print('Download GRAIL clock files')
    clock_files = ["gra_sclkscet_00013.tsc", "gra_sclkscet_00014.tsc"]
    # Define url where clock files can be downloaded for GRAIL
    url_clock_files = "https://naif.jpl.nasa.gov/pub/naif/pds/data/grail-l-spice-6-v1.0/grlsp_1000/data/sclk/"
    for i in range(len(clock_files)):
        # Check if the relevant clock file exists locally
        if (os.path.exists(local_path + clock_files[i]) == False):
            print('download', local_path + clock_files[i])
            # If not, download it
            urlretrieve(url_clock_files + clock_files[i], local_path + clock_files[i])

        # Add local path to clock file name
        clock_files[i] = local_path + clock_files[i]

    # Print and store all relevant clock file names
    print('relevant clock files')
    for f in clock_files:
        print(f)


    # Orientation files (multiple orientation kernels are required, typically covering intervals of a few days)
    print('---------------------------------------------')
    print('Download GRAIL orientation kernels')
    # Define url where orientation kernels can be downloaded for GRAIL
    url_orientation_files = "https://naif.jpl.nasa.gov/pub/naif/pds/data/grail-l-spice-6-v1.0/grlsp_1000/data/ck/"
    # Retrieve the names of all spacecraft orientation kernels required to cover the time interval of interest, and download them if they
    # do not exist locally yet.
    orientation_files_to_load = download_url_files_time_interval(
        local_path=local_path, filename_format='gra_rec_*.bc', start_date=start_date, end_date=end_date,
        url=url_orientation_files, time_interval_format='%y%m%d_%y%m%d')

    # Print and store all relevant orientation file names
    print('relevant orientation files')
    for f in orientation_files_to_load:
        print(f)


    # Tropospheric corrections (multiple tropospheric correction files are required, typically covering intervals of a few days)
    print('---------------------------------------------')
    print('Download GRAIL tropospheric corrections files')
    # Define url where tropospheric correction files can be downloaded for GRAIL
    url_tro_files = "https://pds-geosciences.wustl.edu/grail/grail-l-rss-2-edr-v1/grail_0201/ancillary/tro/"
    # Retrieve the names of all tropospheric correction files required to cover the time interval of interest, and download them if they
    # do not exist locally yet
    tro_files_to_load = download_url_files_time_interval(
        local_path=local_path, filename_format='grxlugf*.tro', start_date=start_date,
        end_date=end_date, url=url_tro_files, time_interval_format='%Y_%j_%Y_%j')

    # Print all relevant tropospheric correction file names
    print('relevant tropospheric corrections files')
    for f in tro_files_to_load:
        print(f)


    # Ionospheric corrections (multiple ionospheric correction files are required, typically covering intervals of a few days)
    print('---------------------------------------------')
    print('Download GRAIL ionospheric corrections files')
    # Define url where ionospheric correction files can be downloaded for GRAIL
    url_ion_files = "https://pds-geosciences.wustl.edu/grail/grail-l-rss-2-edr-v1/grail_0201/ancillary/ion/"
    # Retrieve the names of all ionospheric correction files required to cover the time interval of interest, and download them if they
    # do not exist locally yet
    ion_files_to_load = download_url_files_time_interval(
        local_path=local_path, filename_format='gralugf*.ion', start_date=start_date,
        end_date=end_date, url=url_ion_files, time_interval_format='%Y_%j_%Y_%j')

    # Print all relevant ionospheric correction file names
    print('relevant ionospheric corrections files')
    for f in ion_files_to_load:
        print(f)


    # Manoeuvres file (a single file is necessary, as this file is identical for all dates)
    print('---------------------------------------------')
    print('Download GRAIL manoeuvres file')
    # Define filename and url where a manoeuvre file can be downloaded for GRAIL (the specific date is here arbitrarily chosen
    # since the manoeuvre files are identical for all dates and cover the full mission time span)
    manoeuvres_file = "mas00_2012_04_06_a_04.asc"
    url_manoeuvres_files = "https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-2-edr-v1/grail_0001/level_0/2012_04_06/"
    # Check if the manoeuvre file already exists locally
    if (os.path.exists(local_path + manoeuvres_file) == False):
        print('download', local_path + manoeuvres_file)
        # If not, download it
        urlretrieve(url_manoeuvres_files + manoeuvres_file, local_path + manoeuvres_file)

    # Add local path to manoeuvres file name
    manoeuvres_file = local_path + manoeuvres_file

    # Print the name of the manoeuvre file
    print('relevant manoeuvres files')
    print(manoeuvres_file)


    # Antenna switch files (multiple antenna switch files are required, typically one per day)
    print('---------------------------------------------')
    print('Download antenna switch files')
    # Define url where antenna switch files can be downloaded for GRAIL
    url_antenna_files = ("https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-3-cdr-v1/grail_0101/level_1b/")
    # Retrieve the names of all antenna switch files within the time interval of interest, and download them if they do not exist locally yet
    antenna_files_to_load = download_url_files_time(local_path=local_path, filename_format='*/vgs1b_*_a_04.asc', start_date=start_date,
                            end_date=end_date, url=url_antenna_files, time_format='%Y_%m_%d', indices_date_filename=[0,8])

    # Print the name of all relevant antenna switch files that have been identified over the time interval of interest
    print('relevant antenna files')
    for f in antenna_files_to_load:
        print(f)


    # ODF files (multiple ODF files are required, typically one per day)
    print('---------------------------------------------')
    print('Download GRAIL ODF files')
    # Define url where ODF files can be downloaded for GRAIL
    url_odf = ("https://pds-geosciences.wustl.edu/grail/grail-l-rss-2-edr-v1/grail_0201/odf/")
    # Retrieve the names of all existing ODF files within the time interval of interest, and download them if they do not exist locally yet
    odf_files_to_load = download_url_files_time(
        local_path=local_path, filename_format='gralugf*_\w\w\w\wsmmmv1.odf', start_date=start_date,
        end_date=end_date, url=url_odf, time_format='%Y_%j', indices_date_filename=[7])

    # Print the name of all relevant ODF files that have been identified over the time interval of interest
    print('relevant odf files')
    for f in odf_files_to_load:
        print(f)


    # Retrieve filenames lists for clock files, orientation kernels, tropospheric and ionospheric corrections, manoeuvre files,
    # antenna switch files, and odf files.
    return clock_files, orientation_files_to_load, tro_files_to_load, ion_files_to_load, manoeuvres_file, antenna_files_to_load, odf_files_to_load


def load_clock_kernels(test_index):
    spice.load_kernel(current_directory + "grail_kernels/gra_sclkscet_00013.tsc")
    spice.load_kernel(current_directory + "grail_kernels/gra_sclkscet_00014.tsc")


def load_orientation_kernels(test_index):
    if test_index == 0:
        spice.load_kernel(current_directory + "grail_kernels/gra_rec_120402_120408.bc")
    if (test_index > 0 and test_index <= 6):
        spice.load_kernel(current_directory + "grail_kernels/gra_rec_120409_120415.bc")
    if (test_index >= 4):
        spice.load_kernel(current_directory + "grail_kernels/gra_rec_120416_120422.bc")


def get_grail_odf_file_name(test_index):
    if test_index == 0:
        return current_directory + 'grail_data/gralugf2012_097_0235smmmv1.odf'
    elif test_index == 1:
        return current_directory + 'grail_data/gralugf2012_100_0540smmmv1.odf'
    elif test_index == 2:
        return current_directory + 'grail_data/gralugf2012_101_0235smmmv1.odf'
    elif test_index == 3:
        return current_directory + 'grail_data/gralugf2012_102_0358smmmv1.odf'
    elif test_index == 4:
        return current_directory + 'grail_data/gralugf2012_103_0145smmmv1.odf'
    elif test_index == 5:
        return current_directory + 'grail_data/gralugf2012_105_0352smmmv1.odf'
    elif test_index == 6:
        return current_directory + 'grail_data/gralugf2012_107_0405smmmv1.odf'
    elif test_index == 7:
        return current_directory + 'kernels_test/gralugf2012_109_1227smmmv1.odf' #'grail_data/gralugf2012_108_0450smmmv1.odf'


def get_grail_antenna_file_name(test_index):
    if test_index == 0:
        return current_directory + 'kernels_test/vgs1b_2012_04_07_a_04.asc'
    elif test_index == 1:
        return current_directory + 'kernels_test/vgs1b_2012_04_10_a_04.asc'
    elif test_index == 2:
        return current_directory + 'kernels_test/vgs1b_2012_04_11_a_04.asc'
    elif test_index == 3:
        return current_directory + 'kernels_test/vgs1b_2012_04_12_a_04.asc'
    elif test_index == 4:
        return current_directory + 'kernels_test/vgs1b_2012_04_13_a_04.asc'
    elif test_index == 5:
        return current_directory + 'kernels_test/vgs1b_2012_04_15_a_04.asc'
    elif test_index == 6:
        return current_directory + 'kernels_test/vgs1b_2012_04_17_a_04.asc'
    elif test_index == 7:
        return current_directory + 'kernels_test/vgs1b_2012_04_18_a_04.asc'


def get_grail_panel_geometry():
    # First read the panel data from input file
    this_file_path = os.path.dirname(os.path.abspath(__file__))
    panel_data = pd.read_csv(this_file_path + "/input/grail_macromodel.txt", delimiter=", ", engine="python")
    material_data = pd.read_csv(this_file_path + "/input/grail_materials.txt", delimiter=", ", engine="python")

    # Initialize list to store all panel settings
    all_panel_settings = []

    for i, row in panel_data.iterrows():
        # create panel geometry settings
        # Options are: frame_fixed_panel_geometry, time_varying_panel_geometry, body_tracking_panel_geometry
        panel_geometry_settings = environment_setup.vehicle_systems.frame_fixed_panel_geometry(
            np.array([row["x"], row["y"], row["z"]]),  # panel position in body reference frame
            row["area"]  # panel area
        )

        panel_material_data = material_data[material_data["material"] == row["material"]]

        # create panel radiation settings (for specular and diffuse reflection)
        specular_diffuse_body_panel_reflection_settings = environment_setup.radiation_pressure.specular_diffuse_body_panel_reflection(
            specular_reflectivity=float(panel_material_data["Cs"].iloc[0]),
            diffuse_reflectivity=float(panel_material_data["Cd"].iloc[0]), with_instantaneous_reradiation=True
        )

        # Create settings for complete panel (combining geometry and material properties relevant for radiation pressure calculations)
        complete_panel_settings = environment_setup.vehicle_systems.body_panel_settings(
            panel_geometry_settings,
            specular_diffuse_body_panel_reflection_settings
        )

        # Add panel settings to list of all panel settings
        all_panel_settings.append(
            complete_panel_settings
        )

    # Create settings object for complete vehicle shape
    full_panelled_body_settings = environment_setup.vehicle_systems.full_panelled_body_settings(
        all_panel_settings
    )
    return full_panelled_body_settings


def get_rsw_state_difference(estimated_state_history, spacecraft_name, spacecraft_central_body, global_frame_orientation):

    rsw_state_difference = np.zeros((len(estimated_state_history), 7))
    counter = 0
    for time in estimated_state_history:
        current_estimated_state = estimated_state_history[time]
        current_spice_state = spice.get_body_cartesian_state_at_epoch(spacecraft_name, spacecraft_central_body,
                                                                      global_frame_orientation, "None", time)
        current_state_difference = current_estimated_state - current_spice_state
        current_position_difference = current_state_difference[0:3]
        current_velocity_difference = current_state_difference[3:6]
        rotation_to_rsw = frame_conversion.inertial_to_rsw_rotation_matrix(current_estimated_state)
        # current_rsw_state_difference = np.ndarray([6])
        # current_rsw_state_difference[0:3] = rotation_to_rsw @ current_position_difference
        # current_rsw_state_difference[3:6] = rotation_to_rsw @ current_velocity_difference
        rsw_state_difference[counter, 0] = time
        rsw_state_difference[counter, 1:4] = rotation_to_rsw @ current_position_difference
        rsw_state_difference[counter, 4:7] = rotation_to_rsw @ current_velocity_difference
        # rsw_state_difference[time] = current_rsw_state_difference
        counter = counter + 1

    # rsw_state_difference_array = np.array(list(rsw_state_difference.items()))

    return rsw_state_difference

# MISSING MAIN COMMENT HERE (TO BE MODIFIED)

# This function performs Doppler residual analysis for the MRO spacecraft by adopting the following approach:
# 1) MRO Doppler measurements are loaded from the relevant ODF files
# 2) Synthetic Doppler observables are simulated for all "real" observation times, using the spice kernels as reference for the MRO trajectory
# 3) Residuals are computed as the difference between simulated and real Doppler observations.
#
# The "inputs" variable used as input argument is a list with eight entries:
#   1- the index of the current run (the run_odf_estimation function being run in parallel on several cores in this example)
#   2- the start date of the time interval under consideration
#   3- the end date of the time interval under consideration
#   4- the list of ODF files to be loaded to cover the above-mentioned time interval
#   5- the list of clock files to be loaded
#   6- the list of orientation kernels to be loaded
#   7- the list of tropospheric correction files to be loaded
#   8- the list of ionospheric correction files to be loaded
#   9- the GRAIL manoeuvres file to be loaded
#   10- the antennas switch files to be loaded

def run_odf_estimation(inputs):

    # Unpack various input arguments
    input_index = inputs[0]

    # # TO BE CHECKED AND MODIFIED! Convert start and end datetime objects to Tudat Time variables. A time buffer of one day is subtracted/added to the start/end date
    # # to ensure that the simulation environment covers the full time span of the loaded ODF files. This is mostly needed because some ODF
    # # files - while typically assigned to a certain date - actually spans over (slightly) longer than one day. Without this time buffer,
    # # some observation epochs might thus lie outside the time boundaries within which the dynamical environment is defined.
    # start_time = time_conversion.datetime_to_tudat(inputs[1]).epoch().to_float() - 86400.0
    # end_time = time_conversion.datetime_to_tudat(inputs[2]).epoch().to_float() + 86400.0

    # Retrieve lists of relevant kernels and input files to load (ODF files, clock and orientation kernels,
    # tropospheric and ionospheric corrections, manoeuvres file, antennas switch files)
    odf_files = inputs[3]
    clock_files = inputs[4]
    orientation_files = inputs[5]
    tro_files = inputs[6]
    ion_files = inputs[7]
    manoeuvre_file = inputs[8]
    antennas_switch_files = inputs[9]

    print('list odf files', odf_files)


    with util.redirect_std('grail_odf_estimation_output_' + str(input_index) + ".dat", True, True):

        print("input_index", input_index)

        # Print the list of input arguments, sorted per parallel run
        print('inputs', inputs)

        global_filename_suffix = str(input_index) + '_blip'

        # while True:
        number_of_files = 8
        test_index = input_index % number_of_files

        ### ------------------------------------------------------------------------------------------
        ### LOAD ALL REQUESTED KERNELS AND FILES
        ### ------------------------------------------------------------------------------------------

        # Load standard spice kernels
        spice.load_standard_kernels()

        # Load specific Moon kernels
        spice.load_kernel(current_directory + "grail_kernels/moon_de440_200625.tf")
        spice.load_kernel(current_directory + "grail_kernels/moon_pa_de440_200625.bpc")

        # Load GRAIL frame definition file (useful for spacecraft-fixed frames definition)
        spice.load_kernel(current_directory + "grail_kernels/grail_v07.tf")

        # Load GRAIL orientation kernels (over the entire relevant time period).
        # REMOVE!!! --- Note: each orientation kernel covers a certain time interval, usually spanning over a few days.
        # It must be noted, however, that some dates are not entirely covered, i.e., there is no orientation information available over
        # some short periods of time. This typically happens on dates coinciding with the transition from one orientation file to the
        # next one. As will be shown later in the examples, this requires the user to manually specify which dates should be overlooked,
        # and to filter out observations that were recorded on such dates.
        for orientation_file in orientation_files:
            spice.load_kernel(current_directory + orientation_file)

        # Load GRAIL clock files
        for clock_file in clock_files:
            spice.load_kernel(current_directory + clock_file)

        # Load GRAIL trajectory kernel
        spice.load_kernel(current_directory + "grail_kernels/grail_120301_120529_sci_v02.bsp")

        # load_clock_kernels(test_index)
        # load_orientation_kernels(test_index)


        # Define start and end times for environment
        initial_time_environment = time_conversion.DateTime(2012, 3, 2, 0, 0, 0.0).epoch()
        final_time_environment = time_conversion.DateTime(2012, 5, 29, 0, 0, 0.0).epoch()

        # start_date = datetime(2012, 3, 30)
        # end_date = datetime(2012, 4, 12)

        # clock_files, orientation_files_to_load, tro_files_to_load, ion_files_to_load, manoeuvres_file, \
        # antenna_files_to_load, odf_files_to_load = get_grail_files("kernels_test/", start_date, end_date)


        ### ------------------------------------------------------------------------------------------
        ### LOAD ODF OBSERVATIONS AND PERFORM PRE-PROCESSING STEPS
        ### ------------------------------------------------------------------------------------------

        # Load ODF files ### CHANGE TO MULTIPLE ODF FILES
        # single_odf_file_contents = estimation_setup.observation.process_odf_data_single_file(
        #     get_grail_odf_file_name(test_index), 'GRAIL-A', True)
        multi_odf_file_contents = estimation_setup.observation.process_odf_data_multiple_files(odf_files, 'GRAIL-A', True)

        # Create observation collection from ODF files, only retaining Doppler observations. An observation collection contains
        # multiple "observation sets". Within a given observation set, the observables are of the same type (here Doppler) and
        # defined from the same link ends. However, within the "global" observation collection, multiple observation sets can
        # typically be found for a given observable type and link ends, but they will cover different observation time intervals.
        # When loading ODF data, a separate observation set is created for each ODF file (which means the time intervals of each
        # set match those of the corresponding ODF file).
        original_odf_observations = estimation_setup.observation.create_odf_observed_observation_collection(
            multi_odf_file_contents, [estimation_setup.observation.dsn_n_way_averaged_doppler],
            [numerical_simulation.Time(0, np.nan), numerical_simulation.Time(0, np.nan)])


        observation_time_limits = original_odf_observations.time_bounds
        initial_time = observation_time_limits[0] - 3600.0
        final_time = observation_time_limits[1] + 3600.0

        print('original_odf_observations')
        original_odf_observations.print_observation_sets_start_and_size()

        print('Initial time', initial_time.to_float())
        print('Final time', final_time.to_float())
        print('Time in hours: ', (final_time.to_float() - initial_time.to_float()) / 3600)


        print('original_odf_observations')
        original_odf_observations.print_observation_sets_start_and_size()

        arc_start_times = [initial_time]
        arc_end_times = [final_time]


        ### ------------------------------------------------------------------------------------------
        ### CREATE DYNAMICAL ENVIRONMENT
        ### ------------------------------------------------------------------------------------------

        # Create default body settings for celestial bodies
        bodies_to_create = ["Earth", "Sun", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Moon"]
        global_frame_origin = "SSB"
        global_frame_orientation = "J2000"
        body_settings = environment_setup.get_default_body_settings_time_limited(
            bodies_to_create, initial_time_environment.to_float(), final_time_environment.to_float(),
            global_frame_origin, global_frame_orientation)

        # Modify default shape, rotation, and gravity field settings for the Earth
        body_settings.get('Earth').shape_settings = environment_setup.shape.oblate_spherical_spice()
        body_settings.get('Earth').rotation_model_settings = environment_setup.rotation_model.gcrs_to_itrs(
            environment_setup.rotation_model.iau_2006, global_frame_orientation,
            interpolators.interpolator_generation_settings_float(interpolators.cubic_spline_interpolation(),
                                                                 initial_time_environment.to_float(),
                                                                 final_time_environment.to_float(), 3600.0),
            interpolators.interpolator_generation_settings_float(interpolators.cubic_spline_interpolation(),
                                                                 initial_time_environment.to_float(),
                                                                 final_time_environment.to_float(), 3600.0),
            interpolators.interpolator_generation_settings_float(interpolators.cubic_spline_interpolation(),
                                                                 initial_time_environment.to_float(),
                                                                 final_time_environment.to_float(), 60.0))
        body_settings.get('Earth').gravity_field_settings.associated_reference_frame = "ITRS"

        # Set up DSN ground stations
        body_settings.get("Earth").ground_station_settings = environment_setup.ground_station.dsn_stations()

        # Modify default rotation and gravity field settings for the Moon
        body_settings.get('Moon').rotation_model_settings = environment_setup.rotation_model.spice(
            global_frame_orientation, "MOON_PA_DE440", "MOON_PA_DE440")
        body_settings.get( 'Moon').gravity_field_settings = environment_setup.gravity_field.predefined_spherical_harmonic(
            environment_setup.gravity_field.gggrx1200, 500)
        body_settings.get('Moon').gravity_field_settings.associated_reference_frame = "MOON_PA_DE440"

        # Define gravity field variations for the tides on the Moon
        moon_gravity_field_variations = list()
        moon_gravity_field_variations.append(
            environment_setup.gravity_field_variation.solid_body_tide('Earth', 0.02405, 2))
        moon_gravity_field_variations.append(
            environment_setup.gravity_field_variation.solid_body_tide('Sun', 0.02405, 2))
        body_settings.get('Moon').gravity_field_variation_settings = moon_gravity_field_variations
        body_settings.get('Moon').ephemeris_settings.frame_origin = "Earth"

        # Add Moon radiation properties
        moon_surface_radiosity_models = [
            radiation_pressure.thermal_emission_angle_based_radiosity(  95.0, 385.0, 0.95, "Sun"),
            radiation_pressure.variable_albedo_surface_radiosity(
                radiation_pressure.predefined_spherical_harmonic_surface_property_distribution(radiation_pressure.albedo_dlam1), "Sun")]
        body_settings.get("Moon").radiation_source_settings = radiation_pressure.panelled_extended_radiation_source(
            moon_surface_radiosity_models, [6, 12])

        # Create empty settings for the GRAIL spacecraft
        spacecraft_name = "GRAIL-A"
        spacecraft_central_body = "Moon"
        body_settings.add_empty_settings(spacecraft_name)
        body_settings.get(spacecraft_name).constant_mass = 150

        # Define translational ephemeris from SPICE
        body_settings.get(spacecraft_name).ephemeris_settings = environment_setup.ephemeris.interpolated_spice(
            initial_time_environment.to_float(), final_time_environment.to_float(), 10.0, spacecraft_central_body, global_frame_orientation)

        # Define rotational ephemeris from SPICE
        body_settings.get(spacecraft_name).rotation_model_settings = environment_setup.rotation_model.spice(
            global_frame_orientation, spacecraft_name + "_SPACECRAFT", "")

        # Define GRAIL panel geometry, which will be used for the panel radiation pressure model
        body_settings.get(spacecraft_name).vehicle_shape_settings = get_grail_panel_geometry()

        # Create environment
        bodies = environment_setup.create_system_of_bodies(body_settings)

        # Add radiation pressure target models for GRAIL (cannonball model for the solar radiation pressure,
        # and complete panel model for the radiation pressure from the Moon)
        occulting_bodies = dict()
        occulting_bodies["Sun"] = ["Moon"]
        environment_setup.add_radiation_pressure_target_model(
            bodies, spacecraft_name, radiation_pressure.cannonball_radiation_target(5, 1.5, occulting_bodies))
        environment_setup.add_radiation_pressure_target_model(
            bodies, spacecraft_name, radiation_pressure.panelled_radiation_target(occulting_bodies))

        # Update bodies based on ODF file. This step is necessary to set the antenna transmission frequencies for the GRAIL spacecraft
        estimation_setup.observation.set_odf_information_in_bodies(multi_odf_file_contents, bodies)


        ### ------------------------------------------------------------------------------------------
        ### SET ANTENNA AS REFERENCE POINT FOR DOPPLER OBSERVATIONS
        ### ------------------------------------------------------------------------------------------

        # Load GRAIL's antenna switch files. For each day, the corresponding file contains the position history of the antenna used for radio tracking.
        # As such, they keep track of all switches between the two GRAIL antennas that might have occurred during that day. The GRAIL's antenna positions are
        # provided in the spacecraft-fixed frame.
        antenna_switch_times = []
        antenna_switch_positions = []
        for file in antennas_switch_files:
          antenna_switch_times += grail_antenna_file_reader(current_directory + file)[0]
          antenna_switch_positions += grail_antenna_file_reader(current_directory + file)[1]
        # antenna_switch_times = grail_antenna_file_reader(get_grail_antenna_file_name(test_index))[0]
        # antenna_switch_positions = grail_antenna_file_reader(get_grail_antenna_file_name(test_index))[1]

        print('antenna_switch_times', antenna_switch_times)

        # Reconstruct dictionary containing the antenna switch history (including the GRAIL-fixed position of the relevant antenna at initial and final times)
        antenna_switch_history = dict()
        antenna_switch_history[initial_time.to_float()] = np.array(antenna_switch_positions[0:3])
        for k in range(len(antenna_switch_times)):
            antenna_switch_history[antenna_switch_times[k]] = np.array(antenna_switch_positions[k * 3:(k + 1) * 3])
        antenna_switch_history[final_time.to_float()] = np.array(antenna_switch_positions[-3:])

        print('antenna_switch_history', antenna_switch_history)

        # Set GRAIL's reference point position to follow the antenna switch history (the antennas' positions should be provided in the spacecraft-fixed frame)
        original_odf_observations.set_reference_points(bodies, antenna_switch_history, spacecraft_name, observation.reflector1)

        # NEEDED ????
        # Define arc split times based on antenna switch epochs
        split_times = []
        if (len(antenna_switch_times) > 2):
            for switch_time in antenna_switch_times:
                if (switch_time >= initial_time.to_float() and switch_time <= final_time.to_float()):
                    print("antenna switch detected!")
                    # split_times.append(switch_time)


        ### ------------------------------------------------------------------------------------------
        ### RETRIEVE GRAIL MANOEUVRES EPOCHS
        ### ------------------------------------------------------------------------------------------

        # Load the times at which the spacecraft underwent a manoeuvre from GRAIL's manoeuvres file
        manoeuvres_times = grail_mass_level_0_file_reader(current_directory + manoeuvre_file) # '/grail_data/mas00_2012_04_06_a_04.asc')

        # Store the manoeuvres epochs if they occur within the time interval under consideration # TO BE MODIFIED
        relevant_manoeuvres = []
        for manoeuvre_time in manoeuvres_times:
            if (manoeuvre_time >= initial_time.to_float() and manoeuvre_time <= final_time.to_float()):
                print("manoeuvre detected!")
                split_times.append(manoeuvre_time)
                relevant_manoeuvres.append(manoeuvre_time)



        # Pre-splitting arc definition (TO BE REMOVED)
        for arc in range(len(arc_start_times)):
            print('arc start time', arc_start_times[arc].to_float())
            print('arc end time', arc_end_times[arc].to_float())

        # Define new arcs after splitting based on antenna switches
        for time in split_times:
            print('split set at ', time)
            new_arc_start_dt = datetime.utcfromtimestamp(time) + (datetime(2000, 1, 1, 12) - datetime(1970, 1, 1))
            new_arc_start = time_conversion.datetime_to_tudat(new_arc_start_dt).epoch()

            if (input_index == 0 or input_index==1):
                for arc in range(len(arc_start_times)):
                    if (new_arc_start.to_float() > arc_start_times[arc].to_float() and new_arc_start.to_float() <
                            arc_end_times[arc]):
                        arc_start_times.insert(arc + 1, new_arc_start)
                        arc_end_times.insert(arc, new_arc_start)

        # Post-splitting arc definition (TO BE REMOVED)
        for arc in range(len(arc_start_times)):
            print('arc start time', arc_start_times[arc].to_float())
            print('arc end time', arc_end_times[arc].to_float())

        nb_arcs = len(arc_start_times)

        # times_manoeuvres = []
        # for time in relevant_manoeuvres:
        #     time_dt = datetime.utcfromtimestamp(time) + (datetime(2000, 1, 1, 12) - datetime(1970, 1, 1))
        #     times_manoeuvres.append(time_conversion.datetime_to_tudat(time_dt).epoch())

        np.savetxt('relevant_manoeuvres_' + str(input_index) + '.dat', relevant_manoeuvres, delimiter=',')
        np.savetxt('split_times_' + str(input_index) + '.dat', split_times, delimiter=',')


        ### ------------------------------------------------------------------------------------------
        ### DEFINE PROPAGATION SETTINGS
        ### ------------------------------------------------------------------------------------------

        # Define list of accelerations acting on GRAIL
        accelerations_settings_spacecraft = dict(
            Sun=[
                propagation_setup.acceleration.radiation_pressure(environment_setup.radiation_pressure.paneled_target),
                propagation_setup.acceleration.point_mass_gravity()],
            Earth=[
                propagation_setup.acceleration.point_mass_gravity()],
            Moon=[
                propagation_setup.acceleration.spherical_harmonic_gravity(256, 256),
                propagation_setup.acceleration.radiation_pressure(environment_setup.radiation_pressure.cannonball_target),
                propagation_setup.acceleration.empirical()
            ],
            Mars=[
                propagation_setup.acceleration.point_mass_gravity()],
            Venus=[
                propagation_setup.acceleration.point_mass_gravity()],
            Jupiter=[
                propagation_setup.acceleration.point_mass_gravity()],
            Saturn=[
                propagation_setup.acceleration.point_mass_gravity()]
        )

        # Add manoeuvres if necessary # TO BE MODIFIED POSSIBLY
        if len(relevant_manoeuvres) > 0:
            accelerations_settings_spacecraft[spacecraft_name] = [
                propagation_setup.acceleration.quasi_impulsive_shots_acceleration(relevant_manoeuvres, [np.zeros((3, 1))], 3600.0, 60.0)]

        # Create accelerations settings dictionary
        acceleration_settings = {spacecraft_name: accelerations_settings_spacecraft}

        # Create acceleration models from settings
        bodies_to_propagate = [spacecraft_name]
        central_bodies = [spacecraft_central_body]
        acceleration_models = propagation_setup.create_acceleration_models(bodies, acceleration_settings, bodies_to_propagate, central_bodies)

        # Define integrator settings
        integration_step = 30.0
        integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
            numerical_simulation.Time(0, integration_step), propagation_setup.integrator.rkf_78)

        # Retrieve initial states at the beginning of all arcs
        initial_states = []
        for time in arc_start_times:
            initial_states.append(propagation.get_state_of_bodies(bodies_to_propagate, central_bodies, bodies, time))
        print('initial states', initial_states)

        # Define list of arc-wise propagator settings
        propagator_settings_list = []
        for i in range(nb_arcs):
            arc_wise_propagator_settings = propagation_setup.propagator.translational(
                central_bodies, acceleration_models, bodies_to_propagate, initial_states[i], arc_start_times[i],
                integrator_settings, propagation_setup.propagator.time_termination(arc_end_times[i].to_float()))
            arc_wise_propagator_settings.print_settings.results_print_frequency_in_steps = 3600.0 / integration_step
            propagator_settings_list.append(arc_wise_propagator_settings)
        # propagator_settings = propagation_setup.propagator.multi_arc(propagator_settings_list)


        # Compress Doppler observations from 1.0 s integration time to 60.0 s
        compressed_observations = estimation_setup.observation.create_compressed_doppler_collection(
            original_odf_observations, 60, 10)
        print('Compressed observations: ')
        print(compressed_observations.concatenated_observations.size)


        ### ------------------------------------------------------------------------------------------
        ### DEFINE SETTINGS TO SIMULATE OBSERVATIONS AND COMPUTE RESIDUALS
        ### ------------------------------------------------------------------------------------------

        # Create light-time corrections list
        light_time_correction_list = list()
        light_time_correction_list.append(
            estimation_setup.observation.first_order_relativistic_light_time_correction(["Sun"]))

        # Add tropospheric correction
        # tropospheric_correction_files = [
        #     current_directory + 'grail_data/grxlugf2012_092_2012_122.tro',
        #     current_directory + 'grail_data/grxlugf2012_122_2012_153.tro']
        light_time_correction_list.append(
            estimation_setup.observation.dsn_tabulated_tropospheric_light_time_correction(tro_files))

        # Add ionospheric correction
        # ionospheric_correction_files = [
        #     current_directory + 'grail_data/gralugf2012_092_2012_122.ion',
        #     current_directory + 'grail_data/gralugf2012_122_2012_153.ion']
        spacecraft_name_per_id = dict()
        spacecraft_name_per_id[177] = "GRAIL-A"
        light_time_correction_list.append(
            estimation_setup.observation.dsn_tabulated_ionospheric_light_time_correction(ion_files, spacecraft_name_per_id))

        # Create observation model settings for the Doppler observables. This first implies creating the link ends defining all relevant
        # tracking links between various ground stations and the MRO spacecraft. The list of light-time corrections defined above is then
        # added to each of these link ends.
        doppler_link_ends = compressed_observations.link_definitions_per_observable[
            estimation_setup.observation.dsn_n_way_averaged_doppler]

        observation_model_settings = list()
        for current_link_definition in doppler_link_ends:
            observation_model_settings.append(estimation_setup.observation.dsn_n_way_doppler_averaged(
                current_link_definition, light_time_correction_list))

        # Create observation simulators
        observation_simulators = estimation_setup.create_observation_simulators(observation_model_settings, bodies)

        # ADD DEPENDENT VARIABLES ????
        # # Add elevation and SEP angles dependent variables to the compressed observation collection
        # elevation_angle_settings = observation.elevation_angle_dependent_variable( observation.receiver )
        # elevation_angle_parser = compressed_observations.add_dependent_variable( elevation_angle_settings, bodies )
        # sep_angle_settings = observation.avoidance_angle_dependent_variable("Sun", observation.retransmitter, observation.receiver)
        # sep_angle_parser = compressed_observations.add_dependent_variable( sep_angle_settings, bodies )

        # per_set_time_bounds = compressed_observations.sorted_per_set_time_bounds
        # print('Arc times ================= ')
        # for observable_type in per_set_time_bounds:
        #     for link_end_index in per_set_time_bounds[observable_type]:
        #         current_times_list = per_set_time_bounds[observable_type][link_end_index]
        #         for time_bounds in current_times_list:
        #             print('Arc times', observable_type, ' ', link_end_index, ' ', time_bounds)

        # Compute and set residuals in the compressed observation collection
        estimation.compute_residuals_and_dependent_variables(compressed_observations, observation_simulators, bodies)


        for arc in range(0, nb_arcs):

            filename_suffix = global_filename_suffix
            if (nb_arcs > 1):
                filename_suffix += '_arc_' + str(arc)

            # Create new observation collection based on new arcs definition
            arc_wise_obs_collection = estimation.create_new_observation_collection(
                compressed_observations, estimation.observation_parser((arc_start_times[arc].to_float(), arc_end_times[arc].to_float())))
            print('Arc-wise observations: ')
            print(arc_wise_obs_collection.concatenated_observations.size)

            if (arc_wise_obs_collection.concatenated_observations.size > 0):

                # Save unfiltered residuals
                np.savetxt('unfiltered_residual_' + filename_suffix + '.dat',
                           arc_wise_obs_collection.get_concatenated_residuals(), delimiter=',')
                np.savetxt('unfiltered_obs_' + filename_suffix + '.dat',
                           arc_wise_obs_collection.get_concatenated_observations(), delimiter=',')
                np.savetxt('unfiltered_time_' + filename_suffix + '.dat',
                           arc_wise_obs_collection.concatenated_float_times, delimiter=',')
                np.savetxt('unfiltered_link_end_ids_' + filename_suffix + '.dat',
                           arc_wise_obs_collection.concatenated_link_definition_ids, delimiter=',')

                print('obs sets start and size pre-filtering')
                arc_wise_obs_collection.print_observation_sets_start_and_size()

                # Filter residual outliers
                arc_wise_obs_collection.filter_observations(
                    estimation.observation_filter(estimation.residual_filtering, 0.01))

                # ### ---------------------------------------------------------- TO BE REMOVED
                # print('obs sets start and size pre-splitting')
                # arc_wise_obs_collection.print_observation_sets_start_and_size()
                #
                # obs_sets = arc_wise_obs_collection.get_single_observation_sets()
                # print('nb obs sets', len(obs_sets))
                #
                # for set in obs_sets:
                #     print('time bounds', set.time_bounds[0].to_float(), " - ", set.time_bounds[1].to_float())
                #
                # for time in split_times:
                #     print('split set at ', time)
                # # ----------------------------------------------------------------

                # # Split observation sets based on antenna switch times and manoeuvres
                # arc_wise_obs_collection.split_observation_sets(
                #     estimation.observation_set_splitter(estimation.time_tags_splitter, split_times, 10))
                #
                # print('obs sets start and size post-splitting')
                # arc_wise_obs_collection.print_observation_sets_start_and_size()

                print('Filtered observations: ')
                print(arc_wise_obs_collection.concatenated_observations.size)

                # Save filtered residuals
                np.savetxt('filtered_residual_' + filename_suffix + '.dat',
                           arc_wise_obs_collection.get_concatenated_residuals(), delimiter=',')
                np.savetxt('filtered_time_' + filename_suffix + '.dat',
                           arc_wise_obs_collection.concatenated_float_times, delimiter=',')
                np.savetxt('filtered_link_end_ids_' + filename_suffix + '.dat',
                           arc_wise_obs_collection.concatenated_link_definition_ids, delimiter=',')


                ### ------------------------------------------------------------------------------------------
                ### DEFINE SET OF PARAMETERS TO BE ESTIMATED
                ### ------------------------------------------------------------------------------------------

                # Define parameters to estimate
                parameter_settings = estimation_setup.parameter.initial_states(propagator_settings_list[arc], bodies)

                # Define empirical acceleration components
                empirical_components = dict()
                empirical_components[estimation_setup.parameter.along_track_empirical_acceleration_component] = \
                    list([estimation_setup.parameter.constant_empirical, estimation_setup.parameter.sine_empirical,
                          estimation_setup.parameter.cosine_empirical])

                # Define list of additional parameters
                extra_parameters = [
                    estimation_setup.parameter.radiation_pressure_target_direction_scaling(spacecraft_name, "Sun"),
                    estimation_setup.parameter.radiation_pressure_target_perpendicular_direction_scaling(
                        spacecraft_name, "Sun"),
                    estimation_setup.parameter.radiation_pressure_target_direction_scaling(spacecraft_name, "Moon"),
                    estimation_setup.parameter.radiation_pressure_target_perpendicular_direction_scaling(
                        spacecraft_name, "Moon"),
                    estimation_setup.parameter.empirical_accelerations(spacecraft_name, "Moon", empirical_components),
                ]

                # # Include the estimation of the manoeuvres if any # TO BE MODIFIED (???)
                # if len(relevant_manoeuvres) > 0:
                #     extra_parameters.append(estimation_setup.parameter.quasi_impulsive_shots(spacecraft_name))

                # Add additional parameters settings
                parameter_settings += extra_parameters

                # Create set of parameters to estimate
                parameters_to_estimate = estimation_setup.create_parameter_set(parameter_settings, bodies, propagator_settings_list[arc])
                estimation_setup.print_parameter_names(parameters_to_estimate)

                # Store initial parameters values. This is only necessary to make sure to reset the parameters to their original values from
                # one arc to the next
                # STILL NECESSARY?
                original_parameters = parameters_to_estimate.parameter_vector

                print('parameters', parameters_to_estimate.parameter_vector)


                ### ------------------------------------------------------------------------------------------
                ### DEFINE ESTIMATION SETTINGS AND PERFORM THE FIT
                ### ------------------------------------------------------------------------------------------

                # Create estimator
                estimator = numerical_simulation.Estimator( bodies, parameters_to_estimate, observation_model_settings, propagator_settings_list[arc] )

                # print('test state GRAIL', propagation.get_state_of_bodies(bodies_to_propagate, central_bodies, bodies, arc_start_times[1]))

                estimation_input = estimation.EstimationInput(
                    arc_wise_obs_collection, convergence_checker = estimation.estimation_convergence_checker( 2 ) )
                estimation_input.define_estimation_settings(
                    reintegrate_equations_on_first_iteration = False,
                    reintegrate_variational_equations = False,
                    print_output_to_terminal = True,
                    save_state_history_per_iteration=True)
                estimation_output = estimator.perform_estimation(estimation_input)
                np.savetxt('filtered_postfit_residual_' + filename_suffix + '.dat',
                           estimation_output.final_residuals, delimiter=',')
                estimated_state_history = estimation_output.simulation_results_per_iteration[-1].dynamics_results.state_history_float

                # Estimated parameters
                print("estimated parameters", parameters_to_estimate.parameter_vector)

                # Reset parameters for next arc
                parameters_to_estimate.parameter_vector = original_parameters

                print('Getting RSW difference',len(estimated_state_history),len(estimation_output.simulation_results_per_iteration))
                rsw_state_difference = get_rsw_state_difference(
                    estimated_state_history, spacecraft_name, spacecraft_central_body, global_frame_orientation )
                print('Gotten RSW difference')

                np.savetxt('postfit_rsw_state_difference_' + filename_suffix + '.dat', rsw_state_difference, delimiter = ',')


            # fit_to_kernel = False
            # if fit_to_kernel:
            #     estimation_output = estimation.create_best_fit_to_ephemeris( bodies, acceleration_models, bodies_to_propagate, central_bodies, integrator_settings,
            #                                   initial_time, final_time, numerical_simulation.Time( 0, 60.0 ), extra_parameters )
            #     estimated_state_history = estimation_output.simulation_results_per_iteration[-1].dynamics_results.state_history_float
            #     print('Getting RSW difference',len(estimated_state_history),len(estimation_output.simulation_results_per_iteration))
            #
            #     rsw_state_difference = get_rsw_state_difference(
            #         estimated_state_history, spacecraft_name, spacecraft_central_body, global_frame_orientation)
            #     save2txt(rsw_state_difference, 'fit_spice_rsw_state_difference_' + filename_suffix + '.dat', current_directory)
            #
            #     # Estimated parameters
            #     print("estimated parameters", estimation_output.parameter_history[-1])



if __name__ == "__main__":
    print('Start')
    inputs = []


    # Specify the number of cores over which this example is to run
    nb_cores = 4

    ### COMMENT TO BE MODIFIED
    # Define start and end dates for the six time intervals to be analysed in parallel computations.
    # Each parallel run covers two months of data for the example to parse a total timespan of one year.

    start_dates = [datetime(2012, 4, 10), datetime(2012, 4, 16)]
    end_dates = [datetime(2012, 4, 10), datetime(2012, 4, 16)]

    # (clock_files, orientation_files_to_load, tro_files_to_load, ion_files_to_load, manoeuvres_file,
    #     antenna_files_to_load, odf_files_to_load) = get_grail_files("kernels_test/", start_dates[0], end_dates[1])

    # For each parallel run
    for i in range(nb_cores):

        index_dates = i%2

        start_date = start_dates[i%2]
        end_date = end_dates[i%2]

        # First retrieve the names of all the relevant kernels and data files necessary to cover the specified time interval
        (clock_files, orientation_files_to_load, tro_files_to_load, ion_files_to_load, manoeuvres_file,
         antenna_files_to_load, odf_files_to_load) = get_grail_files("kernels_test/", start_date, end_date)

        # Construct a list of input arguments containing the arguments needed this specific parallel run.
        # These include the start and end dates, along with the names of all relevant kernels and data files that should be loaded
        inputs.append([i, start_date, end_date, odf_files_to_load, clock_files, orientation_files_to_load, tro_files_to_load, ion_files_to_load,
                       manoeuvres_file, antenna_files_to_load])

    # # Print the list of input arguments, sorted per parallel run
    # print('inputs', inputs)

    # Run parallel GRAIL estimations from ODF data
    with mp.get_context("fork").Pool(nb_cores) as pool:
        pool.map(run_odf_estimation, inputs)



