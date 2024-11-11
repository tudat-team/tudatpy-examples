# MRO - Comparing Doppler measurements from ODF files to simulated observables
"""
Copyright (c) 2010-2022, Delft University of Technology. All rights reserved. This file is part of the Tudat. Redistribution and use in source and binary forms, with or without modification, are permitted exclusively under the terms of the Modified BSD license. You should have received a copy of the license with this file. If not, please or visit: http://tudat.tudelft.nl/LICENSE.
"""

## Context
"""
"""

# Load required standard modules
import multiprocessing as mp
import os
import numpy as np
from matplotlib import pyplot as plt

# Load required tudatpy modules
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.astro import time_conversion
from tudatpy.math import interpolators
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import estimation, estimation_setup
from tudatpy.numerical_simulation.estimation_setup import observation
from tudatpy import util

from load_pds_files import download_url_files_time, download_url_files_time_interval
from datetime import datetime
from urllib.request import urlretrieve

current_directory = os.getcwd()

# This function retrieves all relevant files necessary to run the example over the time interval of interest
# (and automatically downloads them if they cannot be found locally). It returns a tuple containing the lists of
# relevant clock files, orientation kernels, tropospheric correction files, ionospheric correction files and odf files
# that should be loaded.
def get_mro_files(local_path, start_date, end_date):

    # Clock file (a single file is necessary)
    print('---------------------------------------------')
    print('Download MRO clock ')
    clock_files = ["mro_sclkscet_00112_65536.tsc"]
    # Define url where clock files can be downloaded for MRO
    url_clock_files = "https://naif.jpl.nasa.gov/pub/naif/pds/data/mro-m-spice-6-v1.0/mrosp_1000/data/sclk/"
    for file in clock_files:
        # Check if the relevant clock file exists locally
        if (os.path.exists(local_path + file) == False):
            print('download', local_path + file)
            # If not, download it
            urlretrieve(url_clock_files + file, local_path + file)

    # Print and store all relevant clock file names
    print('relevant clock files')
    for k in range(len(clock_files)):
        clock_files[k] = local_path + clock_files[k]
        print(clock_files[k])


    # Orientation files (multiple orientation kerneks are required, typically covering intervals of a few days)
    # For this MRO example, orientation kernels should be loaded both for the spacecraft and for the MRO antenna specifically.
    print('---------------------------------------------')
    print('Download MRO orientation kernels')
    # Define url where orientation kernels can be downloaded for MRO
    url_orientation_files = "https://naif.jpl.nasa.gov/pub/naif/pds/data/mro-m-spice-6-v1.0/mrosp_1000/data/ck/"
    # Retrieve the names of all spacecraft orientation kernels required to cover the time interval of interest, and download them if they
    # do not exist locally yet.
    orientation_files_to_load = download_url_files_time_interval(
        local_path=local_path, filename_format='mro_sc_psp_*.bc', start_date=start_date, end_date=end_date,
        url=url_orientation_files, time_interval_format='%y%m%d_%y%m%d')

    # Retrieve the names of all antenna orientation kernels required to cover the time interval of interest, and download them if they
    # do not exist locally yet
    antenna_files_to_load = download_url_files_time_interval(
        local_path=local_path, filename_format='mro_hga_psp_*.bc', start_date=start_date, end_date=end_date,
        url=url_orientation_files, time_interval_format='%y%m%d_%y%m%d')

    # Print and store all relevant orientation file names (both for the MRO spacecraft and antenna)
    for file in antenna_files_to_load:
        orientation_files_to_load.append(file)

    print('relevant orientation files')
    for f in orientation_files_to_load:
        print(f)


    # Tropospheric corrections (multiple tropospheric correction files are required, typically covering intervals of a few days)
    print('---------------------------------------------')
    print('Download MRO tropospheric corrections files')
    # Define url where tropospheric correction files can be downloaded for MRO
    url_tro_files = "https://pds-geosciences.wustl.edu/mro/mro-m-rss-1-magr-v1/mrors_0xxx/ancillary/tro/"
    # Retrieve the names of all tropospheric correction files required to cover the time interval of interest, and download them if they
    # do not exist locally yet
    tro_files_to_load = download_url_files_time_interval(
        local_path=local_path, filename_format='mromagr*.tro', start_date=start_date,
        end_date=end_date, url=url_tro_files, time_interval_format='%Y_%j_%Y_%j')

    # Print all relevant tropospheric correction file names
    print('relevant tropospheric corrections files')
    for f in tro_files_to_load:
        print(f)


    # Ionospheric corrections (multiple ionospheric correction files are required, typically covering intervals of a few days)
    print('---------------------------------------------')
    print('Download MRO ionospheric corrections files')
    # Define url where ionospheric correction files can be downloaded for MRO
    url_ion_files = "https://pds-geosciences.wustl.edu/mro/mro-m-rss-1-magr-v1/mrors_0xxx/ancillary/ion/"
    # Retrieve the names of all ionospheric correction files required to cover the time interval of interest, and download them if they
    # do not exist locally yet
    ion_files_to_load = download_url_files_time_interval(
        local_path=local_path, filename_format='mromagr*.ion', start_date=start_date,
        end_date=end_date, url=url_ion_files, time_interval_format='%Y_%j_%Y_%j')

    # Print all relevant ionospheric correction file names
    print('relevant ionospheric corrections files')
    for f in ion_files_to_load:
        print(f)


    # ODF files (multiple ODF files are required, typically one per day)
    print('---------------------------------------------')
    print('Download MRO ODF files')
    # Define url where ODF files can be downloaded for MRO
    url_odf = ("https://pds-geosciences.wustl.edu/mro/mro-m-rss-1-magr-v1/mrors_0xxx/odf/")
    # Retrieve the names of all existing ODF files within the time interval of interest, and download them if they do not exist locally yet
    odf_files_to_load = download_url_files_time(
        local_path=local_path, filename_format='mromagr*_\w\w\w\wxmmmv1.odf', start_date=start_date,
        end_date=end_date, url=url_odf, time_format='%Y_%j', indices_date_filename=[7])

    # Print the name of all relevant ODF files that have been identified over the time interval of interest
    print('relevant odf files')
    for f in odf_files_to_load:
        print(f)


    # Retrieve filenames lists for clock files, orientation kernels, tropospheric and ionospheric corrections, and odf files.
    return clock_files, orientation_files_to_load, tro_files_to_load, ion_files_to_load, odf_files_to_load


# This function performs Doppler residual analysis for the MRO spacecraft by adopting the following approach:
# 1) MRO Doppler measurements are loaded from the relevant ODF files
# 2) Synthetic Doppler observables are simulated for all "real" observation times, using the spice kernels as reference for the MRO trajectory
# 3) Residuals are computed as the difference between simulated and real Doppler observations.
#
# The "inputs" variable used as input argument is a list with eight entries:
#   1- the index of the current run (the perform_residuals_analysis function being run in parallel on several cores in this example)
#   2- the start date of the time interval under consideration
#   3- the end date of the time interval under consideration
#   4- the list of ODF files to be loaded to cover the above-mentioned time interval
#   5- the list of clock files to be loaded
#   6- the list of orientation kernels to be loaded
#   7- the list of tropospheric correction files to be loaded
#   8- the list of ionospheric correction files to be loaded

def perform_residuals_analysis(inputs):

    # Unpack various input arguments
    input_index = inputs[0]

    # Convert start and end datetime objects to Tudat Time variables. A time buffer of one day is subtracted/added to the start/end date
    # to ensure that the simulation environment covers the full time span of the loaded ODF files. This is mostly needed because some ODF
    # files - while typically assigned to a certain date - actually spans over (slightly) longer than one day. Without this time buffer,
    # some observation epochs might thus lie outside the time boundaries within which the dynamical environment is defined.
    start_time = time_conversion.datetime_to_tudat(inputs[1]).epoch().to_float() - 86400.0
    end_time = time_conversion.datetime_to_tudat(inputs[2]).epoch().to_float() + 86400.0

    # Retrieve lists of relevant kernels and input files to load (ODF files, clock and orientation kernels,
    # tropospheric and ionospheric corrections)
    odf_files = inputs[3]
    clock_files_to_load = inputs[4]
    orientation_files_to_load = inputs[5]
    tro_files_to_load = inputs[6]
    ion_files_to_load = inputs[7]

    with util.redirect_std('mro_estimation_output_' + str(input_index) + ".dat", True, True):

        print("input_index", input_index)

        filename_suffix = str(input_index) + ''

        ### ------------------------------------------------------------------------------------------
        ### LOAD ALL REQUESTED KERNELS AND FILES
        ### ------------------------------------------------------------------------------------------

        # Load MRO orientation kernels (over the entire relevant time period).
        # Note: each orientation kernel covers a certain time interval, usually spanning over a few days.
        # It must be noted, however, that some dates are not entirely covered, i.e., there is no orientation information available over
        # some short periods of time. This typically happens on dates coinciding with the transition from one orientation file to the
        # next one. As will be shown later in the examples, this requires the user to manually specify which dates should be overlooked,
        # and to filter out observations that were recorded on such dates.
        spice.load_standard_kernels()
        for orientation_file in orientation_files_to_load:
            spice.load_kernel(orientation_file)

        # Load MRO clock files
        for clock_file in clock_files_to_load:
            spice.load_kernel(clock_file)

        # Load MRO frame definition file (useful for HGA and spacecraft-fixed frames definition)
        spice.load_kernel(current_directory + "/mro_kernels/mro_v16.tf")

        # Load MRO trajectory kernel
        spice.load_kernel(current_directory + "/mro_kernels/mro_psp21.bsp")
        spice.load_kernel(current_directory + "/mro_kernels/mro_psp22.bsp")
        spice.load_kernel(current_directory + "/mro_kernels/mro_psp23.bsp")
        spice.load_kernel(current_directory + "/mro_kernels/mro_psp24.bsp")
        spice.load_kernel(current_directory + "/mro_kernels/mro_psp25.bsp")

        # Load MRO spacecraft structure file (for antenna position in spacecraft-fixed frame)
        spice.load_kernel(current_directory + "/mro_kernels/mro_struct_v10.bsp")

        # Dates to filter out because of incomplete spacecraft orientation kernels (see note when loading orientation kernels on line 168)
        dates_to_filter = [time_conversion.DateTime(2012, 10, 15, 0, 0, 0.0),
                           time_conversion.DateTime(2012, 10, 30, 0, 0, 0),
                           time_conversion.DateTime(2012, 11, 6, 0, 0, 0),
                           time_conversion.DateTime(2012, 11, 7, 0, 0, 0)]

        # Remove latest ODF file.
        # This step is necessary because some ODF observation sets - although time-tagged to a specific date - might spill over the next day.
        # For the latest ODF data set, this might imply stepping outside the time interval that the loaded spice kernels cover.
        odf_files = odf_files[:-1]


        ### ------------------------------------------------------------------------------------------
        ### LOAD ODF OBSERVATIONS AND PERFORM PRE-PROCESSING STEPS
        ### ------------------------------------------------------------------------------------------

        # Load ODF files
        multi_odf_file_contents = estimation_setup.observation.process_odf_data_multiple_files(odf_files, 'MRO', True)

        # Create observation collection from ODF files, only retaining Doppler observations. An observation collection contains
        # multiple "observation sets". Within a given observation set, the observables are of the same type (here Doppler) and defined from the same link ends.
        # However, within the "global" observation collection, multiple observation sets can typically be found for a given observable type and link ends, but they
        # will cover different observation time intervals. When loading ODF data, a separate observation set is created for each ODF file (which means the time intervals of each
        # set match those of the corresponding ODF file).
        original_odf_observations = estimation_setup.observation.create_odf_observed_observation_collection(
            multi_odf_file_contents, [estimation_setup.observation.dsn_n_way_averaged_doppler],
            [numerical_simulation.Time(0, np.nan), numerical_simulation.Time(0, np.nan)])

        # Filter out observations on dates when orientation kernels are incomplete
        dates_to_filter_float = []
        for date in dates_to_filter:
            dates_to_filter_float.append(date.epoch().to_float())
            # Create filter object for specific date
            date_filter = estimation.observation_filter(
                estimation.time_bounds_filtering, date.epoch().to_float() - 0.0,
                time_conversion.add_days_to_datetime(date, numerical_simulation.Time( 1 ) ).epoch().to_float() + 0.0)
            # Filter out observations from observation collection
            original_odf_observations.filter_observations(date_filter)

        # Remove empty observation sets, if there is any once the filtering is completed
        original_odf_observations.remove_empty_observation_sets()

        # Split observation sets at dates when orientation kernels are incomplete.
        # While all problematic observation epochs have already been filtered out in the previous step, the splitting is still necessary
        # to ensure that the time span of each observation set is fully covered by the available kernels. Without this additional step,
        # one could not parse from the lower to upper time bounds of a given observation set without risking accessing unavailable information from spice.
        # This step only becomes relevant when retrieving the position of the MRO antenna over the observation time intervals, as will be done later in the example.
        date_splitter = estimation.observation_set_splitter(estimation.time_tags_splitter, dates_to_filter_float)
        original_odf_observations.split_observation_sets(date_splitter)

        # Remove empty observation sets, if there is any once both the splitting is completed
        original_odf_observations.remove_empty_observation_sets()

        print('original_odf_observations')
        original_odf_observations.print_observation_sets_start_and_size()

        # Compress Doppler observations from 1.0 s integration time to 60.0 s
        compressed_observations = estimation_setup.observation.create_compressed_doppler_collection(
            original_odf_observations, 60, 10)
        print('Compressed observations: ')
        print(compressed_observations.concatenated_observations.size)


        ### ------------------------------------------------------------------------------------------
        ### CREATE DYNAMICAL ENVIRONMENT
        ### ------------------------------------------------------------------------------------------

        # Create default body settings for celestial bodies
        bodies_to_create = ["Earth", "Sun", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Moon"]
        global_frame_origin = "SSB"
        global_frame_orientation = "J2000"
        body_settings = environment_setup.get_default_body_settings_time_limited(
            bodies_to_create, start_time, end_time, global_frame_origin, global_frame_orientation)

        # Modify Earth default settings
        body_settings.get('Earth').shape_settings = environment_setup.shape.oblate_spherical_spice()
        body_settings.get('Earth').rotation_model_settings = environment_setup.rotation_model.gcrs_to_itrs(
            environment_setup.rotation_model.iau_2006, global_frame_orientation,
            interpolators.interpolator_generation_settings_float(interpolators.cubic_spline_interpolation(),
                                                                 start_time, end_time, 3600.0),
            interpolators.interpolator_generation_settings_float(interpolators.cubic_spline_interpolation(),
                                                                 start_time, end_time, 3600.0),
            interpolators.interpolator_generation_settings_float(interpolators.cubic_spline_interpolation(),
                                                                 start_time, end_time, 60.0))
        body_settings.get('Earth').gravity_field_settings.associated_reference_frame = "ITRS"
        body_settings.get("Earth").ground_station_settings = environment_setup.ground_station.dsn_stations()

        # Create empty settings for the MRO spacecraft
        spacecraft_name = "MRO"
        spacecraft_central_body = "Mars"
        body_settings.add_empty_settings(spacecraft_name)

        # Retrieve translational ephemeris from SPICE
        body_settings.get(spacecraft_name).ephemeris_settings = environment_setup.ephemeris.interpolated_spice(
            start_time, end_time, 10.0, spacecraft_central_body, global_frame_orientation)

        # Retrieve rotational ephemeris from SPICE
        body_settings.get(spacecraft_name).rotation_model_settings = environment_setup.rotation_model.spice(
            global_frame_orientation, spacecraft_name + "_SPACECRAFT", "")

        # Create environment
        bodies = environment_setup.create_system_of_bodies(body_settings)

        # Update bodies based on ODF file. This step is necessary to set the antenna transmission frequencies for the MRO spacecraft
        estimation_setup.observation.set_odf_information_in_bodies(multi_odf_file_contents, bodies)


        ### ------------------------------------------------------------------------------------------
        ### SET ANTENNA AS REFERENCE POINT FOR DOPPLER OBSERVATIONS
        ### ------------------------------------------------------------------------------------------

        # Define MRO center-of-mass (COM) position w.r.t. the origin of the MRO-fixed reference frame (frame spice ID: MRO_SPACECRAFT)
        # This value was taken from Konopliv et al. (2011) doi:10.1016/j.icarus.2010.10.004
        # This is necessary to define the position of the antenna w.r.t. the COM, in the MRO-fixed frame (see below)
        com_position = np.array([0.0, -1.11, 0.0])

        # In the following lines, we create a tabulated history of the position of the MRO antenna with respect to the COM, in the MRO-fixed frame
        # (MRO_SPACECRAFT). This will be used to create a tabulated ephemeris for the antenna, necessary to correct for the position of the spacecraft's
        # reference point (antenna) when computing the Doppler observables.
        # This (manual) extra step is required to account for the offset between the COM and the origin of the MRO-fixed frame, as using spice kernels directly
        # would only provide the antenna position w.r.t. the origin of the MRO-fixed frame (no information on the COM position).
        antenna_position_history = dict()

        # Parsing the observation times in all observation sets.
        # Note: we use a time buffer of one hour with respect to the start and end times of each observation set.
        # This is to ensure that the antenna position history spans over a slightly extended time interval than the observation epochs
        # of the given set. When simulating Doppler data to compute the MRO residuals, we might indeed need to access the antenna position
        # slightly outside the exact time bounds defined by the observation epochs because of the light-time delay.
        for obs_times in compressed_observations.get_observation_times():
            time = obs_times[0].to_float() - 3600.0
            while time <= obs_times[-1].to_float() + 3600.0:
                state = np.zeros((6, 1))

                # For each observation epoch, retrieve the antenna position (spice ID "-74214") w.r.t. the origin of the MRO-fixed frame (spice ID "-74000")
                state[:3,0] = spice.get_body_cartesian_position_at_epoch("-74214", "-74000", "MRO_SPACECRAFT", "none", time)

                # Translate the antenna position to account for the offset between the origin of the MRO-fixed frame and the COM
                state[:3,0] = state[:3,0] - com_position

                # Store antenna position w.r.t. COM in the MRO-fixed frame
                antenna_position_history[time] = state
                time += 60.0

        # Create tabulated ephemeris settings from antenna position history
        antenna_ephemeris_settings = environment_setup.ephemeris.tabulated(antenna_position_history, "-74000",  "MRO_SPACECRAFT")

        # Create tabulated ephemeris for the MRO antenna
        antenna_ephemeris = environment_setup.ephemeris.create_ephemeris(antenna_ephemeris_settings, "Antenna")

        # Set the spacecraft's reference point position to that of the antenna (in the MRO-fixed frame)
        compressed_observations.set_reference_point(bodies, antenna_ephemeris, "Antenna", "MRO", observation.reflector1)


        ### ------------------------------------------------------------------------------------------
        ### DEFINE SETTINGS TO SIMULATE OBSERVATIONS AND COMPUTE RESIDUALS
        ### ------------------------------------------------------------------------------------------

        #  Create light-time corrections list
        light_time_correction_list = list()
        light_time_correction_list.append(
            estimation_setup.observation.first_order_relativistic_light_time_correction(["Sun"]))

        # Add tropospheric correction
        light_time_correction_list.append(
            estimation_setup.observation.dsn_tabulated_tropospheric_light_time_correction(tro_files_to_load))

        # Add ionospheric correction
        spacecraft_name_per_id = dict()
        spacecraft_name_per_id[74] = "MRO"
        light_time_correction_list.append(
            estimation_setup.observation.dsn_tabulated_ionospheric_light_time_correction(ion_files_to_load,
                                                                                         spacecraft_name_per_id))

        # Create observation model settings for the Doppler observables. This first implies creating the link ends defining all relevant
        # tracking links between various ground stations and the MRO spacecraft. The list of light-time corrections defined above is then
        # added to each of these link ends.
        doppler_link_ends = compressed_observations.link_definitions_per_observable[
            estimation_setup.observation.dsn_n_way_averaged_doppler]

        observation_model_settings = list()
        for current_link_definition in doppler_link_ends:
            observation_model_settings.append(estimation_setup.observation.dsn_n_way_doppler_averaged(
                current_link_definition, light_time_correction_list))

        # Create observation simulators.
        observation_simulators = estimation_setup.create_observation_simulators(observation_model_settings, bodies)

        # Add elevation and SEP angles dependent variables to the compressed observation collection
        elevation_angle_settings = observation.elevation_angle_dependent_variable( observation.receiver )
        elevation_angle_parser = compressed_observations.add_dependent_variable( elevation_angle_settings, bodies )
        sep_angle_settings = observation.avoidance_angle_dependent_variable("Sun", observation.retransmitter, observation.receiver)
        sep_angle_parser = compressed_observations.add_dependent_variable( sep_angle_settings, bodies )

        # Compute and set residuals in the compressed observation collection
        estimation.compute_residuals_and_dependent_variables(compressed_observations, observation_simulators, bodies)


        ### ------------------------------------------------------------------------------------------
        ### RETRIEVE AND SAVE VARIOUS OBSERVATION OUTPUTS
        ### ------------------------------------------------------------------------------------------

        # Retrieve RMS and mean of the residuals, sorted per observation set
        rms_residuals = compressed_observations.get_rms_residuals()
        mean_residuals = compressed_observations.get_mean_residuals()

        np.savetxt('mro_unfiltered_residuals_rms_' + filename_suffix + '.dat',
                   np.vstack(rms_residuals), delimiter=',')
        np.savetxt('mro_unfiltered_residuals_mean_' + filename_suffix + '.dat',
                   np.vstack(mean_residuals), delimiter=',')

        # Retrieve the time bounds of each observation set within the observation collection
        time_bounds_per_set = compressed_observations.get_time_bounds_per_set()
        time_bounds_array = np.zeros((len(time_bounds_per_set), 2))
        for j in range(len(time_bounds_per_set)):
            time_bounds_array[j, 0] = time_bounds_per_set[j][0].to_float()
            time_bounds_array[j, 1] = time_bounds_per_set[j][1].to_float()

        # Save time bounds of the (unfiltered) observation sets
        np.savetxt('mro_time_bounds_' + filename_suffix + '.dat', time_bounds_array, delimiter=',')

        # Filter out outliers (i.e., residuals > 0.1 Hz)
        filter_residuals = estimation.observation_filter(estimation.residual_filtering, 0.1)
        compressed_observations.filter_observations(filter_residuals)

        # Remove empty observation sets, if there is any once the filtering is performed
        compressed_observations.remove_empty_observation_sets()

        # Save unfiltered residuals, observation times and link end IDs.
        np.savetxt('mro_filtered_residual_' + filename_suffix + '.dat',
                   compressed_observations.get_concatenated_residuals(), delimiter=',')
        np.savetxt('mro_filtered_time_' + filename_suffix + '.dat',
                   compressed_observations.concatenated_float_times, delimiter=',')

        # Retrieve RMS and mean residuals after outliers filtering
        rms_filtered_residuals = compressed_observations.get_rms_residuals()
        mean_filtered_residuals = compressed_observations.get_mean_residuals()

        # Save RMS and mean residuals
        np.savetxt('mro_filtered_residuals_rms_' + filename_suffix + '.dat',
                   np.vstack(rms_filtered_residuals), delimiter=',')
        np.savetxt('mro_filtered_residuals_mean_' + filename_suffix + '.dat',
                   np.vstack(mean_filtered_residuals), delimiter=',')

        # Retrieve time bounds per observation set
        time_bounds_per_filtered_set = compressed_observations.get_time_bounds_per_set()
        time_bounds_filtered_array = np.zeros((len(time_bounds_per_filtered_set), 2))
        for j in range(len(time_bounds_per_filtered_set)):
            time_bounds_filtered_array[j, 0] = time_bounds_per_filtered_set[j][0].to_float()
            time_bounds_filtered_array[j, 1] = time_bounds_per_filtered_set[j][1].to_float()

        # Save time bounds of each observation set
        np.savetxt('mro_filtered_time_bounds_' + filename_suffix + '.dat', time_bounds_filtered_array, delimiter=',')


        # Retrieve concatenated elevation angle dependent variables
        concatenated_elevation_angles = compressed_observations.concatenated_dependent_variable(elevation_angle_settings)[0]

        # Retrieve concatenated SEP angle dependent variables
        concatenated_sep_angles = compressed_observations.concatenated_dependent_variable(sep_angle_settings)[0]

        # Save elevation and SEP angles
        np.savetxt('mro_elevation_angles_' + filename_suffix + '.dat', concatenated_elevation_angles, delimiter=',')
        np.savetxt('mro_sep_angles_' + filename_suffix + '.dat', concatenated_sep_angles, delimiter=',')


        if input_index == 0:

            # Create observation parser to retrieve observation-related quantities over the first day of data
            # (starting from the first observation epoch)
            # first_day_parser = estimation.observation_parser((start_time + 2.0 * 86400.0, start_time + 3.0 * 86400.0))
            start_obs_times = time_bounds_per_filtered_set[0][0].to_float()
            first_day_parser = estimation.observation_parser((start_obs_times, start_obs_times + 86400.0))

            # Retrieve residuals, observation times and dependent variables over the first day
            first_day_observation_times = compressed_observations.get_concatenated_float_observation_times(first_day_parser)
            first_day_residuals = compressed_observations.get_concatenated_residuals(first_day_parser)
            first_day_elevation_angles = compressed_observations.concatenated_dependent_variable(
                elevation_angle_settings, observation_parser=first_day_parser)[0]
            first_day_link_ends_ids = compressed_observations.get_concatenated_link_definition_ids(first_day_parser)

            # Save first day results
            np.savetxt('mro_first_day_residuals.dat', first_day_residuals, delimiter=',')
            np.savetxt('mro_first_day_times.dat', first_day_observation_times, delimiter=',')
            np.savetxt('mro_first_day_link_end_ids.dat', first_day_link_ends_ids, delimiter=',')
            np.savetxt('mro_first_day_elevation_angles.dat', first_day_elevation_angles, delimiter=',')





if __name__ == "__main__":
    print('Start')
    inputs = []

    # Specify the number of cores over which this example is to run
    nb_cores = 6

    # Define start and end dates for the six time intervals to be analysed in parallel computations.
    # Each parallel run covers two months of data for the example to parse a total timespan of one year.

    start_dates = [datetime(2012, 1, 1),
                   datetime(2012, 3, 1),
                   datetime(2012, 5, 1),
                   datetime(2012, 7, 1),
                   datetime(2012, 9, 1),
                   datetime(2012, 11, 1)]

    end_dates = [datetime(2012, 2, 29),
                 datetime(2012, 4, 30),
                 datetime(2012, 6, 30),
                 datetime(2012, 8, 31),
                 datetime(2012, 10, 31),
                 datetime(2012, 12, 31)]

    # For each parallel run
    for i in range(nb_cores):

        # First retrieve the names of all the relevant kernels and data files necessary to cover the specified time interval
        clock_files_to_load, orientation_files_to_load, tro_files_to_load, ion_files_to_load, odf_files_to_load = (
            get_mro_files("mro_kernels/", start_dates[i], end_dates[i]))

        # Construct a list of input arguments containing the arguments needed this specific parallel run.
        # These include the start and end dates, along with the names of all relevant kernels and data files that should be loaded
        inputs.append([i, start_dates[i], end_dates[i], odf_files_to_load, clock_files_to_load, orientation_files_to_load,
                       tro_files_to_load, ion_files_to_load])

    # Print the list of input arguments, sorted per parallel run
    print('inputs', inputs)

    # Run parallel residuals analyses over several cores
    with mp.get_context("fork").Pool(nb_cores) as pool:
        pool.map(perform_residuals_analysis, inputs)


    # Create empty lists of results to store the outcomes of all parallel analyses
    filtered_residuals_list = []
    filtered_times_list = []
    elevation_angles_list = []
    sep_angles_list = []
    rms_residuals_list = []
    mean_residuals_list = []
    rms_filtered_residuals_list = []
    mean_filtered_residuals_list = []
    time_bounds_list = []
    time_bounds_filtered_list = []

    # Load and add results from all parallel analyses
    for i in range(nb_cores):
        filtered_times_list.append(np.loadtxt("mro_filtered_time_" + str(i) + ".dat"))
        filtered_residuals_list.append(np.loadtxt( "mro_filtered_residual_" + str(i) + ".dat" ))
        elevation_angles_list.append(np.loadtxt( "mro_elevation_angles_" + str(i) + ".dat" ))
        sep_angles_list.append(np.loadtxt( "mro_sep_angles_" + str(i) + ".dat" ))
        rms_residuals_list.append(np.loadtxt("mro_unfiltered_residuals_rms_" + str(i) + ".dat"))
        mean_residuals_list.append(np.loadtxt("mro_unfiltered_residuals_mean_" + str(i) + ".dat"))
        rms_filtered_residuals_list.append(np.loadtxt("mro_filtered_residuals_rms_" + str(i) + ".dat"))
        mean_filtered_residuals_list.append(np.loadtxt("mro_filtered_residuals_mean_" + str(i) + ".dat"))
        time_bounds_list.append(np.loadtxt("mro_time_bounds_" + str(i) + ".dat", delimiter=','))
        time_bounds_filtered_list.append(np.loadtxt("mro_filtered_time_bounds_" + str(i) + ".dat", delimiter=','))

    # Concatenate the above results into single output variables
    filtered_times = np.concatenate(filtered_times_list, axis=0)
    filtered_residuals = np.concatenate(filtered_residuals_list, axis=0)
    elevation_angles = np.concatenate(elevation_angles_list, axis=0)
    sep_angles = np.concatenate(sep_angles_list, axis=0)
    rms_residuals = np.concatenate(rms_residuals_list, axis=0)
    mean_residuals = np.concatenate(mean_residuals_list, axis=0)
    rms_filtered_residuals = np.concatenate(rms_filtered_residuals_list, axis=0)
    mean_filtered_residuals = np.concatenate(mean_filtered_residuals_list, axis=0)
    time_bounds = np.concatenate(time_bounds_list, axis=0)
    time_bounds_filtered = np.concatenate(time_bounds_filtered_list, axis=0)

    # Load first day detailed results
    first_day_residuals = np.loadtxt("mro_first_day_residuals.dat")
    first_day_times = np.loadtxt("mro_first_day_times.dat")
    first_day_elevation_angles = np.loadtxt("mro_first_day_elevation_angles.dat")
    first_day_link_ends_ids = np.loadtxt("mro_first_day_link_end_ids.dat")


    # Plot residuals over time
    fig = plt.figure()
    plt.scatter((filtered_times - np.min(filtered_times))/86400.0, filtered_residuals, s=2)
    plt.grid()
    plt.ylim([-0.1, 0.1])
    plt.xlim([0, 365])
    plt.xlabel('Time [days]')
    plt.ylabel('Residuals [Hz]')
    fig.tight_layout()


    # Plot RMS and mean residuals, both pre- and post-filtering of the outliers (residuals > 0.1)
    # The elevation and SEP angles are also plotted.
    fig2, axs = plt.subplots(3, 2, figsize=(10,8))

    axs[0,0].plot((time_bounds[:,0]-np.min(time_bounds))/86400, rms_residuals, '.', markersize=1)
    axs[0,0].grid()
    axs[0,0].set_ylim([-0.01, 0.1])
    axs[0,0].set_xlim([0, 365])
    axs[0,0].set_xlabel('Time [days]')
    axs[0,0].set_ylabel('RMS residuals [Hz]')
    axs[0,0].set_title('RMS residuals')

    axs[0,1].plot((time_bounds[:,0] - np.min(time_bounds)) / 86400, mean_residuals, '.', markersize=1)
    axs[0,1].grid()
    axs[0,1].set_ylim([-0.02, 0.02])
    axs[0,1].set_xlim([0, 365])
    axs[0,1].set_xlabel('Time [days]')
    axs[0,1].set_ylabel('Mean residuals [Hz]')
    axs[0,1].set_title('Mean residuals')

    axs[1, 0].plot((time_bounds_filtered[:,0] - np.min(time_bounds_filtered)) / 86400, rms_filtered_residuals, '.', markersize=1)
    axs[1, 0].grid()
    axs[1, 0].set_ylim([-0.01, 0.1])
    axs[1, 0].set_xlim([0, 365])
    axs[1, 0].set_xlabel('Time [days]')
    axs[1, 0].set_ylabel('RMS residuals [Hz]')
    axs[1, 0].set_title('RMS residuals (post-filtering)')

    axs[1, 1].plot((time_bounds_filtered[:,0] - np.min(time_bounds_filtered)) / 86400, mean_filtered_residuals, '.', markersize=1)
    axs[1, 1].grid()
    axs[1, 1].set_ylim([-0.02, 0.02])
    axs[1, 1].set_xlim([0, 365])
    axs[1, 1].set_xlabel('Time [days]')
    axs[1, 1].set_ylabel('Mean residuals [Hz]')
    axs[1, 1].set_title('Mean residuals (post-filtering)')

    axs[2, 0].plot((filtered_times - np.min(filtered_times)) / 86400, elevation_angles*180/np.pi, '.', markersize=1)
    axs[2, 0].grid()
    axs[2, 0].set_ylim([0, 90])
    axs[2, 0].set_xlim([0, 365])
    axs[2, 0].set_xlabel('Time [days]')
    axs[2, 0].set_ylabel('Elevation angle [deg]')
    axs[2, 0].set_title('Elevation angle')

    axs[2, 1].plot((filtered_times - np.min(filtered_times)) / 86400, sep_angles * 180 / np.pi, '.', markersize=1)
    axs[2, 1].grid()
    axs[2, 1].set_xlim([0, 365])
    axs[2, 1].set_xlabel('Time [days]')
    axs[2, 1].set_ylabel('SEP angle [deg]')
    axs[2, 1].set_title('SEP angle')

    fig2.tight_layout()


    # Plot residuals and elevation angles over one day
    fig3, ax1 =  plt.subplots()
    ax1.scatter((first_day_times - np.min(first_day_times)) / 3600.0, first_day_residuals,
                c=first_day_link_ends_ids, s=10)
    ax1.set_ylim([-0.02, 0.02])
    ax1.set_ylabel('Residuals [Hz]')

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Elevation angle [deg]', color=color)
    ax2.plot((first_day_times - np.min(first_day_times)) / 3600.0, first_day_elevation_angles*180/np.pi, '.',
             markersize=1, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.grid()
    ax1.set_xlim([0, 24])
    ax1.set_xlabel('Time [hours]')
    fig3.tight_layout()
    ax1.set_title('Residuals over one day')


    plt.show()




