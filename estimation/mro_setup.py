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
from tudatpy.io import save2txt
# from tudatpy.io import grail_mass_level_0_file_reader
# from tudatpy.io import grail_antenna_file_reader
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

current_directory = os.getcwd()


def get_mro_files(local_path, start_date, end_date):
    all_dates = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]

    # Clock files
    print('---------------------------------------------')
    print('Download MRO clock files')
    clock_files = ["mro_sclkscet_00112_65536.tsc"]
    url_clock_files = "https://naif.jpl.nasa.gov/pub/naif/pds/data/mro-m-spice-6-v1.0/mrosp_1000/data/sclk/"
    for file in clock_files:
        if (os.path.exists(local_path + file) == False):
            print('download', local_path + file)
            urlretrieve(url_clock_files + file, local_path + file)

    print('relevant clock files')
    for k in range(len(clock_files)):
        clock_files[k] = local_path + clock_files[k]
        print(clock_files[k])

    # Orientation files
    print('---------------------------------------------')
    print('Download MRO orientation kernels')
    url_orientation_files = "https://naif.jpl.nasa.gov/pub/naif/pds/data/mro-m-spice-6-v1.0/mrosp_1000/data/ck/"
    orientation_files_to_load = download_url_files_time_interval(
        local_path=local_path, filename_format='mro_sc_psp_*.bc', start_date=start_date, end_date=end_date,
        url=url_orientation_files, time_interval_format='%y%m%d_%y%m%d')

    antenna_files_to_load = download_url_files_time_interval(
        local_path=local_path, filename_format='mro_hga_psp_*.bc', start_date=start_date, end_date=end_date,
        url=url_orientation_files, time_interval_format='%y%m%d_%y%m%d')

    for file in antenna_files_to_load:
        orientation_files_to_load.append(file)

    print('relevant orientation files')
    for f in orientation_files_to_load:
        print(f)

    # Tropospheric corrections
    print('---------------------------------------------')
    print('Download MRO tropospheric corrections files')
    url_tro_files = "https://pds-geosciences.wustl.edu/mro/mro-m-rss-1-magr-v1/mrors_0xxx/ancillary/tro/"
    tro_files_to_load = download_url_files_time_interval(
        local_path=local_path, filename_format='mromagr*.tro', start_date=start_date,
        end_date=end_date, url=url_tro_files, time_interval_format='%Y_%j_%Y_%j')

    print('relevant tropospheric corrections files')
    for f in tro_files_to_load:
        print(f)

    # Ionospheric corrections
    print('---------------------------------------------')
    print('Download MRO ionospheric corrections files')
    url_ion_files = "https://pds-geosciences.wustl.edu/mro/mro-m-rss-1-magr-v1/mrors_0xxx/ancillary/ion/"
    ion_files_to_load = download_url_files_time_interval(local_path=local_path, filename_format='mromagr*.ion',
                                                         start_date=start_date,
                                                         end_date=end_date, url=url_ion_files,
                                                         time_interval_format='%Y_%j_%Y_%j')

    print('relevant ionospheric corrections files')
    for f in ion_files_to_load:
        print(f)

    # ODF files
    print('---------------------------------------------')
    print('Download MRO ODF files')
    url_odf = ("https://pds-geosciences.wustl.edu/mro/mro-m-rss-1-magr-v1/mrors_0xxx/odf/")
    odf_files_to_load = download_url_files_time(
        local_path=local_path, filename_format='mromagr*_\w\w\w\wxmmmv1.odf', start_date=start_date,
        end_date=end_date, url=url_odf, time_format='%Y_%j', filename_size=30, indices_date_filename=[7])

    print('relevant odf files')
    for f in odf_files_to_load:
        print(f)

    return clock_files, orientation_files_to_load, tro_files_to_load, ion_files_to_load, odf_files_to_load


def get_rsw_state_difference(
        estimated_state_history,
        spacecraft_name,
        spacecraft_central_body,
        global_frame_orientation):
    rsw_state_difference = dict()
    counter = 0
    for time in estimated_state_history:
        current_estimated_state = estimated_state_history[time]
        current_spice_state = spice.get_body_cartesian_state_at_epoch(spacecraft_name, spacecraft_central_body,
                                                                      global_frame_orientation, "None", time)
        current_state_difference = current_estimated_state - current_spice_state
        current_position_difference = current_state_difference[0:3]
        current_velocity_difference = current_state_difference[3:6]
        rotation_to_rsw = frame_conversion.inertial_to_rsw_rotation_matrix(current_estimated_state)
        current_rsw_state_difference = np.ndarray([6])
        current_rsw_state_difference[0:3] = rotation_to_rsw @ current_position_difference
        current_rsw_state_difference[3:6] = rotation_to_rsw @ current_velocity_difference
        rsw_state_difference[time] = current_rsw_state_difference
        counter = counter + 1
    return rsw_state_difference


def run_estimation(inputs):

    # Unpack various input arguments
    input_index = inputs[0]

    # Convert start and end datetime objects to Tudat Time variables
    start_time = time_conversion.datetime_to_tudat(inputs[1]).epoch()
    end_time = time_conversion.datetime_to_tudat(inputs[2]).epoch()

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

        # CHECK IF STILL NECESSARY
        odf_files = odf_files[:-1]

        # Load ODF file
        multi_odf_file_contents = estimation_setup.observation.process_odf_data_multiple_files(odf_files, 'MRO', True)

        # Create observation collection from ODF file
        original_odf_observations = estimation_setup.observation.create_odf_observed_observation_collection(
            multi_odf_file_contents, list(),
            [numerical_simulation.Time(0, np.nan), numerical_simulation.Time(0, np.nan)])
        observation_time_limits = original_odf_observations.time_bounds
        initial_time = observation_time_limits[0] - 3600.0
        final_time = observation_time_limits[1] + 3600.0

        # Filter out observations on dates when orientation kernels are incomplete
        dates_to_filter_float = []
        for date in dates_to_filter:
            dates_to_filter_float.append(date.epoch().to_float())
            print('filter day', date.epoch().to_float())
            print("filter day of year", date.day_of_year())
            date_filter = estimation.observation_filter(
                estimation.time_bounds_filtering, date.epoch().to_float() - 3600.0,
                time_conversion.add_days_to_datetime(date, numerical_simulation.Time( 1 ) ).epoch().to_float() + 3600.0)
            original_odf_observations.filter_observations(date_filter)

        # Remove empty single observation sets, if there is any once the filtering is performed
        original_odf_observations.remove_empty_observation_sets()

        # Split observation sets at dates when orientation kernels are incomplete
        date_splitter = estimation.observation_set_splitter(estimation.time_tags_splitter, dates_to_filter_float)
        original_odf_observations.split_observation_sets(date_splitter)

        # Remove empty single observation sets, if there is any once the splitting is performed
        original_odf_observations.remove_empty_observation_sets()

        print('Initial time', initial_time.to_float())
        print('Final time', final_time.to_float())
        print('Time in hours: ', (final_time.to_float() - initial_time.to_float()) / 3600)

        print('original_odf_observations')
        original_odf_observations.print_observation_sets_start_and_size()

        # Create default body settings for celestial bodies
        bodies_to_create = ["Earth", "Sun", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Moon"]
        global_frame_origin = "SSB"
        global_frame_orientation = "J2000"
        body_settings = environment_setup.get_default_body_settings_time_limited(
            bodies_to_create, start_time.to_float(), end_time.to_float(),
            global_frame_origin, global_frame_orientation)

        # Modify Earth default settings
        body_settings.get('Earth').shape_settings = environment_setup.shape.oblate_spherical_spice()
        body_settings.get('Earth').rotation_model_settings = environment_setup.rotation_model.gcrs_to_itrs(
            environment_setup.rotation_model.iau_2006, global_frame_orientation,
            interpolators.interpolator_generation_settings_float(interpolators.cubic_spline_interpolation(),
                                                                 start_time.to_float(),
                                                                 end_time.to_float(), 3600.0),
            interpolators.interpolator_generation_settings_float(interpolators.cubic_spline_interpolation(),
                                                                 start_time.to_float(),
                                                                 end_time.to_float(), 3600.0),
            interpolators.interpolator_generation_settings_float(interpolators.cubic_spline_interpolation(),
                                                                 start_time.to_float(),
                                                                 end_time.to_float(), 60.0))
        body_settings.get('Earth').gravity_field_settings.associated_reference_frame = "ITRS"
        body_settings.get("Earth").ground_station_settings = environment_setup.ground_station.dsn_stations()

        # Create MRO spacecraft settings
        spacecraft_name = "MRO"
        spacecraft_central_body = "Mars"
        body_settings.add_empty_settings(spacecraft_name)

        # Retrieve translational ephemeris from SPICE
        body_settings.get(spacecraft_name).ephemeris_settings = environment_setup.ephemeris.interpolated_spice(
            start_time.to_float(), end_time.to_float(), 10.0, spacecraft_central_body,
            global_frame_orientation)

        # Retrieve rotational ephemeris from SPICE
        body_settings.get(spacecraft_name).rotation_model_settings = environment_setup.rotation_model.spice(
            global_frame_orientation, spacecraft_name + "_SPACECRAFT", "")


        # Create environment
        bodies = environment_setup.create_system_of_bodies(body_settings)

        # Define tabulated antenna MRO-fixed ephemeris wrt spacecraft centre-of-mass
        com_position = np.array([0.0, -1.11, 0.0])

        # Define tabulated position of the MRO antenna with respect to the spacecraft's centre-of-mass in the spacecraft-fixed frame
        antenna_position_tabulated = dict()
        # Parsing all single observation sets
        for single_set_obs_times in original_odf_observations.get_observation_times():
            # For each single observation set, compute the antenna position (spice ID "-74214")
            # with respect to the origin of the MRO-fixed frame (spice ID "-74000") and retrieve the offset between the spacecraft's
            # centre-of-mass and the spacecraft-fixed frame origin
            time = single_set_obs_times[ 0 ].to_float() - 3600.0
            while time <= single_set_obs_times[ -1].to_float() + 3600.0:
                state = np.zeros((6, 1))
                state[:3,0] = spice.get_body_cartesian_position_at_epoch("-74214", "-74000", "MRO_SPACECRAFT", "none", time) - com_position
                antenna_position_tabulated[time] = state
                time += 60.0
        antenna_tabulated_settings = environment_setup.ephemeris.tabulated(antenna_position_tabulated, "-74000",  "MRO_SPACECRAFT")
        antenna_tabulated_ephemeris = environment_setup.ephemeris.create_ephemeris(antenna_tabulated_settings, "Antenna")

        # Update bodies based on ODF file
        estimation_setup.observation.set_odf_information_in_bodies(multi_odf_file_contents, bodies)

        # Set MRO antenna ephemeris (expressed in the spacecraft-fixed frame once corrected for the centre-of-mass / MRO-fixed frame origin offset)
        original_odf_observations.set_reference_point( bodies, antenna_tabulated_ephemeris, "Antenna", "MRO", observation.reflector1)

        # Compress Doppler observations to 60.0 s
        compressed_observations = estimation_setup.observation.create_compressed_doppler_collection(
            original_odf_observations, 60, 10)
        print('Compressed observations: ')
        print(compressed_observations.concatenated_observations.size)

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

        # Create observation model settings
        doppler_link_ends = compressed_observations.link_definitions_per_observable[
            estimation_setup.observation.dsn_n_way_averaged_doppler]

        observation_model_settings = list()
        for current_link_definition in doppler_link_ends:
            observation_model_settings.append(estimation_setup.observation.dsn_n_way_doppler_averaged(
                current_link_definition, light_time_correction_list))

        # Create observation simulators
        observation_simulators = estimation_setup.create_observation_simulators(observation_model_settings, bodies)

        per_set_time_bounds = compressed_observations.sorted_per_set_time_bounds
        print('Arc times ================= ')
        for observable_type in per_set_time_bounds:
            for link_end_index in per_set_time_bounds[observable_type]:
                current_times_list = per_set_time_bounds[observable_type][link_end_index]
                for time_bounds in current_times_list:
                    print('Arc times', observable_type, ' ', link_end_index, ' ', time_bounds)

        # Add elevation and SEP angles dependent variables to the compressed observation collection
        elevation_angle_settings = observation.elevation_angle_dependent_variable( observation.receiver )
        elevation_angle_parser = compressed_observations.add_dependent_variable( elevation_angle_settings, bodies )
        sep_angle_settings = observation.avoidance_angle_dependent_variable("Sun", observation.retransmitter, observation.receiver)
        sep_angle_parser = compressed_observations.add_dependent_variable( sep_angle_settings, bodies )

        # Compute and set residuals in the compressed observation collection
        estimation.compute_residuals_and_dependent_variables(compressed_observations, observation_simulators, bodies)

        # Retrieve RMS and mean of the residuals, sorted per observation set
        rms_residuals = compressed_observations.get_rms_residuals()
        mean_residuals = compressed_observations.get_mean_residuals()

        np.savetxt('mro_unfiltered_residuals_rms_' + filename_suffix + '.dat',
                   np.vstack(rms_residuals), delimiter=',')
        np.savetxt('mro_unfiltered_residuals_mean_' + filename_suffix + '.dat',
                   np.vstack(mean_residuals), delimiter=',')

        # Retrieve time bounds per observation set
        time_bounds_per_set = compressed_observations.get_time_bounds_per_set()
        time_bounds_array = np.zeros((len(time_bounds_per_set), 2))
        for j in range(len(time_bounds_per_set)):
            time_bounds_array[j, 0] = time_bounds_per_set[j][0].to_float()
            time_bounds_array[j, 1] = time_bounds_per_set[j][1].to_float()

        np.savetxt('mro_time_bounds_' + filename_suffix + '.dat', time_bounds_array, delimiter=',')

        # Filter out residuals > 0.1 Hz
        filter_residuals = estimation.observation_filter(estimation.residual_filtering, 0.1)
        compressed_observations.filter_observations(filter_residuals)

        # Remove empty single observation sets, if there is any once the filtering is performed
        compressed_observations.remove_empty_observation_sets()

        # Save unfiltered residuals, observation times and link end IDs.
        np.savetxt('mro_filtered_residual_' + filename_suffix + '.dat',
                   compressed_observations.get_concatenated_residuals(), delimiter=',')
        np.savetxt('mro_filtered_time_' + filename_suffix + '.dat',
                   compressed_observations.concatenated_float_times, delimiter=',')

        # Retrieve RMS and mean residuals after outliers filtering
        rms_filtered_residuals = compressed_observations.get_rms_residuals()
        mean_filtered_residuals = compressed_observations.get_mean_residuals()

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

        np.savetxt('mro_filtered_time_bounds_' + filename_suffix + '.dat', time_bounds_filtered_array, delimiter=',')


        # Retrieve concatenated elevation angle dependent variables
        concatenated_elevation_angles = compressed_observations.concatenated_dependent_variable(elevation_angle_settings)[0]

        # Retrieve concatenated SEP angle dependent variables
        concatenated_sep_angles = compressed_observations.concatenated_dependent_variable(sep_angle_settings)[0]

        np.savetxt('mro_elevation_angles_' + filename_suffix + '.dat', concatenated_elevation_angles, delimiter=',')
        np.savetxt('mro_sep_angles_' + filename_suffix + '.dat', concatenated_sep_angles, delimiter=',')


        if input_index == 0:

            # Create observation parser to retrieve observation-related quantities over the first week of data
            first_week_parser = estimation.observation_parser((start_time.to_float(), start_time.to_float() + 8.0 * 86400.0))

            # Retrieve residuals, observation times and dependent variables over the first week
            first_week_observation_times = compressed_observations.get_concatenated_float_observation_times(first_week_parser)
            first_week_residuals = compressed_observations.get_concatenated_residuals(first_week_parser)
            first_week_elevation_angles = compressed_observations.concatenated_dependent_variable(
                elevation_angle_settings, observation_parser=first_week_parser)[0]
            first_week_link_ends_ids = compressed_observations.get_concatenated_link_definition_ids(first_week_parser)

            print('residuals', len(first_week_residuals))
            print('times', len(first_week_observation_times))

            np.savetxt('mro_first_week_residuals.dat', first_week_residuals, delimiter=',')
            np.savetxt('mro_first_week_times.dat', first_week_observation_times, delimiter=',')
            np.savetxt('mro_first_week_link_end_ids.dat', first_week_link_ends_ids, delimiter=',')
            np.savetxt('mro_first_week_elevation_angles.dat', first_week_elevation_angles, delimiter=',')





if __name__ == "__main__":
    print('Start')
    inputs = []

    nb_cores = 6

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

    trajectory_kernels = []

    for i in range(nb_cores):
        clock_files_to_load, orientation_files_to_load, tro_files_to_load, ion_files_to_load, odf_files_to_load = (
            get_mro_files("mro_kernels/", start_dates[i], end_dates[i]))

        inputs.append([i, start_dates[i], end_dates[i], odf_files_to_load, clock_files_to_load, orientation_files_to_load,
                       tro_files_to_load, ion_files_to_load])


    print('inputs', inputs)

    # Run parallel MC analysis
    with mp.get_context("fork").Pool(nb_cores) as pool:
        pool.map(run_estimation, inputs)


    # Load and concatenated results from all parallel analyses
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

    # Load first week detailed results
    first_week_residuals = np.loadtxt("mro_first_week_residuals.dat")
    first_week_times = np.loadtxt("mro_first_week_times.dat")
    first_week_elevation_angles = np.loadtxt("mro_first_week_elevation_angles.dat")
    first_week_link_ends_ids = np.loadtxt("mro_first_week_link_end_ids.dat")


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
    ax1.scatter((first_week_times - np.min(first_week_times)) / 3600.0, first_week_residuals,
                c=first_week_link_ends_ids, s=10)
    ax1.set_ylim([-0.02, 0.02])
    ax1.set_ylabel('Residuals [Hz]')

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Elevation angle [deg]', color=color)
    ax2.plot((first_week_times - np.min(first_week_times)) / 3600.0, first_week_elevation_angles*180/np.pi, '.',
             markersize=1, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.grid()
    ax1.set_xlim([0, 24])
    ax1.set_xlabel('Time [hours]')
    fig3.tight_layout()
    ax1.set_title('Residuals over one day')


    plt.show()




