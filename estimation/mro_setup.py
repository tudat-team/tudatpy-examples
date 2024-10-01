# MARS EXPRESS - Using Different Dynamical Models for the Simulation of Observations and the Estimation
"""
Copyright (c) 2010-2022, Delft University of Technology. All rights reserved. This file is part of the Tudat. Redistribution and use in source and binary forms, with or without modification, are permitted exclusively under the terms of the Modified BSD license. You should have received a copy of the license with this file. If not, please or visit: http://tudat.tudelft.nl/LICENSE.
"""


## Context
"""
"""

import sys
sys.path.insert(0, "/home/mfayolle/Tudat/tudat-bundle/cmake-build-release/tudatpy")

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

    all_dates = [start_date+timedelta(days=x) for x in range((end_date-start_date).days+1)]

    # Clock files
    print('---------------------------------------------')
    print('Download MRO clock files')
    clock_files=["mro_sclkscet_00112_65536.tsc"]
    url_clock_files="https://naif.jpl.nasa.gov/pub/naif/pds/data/mro-m-spice-6-v1.0/mrosp_1000/data/sclk/"
    for file in clock_files:
        if ( os.path.exists(local_path+file) == False ):
            print('download', local_path+file)
            urlretrieve(url_clock_files+file, local_path+file)

    print('relevant clock files')
    for k in range(len(clock_files)):
        clock_files[ k ] = local_path+clock_files[k]
        print(clock_files[k])


    # Orientation files
    print('---------------------------------------------')
    print('Download MRO orientation kernels')
    url_orientation_files = "https://naif.jpl.nasa.gov/pub/naif/pds/data/mro-m-spice-6-v1.0/mrosp_1000/data/ck/"
    orientation_files_to_load = download_url_files_time_interval(
        local_path=local_path, filename_format='mro_sc_psp_*.bc', start_date=start_date,end_date=end_date,
        url=url_orientation_files, time_interval_format='%y%m%d_%y%m%d')

    antenna_files_to_load = download_url_files_time_interval(
        local_path=local_path, filename_format='mro_hga_psp_*.bc', start_date=start_date,end_date=end_date,
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
        end_date=end_date, url=url_tro_files, time_interval_format='%Y_%j_%Y_%j' )

    print('relevant tropospheric corrections files')
    for f in tro_files_to_load:
        print(f)


    # Ionospheric corrections
    print('---------------------------------------------')
    print('Download MRO ionospheric corrections files')
    url_ion_files = "https://pds-geosciences.wustl.edu/mro/mro-m-rss-1-magr-v1/mrors_0xxx/ancillary/ion/"
    ion_files_to_load = download_url_files_time_interval(local_path=local_path, filename_format='mromagr*.ion', start_date=start_date,
                                     end_date=end_date, url=url_ion_files, time_interval_format='%Y_%j_%Y_%j' )

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
        global_frame_orientation ):
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
        counter = counter+1
    return rsw_state_difference


def run_estimation( inputs ):

    input_index = inputs[0]
    # start_time = inputs[1].epoch()
    # end_time = inputs[2].epoch()
    odf_files = inputs[3]
    clock_files_to_load = inputs[4]
    orientation_files_to_load = inputs[5]
    tro_files_to_load = inputs[6]
    ion_files_to_load = inputs[7]

    with util.redirect_std( 'mro_estimation_output_' + str( input_index ) + ".dat", True, True ):

        print("input_index", input_index)

        filename_suffix = str(input_index) + ''

        number_of_files = 8
        test_index = input_index % number_of_files

        # Load MRO orientation kernels
        spice.load_standard_kernels()
        for orientation_file in orientation_files_to_load:
            spice.load_kernel(orientation_file)
        # Load MRO clock files
        for clock_file in clock_files_to_load:
            spice.load_kernel(clock_file)
        # Load MRO frame definition file
        spice.load_kernel(current_directory + "/mro_kernels/mro_v16.tf")
        # Load MRO trajectory kernel
        spice.load_kernel(current_directory + "/mro_kernels/mro_psp2.bsp")
        # Load MRO spacecraft structure file (for antenna position in spacecraft-fixed frame)
        spice.load_kernel(current_directory + "/mro_kernels/mro_struct_v10.bsp")

        # Define start and end times for environment
        initial_time_environment = time_conversion.DateTime(2007, 1, 2, 0, 0, 0.0).epoch()
        final_time_environment = time_conversion.DateTime(2007, 1, 10, 0, 0, 0.0).epoch()

        # Load ODF file
        multi_odf_file_contents = estimation_setup.observation.process_odf_data_multiple_files(odf_files, 'MRO', True)

        # Create observation collection from ODF file
        original_odf_observations = estimation_setup.observation.create_odf_observed_observation_collection(
            multi_odf_file_contents, list(), [numerical_simulation.Time(0, np.nan), numerical_simulation.Time(0, np.nan)])
        observation_time_limits = original_odf_observations.time_bounds
        initial_time = observation_time_limits[0] - 3600.0
        final_time = observation_time_limits[1] + 3600.0

        print('Initial time', initial_time.to_float())
        print('Final time', final_time.to_float())
        print('Time in hours: ', (final_time.to_float() - initial_time.to_float()) / 3600)

        print('original_odf_observations')
        original_odf_observations.print_observation_sets_start_and_size()

        # Create default body settings for celestial bodies
        bodies_to_create = [ "Earth", "Sun", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Moon" ]
        global_frame_origin = "SSB"
        global_frame_orientation = "J2000"
        body_settings = environment_setup.get_default_body_settings_time_limited(
            bodies_to_create, initial_time_environment.to_float( ), final_time_environment.to_float( ), global_frame_origin, global_frame_orientation)

        # Modify Earth default settings
        body_settings.get( 'Earth' ).shape_settings = environment_setup.shape.oblate_spherical_spice( )
        body_settings.get( 'Earth' ).rotation_model_settings = environment_setup.rotation_model.gcrs_to_itrs(
            environment_setup.rotation_model.iau_2006, global_frame_orientation,
            interpolators.interpolator_generation_settings_float( interpolators.cubic_spline_interpolation( ), initial_time_environment.to_float( ), final_time_environment.to_float( ), 3600.0 ),
            interpolators.interpolator_generation_settings_float( interpolators.cubic_spline_interpolation( ), initial_time_environment.to_float( ), final_time_environment.to_float( ), 3600.0 ),
            interpolators.interpolator_generation_settings_float( interpolators.cubic_spline_interpolation( ), initial_time_environment.to_float( ), final_time_environment.to_float( ), 60.0 ) )
        body_settings.get( 'Earth' ).gravity_field_settings.associated_reference_frame = "ITRS"
        body_settings.get( "Earth" ).ground_station_settings = environment_setup.ground_station.dsn_stations( )


        # Create vehicle properties
        spacecraft_name = "MRO"
        spacecraft_central_body = "Mars"
        body_settings.add_empty_settings(spacecraft_name)

        body_settings.get(spacecraft_name).ephemeris_settings = environment_setup.ephemeris.interpolated_spice(
            initial_time_environment.to_float(), final_time_environment.to_float(), 10.0, spacecraft_central_body, global_frame_orientation )

        body_settings.get( spacecraft_name ).rotation_model_settings = environment_setup.rotation_model.spice( global_frame_orientation, spacecraft_name + "_SPACECRAFT", "" )
        body_settings.get( spacecraft_name ).constant_mass = 150

        # Create environment
        bodies = environment_setup.create_system_of_bodies(body_settings)

        # Define antenna position
        position_antenna = spice.get_body_cartesian_position_at_epoch("-74214", "-74000", "MRO_SPACECRAFT", "None", final_time.to_float())
        bodies.get( spacecraft_name ).system_models.set_reference_point( "Antenna", np.array( position_antenna ) )

        # Update bodies based on ODF file
        estimation_setup.observation.set_odf_information_in_bodies(multi_odf_file_contents, bodies)

        # Compress observations
        compressed_observations = estimation_setup.observation.create_compressed_doppler_collection(original_odf_observations, 60, 10)
        print('Compressed observations: ')
        print(compressed_observations.concatenated_observations.size)

        #  Create light-time corrections list
        light_time_correction_list = list()
        light_time_correction_list.append(
            estimation_setup.observation.first_order_relativistic_light_time_correction(["Sun"]))

        light_time_correction_list.append(
            estimation_setup.observation.dsn_tabulated_tropospheric_light_time_correction(tro_files_to_load))

        spacecraft_name_per_id = dict()
        spacecraft_name_per_id[74] = "MRO"
        light_time_correction_list.append(
            estimation_setup.observation.dsn_tabulated_ionospheric_light_time_correction(ion_files_to_load, spacecraft_name_per_id))

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

        # Compute residuals
        estimation.compute_and_set_residuals(compressed_observations, observation_simulators, bodies)

        # Save unfiltered residuals
        np.savetxt('mro_unfiltered_residual_' + filename_suffix + '.dat',
                   compressed_observations.get_concatenated_residuals(), delimiter=',')
        np.savetxt('mro_unfiltered_time_' + filename_suffix + '.dat',
                   compressed_observations.concatenated_float_times, delimiter=',')
        np.savetxt('mro_unfiltered_link_end_ids_' + filename_suffix + '.dat',
                   compressed_observations.concatenated_link_definition_ids, delimiter=',')

 


if __name__ == "__main__":
    print('Start')
    inputs = []

    nb_cores = 1

    start_date = datetime(2007, 1, 3)
    end_date = datetime(2007, 1, 7)

    clock_files_to_load, orientation_files_to_load, tro_files_to_load, ion_files_to_load, odf_files_to_load = \
        get_mro_files("mro_kernels/", start_date, end_date)


    for i in range(nb_cores):
        inputs.append([i, start_date, end_date, odf_files_to_load, clock_files_to_load, orientation_files_to_load, tro_files_to_load, ion_files_to_load])



    print('inputs', inputs)

    # Run parallel MC analysis
    with mp.get_context("fork").Pool(nb_cores) as pool:
        pool.map(run_estimation,inputs)




