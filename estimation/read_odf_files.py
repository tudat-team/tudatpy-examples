# MARS EXPRESS - Using Different Dynamical Models for the Simulation of Observations and the Estimation
"""
Copyright (c) 2010-2022, Delft University of Technology. All rights reserved. This file is part of the Tudat. Redistribution and use in source and binary forms, with or without modification, are permitted exclusively under the terms of the Modified BSD license. You should have received a copy of the license with this file. If not, please or visit: http://tudat.tudelft.nl/LICENSE.
"""


## Context
"""
"""

import sys
sys.path.insert(0, "/home/dominic/Tudat/tudat-bundle/tudat-bundle/cmake-build-default/tudatpy")

# Load required standard modules
import multiprocessing as mp
import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# Load required tudatpy modules
from tudatpy import constants
from tudatpy.data import save2txt
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

current_directory = os.getcwd()

def load_clock_kernels( arc_index ):
    spice.load_kernel(current_directory + "/grail_kernels/gra_sclkscet_00013.tsc")
    spice.load_kernel(current_directory + "/grail_kernels/gra_sclkscet_00014.tsc")

def load_orientation_kernels( arc_index ):
    if arc_index == 0:
        spice.load_kernel(current_directory + "/grail_kernels/gra_rec_120402_120408.bc")
    if( arc_index > 0 and arc_index < 6 ):
        spice.load_kernel(current_directory + "/grail_kernels/gra_rec_120409_120415.bc")
    if( arc_index > 4 ):
        spice.load_kernel(current_directory + "/grail_kernels/gra_rec_120416_120422.bc")

def get_grail_odf_file_name( arc_index ):
    if arc_index == 0:
        return current_directory + '/grail_data/gralugf2012_097_0235smmmv1.odf'
    elif arc_index == 1:
        return current_directory + '/grail_data/gralugf2012_100_0540smmmv1.odf'
    elif arc_index == 2:
        return current_directory + '/grail_data/gralugf2012_101_0235smmmv1.odf'
    elif arc_index == 3:
        return current_directory + '/grail_data/gralugf2012_102_0358smmmv1.odf'
    elif arc_index == 4:
        return current_directory + '/grail_data/gralugf2012_103_0145smmmv1.odf'
    elif arc_index == 5:
        return current_directory + '/grail_data/gralugf2012_105_0352smmmv1.odf'
    elif arc_index == 6:
        return current_directory + '/grail_data/gralugf2012_107_0405smmmv1.odf'
    elif arc_index == 7:
        return current_directory + '/grail_data/gralugf2012_108_0450smmmv1.odf'

        # Load standard spice kernels as well as the one describing the orbit of Mars Express
def load_relevant_spice_kernels( arc_index ):
    spice.load_standard_kernels()
    spice.load_kernel(current_directory + "/grail_kernels/moon_de440_200625.tf")
    spice.load_kernel(current_directory + "/grail_kernels/grail_v07.tf")
    load_clock_kernels(arc_index)
    spice.load_kernel(current_directory + "/grail_kernels/grail_120301_120529_sci_v02.bsp")
    load_orientation_kernels(arc_index)
    spice.load_kernel(current_directory + "/grail_kernels/moon_pa_de440_200625.bpc")

def create_bodies( initial_time_environment, final_time_environment ):
    # Create default body settings for celestial bodies
    bodies_to_create = ["Earth", "Sun", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Moon"]
    global_frame_origin = "SSB"
    global_frame_orientation = "J2000"
    body_settings = environment_setup.get_default_body_settings_time_limited(
        bodies_to_create, initial_time_environment.to_float(), final_time_environment.to_float(), global_frame_origin,
        global_frame_orientation)

    # Modify Earth default settings
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
    body_settings.get("Earth").ground_station_settings = environment_setup.ground_station.dsn_stations()
    shape_deformation_list = list()
    shape_deformation_list.append( environment_setup.shape_deformation.iers_2010_solid_body_tidal( ) )
    body_settings.get("Earth").shape_deformation_settings = shape_deformation_list
    # Modify Moon default settings
    body_settings.get('Moon').rotation_model_settings = environment_setup.rotation_model.spice(global_frame_orientation,
                                                                                               "MOON_PA_DE440",
                                                                                               "MOON_PA_DE440")
    body_settings.get('Moon').gravity_field_settings = environment_setup.gravity_field.predefined_spherical_harmonic(
        environment_setup.gravity_field.gggrx1200, 500)
    body_settings.get('Moon').gravity_field_settings.associated_reference_frame = "MOON_PA_DE440"
    moon_gravity_field_variations = list()
    moon_gravity_field_variations.append(environment_setup.gravity_field_variation.solid_body_tide('Earth', 0.02405, 2))
    moon_gravity_field_variations.append(environment_setup.gravity_field_variation.solid_body_tide('Sun', 0.02405, 2))
    body_settings.get('Moon').gravity_field_variation_settings = moon_gravity_field_variations
    body_settings.get('Moon').ephemeris_settings.frame_origin = "Earth"

    # Add Moon radiation properties
    moon_surface_radiosity_models = [
        radiation_pressure.thermal_emission_angle_based_radiosity(
            95.0, 385.0, 0.95, "Sun"),
        radiation_pressure.variable_albedo_surface_radiosity(
            radiation_pressure.predefined_spherical_harmonic_surface_property_distribution(
                radiation_pressure.albedo_dlam1), "Sun")]
    body_settings.get("Moon").radiation_source_settings = radiation_pressure.panelled_extended_radiation_source(
        moon_surface_radiosity_models, [6, 12, 18])

    # Create vehicle properties
    spacecraft_name = "GRAIL-A"
    spacecraft_central_body = "Moon"
    body_settings.add_empty_settings(spacecraft_name)
    body_settings.get(spacecraft_name).ephemeris_settings = environment_setup.ephemeris.interpolated_spice(
        initial_time_environment.to_float(), final_time_environment.to_float(), 10.0, spacecraft_central_body,
        global_frame_orientation)
    body_settings.get(spacecraft_name).rotation_model_settings = environment_setup.rotation_model.spice(
        global_frame_orientation, spacecraft_name + "_SPACECRAFT", "")
    occulting_bodies = dict()
    occulting_bodies["Sun"] = ["Moon"]
    body_settings.get(
        spacecraft_name).radiation_pressure_target_settings = radiation_pressure.cannonball_radiation_target(
        5, 1.5, occulting_bodies)
    body_settings.get(spacecraft_name).constant_mass = 150;

    # Create environment
    bodies = environment_setup.create_system_of_bodies(body_settings)
    bodies.get(spacecraft_name).system_models.set_reference_point("Antenna", np.array([-0.082, 0.152, -0.810]))

    return bodies


def create_observation_model_settings( observation_collection ):
    #  Create observation model settings
    light_time_correction_list = list()
    light_time_correction_list.append(
        estimation_setup.observation.first_order_relativistic_light_time_correction(["Sun"]))

    tropospheric_correction_files = [
        current_directory + '/grail_data/grxlugf2012_092_2012_122.tro',
        current_directory + '/grail_data/grxlugf2012_122_2012_153.tro']
    light_time_correction_list.append(
        estimation_setup.observation.dsn_tabulated_tropospheric_light_time_correction(
            tropospheric_correction_files))

    ionospheric_correction_files = [
        current_directory + '/grail_data/gralugf2012_092_2012_122.ion',
        current_directory + '/grail_data/gralugf2012_122_2012_153.ion']
    spacecraft_name_per_id = dict()
    spacecraft_name_per_id[177] = "GRAIL-A"
    light_time_correction_list.append(
        estimation_setup.observation.dsn_tabulated_ionospheric_light_time_correction(
            ionospheric_correction_files, spacecraft_name_per_id))

    # Create observation model settings
    doppler_link_ends = observation_collection.link_definitions_per_observable[
        estimation_setup.observation.dsn_n_way_averaged_doppler]
    observation_model_settings = list()
    for current_link_definition in doppler_link_ends:
        observation_model_settings.append(estimation_setup.observation.dsn_n_way_doppler_averaged(
            current_link_definition, light_time_correction_list))

    return observation_model_settings

def save_residuals_to_files( arc_index, residual_collection, file_prefix ):

    print(residual_collection.concatenated_observations)

    np.savetxt( file_prefix + '_residual_' + str(arc_index) + '.dat',
               residual_collection.concatenated_observations, delimiter=',')
    np.savetxt( file_prefix + '_time_' + str(arc_index) + '.dat',
               residual_collection.concatenated_float_times, delimiter=',')
    np.savetxt( file_prefix + '_link_end_ids_' + str(arc_index) + '.dat',
               residual_collection.concatenated_link_definition_ids, delimiter=',')

def run_estimation( arc_index ):

    with util.redirect_std( 'parallel_output_' + str( arc_index ) + ".dat", True, True ):

        number_of_files = 8
        if( arc_index >= number_of_files ):
            print("Error, arc index is too high!" )

        arc_index = arc_index % number_of_files
        load_relevant_spice_kernels( arc_index )
    
        # Define start and end times for environment
        initial_time_environment = time_conversion.DateTime( 2012, 3, 2, 0, 0, 0.0 ).epoch( )
        final_time_environment = time_conversion.DateTime( 2012, 5, 29, 0, 0, 0.0 ).epoch( )
        bodies = create_bodies( initial_time_environment, final_time_environment )

        print('Created bodies')
        # Load ODF file
        single_odf_file_contents = estimation_setup.observation.process_odf_data_single_file(
            get_grail_odf_file_name( arc_index ), 'GRAIL-A', True )
        single_odf_file_contents.define_antenna_id( 'GRAIL-A', 'Antenna' )
        estimation_setup.observation.set_odf_information_in_bodies( single_odf_file_contents, bodies )
        print('Created observations')

        # Create observation collection, split into arcs, and compress to 60 seconds
        uncompressed_observations = estimation_setup.observation.split_observation_sets_into_arc(
                estimation_setup.observation.create_odf_observed_observation_collection(
            single_odf_file_contents, list( ), [ numerical_simulation.Time( 0, np.nan ), numerical_simulation.Time( 0, np.nan ) ] ), 60.0, 10 )
        compressed_observations = estimation_setup.observation.create_compressed_doppler_collection(
            uncompressed_observations, 60)

        print('A')

        # Create observation dependent variables
        dependent_variable_list = list( )
        dependent_variable_list.append(
            observation.elevation_at_link_end_type(
            observation.receiver,
            observation.interval_start ) )
        print('A')

        # Create observation simulators
        observation_model_settings = create_observation_model_settings( compressed_observations )
        observation_simulators = estimation_setup.create_observation_simulators(observation_model_settings, bodies)

        # Compute simulated observations and residuals
        observation_simulation_settings = estimation_setup.observation.observation_settings_from_collection(
            compressed_observations)
        print('A')
        observation.add_dependent_variables_to_all( observation_simulation_settings, dependent_variable_list, bodies )
        print('B')
        simulated_observations = estimation.simulate_observations(observation_simulation_settings,
                                                                  observation_simulators, bodies)

        residual_collection = estimation.create_residual_collection(compressed_observations, simulated_observations)
        save_residuals_to_files( arc_index, residual_collection, 'unfiltered')

        # Filter observations
        residual_filter_cutoff = dict()
        residual_filter_cutoff[observation.dsn_n_way_averaged_doppler] = 0.01
        filtered_compressed_observations = estimation.create_filtered_observation_collection(
            compressed_observations, residual_collection, residual_filter_cutoff)

        # Compute simulated filtered observations and residuals
        observation_simulation_settings = estimation_setup.observation.observation_settings_from_collection(
            filtered_compressed_observations)
        observation.add_dependent_variables_to_all( observation_simulation_settings, dependent_variable_list, bodies )
        simulated_filtered_observations = estimation.simulate_observations(observation_simulation_settings,
                                                                           observation_simulators, bodies)
        print('C')
        [dependent_variable_times, dependent_variables] = simulated_filtered_observations.get_full_dependent_variable_vector( dependent_variable_list[ 0 ] )
        np.savetxt('dependent_variable' + '_time_' + str(arc_index) + '.dat',
                   dependent_variable_times, delimiter=',')
        np.savetxt('dependent_variables_' + str(arc_index) + '.dat',
                   dependent_variables, delimiter=',')
        print('D')
        residual_filtered_collection = estimation.create_residual_collection(filtered_compressed_observations,
                                                                             simulated_filtered_observations)
        save_residuals_to_files( arc_index, residual_filtered_collection, 'filtered')


if __name__ == "__main__":
    print('Start')
    inputs = []
    for i in range(8):
        inputs.append(i)
    # Run parallel MC analysis
    with mp.get_context("fork").Pool(8) as pool:
        pool.map(run_estimation,inputs)




