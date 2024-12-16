import os
from xmlrpc.client import DateTime

import numpy as np
from matplotlib import pyplot as plt

# Load required tudatpy modules
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.astro import time_conversion, element_conversion
from tudatpy.math import interpolators
from tudatpy.numerical_simulation import environment_setup, environment
from tudatpy.numerical_simulation import estimation, estimation_setup
from tudatpy.numerical_simulation.estimation_setup import observation
from tudatpy import util
from load_pds_files import download_url_files_time, download_url_files_time_interval
from datetime import datetime

from urllib.request import urlretrieve

# Unpack various input arguments

spice.load_standard_kernels()
spice.load_kernel('./mex_phobos_flyby/ORMM_T19_131201000000_01033.BSP')
start = datetime(2013, 12, 27)
end = datetime(2013, 12, 30)

start_time = time_conversion.datetime_to_tudat(start).epoch() - 86400.0
end_time = time_conversion.datetime_to_tudat(end).epoch() + 86400.0

#date_to_filter_float = date_to_filter.epoch().to_float()
#original_odf_observations.filter_observations(date_to_filter_float)
# Filter out observations from observation collection
#original_odf_observations.filter_observations(date_filter)

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
                                                         start_time, end_time, 60.0))# Add the ground station to the environment

body_settings.get('Earth').gravity_field_settings.associated_reference_frame = "ITRS"

# Add New Norcia ground station
station_altitude = 252.0  # Altitude of the New Norcia station
new_norcia_latitude = np.deg2rad(-31.0482)  # Latitude of New Norcia in radians
new_norcia_longitude = np.deg2rad(116.191)  # Longitude of New Norcia in radians

spacecraft_name = "MEX"
dsn_station_name = 'NWNORCIA' #new norcia
spacecraft_central_body = "Mars"
body_settings.add_empty_settings(spacecraft_name)
# Retrieve translational ephemeris from SPICE
body_settings.get(spacecraft_name).ephemeris_settings = environment_setup.ephemeris.interpolated_spice(
    start_time, end_time, 10.0, spacecraft_central_body, global_frame_orientation)

# Retrieve rotational ephemeris from SPICE
body_settings.get(spacecraft_name).rotation_model_settings = environment_setup.rotation_model.spice(
    global_frame_orientation, spacecraft_name + "_SPACECRAFT", "")

bodies = environment_setup.create_system_of_bodies(body_settings)

########## IMPORTANT STEP ###################################
vehicleSys = environment.VehicleSystems()
vehicleSys.set_default_transponder_turnaround_ratio_function()
bodies.get_body("MEX").system_models = vehicleSys
###############################################################
# Add the ground station to the environment
dict_stations = environment_setup.ground_station.approximate_ground_stations_position()
NWNORCIA_position = dict_stations['NWNORCIA']
ground_station_settings = environment_setup.ground_station.basic_station(
"NWNORCIA",
NWNORCIA_position)

environment_setup.add_ground_station(
bodies.get_body("Earth"),
ground_station_settings )
#body_settings.get( "Earth" ).ground_station_settings = environment_setup.ground_station.dsn_stations()

#print(len(body_settings.get( "Earth" ).ground_station_settings))
# Create ground station settings



ifms_file = ['./mex_phobos_flyby/M32ICL3L02_D2S_133621904_00_FILTERED.TAB']
reception_band = observation.FrequencyBands.s_band
transmission_band = observation.FrequencyBands.x_band

ifms_collection = observation.observations_from_ifms_files(ifms_file, bodies, spacecraft_name, dsn_station_name, reception_band, transmission_band)
# Compress Doppler observations from 1.0 s integration time to 60.0 s
print('Observations: ')

#dates_to_filter = [time_conversion.DateTime(2013, 12, 29, 00, 25, 41.500)]
#dates_to_filter_float = []
#for date in dates_to_filter:
#    dates_to_filter_float.append(date.epoch().to_float())
#    # Create filter object for specific date
#    date_filter = estimation.observation_filter(
#        estimation.time_bounds_filtering, date, date + 86400)
#        # Filter out observations from observation collection
#    ifms_collection.filter_observations(date_filter)
#print('Filtered Observations: ')
#print(ifms_collection.concatenated_observations.size)

### ------------------------------------------------------------------------------------------
### DEFINE SETTINGS TO SIMULATE OBSERVATIONS AND COMPUTE RESIDUALS
### ------------------------------------------------------------------------------------------

#  Create light-time corrections list
light_time_correction_list = list()
light_time_correction_list.append(
    estimation_setup.observation.first_order_relativistic_light_time_correction(["Sun"]))


##################################################################################################################
# Add tropospheric correction!
#light_time_correction_list.append(
#    estimation_setup.observation.dsn_tabulated_tropospheric_light_time_correction(tro_files))
##################################################################################################################

##################################################################################################################
# Add ionospheric correction
#spacecraft_name_per_id = dict()
#spacecraft_name_per_id[74] = "MEX"
#light_time_correction_list.append(
#    estimation_setup.observation.dsn_tabulated_ionospheric_light_time_correction(ion_files, spacecraft_name_per_id))
#####################################################################################################################################
# Create observation model settings for the Doppler observables. This first implies creating the link ends defining all relevant
# tracking links between various ground stations and the MEX spacecraft. The list of light-time corrections defined above is then
# added to each of these link ends.

doppler_link_ends = ifms_collection.link_definitions_per_observable[
    estimation_setup.observation.dsn_n_way_averaged_doppler]


# IMPORTANT: ADD subtract_doppler_signature = False or it wont work
observation_model_settings = list()
for current_link_definition in doppler_link_ends:
    print(current_link_definition)
    observation_model_settings.append(estimation_setup.observation.dsn_n_way_doppler_averaged(
        current_link_definition, light_time_correction_list, subtract_doppler_signature = False ))

# Create observation simulators.
observation_simulators = estimation_setup.create_observation_simulators(observation_model_settings, bodies)

# Add elevation and SEP angles dependent variables to the IFMS observation collection
elevation_angle_settings = observation.elevation_angle_dependent_variable( observation.receiver )
elevation_angle_parser = ifms_collection.add_dependent_variable( elevation_angle_settings, bodies )
sep_angle_settings = observation.avoidance_angle_dependent_variable("Sun", observation.retransmitter, observation.receiver)
sep_angle_parser = ifms_collection.add_dependent_variable( sep_angle_settings, bodies )

# Compute and set residuals in the IFMS observation collection
estimation.compute_residuals_and_dependent_variables(ifms_collection, observation_simulators, bodies)

### ------------------------------------------------------------------------------------------
### RETRIEVE AND SAVE VARIOUS OBSERVATION OUTPUTS
### ------------------------------------------------------------------------------------------

concatenated_obs = ifms_collection.get_concatenated_observations()
concatenated_computed_obs = ifms_collection.get_concatenated_computed_observations()
# Retrieve RMS and mean of the residuals, sorted per observation set
concatenated_residuals = ifms_collection.get_concatenated_residuals()
rms_residuals = ifms_collection.get_rms_residuals()
mean_residuals = ifms_collection.get_mean_residuals()
print(concatenated_obs)
print(concatenated_computed_obs)
print(f'rms_residual: {rms_residuals}')
print(f'mean_residual: {mean_residuals}')

#np.savetxt('mex_unfiltered_residuals_rms' + '.dat',
#           np.vstack(rms_residuals), delimiter=',')
#np.savetxt('mex_unfiltered_residuals_mean' + '.dat',
#           np.vstack(mean_residuals), delimiter=',')

time_bounds_per_set = ifms_collection.get_time_bounds_per_set()