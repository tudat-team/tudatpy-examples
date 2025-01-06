######################### # IMPORTANT #############################################################################

# In order to test this example, I am using a Phobos Flyby IFMS file missing the few last/first lines...
# The removed lines were classified as outliers, but they should be filtered with the proper tudat functionality,
# rather than manually (as done for now)

##################################################################################################################
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
spice.load_kernel('./mex_phobos_flyby/MEX_STRUCT_V01.BSP')
spice.load_kernel('./mex_phobos_flyby/MEX_V16.TF')
spice.load_kernel('./mex_phobos_flyby/MEX_241229_STEP.TSC')
spice.load_kernel('./mex_phobos_flyby/NAIF0012.TLS')
spice.load_kernel('./mex_phobos_flyby/ATNM_MEASURED_2013_V04.BC')
start = datetime(2013, 12, 27)
end = datetime(2013, 12, 30)

start_time = time_conversion.datetime_to_tudat(start).epoch().to_float() - 86400.0
end_time = time_conversion.datetime_to_tudat(end).epoch().to_float() + 86400.0

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


spacecraft_name = "MEX" # Set Spacecraft Name
dsn_station_name = 'NWNORCIA' # Set New Norcia Ground Station
spacecraft_central_body = "Mars" # Set Central Body (Mars)
body_settings.add_empty_settings(spacecraft_name) # Create empty settings for spacecraft

# Retrieve translational ephemeris from SPICE
body_settings.get(spacecraft_name).ephemeris_settings = environment_setup.ephemeris.interpolated_spice(
    start_time, end_time, 10.0, spacecraft_central_body, global_frame_orientation)

# Retrieve rotational ephemeris from SPICE
body_settings.get(spacecraft_name).rotation_model_settings = environment_setup.rotation_model.spice(
    global_frame_orientation, spacecraft_name + "_SPACECRAFT", "")

# Create System of Bodies using the above-defined body_settings
bodies = environment_setup.create_system_of_bodies(body_settings)

########## IMPORTANT STEP ###################################
# Set the transponder turnaround ratio function
vehicleSys = environment.VehicleSystems()
vehicleSys.set_default_transponder_turnaround_ratio_function()
bodies.get_body("MEX").system_models = vehicleSys
###############################################################

# Add the New Norcia ground station to the environment
dict_stations = environment_setup.ground_station.approximate_ground_stations_position()
NWNORCIA_position = dict_stations['NWNORCIA']
ground_station_settings = environment_setup.ground_station.basic_station("NWNORCIA",NWNORCIA_position)

environment_setup.add_ground_station(bodies.get_body("Earth"), ground_station_settings )

# Load IFMS file
ifms_file = ['./mex_phobos_flyby/M32ICL3L02_D2S_133621904_00_FILTERED.TAB'] # Phobos Flyby
reception_band = observation.FrequencyBands.s_band
transmission_band = observation.FrequencyBands.x_band

# Create collection from IFMS file
ifms_collection = observation.observations_from_ifms_files(ifms_file, bodies, spacecraft_name, dsn_station_name, reception_band, transmission_band)

antenna_position_history = dict()
com_position = [-1.3,0.0,0.0] # estimated based on the MEX_V16.TF file description
for obs_times in ifms_collection.get_observation_times():
    time = obs_times[0].to_float() - 3600.0
    while time <= obs_times[-1].to_float() + 3600.0:
        state = np.zeros((6, 1))

        # For each observation epoch, retrieve the antenna position (spice ID "-41020") w.r.t. the origin of the MEX-fixed frame (spice ID "-41000")
        state[:3,0] = spice.get_body_cartesian_position_at_epoch("-41020", "-41000", "MEX_SPACECRAFT", "none", time)

        # Translate the antenna position to account for the offset between the origin of the MEX-fixed frame and the COM
        state[:3,0] = state[:3,0] - com_position

        # Store antenna position w.r.t. COM in the MEX-fixed frame
        antenna_position_history[time] = state
        time += 10.0

    # Create tabulated ephemeris settings from antenna position history
    antenna_ephemeris_settings = environment_setup.ephemeris.tabulated(antenna_position_history, "-41000",  "MEX_SPACECRAFT")

    # Create tabulated ephemeris for the MEX antenna
    antenna_ephemeris = environment_setup.ephemeris.create_ephemeris(antenna_ephemeris_settings, "Antenna")

    # Set the spacecraft's reference point position to that of the antenna (in the MEX-fixed frame)
    ifms_collection.set_reference_point(bodies, antenna_ephemeris, "Antenna", "MEX", observation.reflector1)

### ------------------------------------------------------------------------------------------
### DEFINE SETTINGS TO SIMULATE OBSERVATIONS AND COMPUTE RESIDUALS
### ------------------------------------------------------------------------------------------

#  Create light-time corrections list
light_time_correction_list = list()
light_time_correction_list.append(
    estimation_setup.observation.first_order_relativistic_light_time_correction(["Sun"]))

##############################
 # ATMOSPHERIC CORRECTION #
###############################

############### If this piece of code is triggered, the residuals go up from .007 to .008 ########################
#light_time_correction_list.append(
#    estimation_setup.observation.jakowski_ionospheric_light_time_correction()
#)
##################################################################################################################

##################################################################################################################

atmospheric_corrections = np.loadtxt('./mex_phobos_flyby/M32ICL3L02_D2S_133621904_00_FILTERED.TAB', usecols = 10)


#####################################################################################################################################
# Create observation model settings for the Doppler observables. This first implies creating the link ends defining all relevant
# tracking links between various ground stations and the MEX spacecraft. The list of light-time corrections defined above is then
# added to each of these link ends.

doppler_link_ends = ifms_collection.link_definitions_per_observable[
    estimation_setup.observation.dsn_n_way_averaged_doppler]

########## IMPORTANT STEP #######################################################################
# Add: subtract_doppler_signature = False, or it won't work
observation_model_settings = list()
for current_link_definition in doppler_link_ends:
    observation_model_settings.append(estimation_setup.observation.dsn_n_way_doppler_averaged(
        current_link_definition, light_time_correction_list, subtract_doppler_signature = False ))
###################################################################################################
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

#print(concatenated_obs - atmospheric_corrections)
#print(concatenated_computed_obs)

####################################################################################################
##### COMPUTE RESIDUALS BY HAND, INCORPORATING ATMOSPHERIC CORRECTIONS PROVIDED IN IFMS FILES #####
residuals_by_hand =(concatenated_computed_obs - (concatenated_obs - atmospheric_corrections))
#print(f'residuals_array: {abs(residuals_by_hand)}')
print('Residuals by Hand, Atmospheric Corrections')
print(f'rms_residuals: {abs(np.sqrt(np.mean(residuals_by_hand**2)))}')
print(f'mean_residuals: {abs(np.mean(residuals_by_hand))}\n')
####################################################################################################

####################################################################################################
##### COMPUTE RESIDUALS BY HAND, WITHOUT ATMOSPHERIC CORRECTIONS #####
residuals_by_hand_no_atm_corr =(concatenated_computed_obs - concatenated_obs)
#print(f'residuals_array: {abs(residuals_by_hand)}')
print('Residuals by Hand, NO Atmospheric Corrections')
print(f'rms_residuals: {abs(np.sqrt(np.mean(residuals_by_hand_no_atm_corr**2)))}')
print(f'mean_residuals: {abs(np.mean(residuals_by_hand_no_atm_corr))}\n')
####################################################################################################

####################################################################################################
# TUDATPY-PROVIDED RESIDUALS
print('Tudatpy Residuals')
print(f'rms_residuals: {rms_residuals}')
print(f'mean_residuals: {mean_residuals}\n')
####################################################################################################

### SAVING FILES ####
#np.savetxt('mex_unfiltered_residuals_rms' + '.dat',
#           np.vstack(rms_residuals), delimiter=',')
#np.savetxt('mex_unfiltered_residuals_mean' + '.dat',
#           np.vstack(mean_residuals), delimiter=',')
#####################

# Retrieve the time bounds of each observation set within the observation collection
time_bounds_per_set = ifms_collection.get_time_bounds_per_set()