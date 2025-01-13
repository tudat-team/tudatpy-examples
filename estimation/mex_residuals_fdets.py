######################### # IMPORTANT #############################################################################

# In order to test this example, I am using a Phobos Flyby fdets file missing the few last/first lines...
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
from datetime import datetime

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
nwnorcia_ground_station_settings = environment_setup.ground_station.basic_station("NWNORCIA",NWNORCIA_position)
environment_setup.add_ground_station(bodies.get_body("Earth"), nwnorcia_ground_station_settings )
CEDUNA_position = dict_stations['CEDUNA']
ceduna_ground_station_settings = environment_setup.ground_station.basic_station("CEDUNA",NWNORCIA_position)
environment_setup.add_ground_station(bodies.get_body("Earth"), ceduna_ground_station_settings )

# Load FDETS file
fdets_file = ['./mex_phobos_flyby/Fdets.mex2013.12.28.Cd.r2i.txt'] # Phobos Flyby X band
ifms_file = ['./mex_phobos_flyby/M32ICL3L02_D2S_133621904_00_FILTERED.TAB']
base_frequency = 8412e6
column_types = ["utc_datetime_string", "signal_to_noise_ratio", "normalised_spectral_max", "doppler_measured_frequency_hz", "doppler_noise_hz"]

target_name = 'MEX'
transmitting_station_name = 'NWNORCIA'
receiving_station_name = 'CEDUNA'
reception_band = observation.FrequencyBands.x_band
transmission_band = observation.FrequencyBands.x_band
# Create collection from fdets file
fdets_collection = observation.observations_from_fdets_files(fdets_file[0], base_frequency, column_types, spacecraft_name, transmitting_station_name, receiving_station_name, reception_band, transmission_band)
ifms_collection = observation.observations_from_ifms_files(ifms_file, bodies, spacecraft_name, transmitting_station_name, reception_band, transmission_band)

antenna_position_history = dict()
com_position = [-1.3,0.0,0.0] # estimated based on the MEX_V16.TF file description
for obs_times in fdets_collection.get_observation_times():
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
    fdets_collection.set_reference_point(bodies, antenna_ephemeris, "Antenna", "MEX", observation.reflector1)

### ------------------------------------------------------------------------------------------
### DEFINE SETTINGS TO SIMULATE OBSERVATIONS AND COMPUTE RESIDUALS
### ------------------------------------------------------------------------------------------

#  Create light-time corrections list
light_time_correction_list = list()
light_time_correction_list.append(
    estimation_setup.observation.first_order_relativistic_light_time_correction(["Sun"]))

##############################
# ATMOSPHERIC CORRECTION #

# EMPTY FOR NOW
###############################


# Create observation model settings for the Doppler observables. This first implies creating the link ends defining all relevant
# tracking links between various ground stations and the MEX spacecraft. The list of light-time corrections defined above is then
# added to each of these link ends.

link_ends = {
    observation.receiver: observation.body_reference_point_link_end_id('Earth', "CEDUNA"),
    observation.retransmitter: observation.body_reference_point_link_end_id('MEX','Antenna'),
    observation.transmitter: observation.body_reference_point_link_end_id('Earth', "NWNORCIA"),
}

# Create a single link definition from the link ends
link_definition = observation.LinkDefinition(link_ends)

# Define the observation model settings
observation_model_settings = [
    estimation_setup.observation.doppler_measured_frequency(
        link_definition, light_time_correction_list
    )
]
print("Ground stations on Earth:", bodies.get_body("Earth").ground_station_list)
###################################################################################################
# Create observation simulators.
observation_simulators = estimation_setup.create_observation_simulators(observation_model_settings, bodies)

# Add elevation and SEP angles dependent variables to the fdets observation collection
elevation_angle_settings = observation.elevation_angle_dependent_variable( observation.receiver )
elevation_angle_parser = fdets_collection.add_dependent_variable( elevation_angle_settings, bodies )
sep_angle_settings = observation.avoidance_angle_dependent_variable("Sun", observation.retransmitter, observation.receiver)
sep_angle_parser = fdets_collection.add_dependent_variable( sep_angle_settings, bodies )

# Compute and set residuals in the fdets observation collection
estimation.compute_residuals_and_dependent_variables(fdets_collection, observation_simulators, bodies)

### ------------------------------------------------------------------------------------------
### RETRIEVE AND SAVE VARIOUS OBSERVATION OUTPUTS
### ------------------------------------------------------------------------------------------

concatenated_obs = fdets_collection.get_concatenated_observations()
concatenated_computed_obs = fdets_collection.get_concatenated_computed_observations()

# Retrieve RMS and mean of the residuals
concatenated_residuals = fdets_collection.get_concatenated_residuals()
rms_residuals = fdets_collection.get_rms_residuals()
mean_residuals = fdets_collection.get_mean_residuals()
print(concatenated_obs)
print(concatenated_computed_obs)
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

# Retrieve the observation times list
times = fdets_collection.get_observation_times()
times = [time.to_float() for time in times[0]]
times = np.array(times)
# Residuals Plot
plt.scatter(times, concatenated_residuals, s = 6, marker = '+', label = 'Atm. Corr.')
plt.axhline(mean_residuals, label = f'mean residuals = {mean_residuals,6}', color = 'black', linestyle = '--')
plt.axhline(rms_residuals, label = f'rms residuals = {rms_residuals,6}', linestyle = '--')
plt.axhline(rms_residuals, linestyle = '--')
plt.legend()
plt.title('Mex Residuals')
plt.xlabel('Time (s)')
plt.ylabel('Residuals (Hz)')
plt.show()