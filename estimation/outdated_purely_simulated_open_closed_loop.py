"""
THIS SCRIPT SHOWCASES HOW TO GO FROM SIMULATED OPEN-LOOP DATA (within TUDAT) to SIMULATED EQUIVALENT CLOSED-LOOP DATA.
- TROPOSPHERIC CORRECTION IS IMPLEMENTED.
- USE DEVELOP BRANCH for TUDAT AND TUDATPY, tudat::Time scalar type
"""
################### Import Modules #####################
import os
from tudatpy.interface import spice
from tudatpy.astro import time_conversion
from tudatpy.math import interpolators
from tudatpy.numerical_simulation import environment_setup, environment
from tudatpy.numerical_simulation import estimation, estimation_setup
from tudatpy.numerical_simulation.estimation_setup import observation
from datetime import datetime
from tudatpy.numerical_simulation import Time
import matplotlib.pyplot as plt
import numpy as np
from tudatpy import data
from scipy.interpolate import interp1d
from scipy.integrate import fixed_quad
#########################################################

def compute_scipy_quadrature(interpolated_function, times, integration_time = 10):

    results = []  # To store the integral results
    interval_centers = []  # To store the midpoints of the intervals

    a = min(times)
    while a + integration_time <= max(times):
        b = a + integration_time  # End of the current interval
        midpoint = (a + b) / 2

        if any(abs(midpoint - t) < 1 for t in times):  # Tolerance-based match
            result, _ = fixed_quad(interpolated_function, a, b, n = 7)
            normalized_result = result / (b - a)
            results.append(normalized_result)
            interval_centers.append(midpoint)  # Store time tag
            a = b  # Move to next interval
        else:
            # Skip gaps
            a = b  # Ensure progression

    return results, interval_centers


################################## COMMON SETTINGS and RELEVANT QUANTITIES ##################################
# Set Folders Containing Relevant Files
mex_kernels_folder = '/Users/lgisolfi/Desktop/mex_phobos_flyby/kernels'
mex_fdets_folder = '/Users/lgisolfi/Desktop/mex_phobos_flyby/fdets/complete'
mex_ifms_folder = '/Users/lgisolfi/Desktop/mex_phobos_flyby/ifms/filtered'
mex_odf_folder = '/Users/lgisolfi/Desktop/mex_phobos_flyby/odf/'

# Load Required Spice Kernels
spice.load_standard_kernels()
for kernel in os.listdir(mex_kernels_folder):
    kernel_path = os.path.join(mex_kernels_folder, kernel)
    spice.load_kernel(kernel_path)

# Define Start and end Dates of Simulation
start = datetime(2013, 12, 29, 1, 00, 00)
end = datetime(2013, 12, 29, 11, 00, 00)

# Add a time buffer of one day
start_time = time_conversion.datetime_to_tudat(start).epoch().to_float()
end_time = time_conversion.datetime_to_tudat(end).epoch().to_float()
times_linspace = np.arange(start_time, end_time, step = 1)
tudat_times_linspace = [Time(time) for time in times_linspace]

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
spacecraft_name = "MEX" # Set Spacecraft Name
spacecraft_central_body = "Mars" # Set Central Body (Mars)
body_settings.add_empty_settings(spacecraft_name) # Create empty settings for spacecraft
body_settings.get(spacecraft_name).ephemeris_settings = environment_setup.ephemeris.interpolated_spice(
    start_time, end_time, 10.0, spacecraft_central_body, global_frame_orientation)

# Retrieve rotational ephemeris from SPICE
body_settings.get(spacecraft_name).rotation_model_settings = environment_setup.rotation_model.spice(
    global_frame_orientation, spacecraft_name + "_SPACECRAFT", "")

body_settings.get("Earth").ground_station_settings = environment_setup.ground_station.radio_telescope_stations()

# Create System of Bodies using the above-defined body_settings
bodies = environment_setup.create_system_of_bodies(body_settings)

######## APPLY TROPOSPHERIC CORRECTION FOR UPLINK AND DOWNLINK ########################################################################################################
observation.set_vmf_troposphere_data(
    [ "/Users/lgisolfi/Desktop/mex_phobos_flyby/VMF/y2013.vmf3_r" ], True, False, bodies, False, True
)
# Load meteorological uplink and downlink corrections
weather_files = ([os.path.join('/Users/lgisolfi/Desktop/data_archiving-1.0/dataset/mex/gr035/downloaded/met', met_file) for met_file in os.listdir('/Users/lgisolfi/Desktop/data_archiving-1.0/dataset/mex/gr035/downloaded/met')])
#body_settings.get("Earth").ground_station_settings.append(data.set_estrack_weather_data_in_ground_stations(bodies,weather_files, 'NWNORCIA'))
#body_settings.get("Earth").ground_station_settings.append(data.set_estrack_weather_data_in_ground_stations(bodies,weather_files, 'ONSALA60'))
#####################################################################################################################################################################

########## Set the transponder turnaround ratio function ###################################
vehicleSys = environment.VehicleSystems()
vehicleSys.set_default_transponder_turnaround_ratio_function()
bodies.get_body("MEX").system_models = vehicleSys
base_frequency = 8412e6 # MEX Base frequency
column_types = ["utc_datetime_string", "signal_to_noise_ratio", "normalised_spectral_max","doppler_measured_frequency_hz", "doppler_noise_hz"]
reception_band = observation.FrequencyBands.x_band
transmission_band = observation.FrequencyBands.x_band
turnaround_ratio = observation.dsn_default_turnaround_ratios( observation.FrequencyBands.x_band,observation.FrequencyBands.x_band)
sites_list = []
fdets_files = []
ifms_files = []
######################################################## TEMPORARY LINE TO FIX FREQUENCY CALCULATOR ERROR ###########################################################
bodies.get( "Earth" ).get_ground_station( "NWNORCIA" ).transmitting_frequency_calculator = environment.ConstantTransmittingFrequencyCalculator( 7166445042.992178 )
######################################################## TEMPORARY LINE TO FIX FREQUENCY CALCULATOR ERROR ###########################################################

######################### Set Involved Link Ends ######################################
link_ends = {
    observation.receiver: observation.body_reference_point_link_end_id('Earth', 'ONSALA60'),
    observation.retransmitter: observation.body_origin_link_end_id('MEX'),
    observation.transmitter: observation.body_reference_point_link_end_id('Earth', 'NWNORCIA'),
}

# Create a single link definition from the link ends
link_definition = observation.LinkDefinition(link_ends)
light_time_correction_list = list()
light_time_correction_list.append(
    estimation_setup.observation.first_order_relativistic_light_time_correction(["Sun"]))
######################################################################################

################### Define Open Loop Observation and Ancillary settings ###################
open_loop_observation_model_settings = [
    estimation_setup.observation.doppler_measured_frequency(
        link_definition, light_time_correction_list)
]
open_loop_ancillary_settings = observation.doppler_measured_frequency_ancillary_settings(
    frequency_bands =  [reception_band, transmission_band]
)
open_loop_observation_simulation_settings = [observation.tabulated_simulation_settings(
    observable_type = estimation_setup.observation.doppler_measured_frequency_type,
    link_ends = link_definition,
    simulation_times = tudat_times_linspace,
    ancilliary_settings = open_loop_ancillary_settings
)]
#############################################################################################

################### Define Closed Loop Observation and Ancillary settings ###################
closed_loop_observation_model_settings = [
    estimation_setup.observation.dsn_n_way_doppler_averaged(
        link_definition, light_time_correction_list, subtract_doppler_signature = False)
]
closed_loop_ancillary_settings = observation.dsn_n_way_doppler_ancilliary_settings(
    frequency_bands =  [reception_band, transmission_band],
    reference_frequency_band = reception_band,
    reference_frequency = 0,
    integration_time = 60
)

closed_loop_observation_simulation_settings = [observation.tabulated_simulation_settings(
    observable_type = estimation_setup.observation.dsn_n_way_averaged_doppler,
    link_ends = link_definition,
    simulation_times = tudat_times_linspace,
    ancilliary_settings = closed_loop_ancillary_settings
)]
#############################################################################################

############### Create observation simulators ###############
closed_loop_observation_simulators = estimation_setup.create_observation_simulators(closed_loop_observation_model_settings, bodies)
open_loop_observation_simulators = estimation_setup.create_observation_simulators(open_loop_observation_model_settings, bodies)
#############################################################

############### Retrieve Collections ########################
open_loop_collection = estimation.simulate_observations(open_loop_observation_simulation_settings, open_loop_observation_simulators, bodies)
closed_loop_collection = estimation.simulate_observations(closed_loop_observation_simulation_settings, closed_loop_observation_simulators, bodies)
#############################################################

############### Compute Residuals and Dependent Variables ########################
estimation.compute_residuals_and_dependent_variables(open_loop_collection, open_loop_observation_simulators, bodies) # fdets simulator
estimation.compute_residuals_and_dependent_variables(closed_loop_collection, closed_loop_observation_simulators, bodies) # ifms simulator
##################################################################################

############### Retrieve Simulated Observations ########################
simulated_observations_fdets = open_loop_collection.get_observations()[0] # simulated open-loop (FDETS)
simulated_observations_ifms = closed_loop_collection.get_observations()[0] # simulated closed-loop (IFMS)
simulated_times_fdets = open_loop_collection.get_observation_times()[0] # simulated open-loop (FDETS)
simulated_times_ifms = closed_loop_collection.get_observation_times()[0] # simulated closed-loop (IFMS)
simulated_times_ifms_float = [time.to_float() for time in simulated_times_ifms]
simulated_times_fdets_float = [time.to_float() for time in simulated_times_fdets]

simulated_times_fdets_utc = [time_conversion.julian_day_to_calendar_date(time_conversion.seconds_since_epoch_to_julian_day(time)) for time in simulated_times_fdets_float]
simulated_times_ifms_utc = [time_conversion.julian_day_to_calendar_date(time_conversion.seconds_since_epoch_to_julian_day(time)) for time in simulated_times_ifms_float]

################ Interpolated, simulated closed loop continous functions (IFMS and FDETS) ##############################
simulated_closed_loop_continuous_function = interp1d(simulated_times_ifms_float_utc, simulated_observations_ifms, kind='cubic', fill_value='extrapolate')
simulated_open_loop_continuous_function = interp1d(simulated_times_fdets_float, simulated_observations_fdets, kind='cubic', fill_value='extrapolate')
########################################################################################################################

################ Compute quadrature to convert simulated open_loop into equivalent simulated closed loop ##############
simulated_equivalent_closed_loop_observables, simulated_equivalent_closed_loop_times = compute_scipy_quadrature(
    simulated_open_loop_continuous_function,
    times_linspace,
    integration_time = 60)
#######################################################################################################################

################# Convert times to UTC ################
simulated_equivalent_closed_loop_times_utc = [time_conversion.julian_day_to_calendar_date(time_conversion.seconds_since_epoch_to_julian_day(time)) for time in simulated_equivalent_closed_loop_times]
#######################################################

################ Retrieve the tones ###################
simulated_open_loop_tone = np.array(simulated_observations_fdets) - base_frequency #retrieve f tone from open-loop observable
simulated_equivalent_closed_loop_tone = np.array(simulated_equivalent_closed_loop_observables) - base_frequency
simulated_observations_ifms_tone =  np.array(simulated_observations_ifms) - base_frequency
#######################################################

################# Retrieve differences between the two simulated closed loop (equivalent-ifms) ################
simulated_closed_loop_at_interval_centers = [
    simulated_closed_loop_continuous_function(interval_center) - base_frequency
    for interval_center in simulated_equivalent_closed_loop_times]
difference_between_closed_loops = [j-i for i, j in zip(simulated_equivalent_closed_loop_tone,simulated_closed_loop_at_interval_centers)]
###############################################################################################################

######################## Retrieve Statistics (mean and rms) #######################
rms_closed_loop_difference_simulated = np.std(difference_between_closed_loops)
mean_closed_loop_difference_simulated = np.mean(difference_between_closed_loops)
mean_pride_residuals = np.mean(open_loop_collection.get_concatenated_residuals())
rms_pride_residuals = np.std(open_loop_collection.get_concatenated_residuals())
mean_ifms_residuals = np.mean(closed_loop_collection.get_concatenated_residuals())
rms_ifms_residuals = np.std(closed_loop_collection.get_concatenated_residuals())
###################################################################################

######################## Visualize data and residuals  ###########################
fig, axs = plt.subplots(3, 1, figsize=(10, 12))  # 2 rows, 1 column of subplots
axs[0].scatter(simulated_times_fdets_utc, simulated_open_loop_tone, marker='o', label='Simulated Open-Loop Tone', s=15, alpha=0.5)
axs[0].scatter(simulated_equivalent_closed_loop_times_utc, simulated_equivalent_closed_loop_tone, marker='o', label='Simulated Equivalent Closed-Loop Tone', s=15, alpha=0.5)
axs[0].set_xlabel('Time [s] (Midpoint of Interval)')
axs[0].set_ylabel('$f_{tone} = f_{R} - f_{base}$ [Hz], $f_{base} = 8412$ MHz')
axs[0].legend()
axs[0].grid(True)

axs[1].scatter(simulated_times_ifms_utc, simulated_observations_ifms_tone, marker='o', label='Simulated Closed-Loop Tone', s=15, alpha=0.5)
axs[1].scatter(simulated_equivalent_closed_loop_times_utc, simulated_equivalent_closed_loop_tone, marker='o', label='Simulated Equivalent Closed-Loop Tone', s=15, alpha=0.5)
axs[1].set_xlabel('Time [s] (Midpoint of Interval)')
axs[1].set_ylabel('$f_{tone} = f_{R} - f_{base}$ [Hz], $f_{base} = 8412$ MHz')
axs[1].legend()
axs[1].grid(True)

filtered_difference = np.array(difference_between_closed_loops)[np.abs(difference_between_closed_loops) <= 0.1]
filtered_simulated_equivalent_closed_loop_times_utc = np.array(simulated_equivalent_closed_loop_times_utc)[np.abs(difference_between_closed_loops) <= 0.1]

mean_filtered_difference = np.mean(filtered_difference)
rms_filtered_difference = np.std(filtered_difference)

axs[2].scatter(filtered_simulated_equivalent_closed_loop_times_utc, filtered_difference, marker='o', label = f'Simulated-Data Eq. Closed-Loop\nmean:{mean_filtered_difference:.2g}\nrms = {rms_filtered_difference:.2g}',s=10, alpha=0.6)
axs[2].set_xlabel('Time [s] (Midpoint of Interval)')
axs[2].set_ylabel('Residuals [Hz]')
axs[2].legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=7)
axs[2].grid(True)
plt.show()
###################################################################################
