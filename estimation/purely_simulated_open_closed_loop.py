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
from scipy.integrate import fixed_quad, quad



#########################################################

def compute_scipy_quadrature(interpolated_function, times, integration_time=10):
    results = list()
    midpoints = list()

    a = min(times)
    max_time = max(times)

    while a + integration_time <= max_time:
        b = a + integration_time
        midpoint = (a + b) / 2
        result, _ = quad(interpolated_function, a, b)
        normalized_result = result / (b - a)
        results.append(normalized_result)
        midpoints.append(midpoint)
        # Always move forward
        a = b

    return results, midpoints
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

integration_time = 200
open_loop_cadence = 1

start_time = time_conversion.datetime_to_tudat(start).epoch().to_float() # in utc
end_time = time_conversion.datetime_to_tudat(end).epoch().to_float()  # in utc
times_linspace = np.arange(start_time, end_time, step = open_loop_cadence) # in utc
tudat_times_linspace_utc = [Time(time) for time in times_linspace] # in utc

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
                                                         start_time, end_time, 10.0))

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

body_fixed_station_position = bodies.get('Earth').get_ground_station('ONSALA60').station_state.get_cartesian_position(0)

################# First conversion: Convert UTC times to TDB. ########################################
time_scale_converter = time_conversion.default_time_scale_converter()
tudat_times_linspace_tdb = list() #prepare TDB tudat::Time list
for time_utc in tudat_times_linspace_utc: # for each UTC epoch, convert it to TDB
    tudat_times_linspace_tdb.append( time_scale_converter.convert_time(
        input_scale = time_conversion.utc_scale,
        output_scale = time_conversion.tdb_scale,
        input_value = time_utc,
        earth_fixed_position = body_fixed_station_position))

times_linspace_tdb = [tudat_times_tdb.to_float() for tudat_times_tdb in tudat_times_linspace_tdb]
########################################################################################################

######## APPLY TROPOSPHERIC CORRECTION FOR UPLINK AND DOWNLINK ########################################################################################################
#observation.set_vmf_troposphere_data(
#    [ "/Users/lgisolfi/Desktop/mex_phobos_flyby/VMF/y2013.vmf3_r" ], True, False, bodies, False, True
#)
# Load meteorological uplink and downlink corrections
#weather_files = ([os.path.join('/Users/lgisolfi/Desktop/data_archiving-1.0/dataset/mex/gr035/downloaded/met', met_file) for met_file in os.listdir('/Users/lgisolfi/Desktop/data_archiving-1.0/dataset/mex/gr035/downloaded/met')])
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
    simulation_times = tudat_times_linspace_tdb,
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
    integration_time = integration_time
)

closed_loop_observation_simulation_settings = [observation.tabulated_simulation_settings(
    observable_type = estimation_setup.observation.dsn_n_way_averaged_doppler,
    link_ends = link_definition,
    simulation_times = tudat_times_linspace_tdb,
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
simulated_times_fdets = open_loop_collection.get_observation_times()[0] # simulated open-loop (FDETS) # TDB
simulated_times_ifms = closed_loop_collection.get_observation_times()[0] # simulated closed-loop (IFMS) # TDB
simulated_times_ifms_float = [time.to_float() for time in simulated_times_ifms] # TDB
simulated_times_fdets_float = [time.to_float() for time in simulated_times_fdets] # TDB

# These are the calendar date corresponding to the converted UTC times (in other words, these are the UTC timetags in the fdets files)
simulated_times_fdets_utc = [time_conversion.julian_day_to_calendar_date(time_conversion.seconds_since_epoch_to_julian_day(time)) for time in times_linspace]
simulated_times_ifms_utc = [time_conversion.julian_day_to_calendar_date(time_conversion.seconds_since_epoch_to_julian_day(time)) for time in times_linspace]

################ Interpolated, simulated closed loop continous functions (IFMS and FDETS) ##############################
# Interpolate the simulated open loop and closed loop, using the UTC times (these are basically the float versions of the fdets/ifms time tags)
simulated_closed_loop_continuous_function_utc = interp1d(times_linspace, simulated_observations_ifms, kind='cubic', fill_value='extrapolate')
simulated_open_loop_continuous_function_utc = interp1d(times_linspace, simulated_observations_fdets, kind='cubic', fill_value='extrapolate')
########################################################################################################################

################ Compute quadrature to convert simulated open_loop into equivalent simulated closed loop ##############
simulated_equivalent_closed_loop_observables, simulated_equivalent_closed_loop_times = compute_scipy_quadrature(
    simulated_open_loop_continuous_function_utc,
    times_linspace, # quadrature: times enter in UTC
    integration_time = integration_time)
#######################################################################################################################

################# Format times to CALENDAR UTC ################
simulated_equivalent_closed_loop_times_utc = [time_conversion.julian_day_to_calendar_date(time_conversion.seconds_since_epoch_to_julian_day(time)) for time in simulated_equivalent_closed_loop_times]
#######################################################

################ Retrieve the tones ###################
simulated_open_loop_tone = simulated_observations_fdets - base_frequency # (this is in tdb)
simulated_equivalent_closed_loop_tone = np.array(simulated_equivalent_closed_loop_observables) - base_frequency # (this is evaluated at utc)
simulated_observations_ifms_tone =  simulated_observations_ifms - base_frequency # (this is in tdb)
#######################################################

################# Retrieve differences between the two simulated closed loop (equivalent-ifms) ################
simulated_closed_loop_tone = [
    simulated_closed_loop_continuous_function_utc(midpoint) - base_frequency
    for midpoint in simulated_equivalent_closed_loop_times]

difference_between_closed_loops_utc = [j-i for i, j in zip(simulated_equivalent_closed_loop_tone,simulated_closed_loop_tone)]
###############################################################################################################

######################## Retrieve Statistics (mean and rms) #######################
rms_closed_loop_difference_simulated = np.std(difference_between_closed_loops_utc)
mean_closed_loop_difference_simulated = np.mean(difference_between_closed_loops_utc)
mean_pride_residuals = np.mean(open_loop_collection.get_concatenated_residuals())
rms_pride_residuals = np.std(open_loop_collection.get_concatenated_residuals())
mean_ifms_residuals = np.mean(closed_loop_collection.get_concatenated_residuals())
rms_ifms_residuals = np.std(closed_loop_collection.get_concatenated_residuals())
###################################################################################

######################## Visualize data and residuals  ###########################
fig, axs = plt.subplots(3, 1, figsize=(10, 12))
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

filtered_difference = np.array(difference_between_closed_loops_utc)[np.abs(difference_between_closed_loops_utc) <= 0.1]
filtered_simulated_equivalent_closed_loop_times_utc = np.array(simulated_equivalent_closed_loop_times_utc)[np.abs(difference_between_closed_loops_utc) <= 0.1]

mean_filtered_difference = np.mean(filtered_difference)
rms_filtered_difference = np.std(filtered_difference)

print(mean_filtered_difference)
print(rms_filtered_difference)
axs[2].scatter(filtered_simulated_equivalent_closed_loop_times_utc, filtered_difference, marker='o', label = f'Simulated-Data Eq. Closed-Loop\nmean:{mean_filtered_difference:.2g}\nrms = {rms_filtered_difference:.2g}',s=10, alpha=0.6)
axs[2].set_xlabel('Time [s] (Midpoint of Interval)')
axs[2].set_ylabel('Residuals [Hz]')
axs[2].legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=7)
axs[2].grid(True)
plt.show()
###################################################################################
