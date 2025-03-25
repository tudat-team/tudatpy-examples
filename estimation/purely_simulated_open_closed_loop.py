"""
THIS SCRIPT SHOWCASES HOW TO GO FROM SIMULATED OPEN-LOOP DATA (within TUDAT) to SIMULATED EQUIVALENT CLOSED-LOOP DATA.
WE DO NOT IMPLEMENT TROPOSPHERIC CORRECTION YET.
USE DEVELOP BRANCH for TUDTA AND TUDATPY, tudat::Time scalar type
"""
import os
from tudatpy.interface import spice
from tudatpy.astro import time_conversion, element_conversion
from tudatpy.math import interpolators
from tudatpy.numerical_simulation import environment_setup, environment
from tudatpy.numerical_simulation import estimation, estimation_setup
from tudatpy.numerical_simulation.estimation_setup import observation
from datetime import datetime
from astropy.time import Time
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import fixed_quad


def compute_scipy_quadrature(interpolated_function, times, integration_time = 10):

    results = []  # To store the integral results
    interval_centers = []  # To store the midpoints of the intervals

    a = min(times)
    while a + integration_time <= max(times):
        b = a + integration_time  # End of the current interval
        midpoint = (a + b) / 2

        if any(abs(midpoint - t) < 1 for t in times):  # Tolerance-based match
            result, _ = fixed_quad(interpolated_function, a, b)
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
start = datetime(2013, 12, 28)
end = datetime(2013, 12, 29)

# Add a time buffer of one day
start_time = time_conversion.datetime_to_tudat(start).epoch().to_float() - 86400.0
end_time = time_conversion.datetime_to_tudat(end).epoch().to_float() + 86400.0

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

######## NO TROPO CORRECTION FOR NOW #########
#observation.set_vmf_troposphere_data(
#    [ "/Users/lgisolfi/Desktop/mex_phobos_flyby/VMF/y2013.vmf3_r.txt" ], True, False, bodies, False, True )
# Meteorological (tropospheric) uplink and downlink corrections
#weather_files = ([os.path.join('/Users/lgisolfi/Desktop/data_archiving-1.0/dataset/mex/gr035/downloaded/met', met_file) for met_file in os.listdir('/Users/lgisolfi/Desktop/data_archiving-1.0/dataset/mex/gr035/downloaded/met')])
#body_settings.get("Earth").ground_station_settings.append(data.set_estrack_weather_data_in_ground_stations(bodies,weather_files, 'NWNORCIA'))
######################################

########## IMPORTANT STEP ###################################
# Set the transponder turnaround ratio function
vehicleSys = environment.VehicleSystems()
vehicleSys.set_default_transponder_turnaround_ratio_function()
bodies.get_body("MEX").system_models = vehicleSys
###############################################################
base_frequency = 8412e6
column_types = ["utc_datetime_string", "signal_to_noise_ratio", "normalised_spectral_max","doppler_measured_frequency_hz", "doppler_noise_hz"]
reception_band = observation.FrequencyBands.x_band
transmission_band = observation.FrequencyBands.x_band
turnaround_ratio = observation.dsn_default_turnaround_ratios( observation.FrequencyBands.x_band,observation.FrequencyBands.x_band)
sites_list = []
fdets_files = []
ifms_files = []
###############################################################################################################
# Load IFMS and FDETS Files to be Analysed
fdets_files = os.path.join(mex_fdets_folder, 'Fdets.mex2013.12.28.On.complete.r2i.txt')
fdets_collection = observation.observations_from_fdets_files(fdets_files, base_frequency, column_types, 'MEX','NWNORCIA', 'ONSALA60', reception_band, transmission_band)
####################################### FDETS Collection Times ################################################
# Retrieve Fdets times (tudat::Time type) and convert them into float for later use
fdets_times_tudat_type = fdets_collection.get_observation_times()
fdets_times = [time.to_float()  for time in fdets_times_tudat_type[0]]
start_fdets_time = min(fdets_times)
end_fdets_time = max(fdets_times)

# Set MEX Antenna COM position (As done in mex_residuals_fdets.py)
com_position = [-1.3,0.0,0.0] # estimated based on the MEX_V16.TF file description
antenna_state = np.zeros((6, 1))
antenna_state[:3,0] = spice.get_body_cartesian_position_at_epoch("-41020", "-41000", "MEX_SPACECRAFT", "none", fdets_times[0])
antenna_state[:3,0] = antenna_state[:3,0] - com_position
antenna_ephemeris_settings = environment_setup.ephemeris.constant(antenna_state, "-41000",  "MEX_SPACECRAFT")
# Create tabulated ephemeris for the MEX antenna
antenna_ephemeris = environment_setup.ephemeris.create_ephemeris(antenna_ephemeris_settings, "Antenna")

# Load IFMS and FDETS Files to be Analysed
fdets_files = os.path.join(mex_fdets_folder, 'Fdets.mex2013.12.28.On.complete.r2i.txt')
ifms_file = [os.path.join(mex_ifms_folder, 'M32ICL1L02_D2X_133630203_00.TAB')]
fdets_collection = observation.observations_from_fdets_files(fdets_files, base_frequency, column_types, 'MEX','NWNORCIA', 'ONSALA60', reception_band, transmission_band)
ifms_collection = observation.observations_from_ifms_files(ifms_file, bodies, spacecraft_name, 'NWNORCIA', reception_band, transmission_band, apply_troposphere_correction = False)

# Set the spacecraft's reference point position to that of the antenna (in the MEX-fixed frame)
fdets_collection.set_reference_point(bodies, antenna_ephemeris, "Antenna", "MEX", observation.reflector1)

# Set Involved Link Ends
link_ends = {
    observation.receiver: observation.body_reference_point_link_end_id('Earth', 'ONSALA60'),
    observation.retransmitter: observation.body_reference_point_link_end_id('MEX', 'Antenna'),
    observation.transmitter: observation.body_reference_point_link_end_id('Earth', 'NWNORCIA'),
}

# Create a single link definition from the link ends
link_definition = observation.LinkDefinition(link_ends)
light_time_correction_list = list()
light_time_correction_list.append(
    estimation_setup.observation.first_order_relativistic_light_time_correction(["Sun"]))

# Define the observation model settings
open_loop_observation_model_settings = [
    estimation_setup.observation.doppler_measured_frequency(
        link_definition, light_time_correction_list
    )
]
closed_loop_observation_model_settings = [
    estimation_setup.observation.dsn_n_way_doppler_averaged(
        link_definition, light_time_correction_list, subtract_doppler_signature = False)
]

closed_loop_ancillary_settings = observation.dsn_n_way_doppler_ancilliary_settings(
    frequency_bands =  [reception_band, transmission_band],
    reference_frequency_band = reception_band,
    reference_frequency = 0,
    integration_time = 10
)

closed_loop_observation_simulation_settings = [observation.tabulated_simulation_settings(
    observable_type = estimation_setup.observation.dsn_n_way_averaged_doppler,
    link_ends = link_definition,
    simulation_times = fdets_times_tudat_type[0],
    ancilliary_settings = closed_loop_ancillary_settings
    )]

open_loop_observation_simulation_settings = estimation_setup.observation.observation_settings_from_collection(fdets_collection, bodies)
# Create observation simulators
closed_loop_observation_simulators = estimation_setup.create_observation_simulators(closed_loop_observation_model_settings, bodies)
open_loop_observation_simulators = estimation_setup.create_observation_simulators(open_loop_observation_model_settings, bodies)
open_loop_collection = estimation.simulate_observations(open_loop_observation_simulation_settings, open_loop_observation_simulators, bodies)

estimation.compute_residuals_and_dependent_variables(open_loop_collection, open_loop_observation_simulators, bodies) # fdets simulator
closed_loop_collection = estimation.simulate_observations(closed_loop_observation_simulation_settings, closed_loop_observation_simulators, bodies)
estimation.compute_residuals_and_dependent_variables(closed_loop_collection, closed_loop_observation_simulators, bodies) # ifms simulator
exit()

# END OF INTERESTING PART OF THE CODE, UP TO THE FIRST TRIGGERED ERROR (on line 200, in simulate_observations)
# RuntimeError: Error in nearest neighbour search, size of input vector is 1
###################################################################################################
###################################################################################################
############################################################################################################

simulated_observations_fdets = fdets_collection.get_computed_observations() # simulated open-loop (FDETS)
simulated_observations_ifms = ifms_collection.get_computed_observations() # simulated closed-loop (IFMS)

# Interpolated, simulated closed loop continous function (IFMS)
simulated_closed_loop_continuous_function = interp1d(ifms_times, simulated_observations_ifms, kind='cubic', fill_value='extrapolate')
# Interpolated, simulated open loop continous function (fdets)
simulated_open_loop_continuous_function = interp1d(fdets_times, simulated_observations_fdets, kind='cubic', fill_value='extrapolate')

# Compute quadrature to convert simulated open_loop into equivalent simulated closed loop
simulated_equivalent_closed_loop_observables, simulated_equivalent_closed_loop_times = compute_scipy_quadrature(simulated_open_loop_continuous_function,
                                                                                                                fdets_times,
                                                                                                                integration_time = 10)

fig, axs = plt.subplots(3, 1, figsize=(10, 12))  # 2 rows, 1 column of subplots

simulated_open_loop_tone = np.array(simulated_observations_fdets[0]) - base_frequency #retrieve f tone from open-loop observable
simulated_equivalent_closed_loop_tone = np.array(simulated_equivalent_closed_loop_observables) - base_frequency
equivalent_closed_loop_continuous_function_from_data = interp1d(equivalent_closed_loop_times, equivalent_closed_loop_observables, kind='cubic', fill_value='extrapolate')
equivalent_closed_loop_tone_from_data = equivalent_closed_loop_continuous_function_from_data(simulated_equivalent_closed_loop_times) - base_frequency

simulated_closed_loop_at_interval_centers = [
    simulated_closed_loop_continuous_function(interval_center) - base_frequency
    for interval_center in simulated_equivalent_closed_loop_times
]


# Compute Differences
difference_to_subtract_due_to_station_position_bias = ifms_closed_loop_continuous_function(simulated_equivalent_closed_loop_times) - fdets_open_loop_continuous_function(simulated_equivalent_closed_loop_times)
difference_between_closed_loops_subtracting_station_position_bias = [j-i-z for i, j, z in zip(simulated_equivalent_closed_loop_tone,simulated_closed_loop_at_interval_centers, difference_to_subtract_due_to_station_position_bias)]

difference_between_closed_loops = [j-i for i, j in zip(simulated_equivalent_closed_loop_tone,simulated_closed_loop_at_interval_centers)]
difference_between_simulated_and_data_equivalent_closed_loop = [j-i for i, j in zip(simulated_equivalent_closed_loop_tone,equivalent_closed_loop_tone_from_data)]

rms_difference = np.std(difference_between_simulated_and_data_equivalent_closed_loop)
mean_difference = np.mean(difference_between_simulated_and_data_equivalent_closed_loop)

rms_closed_loop_difference = np.std(difference_between_closed_loops_subtracting_station_position_bias)
mean_closed_loop_difference = np.mean(difference_between_closed_loops_subtracting_station_position_bias)

rms_closed_loop_difference_simulated = np.std(difference_between_closed_loops)
mean_closed_loop_difference_simulated = np.mean(difference_between_closed_loops)

mean_pride_residuals = np.mean(fdets_collection.get_concatenated_residuals())
rms_pride_residuals = np.std(fdets_collection.get_concatenated_residuals())

mean_ifms_residuals = np.mean(ifms_collection.get_concatenated_residuals())
rms_ifms_residuals = np.std(ifms_collection.get_concatenated_residuals())

axs[0].scatter(fdets_times, simulated_open_loop_tone, marker='o', label='Simulated Open-Loop Observable', s=15, alpha=0.5)
axs[0].scatter(simulated_equivalent_closed_loop_times, simulated_equivalent_closed_loop_tone, marker='+', label='Simulated Equivalent Closed-Loop', s=5)
axs[0].set_xlabel('Time [s] (Midpoint of Interval)')
axs[0].set_ylabel('$f_{tone} = f_{R} - f_{base}$ [Hz], $f_{base} = 8412$ MHz')
axs[0].legend()
axs[0].grid(True)

axs[1].scatter(simulated_equivalent_closed_loop_times, simulated_equivalent_closed_loop_tone, marker='o', label='Simulated Equivalent Closed-Loop', s=15, alpha=0.5)
axs[1].scatter(simulated_equivalent_closed_loop_times, equivalent_closed_loop_tone_from_data, marker='+', label='Equivalent Closed-Loop from Data', s=5)
axs[1].set_xlabel('Time [s] (Midpoint of Interval)')
axs[1].set_ylabel('$f_{tone} = f_{R} - f_{base}$ [Hz], $f_{base} = 8412$ MHz')
axs[1].legend()
axs[1].grid(True)


axs[2].scatter(simulated_equivalent_closed_loop_times,difference_between_closed_loops_subtracting_station_position_bias, marker='o', label = f'Simulated Closed-Loops - station_bias\nmean:{mean_closed_loop_difference:.2g}\nrms = {rms_closed_loop_difference:.2g}',s=10, alpha=0.6)
#axs[2].scatter(simulated_equivalent_closed_loop_times,difference_between_simulated_and_data_equivalent_closed_loop, marker='o', label = f'Simulated Open-Loops\nmean:{mean_open_loop_difference_simulated:.2g}\nrms = {rms_open_loop_difference_simulated:.2g}',s=10, alpha=0.6)
#axs[2].scatter(simulated_equivalent_closed_loop_times,difference_between_closed_loops, marker='o', label = f'Simulated Closed-Loops\nmean:{mean_closed_loop_difference_simulated:.2g}\nrms = {rms_closed_loop_difference_simulated:.2g}',s=10, alpha=0.6)
axs[2].scatter(fdets_times, fdets_collection.get_concatenated_residuals(), marker='o', label = f'FDETS\nmean:{mean_pride_residuals:.2g}\nrms:{rms_pride_residuals:.2g}',s=10, alpha=0.3)
#axs[2].scatter(simulated_equivalent_closed_loop_times, difference_between_simulated_and_data_equivalent_closed_loop, marker='o', label = f'Simulated-Data Eq. Closed-Loop\nmean:{mean_difference:.2g}\nrms = {rms_difference:.2g}',s=10, alpha=0.6)
#axs[2].scatter(ifms_times, ifms_collection.get_concatenated_residuals(), marker='o', label = f'IFMS\nmean:{mean_difference:.2g}\nrms = {rms_difference:.2g}',s=10, alpha=0.2)
axs[2].set_xlabel('Time [s] (Midpoint of Interval)')
axs[2].set_ylabel('Residuals [Hz] (No Tropo Corr.)')
axs[2].legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=7)
axs[2].grid(True)
plt.show()
exit()
###################################################################################################


###############################################################################################################
############################################ PRIDE/ODF CASE ###################################################

fdets_files_1 = os.path.join(mex_fdets_folder, 'Fdets.mex2013.12.28.On.complete.r2i.txt')
odf_file = [os.path.join(mex_odf_folder, 'M00ODF0L1A_ODF_133630350_00.DAT')]
fdets_collection_1  = observation.observations_from_fdets_files(
    fdets_files, base_frequency, column_types, 'MEX',
    'DSS-63', 'ONSALA60', reception_band, transmission_band
)
# Loading ODF file
odf_collection = observation.observations_from_odf_files(
    bodies, odf_file, spacecraft_name
)
fdets_times_1 = fdets_collection_1.get_observation_times()
fdets_times_1 = [time.to_float() for time in fdets_times_1[0]]
start_fdets_time_1 = min(fdets_times_1)
end_fdets_time_1 = max(fdets_times_1)
print(f'Start fdets time 1: {start_fdets_time_1}')
print(f'End fdets time 1: {end_fdets_time_1}')
odf_times = odf_collection.get_observation_times()
odf_times = [time.to_float() for time in odf_times[0]]
start_odf_time = min(odf_times)
end_odf_time = max(odf_times)
print(f'Start odf time: {start_odf_time}')
print(f'End odf time: {end_odf_time}')
time_filter_1 = estimation.observation_filter(
    estimation.time_bounds_filtering, start_odf_time, end_odf_time, use_opposite_condition = True)
fdets_collection_1.filter_observations(time_filter_1)

# Example: Replace this with your actual data
n_points_1 = fdets_collection_1.get_observations()[0]
n_times_1 = fdets_collection_1.get_observation_times()
n_times_1 = [time.to_float() for time in n_times_1[0]]
n_points_odf = np.array(odf_collection.get_observations()[0])
n_times_odf = odf_collection.get_observation_times()
n_times_odf = [time.to_float() for time in n_times_odf[0]]

# Split the data into subarrays of 10 elements
weights_1 = np.ones(len(n_points_1))
subarrays_1 = [n_points_1[i:i + T] for i in range(0, len(n_points_1), T)]
weight_subarrays_1 = [weights_1[i:i + T] for i in range(0, len(weights_1), T)]
time_subarrays_1 = [n_times_1[i:i + T] for i in range(0, len(n_points_1), T)]
# Ensure each subarray and weight array has exactly 10 elements
valid_subarrays_1 = [(data_sub_1, weight_sub_1) for data_sub_1, weight_sub_1 in zip(subarrays_1, weight_subarrays_1) if len(data_sub_1) == T]
valid_time_subarrays_1 = [time_sub_1 for time_sub_1 in time_subarrays_1 if len(time_sub_1) == T]
result_array = np.array([np.sum(data_sub_1 * weight_sub) / T for data_sub_1, weight_sub in valid_subarrays_1])
time_result_array_1 = np.array([time_sub_1[T//2 -1] for time_sub_1 in valid_time_subarrays_1])
original_mjd_times_1 = [time_conversion.seconds_since_epoch_to_julian_day(time_1) for time_1 in n_times_1]
original_utc_times_1 = [Time(mjd_time_1, format='jd', scale='utc').datetime for mjd_time_1 in original_mjd_times_1]
compressed_mjd_times_1 = [time_conversion.seconds_since_epoch_to_julian_day(time_1) for time_1 in time_result_array_1]
compressed_utc_times_1 = [Time(mjd_time_1, format='jd', scale='utc').datetime for mjd_time_1 in compressed_mjd_times_1]
##################################################################################
# Trial: using scipy tools to compute quadrature (ODF case)
x_data = n_times_1
y_data = n_points_1

# Step 1: Fit the data to create a continuous function
# We use cubic interpolation
equivalent_closed_loop_continuous_function_1 = interp1d(n_times_1, n_points_1, kind='cubic', fill_value='extrapolate')
odf_closed_loop_continuous_function = interp1d(n_times_odf, n_points_odf, kind='cubic', fill_value='extrapolate')
for i, single_observation_set in enumerate(
        odf_collection.get_single_observation_sets()):
    if i != 1:
        # retrieve from link definition
        observation_set_link_def = single_observation_set.link_definition
        observation_set_link_def_str = [
            observation_set_link_def.link_end_id(observation.transmitter).reference_point,
            observation_set_link_def.link_end_id(observation.receiver).reference_point]
        print(observation_set_link_def_str)
        print(single_observation_set)
        observation_set_ancilliary_settings = single_observation_set.ancilliary_settings
        try:
            observation_set_ref_frequency = observation_set_ancilliary_settings.get_float_settings(
                observation.ObservationAncilliarySimulationVariable.doppler_reference_frequency)
        except:
            continue


# Step 2: Loop over the full range of times
results_1 = []  # To store the integral results
interval_centers = []  # To store the midpoints of the intervals

a = min(fdets_times)
while a + T <= max(fdets_times):
    b = a + T  # End of the current interval
    midpoint = (a + b) / 2

    if any(abs(midpoint - t) < 1 for t in fdets_times):  # Tolerance-based match
        result, _ = fixed_quad(equivalent_closed_loop_continuous_function_1, a, b)
        normalized_result = result / (b - a)
        results_1.append(normalized_result)
        interval_centers.append(midpoint)  # Store time tag
        a = b  # Move to next interval
    else:
        # Skip gaps
        a = b  # Ensure progression

# Step 3: Prepare the figure with subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 10))  # 2 rows, 1 column of subplots

# First subplot: Fixed Quadrature Results
n_points_1 =  np.array(n_points_1) - base_frequency #retrieve f tone from open-loop observable
n_points_odf = observation_set_ref_frequency*turnaround_ratio -(np.array(n_points_odf)) - base_frequency # retrieve f_tone from odf observable
results_1 = np.array(results_1) - base_frequency
axs[0].scatter(n_times_1, n_points_1, marker='o', label='Open-Loop Observable', s=15, alpha=0.5)
axs[0].scatter(interval_centers, results_1, marker='+', label='Gaussian Quadrature', s=5)
axs[0].scatter(n_times_odf, n_points_odf, marker='+', label='Closed-Loop ODF Observable', s=5)
axs[0].set_xlabel('Time (Midpoint of Interval)')
axs[0].set_xlim(max(min(n_times_1), min(n_times_odf)), min(max(n_times_1), max(n_times_odf)))
axs[0].set_ylabel('$f_{tone} = f_{R} - f_{base}$ [Hz], $f_{base} = 8412$ MHz')
axs[0].set_title('Open-Loop, Closed-Loop, Equivalent Closed-Loop Comparison (GR035), $\Delta T = 10s$')
axs[0].legend()
axs[0].grid(True)

# Second subplot: Difference Between Interpolated Functions
evaluation_points = np.linspace(max(min(n_times_1), min(n_times_odf)), min(max(n_times_1), max(n_times_odf)), 500)
equivalent_values = equivalent_closed_loop_continuous_function_1(evaluation_points)
odf_values = odf_closed_loop_continuous_function(evaluation_points)
difference = (equivalent_values - base_frequency) - (observation_set_ref_frequency*turnaround_ratio - odf_values - base_frequency)
mean_diff = np.mean(difference)

axs[1].plot(evaluation_points, difference, label="Difference", color='blue')
axs[1].axhline(mean_diff, color='black', linestyle='--', linewidth=0.8, label=f'Mean Difference: {mean_diff}')
axs[1].set_xlim(max(min(n_times_1), min(n_times_odf)), min(max(n_times_1), max(n_times_odf)))
axs[1].set_title("Difference in the Data (Cubic Spline Interpolation)")
axs[1].set_xlabel("Time [s]")
axs[1].set_ylabel("Difference [Hz]")
axs[1].legend()
axs[1].grid(True)
# Adjust layout and show the combined plot
plt.tight_layout()
#plt.show()