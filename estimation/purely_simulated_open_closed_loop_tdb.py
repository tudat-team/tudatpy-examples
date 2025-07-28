"""
THIS SCRIPT SHOWCASES HOW TO GO FROM SIMULATED OPEN-LOOP DATA (within TUDAT) to SIMULATED EQUIVALENT CLOSED-LOOP DATA.
- TROPOSPHERIC CORRECTION IS IMPLEMENTED.
- USE DEVELOP BRANCH for TUDAT AND TUDATPY, tudat::Time scalar type
"""
################### Import Modules #####################
import os
from tudatpy.interface import spice
from tudatpy.astro import time_conversion, element_conversion
from tudatpy.math import interpolators
from tudatpy.numerical_simulation import environment_setup, environment
from tudatpy.numerical_simulation import estimation, estimation_setup
from tudatpy.numerical_simulation.estimation_setup import observation
from datetime import datetime
from tudatpy.numerical_simulation import Time
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
########################################################

def make_state_interpolator(times: np.ndarray, states: np.ndarray):
    interpolators = [interp1d(times, states[:, i], kind='linear', fill_value="extrapolate") for i in range(6)]

    def state_function(t: float) -> np.ndarray:
        return np.array([[f(t)] for f in interpolators])  # shape (6, 1)

    return state_function

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

fixed_point_trajectory_flag= True
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
start = datetime(2000, 12, 28, 00, 00, 00) # simulate over the relevant GR035 timeframe
end = datetime(2000, 12, 28, 2, 00, 00)
integration_time = 600
open_loop_cadence = 10

start_time = time_conversion.datetime_to_tudat(start).epoch().to_float() # in fake tdb
end_time = time_conversion.datetime_to_tudat(end).epoch().to_float()  # in fake tdb

# Create default body settings for celestial bodies
bodies_to_create = ["Earth","Sun", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Moon"]
global_frame_origin = "SSB"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation)

rotation_function = lambda t: np.identity(3)
body_settings.get('Earth').rotation_model_settings= environment_setup.rotation_model.custom_rotation_model(
    base_frame='J2000',
    target_frame='IAU_EARTH',
    custom_rotation_matrix_function=rotation_function,
    finite_difference_time_step=3600.0  # example timestep in seconds
)

# Modify Earth default settings
body_settings.get('Earth').shape_settings = environment_setup.shape.oblate_spherical_spice()
body_settings.get('Earth').gravity_field_settings.associated_reference_frame = "ITRS"
spacecraft_name = "MEX" # Set Spacecraft Name
spacecraft_central_body = "Mars" # Set Central Body (Mars)
body_settings.add_empty_settings(spacecraft_name) # Create empty settings for spacecraft

times_linspace_tdb = np.arange(start_time, end_time, step = open_loop_cadence) # in tdb, float type
tudat_times_linspace_tdb = [Time(time) for time in times_linspace_tdb] # in tdb, tudat::Time type

if fixed_point_trajectory_flag:
    initial_state_mex = np.array([1e8,0,0,0,0,0])
    final_state_mex = initial_state_mex
    p0 =np.array(initial_state_mex[:3])
    p1 =np.array(final_state_mex[:3])
    N = len(times_linspace_tdb)
    v = (p1 - p0) / (start_time - end_time)
    positions = p0 + np.outer(times_linspace_tdb - start_time, v) # Compute position over time
    velocities = np.tile(v, (N, 1)) # Repeat velocity for each time
    straight_trajectory = np.hstack((positions, velocities)) # stack them
    state_function = make_state_interpolator(times_linspace_tdb, straight_trajectory) # create state function for custom_ephemeris
    custom_ephem = environment_setup.ephemeris.custom_ephemeris(
        custom_state_function=state_function,
        frame_origin='Earth',
        frame_orientation=global_frame_orientation
    )
    body_settings.get(spacecraft_name).ephemeris_settings = custom_ephem

else:
    body_settings.get(spacecraft_name).ephemeris_settings = environment_setup.ephemeris.interpolated_spice(
        start_time, end_time, 10.0, spacecraft_central_body, global_frame_orientation)

body_settings.get(spacecraft_name).rotation_model_settings = environment_setup.rotation_model.spice(
    global_frame_orientation, spacecraft_name + "_SPACECRAFT", "")
body_settings.get("Earth").ground_station_settings = environment_setup.ground_station.radio_telescope_stations()

# Create System of Bodies using the above-defined body_settings
bodies = environment_setup.create_system_of_bodies(body_settings)
receiver_station_name = 'ONSALA60' # you can change this to CEDUNA or PARKES to test different stations

times_linspace_tdb = [tudat_times_tdb.to_float() for tudat_times_tdb in tudat_times_linspace_tdb] # tdb, float type
########## Set the transponder turnaround ratio function ###################################
vehicleSys = environment.VehicleSystems()
vehicleSys.set_default_transponder_turnaround_ratio_function()
bodies.get_body("MEX").system_models = vehicleSys
base_frequency = 8412e6 # MEX Base frequency
reception_band = observation.FrequencyBands.x_band
transmission_band = observation.FrequencyBands.x_band
turnaround_ratio = observation.dsn_default_turnaround_ratios( observation.FrequencyBands.x_band,observation.FrequencyBands.x_band)
######################################################## TEMPORARY LINE TO FIX FREQUENCY CALCULATOR ERROR ###########################################################
#bodies.get( "Earth" ).get_ground_station( "NWNORCIA" ).transmitting_frequency_calculator = environment.ConstantTransmittingFrequencyCalculator(7166445042.992178)
bodies.get( "Earth" ).get_ground_station( "NWNORCIA" ).transmitting_frequency_calculator = environment.ConstantTransmittingFrequencyCalculator(749/880*7166445042)
######################################################## TEMPORARY LINE TO FIX FREQUENCY CALCULATOR ERROR ###########################################################

######################### Set Involved Link Ends ######################################
link_ends = {
    observation.receiver: observation.body_reference_point_link_end_id('Earth', receiver_station_name),
    observation.retransmitter: observation.body_origin_link_end_id('MEX'),
    observation.transmitter: observation.body_reference_point_link_end_id('Earth', 'NWNORCIA'),
}
# Create a single link definition from the link ends
link_definition = observation.LinkDefinition(link_ends)
light_time_correction_list = list() # EMPTY RELATIVISTIC CORRECTIONS
#light_time_correction_list.append(
#    estimation_setup.observation.first_order_relativistic_light_time_correction(["Sun"]))
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

############### Create OPEN LOOP observation simulator ###############
open_loop_observation_simulators = estimation_setup.create_observation_simulators(open_loop_observation_model_settings, bodies)
#############################################################

############### Retrieve OPEN LOOP Collection ########################
open_loop_collection = estimation.simulate_observations(open_loop_observation_simulation_settings, open_loop_observation_simulators, bodies)
############### Compute OPEN LOOOP Residuals and Dependent Variables ########################
estimation.compute_residuals_and_dependent_variables(open_loop_collection, open_loop_observation_simulators, bodies) # fdets simulator
##################################################################################

############### Retrieve OPEN LOOP Simulated Observations ########################
simulated_observations_fdets = open_loop_collection.get_observations()[0] # simulated open-loop (FDETS)
simulated_times_fdets_tdb = [time_conversion.DateTime.from_julian_day(time_conversion.seconds_since_epoch_to_julian_day(time)).to_python_datetime() for time in times_linspace_tdb]
################ Interpolated, simulated open loop (FDETS) continous function ##############################
simulated_open_loop_continuous_function_tdb = interp1d(times_linspace_tdb, simulated_observations_fdets, kind='cubic', fill_value='extrapolate')
################ Compute quadrature to convert simulated open_loop into equivalent simulated closed loop ##############
simulated_equivalent_closed_loop_observables, simulated_equivalent_closed_loop_times = compute_scipy_quadrature(
    simulated_open_loop_continuous_function_tdb,
    times_linspace_tdb, # quadrature: times enter in tdb
    integration_time = integration_time)

tudat_simulated_equivalent_closed_loop_times = [Time(time_tdb) for time_tdb in simulated_equivalent_closed_loop_times] # in tdb
closed_loop_times_linspace_tdb = [time_tudat_tdb.to_float() for time_tudat_tdb in tudat_simulated_equivalent_closed_loop_times]
#######################################################################################################################
################# Format times to CALENDAR TDB ################
simulated_equivalent_closed_loop_times_tdb = [time_conversion.DateTime.from_julian_day(time_conversion.seconds_since_epoch_to_julian_day(time)).to_python_datetime() for time in simulated_equivalent_closed_loop_times]
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
    simulation_times = tudat_simulated_equivalent_closed_loop_times,
    ancilliary_settings = closed_loop_ancillary_settings
)]
closed_loop_observation_simulators = estimation_setup.create_observation_simulators(closed_loop_observation_model_settings, bodies)
closed_loop_collection = estimation.simulate_observations(closed_loop_observation_simulation_settings, closed_loop_observation_simulators, bodies)
estimation.compute_residuals_and_dependent_variables(closed_loop_collection, closed_loop_observation_simulators, bodies) # ifms simulator
simulated_observations_ifms = closed_loop_collection.get_observations()[0] # simulated closed-loop (IFMS)

################ Retrieve the tones ###################
simulated_open_loop_tone = simulated_observations_fdets #- base_frequency # (this is in tdb)
simulated_equivalent_closed_loop_tone = np.array(simulated_equivalent_closed_loop_observables) #- base_frequency # (this is evaluated at tdb)
simulated_closed_loop_tone =  simulated_observations_ifms #- base_frequency # (this is in tdb equivalent to the tdb times of the equivalent closed-loop doppler)
#######################################################

################# Retrieve differences between the two simulated closed loop (equivalent-ifms) ################
difference_between_closed_loops_tdb = [j-i for i, j in zip(simulated_equivalent_closed_loop_tone,simulated_closed_loop_tone)]
simulated_times_ifms_tdb = [time_conversion.DateTime.from_julian_day(time_conversion.seconds_since_epoch_to_julian_day(time)).to_python_datetime() for time in simulated_equivalent_closed_loop_times]
###############################################################################################################

######################## Retrieve Statistics (mean and rms) #######################
rms_closed_loop_difference_simulated = np.std(difference_between_closed_loops_tdb)
mean_closed_loop_difference_simulated = np.mean(difference_between_closed_loops_tdb)
mean_pride_residuals = np.mean(open_loop_collection.get_concatenated_residuals())
rms_pride_residuals = np.std(open_loop_collection.get_concatenated_residuals())
mean_ifms_residuals = np.mean(closed_loop_collection.get_concatenated_residuals())
rms_ifms_residuals = np.std(closed_loop_collection.get_concatenated_residuals())
###################################################################################

####### ANOTHER VALIDATION PLOT ###########
#print(len(simulated_times_fdets_tdb),len(simulated_times_ifms_tdb),len(simulated_equivalent_closed_loop_times_tdb))
#print(times_linspace_tdb[:20],simulated_equivalent_closed_loop_times[:20])
#########
closed_loop_interpol = interp1d(simulated_equivalent_closed_loop_times,simulated_closed_loop_tone)
cond = (times_linspace_tdb >= np.min(simulated_equivalent_closed_loop_times)) & \
       (times_linspace_tdb <= np.max(simulated_equivalent_closed_loop_times))
valid_times = np.array(times_linspace_tdb)[cond]

shared_times = np.intersect1d(valid_times, simulated_equivalent_closed_loop_times)

closed_loop_values = simulated_closed_loop_tone
open_loop_values = simulated_open_loop_continuous_function_tdb(shared_times) #- base_frequency
residual = (closed_loop_values - open_loop_values)

bias = np.mean(residual)

print(f'Average Bias: {bias}')

first_derivative = np.gradient(open_loop_values, shared_times)
second_derivative = np.gradient(first_derivative, shared_times)
third_derivative = np.gradient(second_derivative, shared_times)
fourth_derivative = np.gradient(third_derivative, shared_times)

fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)


axs[0].scatter(shared_times, open_loop_values, label='Open Loop (FDETS)', color='orange')
axs[0].scatter(shared_times, closed_loop_values, label='Closed Loop (IFMS)', color='blue', alpha = 0.4, marker = '+')
axs[0].set_title('Interpolated (Simulated) Doppler Observations')
axs[0].set_ylabel('Doppler Tone [Hz]')
axs[0].grid(True)
axs[0].legend()

# Second subplot: Residual
axs[1].plot(shared_times, second_derivative*integration_time**2/24, label="Quadratic Term", color='r')
axs[1].plot(shared_times, fourth_derivative*integration_time**4/1920, label="Quartic Term", color='r', linestyle = '--')
axs[1].scatter(shared_times[np.abs(residual)<10000], residual[np.abs(residual)<10000], label='Closed - Open Loop', color='green', s = 8)
axs[1].set_title('Difference Between Closed and Open Loop Doppler')
axs[1].set_xlabel('TDB Time [s]')
axs[1].set_ylabel(r'$h_c(t) - h_o(t) \approx \frac{T^2}{24} \cdot h_o^{\prime\prime}(t)$ [Hz]')
axs[1].grid(True)
axs[1].legend(loc='upper left')

# Add zoom inset to axs[1]
#axins = inset_axes(axs[1], width="30%", height="20%", loc='upper right')  # Adjust loc if needed
#axins.scatter(shared_times, residual, color = 'green', s = 8)
#axins.plot(shared_times, second_derivative*integration_time**2/24, 'red')

# Set zoomed region
#axins.set_xlim(valid_times[1550], valid_times[1730])
#axins.set_ylim(-35, 10)

# Draw lines showing zoomed area
#mark_inset(axs[1], axins, loc1=2, loc2=4, fc="none", ec="0.5")


axs[2].plot(shared_times, second_derivative*integration_time, label="2nd derivative Open Loop", color='r')
axs[2].set_ylabel("2nd Derivative Open-Loop [Hz]")
axs[2].set_xlabel("Time [s]")
axs[2].set_title("Second Derivative of Open-loop")
axs[2].legend()

plt.tight_layout()
plt.show()
########################################################################################################################

######################## Visualize data and residuals  ###########################
fig, axs = plt.subplots(3, 1, figsize=(10, 12))
axs[0].scatter(simulated_times_fdets_tdb, simulated_open_loop_tone, marker='o', label='Simulated Open-Loop Tone', s=15, alpha=0.5)
axs[0].scatter(simulated_equivalent_closed_loop_times_tdb, simulated_equivalent_closed_loop_tone, marker='o', label='Simulated Equivalent Closed-Loop Tone', s=15, alpha=0.5)
axs[0].set_xlabel('Time [s] (Midpoint of Interval)')
axs[0].set_ylabel('$f_{tone} = f_{R} - f_{base}$ [Hz], $f_{base} = 8412$ MHz')
axs[0].legend()
axs[0].grid(True)

print('Difference wrt to Close Loop Tone:',simulated_closed_loop_tone - 7166445042)
print('Difference wrt to Open Loop Tone:',simulated_open_loop_tone - 7166445042)
axs[1].scatter(simulated_times_ifms_tdb, simulated_closed_loop_tone, marker='o', label='Simulated Closed-Loop Tone', s=15, alpha=0.5)
axs[1].scatter(simulated_times_fdets_tdb, simulated_open_loop_tone, marker='o', label='Simulated Open-Loop Tone', s=15, alpha=0.5)
axs[1].scatter(simulated_equivalent_closed_loop_times_tdb, simulated_equivalent_closed_loop_tone, marker='o', label='Simulated Equivalent Closed-Loop Tone', s=15, alpha=0.5)
axs[1].set_xlabel('Time [s] (Midpoint of Interval)')
axs[1].set_ylabel('$f_{tone} = f_{R} - f_{base}$ [Hz], $f_{base} = 8412$ MHz')
axs[1].legend()
axs[1].grid(True)

filtered_difference = np.array(difference_between_closed_loops_tdb)[np.abs(difference_between_closed_loops_tdb) <= 0.1]
filtered_simulated_equivalent_closed_loop_times_tdb = np.array(simulated_equivalent_closed_loop_times_tdb)[np.abs(difference_between_closed_loops_tdb) <= 0.1]

mean_filtered_difference = np.mean(filtered_difference)
rms_filtered_difference = np.std(filtered_difference)

axs[2].scatter(filtered_simulated_equivalent_closed_loop_times_tdb, filtered_difference, marker='o', label = f'Simulated-Data Eq. Closed-Loop\nmean:{mean_filtered_difference:.2g}\nrms = {rms_filtered_difference:.2g}',s=10, alpha=0.6)
axs[2].set_xlabel('Time [s] (Midpoint of Interval)')
axs[2].set_ylabel('Residuals [Hz]')
axs[2].legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=7)
axs[2].grid(True)
plt.show()
###################################################################################

