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

straight_trajectory_flag = True
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
#start = datetime(2013, 12, 28, 00, 00, 00) # simulate over multiple days to see periodic trend
#end = datetime(2013, 12, 31, 23, 00, 00)
start = datetime(2013, 12, 28, 00, 00, 00) # simulate over the relevant GR035 timeframe
end = datetime(2013, 12, 28, 23, 59, 00)
integration_time = 10
open_loop_cadence = 10

start_time = time_conversion.datetime_to_tudat(start).epoch().to_float() # in utc
end_time = time_conversion.datetime_to_tudat(end).epoch().to_float()  # in utc

# Create default body settings for celestial bodies
bodies_to_create = ["Earth", "Sun", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Moon"]
global_frame_origin = "SSB"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings_time_limited(
    bodies_to_create, start_time, end_time, global_frame_origin, global_frame_orientation)

# Modify Earth default settings
body_settings.get('Earth').shape_settings = environment_setup.shape.oblate_spherical_spice()
#body_settings.get('Earth').rotation_model_settings = environment_setup.rotation_model.gcrs_to_itrs(
#    environment_setup.rotation_model.iau_2006, global_frame_orientation,
#    interpolators.interpolator_generation_settings_float(interpolators.cubic_spline_interpolation(),
#                                                         start_time, end_time, 3600.0),
#    interpolators.interpolator_generation_settings_float(interpolators.cubic_spline_interpolation(),
#                                                         start_time, end_time, 3600.0),
#    interpolators.interpolator_generation_settings_float(interpolators.cubic_spline_interpolation(),
#                                                         start_time, end_time, 10.0))

body_settings.get('Earth').gravity_field_settings.associated_reference_frame = "ITRS"
spacecraft_name = "MEX" # Set Spacecraft Name
spacecraft_central_body = "Mars" # Set Central Body (Mars)
body_settings.add_empty_settings(spacecraft_name) # Create empty settings for spacecraft

times_linspace = np.arange(start_time, end_time, step = open_loop_cadence) # in utc, float type
tudat_times_linspace_utc = [Time(time) for time in times_linspace] # in utc, tudat::Time type

if straight_trajectory_flag:

    #initial_state_mex = np.array([ 8.66335594e+05,-6.71856322e+06,1.13422482e+07,-1.11986497e+03,
    #                              -5.49981761e+02,2.67200771e+02]) + start_earth_pos
    #final_state_mex = np.array([-2.99178963e+06, 8.97049285e+05,-3.10138365e+06,3.31557232e+03,
    #                            1.67203612e+03,-8.94232091e+02]) + end_earth_pos

    initial_state_mex = np.array([1e8,0,0,0,0,0])
    #final_state_mex = np.array([2e8,0,0,0,0,0])
    final_state_mex = initial_state_mex
    #final_state_mex = initial_state_mex
    p0 =np.array(initial_state_mex[:3])
    p1 =np.array(final_state_mex[:3])
    N = len(times_linspace)
    v = (p1 - p0) / (start_time - end_time)
    positions = p0 + np.outer(times_linspace - start_time, v) # Compute position over time
    velocities = np.tile(v, (N, 1)) # Repeat velocity for each time
    straight_trajectory = np.hstack((positions, velocities)) # stack them
    state_function = make_state_interpolator(times_linspace, straight_trajectory) # create state function for custom_ephemeris
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
#start_earth_pos = bodies.get('Earth').ephemeris.cartesian_state(start_time)
#end_earth_pos = bodies.get('Earth').ephemeris.cartesian_state(end_time)
#print(start_earth_pos, end_earth_pos)
#exit()
receiver_station_name = 'ONSALA60' # you can chenge this to CEDUNA or PARKES to test different stations
body_fixed_station_position = bodies.get('Earth').get_ground_station(receiver_station_name).station_state.get_cartesian_position(0)

## The following was a trial to see whether changing the geodetic altitude of the station would make the residuals worse (it does not).
#flattening = 1.0 / 298.257223563
#equatorial_radius = 6378137.0
#geodetic_body_fixed_station = element_conversion.convert_cartesian_to_geodetic_coordinates(body_fixed_station_position, equatorial_radius, flattening, 10)
#geodetic_body_fixed_station[0] += 10000
#fake_body_fixed_station_position = element_conversion.convert_position_elements(
#    geodetic_body_fixed_station, element_conversion.geodetic_position_type,element_conversion.cartesian_position_type,bodies.get('Earth').shape_model, 10)
################# First conversion: Convert UTC times to TDB. ########################################

time_scale_converter = time_conversion.default_time_scale_converter()
tudat_times_linspace_tdb = list() #prepare TDB, tudat::Time type
fake_tudat_times_linspace_tdb = list() #prepare TDB, tudat::Time type
for time_utc in tudat_times_linspace_utc: # for each UTC epoch, convert it to TDB
    tudat_times_linspace_tdb.append( time_scale_converter.convert_time(
        input_scale = time_conversion.utc_scale,
        output_scale = time_conversion.tdb_scale,
        input_value = time_utc,
        earth_fixed_position = body_fixed_station_position)) # if trying the geodetic altitude change, use fake_body_fixed_station_position

times_linspace_tdb = [tudat_times_tdb.to_float() for tudat_times_tdb in tudat_times_linspace_tdb] # tdb, float type
#################################### ANOTHER VALIDATION PLOT ######################################################
tdb_utc_diff = [tdb - utc for tdb, utc in zip(times_linspace_tdb, times_linspace)] # Compute TDB - UTC difference
utc_spacing = [times_linspace[i+1] - times_linspace[i] for i in range(len(times_linspace)-1)] # Compute UTC spacing
tdb_spacing = [times_linspace_tdb[i+1] - times_linspace_tdb[i] for i in range(len(times_linspace_tdb)-1)] # Compute TDB spacing
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
# First subplot: TDB - UTC
axs[0].scatter(times_linspace, tdb_utc_diff, label='TDB - UTC', color='blue', s=10)
axs[0].set_title('Difference Between TDB and UTC')
axs[0].set_ylabel('TDB - UTC [s]')
axs[0].grid(True)
# Second subplot: UTC spacing
axs[1].scatter(times_linspace[1:], utc_spacing, label='UTC Spacing', color='green')
axs[1].set_title('UTC Time Step Spacing')
axs[1].set_ylabel('Δt [s]')
axs[1].grid(True)
# Third subplot: TDB spacing
axs[2].plot(times_linspace_tdb[1:], tdb_spacing, label='TDB Spacing', color='red', linewidth=0.02)
axs[2].set_title('TDB Time Step Spacing')
axs[2].set_xlabel('Time [s]')
axs[2].set_ylabel('Δt [s]')
axs[2].grid(True)
plt.tight_layout()
plt.show()
########################################################################################################

######## OPTIONALLY APPLY TROPOSPHERIC CORRECTION FOR UPLINK AND DOWNLINK ########################################################################################################
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
reception_band = observation.FrequencyBands.x_band
transmission_band = observation.FrequencyBands.x_band
turnaround_ratio = observation.dsn_default_turnaround_ratios( observation.FrequencyBands.x_band,observation.FrequencyBands.x_band)
######################################################## TEMPORARY LINE TO FIX FREQUENCY CALCULATOR ERROR ###########################################################
bodies.get( "Earth" ).get_ground_station( "NWNORCIA" ).transmitting_frequency_calculator = environment.ConstantTransmittingFrequencyCalculator( 7166445042.992178 )
######################################################## TEMPORARY LINE TO FIX FREQUENCY CALCULATOR ERROR ###########################################################

######################### Set Involved Link Ends ######################################
link_ends = {
    observation.receiver: observation.body_reference_point_link_end_id('Earth', receiver_station_name),
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
########## ANOTHER VALIDATION PLOT #################
simulated_fdets = np.array(times_linspace) #utc
original_tdb = np.array(times_linspace_tdb) #tdb
diff_fdets = simulated_fdets - original_tdb #utc-tdb

fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# First subplot: FDETS
axs[0].plot(simulated_fdets, diff_fdets, label='FDETS_TDB - TDB', color='blue')
axs[0].set_title('FDETS Simulation TDB − Original TDB')
axs[0].set_ylabel('Difference [s]')
axs[0].grid(True)

plt.tight_layout()
plt.show()
##############################################################
# These are the calendar date corresponding to the converted UTC times (in other words, these are the UTC timetags in the fdets files)
simulated_times_fdets_utc = [time_conversion.julian_day_to_calendar_date(time_conversion.seconds_since_epoch_to_julian_day(time)) for time in times_linspace]
################ Interpolated, simulated open loop (FDETS) continous function ##############################
# Interpolate the simulated open loop and closed loop, using the UTC times (these are basically the float versions of the fdets/ifms time tags)
simulated_open_loop_continuous_function_utc = interp1d(times_linspace, simulated_observations_fdets, kind='cubic', fill_value='extrapolate')
################ Compute quadrature to convert simulated open_loop into equivalent simulated closed loop ##############
simulated_equivalent_closed_loop_observables, simulated_equivalent_closed_loop_times = compute_scipy_quadrature(
    simulated_open_loop_continuous_function_utc,
    times_linspace, # quadrature: times enter in UTC
    integration_time = integration_time)

tudat_simulated_equivalent_closed_loop_times = [Time(time_utc) for time_utc in simulated_equivalent_closed_loop_times]
closed_loop_tudat_times_linspace_tdb = list()
for time_utc in tudat_simulated_equivalent_closed_loop_times: # for each UTC epoch, convert it to TDB
    closed_loop_tudat_times_linspace_tdb.append( time_scale_converter.convert_time(
        input_scale = time_conversion.utc_scale,
        output_scale = time_conversion.tdb_scale,
        input_value = time_utc,
        earth_fixed_position = body_fixed_station_position))

closed_loop_times_linspace_tdb = [time_tudat_tdb.to_float() for time_tudat_tdb in closed_loop_tudat_times_linspace_tdb]
#######################################################################################################################
################# Format times to CALENDAR UTC ################
simulated_equivalent_closed_loop_times_utc = [time_conversion.julian_day_to_calendar_date(time_conversion.seconds_since_epoch_to_julian_day(time)) for time in simulated_equivalent_closed_loop_times]
#######################################################
# Calculate spacing (differences between consecutive elements)
spacing_open_loop_utc = np.diff(times_linspace)
spacing_closed_loop_utc = np.diff(simulated_equivalent_closed_loop_times)
spacing_open_loop_tdb = np.diff(times_linspace_tdb)
spacing_closed_loop_tdb = np.diff(closed_loop_times_linspace_tdb)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Create a figure with two subplots

# First subplot: Line plots
axes[0].plot(spacing_open_loop_utc, marker='o', linestyle='-', alpha=0.7, label='Open loop UTC spacing')
axes[0].plot(spacing_closed_loop_utc, marker='x', linestyle='--', alpha=0.7, label='Closed loop UTC spacing')
axes[0].set_xlabel('Index')
axes[0].set_ylabel('Spacing between consecutive UTC elements')
axes[0].set_title('Spacing Between Consecutive Elements in Each UTC Array')
axes[0].legend()
axes[0].grid(True)

# Second subplot: Scatter plots
axes[1].plot(times_linspace, times_linspace, 'o', alpha=0.5, label='Open Loop UTC')
axes[1].plot(simulated_equivalent_closed_loop_times,simulated_equivalent_closed_loop_times, '+', label='Closed Loop UTC')
axes[1].set_xlabel('UTC Time')
axes[1].set_title('UTC Time Comparisons')
axes[1].legend()

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Create a figure with two subplots

# First subplot: Line plots
axes[0].plot(spacing_open_loop_tdb, marker='o', linestyle='-', alpha=0.7, label='Open loop UTC spacing')
axes[0].plot(spacing_closed_loop_tdb, marker='x', linestyle='--', alpha=0.7, label='Closed loop UTC spacing')
axes[0].set_xlabel('Index')
axes[0].set_ylabel('Spacing between consecutive UTC elements')
axes[0].set_title('Spacing Between Consecutive Elements in Each UTC Array')
axes[0].legend()
axes[0].grid(True)

# Second subplot: Scatter plots
axes[1].plot(times_linspace, times_linspace, 'o', alpha=0.5, label='Open Loop TDB')
axes[1].plot(closed_loop_times_linspace_tdb,closed_loop_times_linspace_tdb, '+', label='Closed Loop TDB')
axes[1].set_xlabel('TDB Time')
axes[1].set_title('TDB Time Comparisons')
axes[1].legend()

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()
plt.close()
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
    simulation_times = closed_loop_tudat_times_linspace_tdb,
    ancilliary_settings = closed_loop_ancillary_settings
)]

closed_loop_observation_simulators = estimation_setup.create_observation_simulators(closed_loop_observation_model_settings, bodies)
closed_loop_collection = estimation.simulate_observations(closed_loop_observation_simulation_settings, closed_loop_observation_simulators, bodies)
estimation.compute_residuals_and_dependent_variables(closed_loop_collection, closed_loop_observation_simulators, bodies) # ifms simulator
simulated_observations_ifms = closed_loop_collection.get_observations()[0] # simulated closed-loop (IFMS)
################ Retrieve the tones ###################
simulated_open_loop_tone = simulated_observations_fdets - base_frequency # (this is in tdb)
simulated_equivalent_closed_loop_tone = np.array(simulated_equivalent_closed_loop_observables) - base_frequency # (this is evaluated at utc)
simulated_closed_loop_tone =  simulated_observations_ifms - base_frequency # (this is in tdb equivalent to the utc times of the equivalent closed-loop doppler)
#######################################################

################# Retrieve differences between the two simulated closed loop (equivalent-ifms) ################
difference_between_closed_loops_utc = [j-i for i, j in zip(simulated_equivalent_closed_loop_tone,simulated_closed_loop_tone)]
simulated_times_ifms_utc = [time_conversion.julian_day_to_calendar_date(time_conversion.seconds_since_epoch_to_julian_day(time)) for time in simulated_equivalent_closed_loop_times]
###############################################################################################################

######################## Retrieve Statistics (mean and rms) #######################
rms_closed_loop_difference_simulated = np.std(difference_between_closed_loops_utc)
mean_closed_loop_difference_simulated = np.mean(difference_between_closed_loops_utc)
mean_pride_residuals = np.mean(open_loop_collection.get_concatenated_residuals())
rms_pride_residuals = np.std(open_loop_collection.get_concatenated_residuals())
mean_ifms_residuals = np.mean(closed_loop_collection.get_concatenated_residuals())
rms_ifms_residuals = np.std(closed_loop_collection.get_concatenated_residuals())
###################################################################################

####### ANOTHER VALIDATION PLOT ###########
closed_loop_interpol = interp1d(simulated_equivalent_closed_loop_times,simulated_closed_loop_tone)
cond = (times_linspace >= np.min(simulated_equivalent_closed_loop_times)) & \
       (times_linspace <= np.max(simulated_equivalent_closed_loop_times))

valid_times = times_linspace[cond]

shared_times = np.intersect1d(valid_times, simulated_equivalent_closed_loop_times)
closed_loop_values = simulated_closed_loop_tone
open_loop_values = simulated_open_loop_continuous_function_utc(shared_times) - base_frequency
residual = closed_loop_values - open_loop_values

first_derivative = np.gradient(open_loop_values, shared_times)
second_derivative = np.gradient(first_derivative, shared_times)
fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)


axs[0].scatter(shared_times, open_loop_values, label='Open Loop (FDETS)', color='orange')
axs[0].scatter(shared_times, closed_loop_values, label='Closed Loop (IFMS)', color='blue', alpha = 0.4, marker = '+')
axs[0].set_title('Interpolated (Simulated) Doppler Observations')
axs[0].set_ylabel('Doppler Tone [Hz]')
axs[0].grid(True)
axs[0].legend()

# Second subplot: Residual
axs[1].plot(shared_times, second_derivative*integration_time**2/24, label="Quadratic Term", color='r')
axs[1].scatter(shared_times[np.abs(residual)<1000], residual[np.abs(residual)<1000], label='Closed - Open Loop', color='green', s = 8)
axs[1].set_title('Difference Between Closed and Open Loop Doppler')
axs[1].set_xlabel('UTC Time [s]')
axs[1].set_ylabel(r'$h_c(t) - h_o(t) \approx \frac{T^2}{24} \cdot h_o^{\prime\prime}(t)$ [Hz]')
axs[1].grid(True)
axs[1].legend(loc='upper left')

# Add zoom inset to axs[1]
axins = inset_axes(axs[1], width="30%", height="20%", loc='upper right')  # Adjust loc if needed
axins.scatter(shared_times, residual, color = 'green', s = 8)
axins.plot(shared_times, second_derivative*integration_time**2/24, 'red')

# Set zoomed region
axins.set_xlim(valid_times[1550], valid_times[1730])
axins.set_ylim(-35, 10)

# Draw lines showing zoomed area
mark_inset(axs[1], axins, loc1=2, loc2=4, fc="none", ec="0.5")


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
axs[0].scatter(simulated_times_fdets_utc, simulated_open_loop_tone, marker='o', label='Simulated Open-Loop Tone', s=15, alpha=0.5)
axs[0].scatter(simulated_equivalent_closed_loop_times_utc, simulated_equivalent_closed_loop_tone, marker='o', label='Simulated Equivalent Closed-Loop Tone', s=15, alpha=0.5)
axs[0].set_xlabel('Time [s] (Midpoint of Interval)')
axs[0].set_ylabel('$f_{tone} = f_{R} - f_{base}$ [Hz], $f_{base} = 8412$ MHz')
axs[0].legend()
axs[0].grid(True)

axs[1].scatter(simulated_times_ifms_utc, simulated_closed_loop_tone, marker='o', label='Simulated Closed-Loop Tone', s=15, alpha=0.5)
axs[1].scatter(simulated_equivalent_closed_loop_times_utc, simulated_equivalent_closed_loop_tone, marker='o', label='Simulated Equivalent Closed-Loop Tone', s=15, alpha=0.5)
axs[1].set_xlabel('Time [s] (Midpoint of Interval)')
axs[1].set_ylabel('$f_{tone} = f_{R} - f_{base}$ [Hz], $f_{base} = 8412$ MHz')
axs[1].legend()
axs[1].grid(True)

filtered_difference = np.array(difference_between_closed_loops_utc)[np.abs(difference_between_closed_loops_utc) <= 0.1]
filtered_simulated_equivalent_closed_loop_times_utc = np.array(simulated_equivalent_closed_loop_times_utc)[np.abs(difference_between_closed_loops_utc) <= 0.1]

mean_filtered_difference = np.mean(filtered_difference)
rms_filtered_difference = np.std(filtered_difference)

axs[2].scatter(filtered_simulated_equivalent_closed_loop_times_utc, filtered_difference, marker='o', label = f'Simulated-Data Eq. Closed-Loop\nmean:{mean_filtered_difference:.2g}\nrms = {rms_filtered_difference:.2g}',s=10, alpha=0.6)
axs[2].set_xlabel('Time [s] (Midpoint of Interval)')
axs[2].set_ylabel('Residuals [Hz]')
axs[2].legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=7)
axs[2].grid(True)
plt.show()
###################################################################################

