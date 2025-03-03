######################### # IMPORTANT #############################################################################

# This code reproduces TUDAT's weird behavior when computing
# residuals for multiple ifms at once (uses observations_from_multi_station_ifms_files function).

# Ramp settings from T. Bocanegra mex ramp file are also implemented.
# (But seem to have no effect, so please check they are indeed set as they should)
##################################################################################################################
import os
import csv
import numpy as np
from matplotlib import pyplot as plt
from astropy.time import Time
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.astro import time_conversion, element_conversion
from tudatpy.math import interpolators
from tudatpy.numerical_simulation import environment_setup, environment
from tudatpy.numerical_simulation import estimation, estimation_setup
from tudatpy.numerical_simulation.estimation_setup import observation
import random
import matplotlib.dates as mdates
from datetime import datetime

def parse_tracking_data(file_path='/Users/lgisolfi/Desktop/data_archiving-1.0/ramp.mex.gr035'):

    """ Ad-hoc function to parse ramp.mex.gr035 ramp file by T. Bocanegra """
    parsed_data = {}

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()

            start_time_str = parts[0] + ' ' + parts[1]
            end_time_str = parts[2] + ' ' + parts[3]
            start_frequency = float(parts[4])
            ramp_rate = float(parts[5])
            transmitter = parts[6]

            # Convert timestamps to datetime objects
            start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S.%f")
            end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S.%f")

            # Convert to UTC seconds
            start_time  = time_conversion.datetime_to_tudat(start_time).epoch()
            end_time = time_conversion.datetime_to_tudat(end_time).epoch()

            # Initialize dictionary entry if the station is not yet present
            if transmitter not in parsed_data:
                parsed_data[transmitter] = {
                    "start_times": [],
                    "end_times": [],
                    "start_frequencies": [],
                    "ramp_rates": []
                }

            # Append values to the corresponding station key
            parsed_data[transmitter]["start_times"].append(start_time)
            parsed_data[transmitter]["end_times"].append(end_time)
            parsed_data[transmitter]["start_frequencies"].append(start_frequency)
            parsed_data[transmitter]["ramp_rates"].append(ramp_rate)

    return parsed_data

# Set Folders Containing Relevant Files
mex_kernels_folder = '/Users/lgisolfi/Desktop/mex_phobos_flyby/kernels/'
mex_fdets_folder = '/Users/lgisolfi/Desktop/mex_phobos_flyby/fdets/complete'
mex_ifms_folder = '/Users/lgisolfi/Desktop/mex_phobos_flyby/ifms/filtered'
mex_odf_folder = '/Users/lgisolfi/Desktop/mex_phobos_flyby/odf/'

# Load Required Spice Kernels
spice.load_standard_kernels()
for kernel in os.listdir(mex_kernels_folder):
    kernel_path = os.path.join(mex_kernels_folder, kernel)
    spice.load_kernel(kernel_path)

# Define Start and end Dates of Simulation.
start = datetime(2013, 12, 28)
end = datetime(2013, 12, 30)

# Add a time buffer of one day to avoid interpolation issues
start_time = time_conversion.datetime_to_tudat(start).epoch().to_float() - 86400.0
end_time = time_conversion.datetime_to_tudat(end).epoch().to_float() + 86400.0

# Create default body settings for celestial bodies and set origin and orientation of global frame
bodies_to_create = ["Earth", "Sun", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Moon"]
global_frame_origin = "SSB"
global_frame_orientation = "J2000"

# Create body settings
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


# Create MEX spacecraft and its settings
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

# Set the transponder turnaround ratio function
vehicleSys = environment.VehicleSystems()
vehicleSys.set_default_transponder_turnaround_ratio_function()
bodies.get_body("MEX").system_models = vehicleSys
###############################################################

# Set reception and transmission bands
reception_band = observation.FrequencyBands.x_band
transmission_band = observation.FrequencyBands.x_band

# Retrieve lists of ramp values and times, as well as start frequencies, for each station
ramp_dictionary = parse_tracking_data()

# Set Ramped Frequencies with the new PiecewiseLinearFrequencyInterpolator function
for key in ramp_dictionary.keys():
    station_start_ramp_times = ramp_dictionary[key]['start_times']
    station_end_ramp_times = ramp_dictionary[key]['end_times']
    station_ramp_rates = ramp_dictionary[key]['ramp_rates']
    station_start_frequencies= ramp_dictionary[key]['start_frequencies']

    station_ramp = numerical_simulation.environment.PiecewiseLinearFrequencyInterpolator(
        station_start_ramp_times ,
        station_end_ramp_times,
        station_ramp_rates,
        station_start_frequencies
    )

    # Set the ramp
    bodies.get('Earth').get_ground_station(key).transmitting_frequency_calculator = station_ramp

# This is the manually ordered list of ifms files
ordered_ifms_list = ['M32ICL2L02_D2X_133621819_00.TAB',
                     'M32ICL2L02_D2X_133621904_00.TAB',
                     'M32ICL1L02_D2X_133630120_00.TAB',
                     'M32ICL1L02_D2X_133630203_00.TAB',
                     'M63ODFXL02_DPX_133630348_00.TAB',
                     'M14ODFXL02_DPX_133631130_00.TAB',
                     'M32ICL1L02_D2X_133631902_00.TAB',
                     'M32ICL1L02_D2X_133632221_00.TAB',
                     'M32ICL1L02_D2X_133632301_00.TAB']

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)  # Two subplots, sharing x-axis
file_list_labels = ["List from os.listdir", "Ordered IFMS List"] # Subplots labels for comparison between  ordered and non-ordered list

# Perform the analysis for both the unordered and ordered list.
for idx, ifms_file_list in enumerate([os.listdir(mex_ifms_folder), ordered_ifms_list]):
    ifms_files = []
    transmitting_stations_list = []

    for ifms_file in ifms_file_list:
        if ifms_file.startswith('.'):
            continue
        station_code = ifms_file[1:3]

        if station_code == '14':
            transmitting_station_name = 'DSS14'
        elif station_code == '63':
            transmitting_station_name = 'DSS63'
        elif station_code == '32':
            transmitting_station_name = 'NWNORCIA'
        else:
            continue  # Skip unknown station codes

        ifms_files.append(os.path.join(mex_ifms_folder, ifms_file))
        transmitting_stations_list.append(transmitting_station_name)

    # Use the observations_from_multi_station_ifms_files function
    ifms_collection = observation.observations_from_multi_station_ifms_files(
        ifms_files, bodies, spacecraft_name, transmitting_stations_list,
        reception_band, transmission_band, apply_troposphere_correction=True
    )

    compressed_observations = ifms_collection # Do not compress for now, since we need to compare them with PRIDE open-loop data
    antenna_position_history = dict()
    com_position = [-1.3,0.0,0.0] # estimated based on the MEX_V16.TF file description
    times = compressed_observations.get_concatenated_observation_times()
    times = [time.to_float() for time in times]
    mjd_times = [time_conversion.seconds_since_epoch_to_julian_day(t) for t in times]
    utc_times = np.array([Time(mjd_time, format='jd', scale='utc').datetime for mjd_time in mjd_times])

    antenna_state = np.zeros((6, 1))
    antenna_state[:3,0] = spice.get_body_cartesian_position_at_epoch("-41020", "-41000", "MEX_SPACECRAFT", "none", times[0])
    antenna_state[:3,0] = antenna_state[:3,0] - com_position
    antenna_ephemeris_settings = environment_setup.ephemeris.constant(antenna_state, "-41000",  "MEX_SPACECRAFT")
    antenna_ephemeris = environment_setup.ephemeris.create_ephemeris(antenna_ephemeris_settings, "Antenna")
    compressed_observations.set_reference_point(bodies, antenna_ephemeris, "Antenna", "MEX", observation.retransmitter)

    light_time_correction_list = list()
    light_time_correction_list.append(
        estimation_setup.observation.first_order_relativistic_light_time_correction(["Sun"]))

    doppler_link_ends = compressed_observations.link_definitions_per_observable[
        estimation_setup.observation.dsn_n_way_averaged_doppler]

    observation_model_settings = list()
    for current_link_definition in doppler_link_ends:
        observation_model_settings.append(estimation_setup.observation.dsn_n_way_doppler_averaged(
            current_link_definition, light_time_correction_list, subtract_doppler_signature = False ))

    observation_simulators = estimation_setup.create_observation_simulators(observation_model_settings, bodies)

    elevation_angle_settings = observation.elevation_angle_dependent_variable( observation.receiver )
    elevation_angle_parser = compressed_observations.add_dependent_variable( elevation_angle_settings, bodies )
    sep_angle_settings = observation.avoidance_angle_dependent_variable("Sun", observation.retransmitter, observation.receiver)
    sep_angle_parser = compressed_observations.add_dependent_variable( sep_angle_settings, bodies )

    estimation.compute_residuals_and_dependent_variables(compressed_observations, observation_simulators, bodies)

    concatenated_obs = compressed_observations.get_concatenated_observations()
    concatenated_computed_obs = compressed_observations.get_concatenated_computed_observations()
    concatenated_residuals = compressed_observations.get_concatenated_residuals()
    rms_residuals = compressed_observations.get_rms_residuals()
    mean_residuals = compressed_observations.get_mean_residuals()

    print(f'Mean Residuals: {mean_residuals}')
    print(f'RMS Residuals: {rms_residuals}')

    # Plot in respective subplot
    axes[idx].scatter(utc_times, concatenated_residuals, s=5)
    axes[idx].set_title(f'IFMS Pre-fit Residuals ({file_list_labels[idx]})')
    axes[idx].set_ylabel('Frequency (Hz)')
    axes[idx].grid(True)
    axes[idx].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    axes[idx].xaxis.set_major_locator(mdates.AutoDateLocator())

# Common x-label and formatting
axes[-1].set_xlabel('UTC Time')
plt.gcf().autofmt_xdate()
plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()
exit()