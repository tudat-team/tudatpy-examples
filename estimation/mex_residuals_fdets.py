######################### # IMPORTANT #############################################################################

# In order to test this example, I am using a Phobos Flyby fdets file missing the few last/first lines...
# The removed lines were classified as outliers, but they should be filtered with the proper tudat functionality,
# rather than manually (as done for now)

##################################################################################################################
import os
from xmlrpc.client import DateTime
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import random

# Load required tudatpy modules
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.astro import time_conversion, element_conversion
from tudatpy.math import interpolators
from tudatpy.numerical_simulation import environment_setup, environment
from tudatpy.numerical_simulation import estimation, estimation_setup
from tudatpy.numerical_simulation.estimation_setup import observation
from datetime import datetime, timezone
from astropy.time import Time
from collections import defaultdict
import matplotlib.dates as mdates



def ID_to_site(site_ID):
    """
    Maps a site ID to its corresponding ground station name.

    Args:
        site_ID (str): The site ID.

    Returns:
        str: The ground station name if the site ID is found, or None otherwise.
    """
    id_to_site_mapping = {
        'Cd': 'CEDUNA',
        'Hb': 'HOBART12',
        'Yg': 'YARRA12M',
        'Ke': 'KATH12M',
        'Ww': 'WARK',
        'Ym': 'YAMAGU32',
        'T6': 'TIANMA65',
        'Km': 'KUNMING',
        'Ku': 'KVNUS',
        'Bd': 'BADARY',
        'Ur': 'URUMQI',
        'Zc': 'ZELENCHK',
        'Hh': 'HART',
        'Wz': 'WETTZELL',
        'Sv': 'SVETLOE',
        'Mc': 'MEDICINA',
        'Wb': 'WSTRBORK',
        'On': 'ONSALA60',
        'Ys': 'YEBES40M',
        'Sc': 'SC-VLBA',
        'Hn': 'HN-VLBA',
        'Nl': 'NL-VLBA',
        'Fd': 'FD-VLBA',
        'La': 'LA-VLBA',
        'Kp': 'KP-VLBA',
        'Pt': 'PR_VLBA',
        'Br': 'BR-VLBA',
        'Ov': 'OV-VLBA',
        'Mk': 'MK-VLBA'
    }

    # Return the corresponding site name or None if the site_ID is not found
    return id_to_site_mapping.get(site_ID, None)

def process_residuals(fdets_files, site_names, ifms_files):
    """
    Plots residuals for multiple FDETS files and sites on the same plot.

    Args:
        fdets_files (list): List of FDETS file paths.
        site_names (list): List of site names corresponding to the FDETS files.
    """
    plt.figure(figsize=(10, 10))

    for fdets_file, site_name in zip(fdets_files,site_names):

        # Set other variables for processing (adapted from your function)
        base_frequency = 8412e6
        column_types = ["utc_datetime_string", "signal_to_noise_ratio", "normalised_spectral_max",
                        "doppler_measured_frequency_hz", "doppler_noise_hz"]
        target_name = 'MEX'

        receiving_station_name = site_name
        print(f'Fdets File Name: {fdets_file}')
        print(f'Receiving station: {receiving_station_name}')

        reception_band = observation.FrequencyBands.x_band
        transmission_band = observation.FrequencyBands.x_band
        added_labels = set()
        residuals_lines = []
        min_residuals = defaultdict(list)
        for ifms_file in ifms_files:
            station_code = ifms_file.split('/')[3][1:3]
            if station_code == '14':
                transmitting_station_name = 'DSS14'

            elif station_code == '63':
                transmitting_station_name = 'DSS63'

            elif station_code == '32':
                transmitting_station_name = 'NWNORCIA'

            print(f'ifms file: {ifms_file}\ntransmitting station: {transmitting_station_name}')

            # Load FDETS file
            try:
                fdets_collection = observation.observations_from_fdets_files(
                    fdets_file, base_frequency, column_types, target_name,
                    transmitting_station_name, receiving_station_name, reception_band, transmission_band
                )
            except:
                continue

            ########## TIME BOUNDS IFMS #########

            ifms_collection = observation.observations_from_ifms_files([ifms_file], bodies, spacecraft_name, transmitting_station_name, reception_band, transmission_band)

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

            link_ends = {
                observation.receiver: observation.body_reference_point_link_end_id('Earth', site_name),
                observation.retransmitter: observation.body_reference_point_link_end_id('MEX','Antenna'),
                observation.transmitter: observation.body_reference_point_link_end_id('Earth', transmitting_station_name),
            }

            # Create a single link definition from the link ends
            link_definition = observation.LinkDefinition(link_ends)

            light_time_correction_list = list()
            light_time_correction_list.append(
                estimation_setup.observation.first_order_relativistic_light_time_correction(["Sun"]))

            # Define the observation model settings
            observation_model_settings = [
                estimation_setup.observation.doppler_measured_frequency(
                    link_definition, light_time_correction_list
                )
            ]
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

            ##################################### HANDLING FDETS #############################################
            # Perform computations as in the original function
            concatenated_obs = fdets_collection.get_concatenated_observations()
            concatenated_computed_obs = fdets_collection.get_concatenated_computed_observations()
            residuals_by_hand_no_atm_corr = concatenated_computed_obs - concatenated_obs

            filtered_residuals = residuals_by_hand_no_atm_corr[abs(residuals_by_hand_no_atm_corr) < 10]
            print(f'filtered residuals for {site_name}: {filtered_residuals}')

            # Get observation times
            times = fdets_collection.get_observation_times()
            times = [time.to_float() for time in times[0]]
            times = np.array(times)

            mjd_times = [time_conversion.seconds_since_epoch_to_julian_day(t) for t in times]
            utc_times = np.array([Time(mjd_time, format='jd', scale='utc').datetime for mjd_time in mjd_times])

            # Convert to UTC
            filtered_utc_times = utc_times[abs(residuals_by_hand_no_atm_corr) < 10]


            if site_name not in added_labels:
                print(site_name, 'not in added labels')

                print(filtered_utc_times, filtered_residuals)
                color = generate_random_color()
                plt.scatter(utc_times, residuals_by_hand_no_atm_corr, color = color, s=10, marker='+', label=f'{site_name}')
                plot_ifms_windows(ifms_file, ifms_collection, color)

            else:
                color = generate_random_color()
                plt.scatter(utc_times, residuals_by_hand_no_atm_corr,color = color, s=10, marker='+')
                plot_ifms_windows(ifms_file, ifms_collection, color)

            added_labels.add(site_name)
            added_labels.add(site_name)

    #######################################################################################################

    # Format the x-axis for dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()  # Auto-rotate date labels for better readability

    plt.title('Residuals from Multiple Sites')
    plt.xlabel('Time [s]')
    plt.ylabel('Residuals [Hz]')
    plt.grid(True)
    # Place the legend outside the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)

    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.show()

######################################################################################################

def plot_ifms_windows(ifms_file,ifms_collection, color):

    # Get IFMS observation times
    ifms_times = ifms_collection.get_observation_times()

    ifms_file_name = ifms_file.split('/')[3]

    # Loop through each element in ifms_times and convert to float
    min_sublist = np.min([time.to_float() for time in ifms_times])
    max_sublist = np.max([time.to_float() for time in ifms_times])
    mjd_min_sublist = time_conversion.seconds_since_epoch_to_julian_day(min_sublist)
    mjd_max_sublist = time_conversion.seconds_since_epoch_to_julian_day(max_sublist)
    utc_min_sublist = Time(mjd_min_sublist, format='jd', scale = 'utc').datetime
    utc_max_sublist = Time(mjd_max_sublist, format='jd', scale = 'utc').datetime
    #(utc_min_sublist, utc_max_sublist)
    plt.axvspan(utc_min_sublist, utc_max_sublist, color=color, alpha=0.2, label = ifms_file_name)

    return(min_sublist, max_sublist)
######################################################################################################
def plot_dsn_windows_from_ramp_file():

    ramp_data = """ 
2013-12-28 17:56:27.905000  2013-12-28 18:40:22.259000  7.1664308519995022e+09   0.0000000000000000e+00  NWNORCIA
2013-12-28 18:40:58.264000  2013-12-29 00:33:05.000000  7.1664308519995022e+09   0.0000000000000000e+00  NWNORCIA
2013-12-29 00:33:05.000000  2013-12-29 00:33:06.000000  7.1664857440000000e+09   0.0000000000000000e+00  DSS63
2013-12-29 00:33:06.000000  2013-12-29 00:36:03.000000  7.1664857440000000e+09   0.0000000000000000e+00  DSS63
2013-12-29 00:36:03.000000  2013-12-29 00:44:17.000000  7.1664857440000000e+09   0.0000000000000000e+00  DSS63
2013-12-29 00:44:17.000000  2013-12-29 00:45:24.000000  7.1664857440000000e+09   0.0000000000000000e+00  DSS63
2013-12-29 00:45:24.000000  2013-12-29 00:45:32.000000  7.1664857440000000e+09   0.0000000000000000e+00  DSS63
2013-12-29 00:45:32.000000  2013-12-29 01:40:29.000000  7.1664450429921780e+09   0.0000000000000000e+00  NWNORCIA
2013-12-29 01:40:29.000000  2013-12-29 03:21:27.000000  7.1664450429921780e+09   0.0000000000000000e+00  NWNORCIA
2013-12-29 03:21:27.000000  2013-12-29 03:21:37.000000  7.1665434266206827e+09  -1.0000000000000000e+03  DSS63
2013-12-29 03:21:37.000000  2013-12-29 03:21:52.000000  7.1665334266206827e+09   0.0000000000000000e+00  DSS63
2013-12-29 03:21:52.000000  2013-12-29 03:21:55.000000  7.1665334266206827e+09   0.0000000000000000e+00  DSS63
2013-12-29 03:21:55.000000  2013-12-29 03:22:07.000000  7.1665334266206827e+09   0.0000000000000000e+00  DSS63
2013-12-29 03:22:07.000000  2013-12-29 03:22:47.000000  7.1665334266206827e+09   5.0000000000000000e+02  DSS63
2013-12-29 03:22:47.000000  2013-12-29 03:23:27.000000  7.1665534266206827e+09  -4.9999999999900001e+02  DSS63
2013-12-29 03:23:27.000000  2013-12-29 03:25:03.000000  7.1665334266206827e+09  -4.9669396544699998e+02  DSS63
2013-12-29 03:25:03.000000  2013-12-29 07:03:36.000000  7.1664857440000000e+09   0.0000000000000000e+00  DSS63
2013-12-29 07:03:36.000000  2013-12-29 07:03:45.000000  7.1664857440000000e+09   0.0000000000000000e+00  DSS63
2013-12-29 07:03:45.000000  2013-12-29 07:03:58.000000  7.1664857440000000e+09   0.0000000000000000e+00  DSS63
2013-12-29 07:03:58.000000  2013-12-29 07:04:00.000000  7.1664857440000000e+09   0.0000000000000000e+00  DSS63
2013-12-29 07:04:00.000000  2013-12-29 07:52:30.000000  7.1664419636242456e+09   0.0000000000000000e+00  DSS63
2013-12-29 07:52:30.000000  2013-12-29 07:52:40.000000  7.1664419636242456e+09  -1.0000000000000000e+03  DSS63
2013-12-29 07:52:40.000000  2013-12-29 07:52:55.000000  7.1664319636242456e+09   0.0000000000000000e+00  DSS63
2013-12-29 07:52:55.000000  2013-12-29 07:52:58.000000  7.1664319636242456e+09   0.0000000000000000e+00  DSS63
2013-12-29 07:52:58.000000  2013-12-29 07:53:10.000000  7.1664319636242456e+09   0.0000000000000000e+00  DSS63
2013-12-29 07:53:10.000000  2013-12-29 07:53:50.000000  7.1664319636242456e+09   5.0000000000000000e+02  DSS63
2013-12-29 07:53:50.000000  2013-12-29 07:54:30.000000  7.1664519636242456e+09  -4.9999999999900001e+02  DSS63
2013-12-29 07:54:30.000000  2013-12-29 07:56:18.000000  7.1664319636242456e+09   4.9796644216700003e+02  DSS63
2013-12-29 07:56:18.000000  2013-12-29 11:01:52.000000  7.1664857440000000e+09   0.0000000000000000e+00  DSS63
2013-12-29 11:01:52.000000  2013-12-29 11:25:55.000000  7.1664857440000000e+09   0.0000000000000000e+00  DSS63
2013-12-29 11:02:25.000000  2013-12-29 14:02:59.000000  7.1664909280000000e+09   0.0000000000000000e+00  DSS14
2013-12-29 14:02:59.000000  2013-12-29 14:03:07.000000  7.1664909280000000e+09   0.0000000000000000e+00  DSS14
2013-12-29 14:03:07.000000  2013-12-29 14:03:20.000000  7.1664909280000000e+09   0.0000000000000000e+00  DSS14
2013-12-29 14:03:20.000000  2013-12-29 14:03:22.000000  7.1664909280000000e+09   0.0000000000000000e+00  DSS14
2013-12-29 14:03:22.000000  2013-12-29 18:36:54.000000  7.1664909280000000e+09   0.0000000000000000e+00  DSS14
2013-12-29 18:36:54.000000  2013-12-29 19:02:04.000000  7.1664909280000000e+09   0.0000000000000000e+00  DSS14
"""

    # Initialize a list to hold the tuples of start and end times for DSN stations
    dsn_boundaries = []

    # Process each line in the data
    for line in ramp_data.strip().split('\n'):
        # Split each line into components
        parts = line.split()

        # Extract the station name from the last column
        station = parts[-1]

        # Check if the station is DSS63 or DSS14
        if station in ['DSS63', 'DSS14']:
            # Extract the start and end times (first two columns)
            start_time_str = parts[0] + ' ' + parts[1]
            end_time_str = parts[2] + ' ' + parts[3]

            # Convert the time strings to datetime objects
            start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S.%f")
            end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S.%f")

            # Append the tuple of (start_time, end_time) to the list
            dsn_boundaries.append((start_time, end_time))

    # Process each tuple in the list
    for start_dt, end_dt in dsn_boundaries:
        # Convert the start time string to a naive datetime object
        start_naive = start_dt
        # Remove microseconds if needed
        start_naive = start_naive.replace(microsecond=0)
        # Assign UTC timezone
        start_utc = start_naive.replace(tzinfo=timezone.utc)

        # Convert the end time string to a naive datetime object
        end_naive = end_dt
        # Remove microseconds if needed
        end_naive = end_naive.replace(microsecond=0)
        # Assign UTC timezone
        end_utc = end_naive.replace(tzinfo=timezone.utc)

        # Print the results for each time boundary
        #print(f"Start time: {start_utc}")
        #print(f"End time: {end_utc}")

        plt.axvspan(start_utc, end_utc, alpha=0.1, color='red')
        plt.show()

def generate_random_color():
    """Generate a random color in hexadecimal format."""
    return "#{:02x}{:02x}{:02x}".format(
        random.randint(0, 255),  # Red
        random.randint(0, 255),  # Green
        random.randint(0, 255)   # Blue
    )
###################################################################
def get_ifms_collection_from_file(
        ifms_file,
        reception_band = observation.FrequencyBands.x_band,
        transmission_band = observation.FrequencyBands.x_band):

    print(f'Creating IFMS collection from {ifms_file}...\n')

    station_code = ifms_file.split('/')[3][1:3]
    if station_code == '14':
        transmitting_station_name = 'DSS14'

    elif station_code == '63':
        transmitting_station_name = 'DSS63'

    elif station_code == '32':
        transmitting_station_name = 'NWNORCIA'

    print(f'Loading IFMS file: {ifms_file}\n with transmitting station: {transmitting_station_name}')

    ifms_collection = observation.observations_from_ifms_files([ifms_file], bodies, spacecraft_name, transmitting_station_name, reception_band, transmission_band)

    return(ifms_collection)

def get_fdets_receiving_station_name(fdets_file):
    site = ID_to_site(fdets_file.split('.')[4])

    return(site)

def get_overlap_windows(
        fdets_file,
        ifms_files,
        reception_band = observation.FrequencyBands.x_band,
        transmission_band = observation.FrequencyBands.x_band,
        base_frequency = 8412e6,
        column_types = ["utc_datetime_string", "signal_to_noise_ratio", "normalised_spectral_max","doppler_measured_frequency_hz", "doppler_noise_hz"],
        target_name = 'MEX'
):


    receiving_station_name = get_fdets_receiving_station_name(fdets_file)

    for ifms_file in ifms_files:
        station_code = ifms_file.split('/')[3][1:3]
        if station_code == '14':
            transmitting_station_name = 'DSS14'

        elif station_code == '63':
            transmitting_station_name = 'DSS63'

        elif station_code == '32':
            transmitting_station_name = 'NWNORCIA'


        # Loading IFMS file
        print(f'Loading IFMS file: {ifms_file}\n with transmitting station: {transmitting_station_name}')
        ifms_collection = observation.observations_from_ifms_files(
            [ifms_file], bodies, spacecraft_name, transmitting_station_name, reception_band, transmission_band
        )

        ifms_times = ifms_collection.get_observation_times()
        start_ifms_time = np.min(ifms_times).to_float()
        end_ifms_time = np.max(ifms_times).to_float()
        ifms_interval = (start_ifms_time, end_ifms_time)
        print(f"IFMS Interval: {ifms_interval}")

        try:
            # Loading FDETS file
            site_name = get_fdets_receiving_station_name(fdets_file)
            fdets_collection = observation.observations_from_fdets_files(
                fdets_file, base_frequency, column_types, target_name,
                transmitting_station_name, receiving_station_name, reception_band, transmission_band
            )

            fdets_times = fdets_collection.get_observation_times()
            start_fdets_time = np.min(fdets_times).to_float()
            end_fdets_time = np.max(fdets_times).to_float()
            fdets_interval = (start_fdets_time, end_fdets_time)
            print(f"FDETS Interval: {fdets_interval}")

            # Check if intervals overlap
            check_overlaps = start_ifms_time < end_fdets_time and start_fdets_time < end_ifms_time

            if check_overlaps:
                print('ye')
                # Initialize overlaps as a list if it’s not already
                if 'overlaps' not in locals():
                    overlap_dict = []

                # Check if the current IFMS file is already in overlaps
                if all(entry['ifms_file_name'] != ifms_file for entry in overlap_dict):
                    overlap_dict.append({
                        'ifms_file_name': ifms_file,
                        'ifms_time_interval': ifms_interval,
                        'transmitting_station_name': transmitting_station_name,
                        'reception_band': reception_band,
                        'transmission_band': transmission_band,
                        'site_name': site_name
                    })

            else:
                continue

            return fdets_collection, overlap_dict

        except:
                continue

def get_filtered_fdets_collection_mex(fdets_collection, overlap_dict):

    # Filter fdets based on overlap dictionary time interval values
    filter_interval = overlap_dict['ifms_time_interval']
    filter_start_time = filter_interval[0]
    filter_end_time = filter_interval[1]
    time_filter = estimation.observation_filter(
        estimation.time_bounds_filtering, filter_start_time, filter_end_time, use_opposite_condition = True)
    filtered_fdets_collection = fdets_collection.filter_observations(time_filter)
    filtered_fdets_collection = filtered_fdets_collection.remove_empty_observation_sets()

    transmitting_station_name = overlap_dict['transmitting_station_name']
    transmission_band = overlap_dict['transmission_band']
    reception_band = overlap_dict['reception_band']

    corresponding_ifms_collection = observation.observations_from_ifms_files([ifms_file], bodies, spacecraft_name, transmitting_station_name, reception_band, transmission_band)

    return filtered_fdets_collection, corresponding_ifms_collection, overlap_dict

def process_residuals_coupling_fdets_ifms(
        filtered_fdets_collection, corresponding_ifms_collection, overlap_dict):

    added_labels = set()

    transmitting_station_name = overlap_dict['transmitting_station_name']
    transmission_band = overlap_dict['transmission_band']
    reception_band = overlap_dict['reception_band']
    site_name = overlap_dict['site_name']
    antenna_position_history = dict()
    com_position = [-1.3,0.0,0.0] # estimated based on the MEX_V16.TF file description
    for obs_times in filtered_fdets_collection.get_observation_times():
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
    filtered_fdets_collection.set_reference_point(bodies, antenna_ephemeris, "Antenna", "MEX", observation.reflector1)

    link_ends = {
        observation.receiver: observation.body_reference_point_link_end_id('Earth', site_name),
        observation.retransmitter: observation.body_reference_point_link_end_id('MEX','Antenna'),
        observation.transmitter: observation.body_reference_point_link_end_id('Earth', transmitting_station_name),
    }

    # Create a single link definition from the link ends
    link_definition = observation.LinkDefinition(link_ends)

    light_time_correction_list = list()
    light_time_correction_list.append(
        estimation_setup.observation.first_order_relativistic_light_time_correction(["Sun"]))

    # Define the observation model settings
    observation_model_settings = [
        estimation_setup.observation.doppler_measured_frequency(
            link_definition, light_time_correction_list
        )
    ]
    ###################################################################################################
    # Create observation simulators.
    observation_simulators = estimation_setup.create_observation_simulators(observation_model_settings, bodies)

    # Add elevation and SEP angles dependent variables to the fdets observation collection
    elevation_angle_settings = observation.elevation_angle_dependent_variable( observation.receiver )
    elevation_angle_parser = filtered_fdets_collection.add_dependent_variable( elevation_angle_settings, bodies )
    sep_angle_settings = observation.avoidance_angle_dependent_variable("Sun", observation.retransmitter, observation.receiver)
    sep_angle_parser = filtered_fdets_collection.add_dependent_variable( sep_angle_settings, bodies )

    # Compute and set residuals in the fdets observation collection
    estimation.compute_residuals_and_dependent_variables(filtered_fdets_collection, observation_simulators, bodies)

    ##################################### HANDLING FDETS #############################################
    # Perform computations as in the original function
    concatenated_obs = filtered_fdets_collection.get_concatenated_observations()
    concatenated_computed_obs = filtered_fdets_collection.get_concatenated_computed_observations()
    residuals_by_hand_no_atm_corr = concatenated_computed_obs - concatenated_obs

    filtered_residuals = residuals_by_hand_no_atm_corr[abs(residuals_by_hand_no_atm_corr) < 10]
    print(f'filtered residuals for {site_name}: {filtered_residuals}')

    # Get observation times
    times = filtered_fdets_collection.get_observation_times()
    times = [time.to_float() for time in times[0]]
    times = np.array(times)

    mjd_times = [time_conversion.seconds_since_epoch_to_julian_day(t) for t in times]
    utc_times = np.array([Time(mjd_time, format='jd', scale='utc').datetime for mjd_time in mjd_times])

    # Convert to UTC
    filtered_utc_times = utc_times[abs(residuals_by_hand_no_atm_corr) < 10]


    if site_name not in added_labels:
        color = generate_random_color()
        plt.scatter(utc_times, residuals_by_hand_no_atm_corr, color = color, s=10, marker='+', label=f'{site_name}')
        #plot_ifms_windows(ifms_file, ifms_collection, color)

    else:
        color = generate_random_color()
        plt.scatter(utc_times, residuals_by_hand_no_atm_corr,color = color, s=10, marker='+')
        #plot_ifms_windows(ifms_file, ifms_collection, color)

    added_labels.add(site_name)
    added_labels.add(site_name)

    #######################################################################################################

    # Format the x-axis for dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()  # Auto-rotate date labels for better readability


if __name__ == "__main__":

    # Unpack various input arguments
    mex_kernels_folder = 'mex_phobos_flyby/kernels/'
    mex_fdets_folder = 'mex_phobos_flyby/fdets/complete'
    mex_ifms_folder = 'mex_phobos_flyby/ifms/filtered'
    mex_odf_folder = 'mex_phobos_flyby/odf/'

    spice.load_standard_kernels()
    for kernel in os.listdir(mex_kernels_folder):
        kernel_path = os.path.join(mex_kernels_folder, kernel)
        spice.load_kernel(kernel_path)

    start = datetime(2013, 12, 27)
    end = datetime(2013, 12, 30)

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
    body_settings.get("Earth").ground_station_settings = environment_setup.ground_station.radio_telescope_stations()
    # Create System of Bodies using the above-defined body_settings
    bodies = environment_setup.create_system_of_bodies(body_settings)

    ########## IMPORTANT STEP ###################################
    # Set the transponder turnaround ratio function
    vehicleSys = environment.VehicleSystems()
    vehicleSys.set_default_transponder_turnaround_ratio_function()
    bodies.get_body("MEX").system_models = vehicleSys
    ###############################################################

    sites_list = []
    fdets_files = []
    ifms_files = []
    odf_files = []


    for fdets_file in os.listdir(mex_fdets_folder):
        fdets_files.append(os.path.join(mex_fdets_folder, fdets_file))
        site = ID_to_site(fdets_file.split('.')[4])
        sites_list.append(site)
    for ifms_file in os.listdir(mex_ifms_folder):
        ifms_files.append(os.path.join(mex_ifms_folder, ifms_file))
    for odf_file in os.listdir(mex_odf_folder):
        odf_files.append(os.path.join(mex_odf_folder, odf_file))

    print(f'IFMS FILES:\n {ifms_files}\n\n')
    # print(f'DSN FILES:\n {odf_files}\n\n')
    print(f'FDETS FILES:\n {fdets_files}\n\n')
    print(f'STATIONS:\n {sites_list}\n\n')

    ifms_collection = get_ifms_collection_from_file(ifms_file = ifms_files[0])

    fdets_collection, full_overlap_dict = get_overlap_windows(fdets_files[0], ifms_files) # full overlap_dict = all ifms included in the dict

    for overlap_dict in full_overlap_dict: #overlap_dict = single ifms included in the dict
        filtered_fdets_collection, corresponding_ifms_collection, overlap_dict = get_filtered_fdets_collection_mex(fdets_collection, overlap_dict)
        process_residuals_coupling_fdets_ifms(filtered_fdets_collection,corresponding_ifms_collection, overlap_dict)



    plt.title('Residuals from Multiple Sites')
    plt.xlabel('Time [s]')
    plt.ylabel('Residuals [Hz]')
    plt.grid(True)
    # Place the legend outside the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)

    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.show()

    exit()

    process_residuals(fdets_files, sites_list, ifms_files)




