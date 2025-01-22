######################### # IMPORTANT #############################################################################

# In order to test this example, I am using a Phobos Flyby fdets file missing the few last/first lines...
# The removed lines were classified as outliers, but they should be filtered with the proper tudat functionality,
# rather than manually (as done for now)

# NOTE: "DSS63" DOES NOT WORK. IT MUST BE "DSS-63". Not the same for DSS14 (it must be DSS-14)...
# NOTE: remember to remove empty sets, or the loaded stations (with empty observations) will cause troubles in the simulation.
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
        'Hh': 'HART15M',
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
#####################################################################################################
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
2013-12-29 00:33:05.000000  2013-12-29 00:33:06.000000  7.1664857440000000e+09   0.0000000000000000e+00  DSS-63
2013-12-29 00:33:06.000000  2013-12-29 00:36:03.000000  7.1664857440000000e+09   0.0000000000000000e+00  DSS-63
2013-12-29 00:36:03.000000  2013-12-29 00:44:17.000000  7.1664857440000000e+09   0.0000000000000000e+00  DSS-63
2013-12-29 00:44:17.000000  2013-12-29 00:45:24.000000  7.1664857440000000e+09   0.0000000000000000e+00  DSS-63
2013-12-29 00:45:24.000000  2013-12-29 00:45:32.000000  7.1664857440000000e+09   0.0000000000000000e+00  DSS-63
2013-12-29 00:45:32.000000  2013-12-29 01:40:29.000000  7.1664450429921780e+09   0.0000000000000000e+00  NWNORCIA
2013-12-29 01:40:29.000000  2013-12-29 03:21:27.000000  7.1664450429921780e+09   0.0000000000000000e+00  NWNORCIA
2013-12-29 03:21:27.000000  2013-12-29 03:21:37.000000  7.1665434266206827e+09  -1.0000000000000000e+03  DSS-63
2013-12-29 03:21:37.000000  2013-12-29 03:21:52.000000  7.1665334266206827e+09   0.0000000000000000e+00  DSS-63
2013-12-29 03:21:52.000000  2013-12-29 03:21:55.000000  7.1665334266206827e+09   0.0000000000000000e+00  DSS-63
2013-12-29 03:21:55.000000  2013-12-29 03:22:07.000000  7.1665334266206827e+09   0.0000000000000000e+00  DSS-63
2013-12-29 03:22:07.000000  2013-12-29 03:22:47.000000  7.1665334266206827e+09   5.0000000000000000e+02  DSS-63
2013-12-29 03:22:47.000000  2013-12-29 03:23:27.000000  7.1665534266206827e+09  -4.9999999999900001e+02  DSS-63
2013-12-29 03:23:27.000000  2013-12-29 03:25:03.000000  7.1665334266206827e+09  -4.9669396544699998e+02  DSS-63
2013-12-29 03:25:03.000000  2013-12-29 07:03:36.000000  7.1664857440000000e+09   0.0000000000000000e+00  DSS-63
2013-12-29 07:03:36.000000  2013-12-29 07:03:45.000000  7.1664857440000000e+09   0.0000000000000000e+00  DSS-63
2013-12-29 07:03:45.000000  2013-12-29 07:03:58.000000  7.1664857440000000e+09   0.0000000000000000e+00  DSS-63
2013-12-29 07:03:58.000000  2013-12-29 07:04:00.000000  7.1664857440000000e+09   0.0000000000000000e+00  DSS-63
2013-12-29 07:04:00.000000  2013-12-29 07:52:30.000000  7.1664419636242456e+09   0.0000000000000000e+00  DSS-63
2013-12-29 07:52:30.000000  2013-12-29 07:52:40.000000  7.1664419636242456e+09  -1.0000000000000000e+03  DSS-63
2013-12-29 07:52:40.000000  2013-12-29 07:52:55.000000  7.1664319636242456e+09   0.0000000000000000e+00  DSS-63
2013-12-29 07:52:55.000000  2013-12-29 07:52:58.000000  7.1664319636242456e+09   0.0000000000000000e+00  DSS-63
2013-12-29 07:52:58.000000  2013-12-29 07:53:10.000000  7.1664319636242456e+09   0.0000000000000000e+00  DSS-63
2013-12-29 07:53:10.000000  2013-12-29 07:53:50.000000  7.1664319636242456e+09   5.0000000000000000e+02  DSS-63
2013-12-29 07:53:50.000000  2013-12-29 07:54:30.000000  7.1664519636242456e+09  -4.9999999999900001e+02  DSS-63
2013-12-29 07:54:30.000000  2013-12-29 07:56:18.000000  7.1664319636242456e+09   4.9796644216700003e+02  DSS-63
2013-12-29 07:56:18.000000  2013-12-29 11:01:52.000000  7.1664857440000000e+09   0.0000000000000000e+00  DSS-63
2013-12-29 11:01:52.000000  2013-12-29 11:25:55.000000  7.1664857440000000e+09   0.0000000000000000e+00  DSS-63
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

        # Check if the station is DSS-63 or DSS14
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

def get_fdets_receiving_station_name(fdets_file):
    site = ID_to_site(fdets_file.split('.')[4])

    return(site)
###################################################################

def get_filtered_fdets_collection(
        fdets_file,
        ifms_files,
        receiving_station_name,
        reception_band = observation.FrequencyBands.x_band,
        transmission_band = observation.FrequencyBands.x_band,
        base_frequency = 8412e6,
        column_types = ["utc_datetime_string", "signal_to_noise_ratio", "normalised_spectral_max","doppler_measured_frequency_hz", "doppler_noise_hz"],
        target_name = 'MEX'
):

    transmitting_stations_list = []
    fdets_collections_list = []
    ifms_collections_list = []
    for ifms_file in ifms_files:
        station_code = ifms_file.split('/')[3][1:3]
        if station_code == '14':
            transmitting_station_name = 'DSS14'

        elif station_code == '63':
            transmitting_station_name = 'DSS63'

        elif station_code == '32':
            transmitting_station_name = 'NWNORCIA'


        #transmitting_stations_list.append(transmitting_station_name)

        # Loading IFMS file
        #print(f'IFMS file: {ifms_file}\n with transmitting station: {transmitting_station_name} will be loaded.')
        ifms_collection = observation.observations_from_ifms_files(
            [ifms_file], bodies, spacecraft_name, transmitting_station_name, reception_band, transmission_band
        )

        ifms_collections_list.append(ifms_collection)
        ifms_times = ifms_collection.get_observation_times()
        ifms_times = [time.to_float() for time in ifms_times[0]]

        start_ifms_time = min(ifms_times)
        end_ifms_time = max(ifms_times)

        print(f'Assigned Transmitting Station: {transmitting_station_name}\nIFMS file name: {ifms_file}')

        fdets_collection = observation.observations_from_fdets_files(
            fdets_file, base_frequency, column_types, target_name,
            transmitting_station_name, receiving_station_name, reception_band, transmission_band
        )

        time_filter = estimation.observation_filter(
            estimation.time_bounds_filtering, start_ifms_time, end_ifms_time, use_opposite_condition = True)
        fdets_collection.filter_observations(time_filter)

        if len(fdets_collection.get_observation_times()[0]) > 1:
            for key, link_end_items in fdets_collection.link_definition_ids.items():
                print(f"Key: {key}")
                for link_type, link_end_id in link_end_items.items():
                    if link_type.name == 'transmitter':
                        # Print LinkEndType name and object memory address
                        print(f"Link Type: {link_type.name} - Object: {link_end_id.reference_point}")
                        transmitter_name = link_end_id.reference_point
                        transmitting_stations_list.append(transmitter_name)
                        print(transmitting_stations_list)
            fdets_collections_list.append(fdets_collection)

    print(f'transmitting stations list:{transmitting_stations_list}')
    print(f'Length of Fdets collections: {len(fdets_collections_list)}')
    print(f'Length of stations list: {len(transmitting_stations_list)}')

    # CREATE MERGED IFMS_COLLECTION
    merged_ifms_collection = estimation.merge_observation_collections(ifms_collections_list)
    return fdets_collections_list, transmitting_stations_list


def get_filtered_and_merged_fdets_collection(
        fdets_file,
        ifms_files,
        receiving_station_name,
        reception_band = observation.FrequencyBands.x_band,
        transmission_band = observation.FrequencyBands.x_band,
        base_frequency = 8412e6,
        column_types = ["utc_datetime_string", "signal_to_noise_ratio", "normalised_spectral_max","doppler_measured_frequency_hz", "doppler_noise_hz"],
        target_name = 'MEX'
):

    transmitting_stations_list = []
    transmitting_stations_list_from_dict = []
    for ifms_file in ifms_files:
        station_code = ifms_file.split('/')[3][1:3]
        if station_code == '14':
            transmitting_station_name = 'DSS14'

        elif station_code == '63':
            transmitting_station_name = 'DSS63'

        elif station_code == '32':
            transmitting_station_name = 'NWNORCIA'


        transmitting_stations_list.append(transmitting_station_name)

    # Loading IFMS file
    #print(f'IFMS file: {ifms_file}\n with transmitting station: {transmitting_station_name} will be loaded.')
    ifms_collection = observation.observations_from_multi_station_ifms_files(
        ifms_files, bodies, spacecraft_name, transmitting_stations_list, reception_band, transmission_band
    )

    time_bounds_per_set = ifms_collection.get_time_bounds_per_set()
    time_bounds_array = np.zeros((len(time_bounds_per_set), 2))
    compare_time_bounds_array = np.zeros((len(time_bounds_per_set), 2))
    ifms_intervals_list = []
    compare_ifms_intervals_list = []
    for j in range(len(time_bounds_per_set)):
        time_bounds_array[j, 0] = time_bounds_per_set[j][0].to_float()
        time_bounds_array[j, 1] = time_bounds_per_set[j][1].to_float()
        compare_time_bounds_array[j, 0] = time_conversion.seconds_since_epoch_to_julian_day(time_bounds_per_set[j][0].to_float())
        compare_time_bounds_array[j, 1] = time_conversion.seconds_since_epoch_to_julian_day(time_bounds_per_set[j][1].to_float())
        ifms_intervals_list.append((time_bounds_array[j, 0], time_bounds_array[j, 1]))
        compare_ifms_intervals_list.append((compare_time_bounds_array[j, 0], compare_time_bounds_array[j, 1]))

    fdets_collections_list = []
    #print(f'lengths: {len(ifms_files)}, {len(transmitting_stations_list)}')
    final_transmitting_stations_list = []
    for ifms_file, transmitting_station_name, ifms_interval in zip(ifms_files, transmitting_stations_list, ifms_intervals_list):

        start_ifms_time = ifms_interval[0]
        end_ifms_time = ifms_interval[1]

        print(f'Assigned transmitting station: {transmitting_station_name}\nIFMS file name: {ifms_file}')

        fdets_collection = observation.observations_from_fdets_files(
            fdets_file, base_frequency, column_types, target_name,
            transmitting_station_name, receiving_station_name, reception_band, transmission_band
        )

        times = fdets_collection.get_observation_times()
        times = np.array([time.to_float() for time in times[0]])
        max_time = np.max(times)
        min_time = np.min(times)

        time_filter = estimation.observation_filter(
            estimation.time_bounds_filtering, start_ifms_time, end_ifms_time, use_opposite_condition = True)
        fdets_collection.filter_observations(time_filter)

        if len(fdets_collection.get_observation_times()[0]) > 0:
            final_transmitting_stations_list.append(transmitting_station_name)

        fdets_collections_list.append(fdets_collection)

    if len(fdets_collections_list) > 0:
        merged_fdets_collection = estimation.merge_observation_collections(fdets_collections_list)
        merged_fdets_collection.remove_empty_observation_sets()

    else:
        print('Impossible to create merged observation collection. Reason: list of fdets collections has length 0.')


    fdets_time_bounds_per_set = merged_fdets_collection.get_time_bounds_per_set()
    fdets_time_bounds_array = np.zeros((len(fdets_time_bounds_per_set), 2))
    fdets_intervals_list = []
    for j in range(len(fdets_time_bounds_per_set)):
        fdets_time_bounds_array[j, 0] = time_conversion.seconds_since_epoch_to_julian_day(fdets_time_bounds_per_set[j][0].to_float())
        fdets_time_bounds_array[j, 1] = time_conversion.seconds_since_epoch_to_julian_day(fdets_time_bounds_per_set[j][1].to_float())
        fdets_intervals_list.append((fdets_time_bounds_array[j, 0], fdets_time_bounds_array[j, 1]))

    fdets_calendar_date_tuples = [
        (
            time_conversion.julian_day_to_calendar_date(julian_day[0]),
            time_conversion.julian_day_to_calendar_date(julian_day[1])
        )
        for julian_day in fdets_intervals_list
    ]

    ifms_calendar_date_tuples = [
        (
            time_conversion.julian_day_to_calendar_date(julian_day[0]),
            time_conversion.julian_day_to_calendar_date(julian_day[1])
        )
        for julian_day in compare_ifms_intervals_list
    ]

    print(f'Fdets intervals list: {fdets_calendar_date_tuples}')
    print(f'IFMS intervals list: {ifms_calendar_date_tuples}')
    print(f'Final stations: {final_transmitting_stations_list}')

    dict_merged_fdets_collection = list(merged_fdets_collection.sorted_observation_sets.values())
    # Iterate over the dictionary
    for key, observation_sets in dict_merged_fdets_collection[0].items():
        print(f"Key: {key}")
        for observation_set in observation_sets:
            # Print object details
            transmitter_name = observation_set.link_definition.link_end_id(observation.transmitter).reference_point
            print(f"Transmitter: {observation_set.link_definition.link_end_id(observation.transmitter).reference_point}")

        transmitting_stations_list_from_dict.append(transmitter_name)

    print(f'Final Stations from dict: {transmitting_stations_list_from_dict}')

    return merged_fdets_collection, transmitting_stations_list_from_dict


def extract_date_from_filename(filename):
    # Extract the numeric date-time portion (e.g., 133630348)
    date_code = filename.split('_')[2]  # "133630348"
    print(date_code)

    # Parse the date information
    year = 2000 + int(date_code[:2])  # 13 -> 2013
    doy = int(date_code[2:5])         # 363 -> DOY
    hour = int(date_code[5:7])        # 03 -> 3 hours
    minute = int(date_code[7:])       # 48 -> 48 minutes

    # Convert to a standard datetime object for easy sorting
    return datetime.strptime(f"{year} {doy} {hour} {minute}", "%Y %j %H %M")


if __name__ == "__main__":
    # Set Folders Containing Relevant Files
    mex_kernels_folder = 'mex_phobos_flyby/kernels/'
    mex_fdets_folder = 'mex_phobos_flyby/fdets/complete'
    mex_ifms_folder = 'mex_phobos_flyby/ifms/filtered'
    mex_odf_folder = 'mex_phobos_flyby/odf/'

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
    # Retrieve translational ephemeris from SPICE
    #body_settings.get(spacecraft_name).ephemeris_settings = environment_setup.ephemeris.direct_spice(
    #    'SSB', 'J2000', 'MEX')
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
    #print(environment_setup.get_ground_station_list(bodies.get_body("Earth")))
    ###############################################################

    sites_list = []
    fdets_files = []
    ifms_files = []

    for fdets_file in os.listdir(mex_fdets_folder):
        fdets_files.append(os.path.join(mex_fdets_folder, fdets_file))
        site = ID_to_site(fdets_file.split('.')[4])
        sites_list.append(site)
    for ifms_file in os.listdir(mex_ifms_folder):
        ifms_files.append(os.path.join(mex_ifms_folder, ifms_file))


    # For now, only try with one Fdets element!
    #fdets_files = [os.path.join(mex_fdets_folder, 'Fdets.mex2013.12.28.Bd.complete.r2i.txt'), os.path.join(mex_fdets_folder, 'Fdets.mex2013.12.28.On.complete.r2i.txt')]



    #fdets_files = ['mex_phobos_flyby/fdets/complete/Fdets.mex2013.12.28.On.complete.r2i.txt', 'mex_phobos_flyby/fdets/complete/Fdets.mex2013.12.28.Bd.complete.r2i.txt'] #['mex_phobos_flyby/fdets/single/fdets.r3i.new.trial.On.txt']

    #fdets_fils = 'mex_phobos_flyby/fdets/complete/Fdets.mex2013.12.28.On.complete.r2i.txt'
    for fdets_file in fdets_files:

        receiving_station_name = get_fdets_receiving_station_name(fdets_file)
        if receiving_station_name == None or receiving_station_name not in  [station[1] for station in environment_setup.get_ground_station_list(bodies.get_body("Earth"))]:
            continue
        added_labels = set()
        label_colors = {}

        filtered_collections_list, transmitting_stations_list = get_filtered_fdets_collection(fdets_file, ifms_files, receiving_station_name)
        site_name = get_fdets_receiving_station_name(fdets_file)

        for filtered_collection, transmitting_station_name in zip(filtered_collections_list,transmitting_stations_list):
            #if not 'VLBA' in site_name:
            #    continue

            print(f'Transmitting station: {transmitting_station_name}')

            antenna_position_history = dict()
            com_position = [-1.3,0.0,0.0] # estimated based on the MEX_V16.TF file description

            times = filtered_collection.get_observation_times()
            times = [time.to_float() for time in times[0]]

            mjd_times = [time_conversion.seconds_since_epoch_to_julian_day(t) for t in times]
            utc_times = np.array([Time(mjd_time, format='jd', scale='utc').datetime for mjd_time in mjd_times])

            antenna_state = np.zeros((6, 1))
            antenna_state[:3,0] = spice.get_body_cartesian_position_at_epoch("-41020", "-41000", "MEX_SPACECRAFT", "none", times[0])
            # Translate the antenna position to account for the offset between the origin of the MEX-fixed frame and the COM
            antenna_state[:3,0] = antenna_state[:3,0] - com_position

            # Create tabulated ephemeris settings from antenna position history
            antenna_ephemeris_settings = environment_setup.ephemeris.constant(antenna_state, "-41000",  "MEX_SPACECRAFT")
            # Create tabulated ephemeris for the MEX antenna
            antenna_ephemeris = environment_setup.ephemeris.create_ephemeris(antenna_ephemeris_settings, "Antenna")
            # Set the spacecraft's reference point position to that of the antenna (in the MEX-fixed frame)
            filtered_collection.set_reference_point(bodies, antenna_ephemeris, "Antenna", "MEX", observation.reflector1)

            print(f'\nSetting uplink: {transmitting_station_name}')
            link_ends = {
                observation.receiver: observation.body_reference_point_link_end_id('Earth', site_name),
                observation.retransmitter: observation.body_reference_point_link_end_id('MEX', 'Antenna'),
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
            # Add elevation and SEP angles dependent variables to the compressed observation collection
            elevation_angle_settings = observation.elevation_angle_dependent_variable( observation.receiver )
            elevation_angle_parser = filtered_collection.add_dependent_variable( elevation_angle_settings, bodies )
            sep_angle_settings = observation.avoidance_angle_dependent_variable("Sun", observation.retransmitter, observation.receiver)
            sep_angle_parser = filtered_collection.add_dependent_variable( sep_angle_settings, bodies )
            # Compute and set residuals in the fdets observation collection
            estimation.compute_residuals_and_dependent_variables(filtered_collection, observation_simulators, bodies)

            # Perform computations
            concatenated_obs = np.array(filtered_collection.get_observations())
            concatenated_computed_obs = np.array(filtered_collection.get_computed_observations())
            residuals_by_hand_no_atm_corr = concatenated_computed_obs - concatenated_obs

            print(f'Residuals: {residuals_by_hand_no_atm_corr}')

            if site_name not in label_colors:
                label_colors[site_name] = generate_random_color()

            # Use the stored color for plotting
            plt.scatter(
                utc_times,
                residuals_by_hand_no_atm_corr,
                color=label_colors[site_name],
                s=10,
                marker='+',
                label=f'{site_name}' if site_name not in added_labels else None
            )
            added_labels.add(site_name)  # Avoid duplicate labels in the legend
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
