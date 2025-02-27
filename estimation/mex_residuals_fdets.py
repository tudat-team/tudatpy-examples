######################### # IMPORTANT #############################################################################

# This script allows to create residuals csv files (PRIDE data).
# These are saved and can later be plotted via the read_mex_ifms_fdets_residuals.py function.
##################################################################################################################
import os
import csv
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
import tudatpy.data as data
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
        'Hh': 'HARTRAO',
        'Ht': 'HART15M',
        'Mh': 'METSAHOV',
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
        'Br': 'BR-VLBA',
        'Ov': 'OV-VLBA',
        'Mk': 'MK-VLBA',
        'Pt': 'PIETOWN',
    }

    # Return the corresponding site name or None if the site_ID is not found
    return id_to_site_mapping.get(site_ID, None)
#####################################################################################################
def generate_random_color():
    """Generates a random color in hexadecimal format."""
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

    '''
    This function processes FDETS and IFMS observation files, filters the observations based on specific criteria,
    and returns filtered collections along with relevant metadata.

    Parameters:
    fdets_file (str): Path to the FDETS file.
    ifms_files (list of str): List of paths to IFMS files.
    receiving_station_name (str): Name of the receiving station.
    reception_band (observation.FrequencyBands, optional): Frequency band of the reception. Default is observation.FrequencyBands.x_band.
    transmission_band (observation.FrequencyBands, optional): Frequency band of the transmission. Default is observation.FrequencyBands.x_band.
    base_frequency (float, optional): Base frequency used for observations. Default is 8412e6 Hz.
    column_types (list of str, optional): List of column types to extract from the FDETS file. Default includes ["utc_datetime_string", "signal_to_noise_ratio", "normalised_spectral_max", "doppler_measured_frequency_hz", "doppler_noise_hz"].
    target_name (str, optional): Name of the target. Default is 'MEX'.

    Returns:
    fdets_collections_list (list): List of filtered FDETS collections.
    transmitting_stations_list (list): List of transmitting stations.
    start_utc_time (datetime): Start time of observations in UTC.
    end_utc_time (datetime): End time of observations in UTC.
    '''

    transmitting_stations_list = []
    fdets_collections_list = []
    ifms_collections_list = []

    for ifms_file in ifms_files:
        if ifms_file.split('/')[7].startswith('.'):
            continue
        station_code = ifms_file.split('/')[7][1:3]

        if station_code == '14':
            transmitting_station_name = 'DSS14'
            continue

        elif station_code == '63':
            transmitting_station_name = 'DSS63'


        elif station_code == '32':
            transmitting_station_name = 'NWNORCIA'

        # Loading IFMS file
        ifms_collection = observation.observations_from_ifms_files(
            [ifms_file], bodies, spacecraft_name, transmitting_station_name, reception_band, transmission_band
        )


        ifms_collections_list.append(ifms_collection)
        ifms_times = ifms_collection.get_observation_times()
        ifms_times = [time.to_float() for time in ifms_times[0]]

        start_ifms_time = min(ifms_times)
        end_ifms_time = max(ifms_times)


        start_mjd_time = time_conversion.seconds_since_epoch_to_julian_day(start_ifms_time)
        end_mjd_time = time_conversion.seconds_since_epoch_to_julian_day(end_ifms_time)

        start_utc_time = Time(start_mjd_time, format='jd', scale='utc').datetime
        end_utc_time = Time(end_mjd_time, format='jd', scale='utc').datetime

        fdets_collection = observation.observations_from_fdets_files(
            fdets_file, base_frequency, column_types, target_name,
            transmitting_station_name, receiving_station_name, reception_band, transmission_band
        )

        time_filter = estimation.observation_filter(
            estimation.time_bounds_filtering, start_ifms_time, end_ifms_time, use_opposite_condition = True)
        filter_residuals = estimation.observation_filter(estimation.residual_filtering, 1)
        fdets_collection.filter_observations(time_filter)
        fdets_collection.filter_observations(filter_residuals)

        if len(fdets_collection.get_observation_times()[0]) > 1:
            for key, link_end_items in fdets_collection.link_definition_ids.items():
                for link_type, link_end_id in link_end_items.items():
                    if link_type.name == 'transmitter':
                        # Print LinkEndType name and object memory address
                        print(f"Link Type: {link_type.name} - Object: {link_end_id.reference_point}")
                        transmitter_name = link_end_id.reference_point
                        transmitting_stations_list.append(transmitter_name)
                        print(transmitting_stations_list)
            fdets_collections_list.append(fdets_collection)

    return fdets_collections_list, transmitting_stations_list, start_utc_time, end_utc_time

if __name__ == "__main__":
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
    observation.set_vmf_troposphere_data(
        [ "/Users/lgisolfi/Desktop/mex_phobos_flyby/VMF/y2013.vmf3_r.txt" ], True, False, bodies, False, True )
    # Meteorological (tropospsheric) uplink and downlink corrections
    weather_files = ([os.path.join('/Users/lgisolfi/Desktop/data_archiving-1.0/dataset/mex/gr035/downloaded/met', met_file) for met_file in os.listdir('/Users/lgisolfi/Desktop/data_archiving-1.0/dataset/mex/gr035/downloaded/met')])
    body_settings.get("Earth").ground_station_settings.append(data.set_estrack_weather_data_in_ground_stations(bodies,weather_files, 'NWNORCIA'))
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

    sites_list = []
    fdets_files = []
    ifms_files = []

    ###############################################################################################################################################

    # Temporary: Change this line if you want to compute residuals for different fdets.
    fdets_list = ['Fdets.mex2013.12.28.Ur.complete.r2i.txt', 'Fdets.mex2013.12.28.Ht.complete.r2i.txt', 'Fdets.mex2013.12.28.On.complete.r2i.txt']

    ###############################################################################################################################################

    for ifms_file in os.listdir(mex_ifms_folder):
        ifms_files.append(os.path.join(mex_ifms_folder, ifms_file))

    filtered_collections_list = []
    transmitting_stations_list = []
    single_ifms_collections_list = []

    fdets_station_residuals = dict()
    for fdets_file in fdets_list:
        fdets_file = os.path.join(mex_fdets_folder, fdets_file)
        site_name = get_fdets_receiving_station_name(fdets_file)
        if site_name == None or site_name not in  [station[1] for station in environment_setup.get_ground_station_list(bodies.get_body("Earth"))]:
            continue

        for ifms_file in ifms_files:
            if ifms_file.split('/')[7].startswith('.'):
                continue
            filtered_collections_list, transmitting_stations_list, start_ifms_utc_time, end_ifms_utc_time = get_filtered_fdets_collection(fdets_file, [ifms_file], site_name)
            for filtered_collection, transmitting_station_name in zip(filtered_collections_list,transmitting_stations_list):

                print(f'Transmitting station: {transmitting_station_name}')

                times = filtered_collection.get_observation_times()
                times = [time.to_float() for time in times[0]]
                mjd_times = [time_conversion.seconds_since_epoch_to_julian_day(t) for t in times]
                utc_times = np.array([Time(mjd_time, format='jd', scale='utc').datetime for mjd_time in mjd_times])

                com_position = [-1.3,0.0,0.0] # estimated based on the MEX_V16.TF file description
                antenna_state = np.zeros((6, 1))
                antenna_state[:3,0] = spice.get_body_cartesian_position_at_epoch("-41020", "-41000", "MEX_SPACECRAFT", "none", times[0])
                antenna_state[:3,0] = antenna_state[:3,0] - com_position
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
                light_time_correction_list.append(
                    estimation_setup.observation.saastamoinen_tropospheric_light_time_correction( ))

                # Define the observation model settings
                observation_model_settings = [
                    estimation_setup.observation.doppler_measured_frequency(
                        link_definition, light_time_correction_list
                    )
                ]
                ###################################################################################################

                # Create observation simulators.
                observation_simulators = estimation_setup.create_observation_simulators(observation_model_settings, bodies)
                # Add elevation and SEP angles dependent variables to the observation collection
                elevation_angle_settings = observation.elevation_angle_dependent_variable( observation.receiver )
                elevation_angle_parser = filtered_collection.add_dependent_variable( elevation_angle_settings, bodies )
                sep_angle_settings = observation.avoidance_angle_dependent_variable("Sun", observation.retransmitter, observation.receiver)
                sep_angle_parser = filtered_collection.add_dependent_variable( sep_angle_settings, bodies )
                # Compute and set residuals in the fdets observation collection
                estimation.compute_residuals_and_dependent_variables(filtered_collection, observation_simulators, bodies)

                # Retrieve RMS and mean of the residuals
                concatenated_residuals = filtered_collection.get_concatenated_residuals()
                rms_residuals = filtered_collection.get_rms_residuals()
                mean_residuals = filtered_collection.get_mean_residuals()

                print(f'Mean Residuals: {mean_residuals}')
                print(f'RMS Residuals: {rms_residuals}')

                ##Populate Station Residuals Dictionary
                if site_name not in fdets_station_residuals.keys():
                    fdets_station_residuals[site_name] = [(times, utc_times, concatenated_residuals, mean_residuals, rms_residuals, start_ifms_utc_time, end_ifms_utc_time, transmitting_station_name)]
                else:
                    fdets_station_residuals[site_name].append((times, utc_times, concatenated_residuals, mean_residuals, rms_residuals, start_ifms_utc_time, end_ifms_utc_time, transmitting_station_name))

                #######################################################################################################

# Output files creation
for site_name, data in fdets_station_residuals.items():
    fdets_residuals_path = '/Users/lgisolfi/Desktop/mex_phobos_flyby/output/fdets_residuals'
    os.makedirs(fdets_residuals_path, exist_ok=True)
    filename = f"{site_name}_residuals.csv"
    file_path = os.path.join(fdets_residuals_path, filename)
    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)

        # Write the header
        writer.writerow([f'# Station: {site_name}'])
        writer.writerow(['# Time | UTC Time | Residuals'])

        # Write the data rows
        for record in data:
            times, utc_times, concatenated_residuals, _, _, _, _, _ = record

            # Write each UTC time and residual in a separate row
            for time, utc_time, residual in zip(times, utc_times, concatenated_residuals):
                writer.writerow([
                    time,
                    utc_time.strftime("%Y-%m-%d %H:%M:%S"),  # Convert datetime to string
                    residual
                ])

    print(f"Residuals File Created for station {site_name}: {file_path}")
    