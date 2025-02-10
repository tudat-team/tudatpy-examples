######################### # IMPORTANT #############################################################################

# In order to test this example, I am using a Phobos Flyby fdets file missing the few last/first lines...
# The removed lines were classified as outliers, but they should be filtered with the proper tudat functionality,
# rather than manually (as done for now)

# NOTE: "DSS63" DOES NOT WORK. IT MUST BE "DSS-63". Not the same for DSS14 (it must be DSS-14)...
# NOTE: remember to remove empty sets, or the loaded stations (with empty observations) will cause troubles in the simulation.
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
def compute_h(p, sat_positions, sat_velocities, transmitting_station):
    """
    Compute Doppler shift for a given station position.

    Parameters:
    - p: np.array, station position [x, y, z] (meters, ECEF)
    - sat_positions: np.array, shape (N,3), satellite positions at observation times
    - sat_velocities: np.array, shape (N,3), satellite velocities at observation times
    - f0: float, transmitted frequency (default: 400 MHz)

    Returns:
    - h_computed: np.array, computed Doppler shift observations
    """

    if transmitting_station == 'NWNORCIA':
        f0 = 7166437036.994461
    elif transmitting_station == 'DSS63':
        f0 = 7166485744.000000
    elif transmitting_station == 'DSS14':
        f0 =  7166490928.000000

    c = 299792458.0  # Speed of light in m/s

    # Compute range vector (satellite to station)
    r = sat_positions - p  # Shape: (N,3)


# Compute range rate (project satellite velocity along range vector)
    range_norm = np.linalg.norm(r, axis=1)  # Compute magnitude of r
    range_rate = np.einsum('ij,ij->i', r, sat_velocities) / range_norm  # Dot product projection

    # Compute Doppler shift
    h_computed = - (f0 / c) * range_rate
    return h_computed
def compute_station_shift(h_observed, h_computed, p_old, sat_positions, sat_velocities, transmitting_station, delta=1e3):
    """
    Compute the station position correction using numerical differentiation.

    Parameters:
    - h_computed: np.array, computed Doppler observations
    - h_observed: np.array, observed Doppler observations
    - p_old: np.array, initial station position [x, y, z] in ECEF (meters)
    - delta: float, finite difference step size (default: 1 km)

    Returns:
    - p_new: np.array, corrected station position
    - delta_p: np.array, computed station shift
    """

    delta_h = h_observed - h_computed
    # Number of observations
    num_obs = len(concatenated_obs)

    # Initialize Jacobian matrix H (size: num_obs x 3 for x, y, z perturbations)
    H = np.zeros((num_obs, 3))

    # Compute numerical derivatives for each coordinate (x, y, z)
    for i in range(3):
        # Perturb station position in + direction
        p_plus = p_old.copy()
        p_plus[i] += delta
        h_plus = compute_h(p_plus, sat_positions, sat_velocities, transmitting_station)  # Function to compute Doppler with perturbed position

        # Perturb station position in - direction
        p_minus = p_old.copy()
        p_minus[i] -= delta
        h_minus = compute_h(p_minus, sat_positions, sat_velocities, transmitting_station)  # Function to compute Doppler with perturbed position

        # Finite difference approximation
        H[:, i] = (h_plus - h_minus) / (2 * delta)

    # Solve for station shift using least squares
    HTH_inv = np.linalg.inv(H.T @ H)  # (H^T H)^(-1)
    delta_p = HTH_inv @ H.T @ delta_h  # (H^T H)^(-1) H^T delta_h

    # Compute corrected station position
    p_new = p_old + delta_p

    print(f'new position: {p_new}')
    print(f'new position norm: {np.linalg.norm(p_new)}') # just a check that it is similar to r_earth
    return p_new, delta_p

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
def plot_ifms_windows(ifms_file, color):

    # Get IFMS observation times
    ifms_times = ifms_file.get_observation_times()
    ifms_file_name = ifms_file.split('/')[3]

    # Loop through each element in ifms_times and convert to float
    min_sublist = np.min([time.to_float() for time in ifms_times[0]])
    max_sublist = np.max([time.to_float() for time in ifms_times[0]])
    mjd_min_sublist = time_conversion.seconds_since_epoch_to_julian_day(min_sublist)
    mjd_max_sublist = time_conversion.seconds_since_epoch_to_julian_day(max_sublist)
    utc_min_sublist = Time(mjd_min_sublist, format='jd', scale = 'utc').datetime
    utc_max_sublist = Time(mjd_max_sublist, format='jd', scale = 'utc').datetime
    #(utc_min_sublist, utc_max_sublist)
    plt.axvspan(utc_min_sublist, utc_max_sublist, color=color, alpha=0.2, label = ifms_file_name)

    return(min_sublist, max_sublist)

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
        if ifms_file.split('/')[7].startswith('.'):
            continue
        #station_code = ifms_file.split('/')[3][1:3]
        print(ifms_file)
        station_code = ifms_file.split('/')[7][1:3]
        print(station_code)

        if station_code == '14':
            transmitting_station_name = 'DSS14'

        elif station_code == '63':
            transmitting_station_name = 'DSS63'

        elif station_code == '32':
            transmitting_station_name = 'NWNORCIA'


        #transmitting_stations_list.append(transmitting_station_name)

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
    #merged_ifms_collection = estimation.merge_observation_collections(ifms_collections_list)

    return fdets_collections_list, transmitting_stations_list, start_utc_time, end_utc_time


def create_single_ifms_collection_from_file(
        ifms_file,
        bodies,
        spacecraft_name,
        transmitting_station_name,
        reception_band,
        transmission_band):

        # Loading IFMS file
        #print(f'IFMS file: {ifms_file}\n with transmitting station: {transmitting_station_name} will be loaded.')
        single_ifms_collection = observation.observations_from_ifms_files(
            [ifms_file], bodies, spacecraft_name, transmitting_station_name, reception_band, transmission_band
        )

        return(single_ifms_collection)

def get_transmitting_station_from_ifms_file(ifms_file):

        station_code = ifms_file.split('/')[3][1:3]
        if station_code == '14':
            transmitting_station_name = 'DSS14'

        elif station_code == '63':
            transmitting_station_name = 'DSS63'

        elif station_code == '32':
            transmitting_station_name = 'NWNORCIA'

        return transmitting_station_name

def get_single_ifms_collection_time_bounds(ifms_collection):


        ifms_times = ifms_collection.get_observation_times()
        ifms_times = [time.to_float() for time in ifms_times[0]]

        start_ifms_time = min(ifms_times)
        end_ifms_time = max(ifms_times)

        return start_ifms_time, end_ifms_time

def create_single_fdets_collection_from_file(
        fdets_file,
        base_frequency,
        column_types,
        spacecraft_name,
        transmitting_station_name,
        reception_band,
        transmission_band ):

        single_fdets_collection = observation.observations_from_fdets_files(
            fdets_file, base_frequency, column_types, spacecraft_name,
            transmitting_station_name, receiving_station_name, reception_band, transmission_band
        )

        return single_fdets_collection


def set_time_filter(single_fdets_collection, start_ifms_time, end_ifms_time):

    time_filter = estimation.observation_filter(
        estimation.time_bounds_filtering, start_ifms_time, end_ifms_time, use_opposite_condition = True)

    return(time_filter)

def get_ordered_ground_stations(single_fdets_collection):

    if len(single_fdets_collection.get_observation_times()[0]) > 1:
        for key, link_end_items in single_fdets_collection.link_definition_ids.items():
            print(f"Key: {key}")
            for link_type, link_end_id in link_end_items.items():
                if link_type.name == 'transmitter':
                    # Print LinkEndType name and object memory address
                    print(f"Link Type: {link_type.name} - Object: {link_end_id.reference_point}")
                    transmitting_station_name = link_end_id.reference_point

                    return transmitting_station_name

def get_filtered_fdets_collection_new(fdets_file, ifms_file, bodies, base_frequency,column_types,spacecraft_name,reception_band,transmission_band):
    transmitting_station_name = get_transmitting_station_from_ifms_file(ifms_file)
    single_ifms_collection = create_single_ifms_collection_from_file(ifms_file, bodies, spacecraft_name,transmitting_station_name,reception_band,transmission_band)
    start_ifms_time, end_ifms_time = get_single_ifms_collection_time_bounds(single_ifms_collection)
    time_filter = set_time_filter(single_ifms_collection, start_ifms_time, end_ifms_time)
    single_fdets_collection = create_single_fdets_collection_from_file(fdets_file,base_frequency,column_types,spacecraft_name,transmitting_station_name,reception_band,transmission_band)
    single_fdets_collection.filter_observations(time_filter)

    if len(single_fdets_collection.get_observation_times()[0]) > 1:
        transmitting_stations_list = get_ordered_ground_stations(single_fdets_collection)

        return single_fdets_collection, transmitting_stations_list, single_ifms_collection
    else:
        return None, None, single_ifms_collection



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


    base_frequency = 8412e6
    column_types = ["utc_datetime_string", "signal_to_noise_ratio", "normalised_spectral_max","doppler_measured_frequency_hz", "doppler_noise_hz"]
    reception_band = observation.FrequencyBands.x_band
    transmission_band = observation.FrequencyBands.x_band

    sites_list = []
    fdets_files = []
    ifms_files = []

    #fdets_files = ['mex_phobos_flyby/fdets/complete/Fdets.mex2013.12.28.Ur.complete.r2i.txt']

    #for fdets_file in os.listdir(mex_fdets_folder):
    for fdets_file in fdets_files:
        if fdets_file == 'Fdets.mex2013.12.28.Wz.complete.r2i.txt':
            continue
        fdets_files.append(os.path.join(mex_fdets_folder, fdets_file))
        site = ID_to_site(fdets_file.split('.')[4])
        sites_list.append(site)
    for ifms_file in os.listdir(mex_ifms_folder):
        #if ifms_file == 'M63ODFXL02_DPX_133630348_00.TAB' or ifms_file == 'M63ODFXL02_DPX_133630348_00.TAB':
        ifms_files.append(os.path.join(mex_ifms_folder, ifms_file))

    filtered_collections_list = []
    transmitting_stations_list = []
    single_ifms_collections_list = []

    fdets_station_residuals = dict()
    for fdets_file in os.listdir(mex_fdets_folder):
        fdets_file = os.path.join(mex_fdets_folder, fdets_file)
        receiving_station_name = get_fdets_receiving_station_name(fdets_file)
        if receiving_station_name == None or receiving_station_name not in  [station[1] for station in environment_setup.get_ground_station_list(bodies.get_body("Earth"))]:
            continue

        for ifms_file in ifms_files:
            if ifms_file.split('/')[7].startswith('.'):
                continue

            filtered_collections_list, transmitting_stations_list, start_ifms_utc_time, end_ifms_utc_time = get_filtered_fdets_collection(fdets_file, [ifms_file], receiving_station_name)
            site_name = get_fdets_receiving_station_name(fdets_file)

            for filtered_collection, transmitting_station_name in zip(filtered_collections_list,transmitting_stations_list):

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
                # Add elevation and SEP angles dependent variables to the observation collection
                elevation_angle_settings = observation.elevation_angle_dependent_variable( observation.receiver )
                elevation_angle_parser = filtered_collection.add_dependent_variable( elevation_angle_settings, bodies )
                sep_angle_settings = observation.avoidance_angle_dependent_variable("Sun", observation.retransmitter, observation.receiver)
                sep_angle_parser = filtered_collection.add_dependent_variable( sep_angle_settings, bodies )
                # Compute and set residuals in the fdets observation collection
                estimation.compute_residuals_and_dependent_variables(filtered_collection, observation_simulators, bodies)

                ### ------------------------------------------------------------------------------------------
                ### RETRIEVE AND SAVE VARIOUS OBSERVATION OUTPUTS
                ### ------------------------------------------------------------------------------------------

                concatenated_obs = filtered_collection.get_concatenated_observations()
                concatenated_computed_obs = filtered_collection.get_concatenated_computed_observations()

                # Retrieve RMS and mean of the residuals
                concatenated_residuals = filtered_collection.get_concatenated_residuals()
                rms_residuals = filtered_collection.get_rms_residuals()
                mean_residuals = filtered_collection.get_mean_residuals()

                #print(f'Residuals: {concatenated_residuals}')
                print(f'Mean Residuals: {mean_residuals}')
                print(f'RMS Residuals: {rms_residuals}')

                ##Populate Station Residuals Dictionary
                site_name = receiving_station_name
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

    print(f"File created: {file_path}")

# Plot Residuals
added_labels = set()
label_colors = dict()
label_ifms_colors = {'NWNORCIA': 'blue', 'DSS63': 'grey', 'DSS14': 'red'}
# Plot residuals for each station
for site_name, data_list in fdets_station_residuals.items():
    if site_name not in label_colors:
        label_colors[site_name] = generate_random_color()

    for times, utc_times, residuals, mean_residuals, rms_residuals, start_ifms_utc_time, end_ifms_utc_time, transmitting_station_name in data_list:
        # Plot all stations' residuals on the same figure
        plt.scatter(
            utc_times, residuals,
            color = label_colors[site_name],
            marker = '+', s=10,
            label=f"{site_name}, mean = {mean_residuals[0]}, rms = {rms_residuals[0]}"
            if site_name not in added_labels else None
        )
        #plt.axvspan(start_ifms_utc_time, end_ifms_utc_time, alpha=0.3, color=label_ifms_colors[transmitting_station_name], label = transmitting_station_name if transmitting_station_name not in added_labels else None)
        added_labels.add(site_name)  # Avoid duplicate labels in the legend
        added_labels.add(transmitting_station_name)

# Format the x-axis for dates
plt.title('Fdets Residuals')
plt.xlabel('Time [s]')
plt.ylabel('Residuals [Hz]')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gcf().autofmt_xdate()  # Auto-rotate date labels for better readability
plt.grid(True)
plt.legend(loc='lower left', bbox_to_anchor=(0.8, 1), borderaxespad=0.)
# Adjust layout to make room for the legend
single_residuals_path = f'/Users/lgisolfi/Desktop/mex_phobos_flyby/output/single_residual_plots/'
os.makedirs(single_residuals_path, exist_ok=True)
plt.show()
plt.savefig('residuals_fdets')

plt.close('all')
exit()
    