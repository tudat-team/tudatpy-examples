######################### # IMPORTANT #############################################################################

# In order to test this example, I am using a Phobos Flyby IFMS file missing the few last/first lines...
# The removed lines were classified as outliers, but they should be filtered with the proper tudat functionality,
# rather than manually (as done for now)

##################################################################################################################
import os
import csv
from lzma import compress
from xmlrpc.client import DateTime
import numpy as np
from matplotlib import pyplot as plt
from astropy.time import Time

# Load required tudatpy modules
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.astro import time_conversion, element_conversion
from tudatpy.math import interpolators
from tudatpy.numerical_simulation import environment_setup, environment
from tudatpy.numerical_simulation import estimation, estimation_setup
from tudatpy.numerical_simulation.estimation_setup import observation
from tudatpy import util
from load_pds_files import download_url_files_time, download_url_files_time_interval
from datetime import datetime
import random
import matplotlib.dates as mdates

from urllib.request import urlretrieve
def generate_random_color():
    """Generate a random color in hexadecimal format."""
    return "#{:02x}{:02x}{:02x}".format(
        random.randint(0, 255),  # Red
        random.randint(0, 255),  # Green
        random.randint(0, 255)   # Blue
    )
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

# Define Start and end Dates of Simulation
start = datetime(2013, 12, 26)
end = datetime(2013, 12, 30)

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

# Load IFMS files
ifms_files = list()
for ifms_file in os.listdir(mex_ifms_folder):
    ifms_files.append(os.path.join(mex_ifms_folder, ifms_file))

single_ifms_collections_list = list()
atmospheric_corrections_list = list()
transmitting_stations_list = list()
reception_band = observation.FrequencyBands.x_band
transmission_band = observation.FrequencyBands.x_band

labels = set()
ifms_station_residuals = dict()
for ifms_file in ifms_files:
    if ifms_file.split('/')[7].startswith('.'):
        continue
    station_code = ifms_file.split('/')[7][1:3]
    if station_code == '14':
        transmitting_station_name = 'DSS14'

    elif station_code == '63':
        transmitting_station_name = 'DSS63'

    elif station_code == '32':
        transmitting_station_name = 'NWNORCIA'

    ifms_collection = observation.observations_from_ifms_files([ifms_file], bodies, spacecraft_name, transmitting_station_name, reception_band, transmission_band, apply_troposphere_correction = True)


    compressed_observations = ifms_collection

    antenna_position_history = dict()
    com_position = [-1.3,0.0,0.0] # estimated based on the MEX_V16.TF file description

    times = compressed_observations.get_concatenated_observation_times()
    times = [time.to_float() for time in times]

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
    compressed_observations.set_reference_point(bodies, antenna_ephemeris, "Antenna", "MEX", observation.retransmitter)

    #  Create light-time corrections list
    light_time_correction_list = list()
    light_time_correction_list.append(
        estimation_setup.observation.first_order_relativistic_light_time_correction(["Sun"]))

    doppler_link_ends = compressed_observations.link_definitions_per_observable[
        estimation_setup.observation.dsn_n_way_averaged_doppler]

    ########## IMPORTANT STEP #######################################################################
    # When woerking with IFMS, Add: subtract_doppler_signature = False, or it won't work
    observation_model_settings = list()


    for current_link_definition in doppler_link_ends:
        print(current_link_definition.link_end_id(observation.retransmitter).reference_point)
        print(current_link_definition.link_end_id(observation.transmitter).reference_point)
        print(current_link_definition.link_end_id(observation.receiver).reference_point)
        observation_model_settings.append(estimation_setup.observation.dsn_n_way_doppler_averaged(
            current_link_definition, light_time_correction_list, subtract_doppler_signature = False ))
    ###################################################################################################
    # Create observation simulators.
    observation_simulators = estimation_setup.create_observation_simulators(observation_model_settings, bodies)

    # Add elevation and SEP angles dependent variables to the IFMS observation collection
    elevation_angle_settings = observation.elevation_angle_dependent_variable( observation.receiver )
    elevation_angle_parser = compressed_observations.add_dependent_variable( elevation_angle_settings, bodies )
    sep_angle_settings = observation.avoidance_angle_dependent_variable("Sun", observation.retransmitter, observation.receiver)
    sep_angle_parser = compressed_observations.add_dependent_variable( sep_angle_settings, bodies )

    # Compute and set residuals in the IFMS observation collection
    estimation.compute_residuals_and_dependent_variables(compressed_observations, observation_simulators, bodies)

    ### ------------------------------------------------------------------------------------------
    ### RETRIEVE AND SAVE VARIOUS OBSERVATION OUTPUTS
    ### ------------------------------------------------------------------------------------------

    concatenated_obs = compressed_observations.get_concatenated_observations()
    concatenated_computed_obs = compressed_observations.get_concatenated_computed_observations()

    # Retrieve RMS and mean of the residuals
    concatenated_residuals = compressed_observations.get_concatenated_residuals()
    rms_residuals = compressed_observations.get_rms_residuals()
    mean_residuals = compressed_observations.get_mean_residuals()

    #print(f'Residuals: {concatenated_residuals}')
    print(f'Mean Residuals: {mean_residuals}')
    print(f'RMS Residuals: {rms_residuals}')

    #Populate Station Residuals Dictionary
    site_name = transmitting_station_name
    if site_name not in ifms_station_residuals.keys():
        ifms_station_residuals[site_name] = [(times, utc_times, concatenated_residuals, mean_residuals, rms_residuals)]
    else:
        ifms_station_residuals[site_name].append((times, utc_times, concatenated_residuals, mean_residuals, rms_residuals))

# Output files creation
for site_name, data in ifms_station_residuals.items():
    ifms_residuals_path = '/Users/lgisolfi/Desktop/mex_phobos_flyby/output/ifms_residuals'
    os.makedirs(ifms_residuals_path, exist_ok=True)
    filename = f"{site_name}_residuals.csv"
    file_path = os.path.join(ifms_residuals_path, filename)
    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)

        # Write the header
        writer.writerow([f'# Station: {site_name}'])
        writer.writerow(['# Time | UTC Time | Residuals'])

        # Write the data rows
        for record in data:
            times, utc_times, concatenated_residuals, _, _ = record

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
# Plot residuals for each station
for site_name, data_list in ifms_station_residuals.items():
    if site_name not in label_colors:
        label_colors[site_name] = generate_random_color()

    for times, utc_times, residuals, mean_residuals, rms_residuals in data_list:
        # Plot all stations' residuals on the same figure
        plt.scatter(
            utc_times, residuals,
            color = label_colors[site_name],
            marker = '+', s=10,
            label=f"{site_name}, mean = {mean_residuals}, rms = {rms_residuals}"
            if site_name not in added_labels else None
        )
        added_labels.add(site_name)  # Avoid duplicate labels in the legend

# Format the x-axis for dates
plt.title('IFMS Residuals')
plt.xlabel('Time [s]')
plt.ylabel('Residuals [Hz]')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gcf().autofmt_xdate()  # Auto-rotate date labels for better readability
plt.grid(True)
plt.legend(loc='upper left', bbox_to_anchor=(1.00, 1.0), borderaxespad=0.)
# Adjust layout to make room for the legend
plt.show()
plt.close('all')
exit()
####################################################################################################
##### COMPUTE RESIDUALS BY HAND, INCORPORATING ATMOSPHERIC CORRECTIONS PROVIDED IN IFMS FILES #####
residuals_by_hand =(concatenated_computed_obs - (concatenated_obs - atmospheric_corrections))
#print(f'residuals_array: {abs(residuals_by_hand)}')
print('Residuals by Hand, Atmospheric Corrections')
print(f'rms_residuals: {abs(np.sqrt(np.mean(residuals_by_hand**2)))}')
print(f'mean_residuals: {abs(np.mean(residuals_by_hand))}\n')

# Filtering Residuals ???
filtered_residuals_by_hand = residuals_by_hand[residuals_by_hand < 0.1]
print(f'mean_filtered_residuals: {abs(np.mean(filtered_residuals_by_hand))}\n')
print(f'rms_filtered_residuals: {abs(np.sqrt(np.mean(filtered_residuals_by_hand**2)))}')
####################################################################################################

####################################################################################################
##### COMPUTE RESIDUALS BY HAND, WITHOUT ATMOSPHERIC CORRECTIONS #####
residuals_by_hand_no_atm_corr =(concatenated_computed_obs - concatenated_obs)
#print(f'residuals_array: {abs(residuals_by_hand)}')
print('Residuals by Hand, NO Atmospheric Corrections')
print(f'rms_residuals: {abs(np.sqrt(np.mean(residuals_by_hand_no_atm_corr**2)))}')
print(f'mean_residuals: {abs(np.mean(residuals_by_hand_no_atm_corr))}\n')
####################################################################################################

####################################################################################################
# TUDATPY-PROVIDED RESIDUALS
print('Tudatpy Residuals')
print(f'rms_residuals: {rms_residuals}')
print(f'mean_residuals: {mean_residuals}\n')
####################################################################################################

### SAVING FILES ####
#np.savetxt('mex_unfiltered_residuals_rms' + '.dat',
#           np.vstack(rms_residuals), delimiter=',')
#np.savetxt('mex_unfiltered_residuals_mean' + '.dat',
#           np.vstack(mean_residuals), delimiter=',')
#####################

# Retrieve the observation times list
times = merged_ifms_collection.get_observation_times()
times = [time.to_float() for time in times[0]]
times = np.array(times)
# Residuals Plot
print(residuals_by_hand < 0.1)
plt.scatter(times, residuals_by_hand, s = 6, marker = '+', label = 'Atm. Corr.')
plt.axhline(abs(np.mean(residuals_by_hand)), label = f'mean residuals = {round(abs(np.mean(residuals_by_hand)),6)}', color = 'black', linestyle = '--')
plt.axhline(abs(np.sqrt(np.mean(residuals_by_hand**2))), label = f'rms residuals = {round(abs(np.sqrt(np.mean(residuals_by_hand**2))),6)}', linestyle = '--')
plt.axhline(-abs(np.sqrt(np.mean(residuals_by_hand**2))), linestyle = '--')
plt.legend()
plt.title('Mex Residuals')
plt.xlabel('Time (s)')
plt.ylabel('Residuals (Hz)')
plt.show()