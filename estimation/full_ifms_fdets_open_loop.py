# # Mars Express Residual Analysis: IFMS and FDETS Data Processing
#
# This notebook provides a comprehensive workflow for analyzing Mars Express tracking data residuals using both closed-loop IFMS (Intermediate Frequency and Modem System) and open-loop FDETS (Frequency and Doppler Extraction and Tracking System) observations.
#
# ## Overview
#
# The analysis covers the Mars Express Phobos flyby period (December 26-30, 2013) and includes:
# - **IFMS Analysis**: Closed-loop Doppler tracking from DSN stations (DSS14, DSS63, NWNORCIA)
# - **FDETS Analysis**: Open-loop VLBI observations from multiple ground stations
# - **Combined Visualization**: Comparative residual plots with statistical analysis
#
# ## Table of Contents
#
# 1. [Configuration and Setup](#1-configuration-and-setup)
# 2. [Environment and Body Setup](#2-environment-and-body-setup)
# 3. [IFMS Data Processing](#3-ifms-data-processing)
# 4. [FDETS Data Processing](#4-fdets-data-processing)
# 5. [Results Visualization](#5-results-visualization)
# 6. [Data Export](#6-data-export)
#

# +
## 1. Configuration and Setup

### 1.1 Import Required Libraries

import os
import csv
import numpy as np
from matplotlib import pyplot as plt
from astropy.time import Time
from datetime import datetime
import random
import matplotlib.dates as mdates
from collections import defaultdict
import pandas as pd
import itertools

from pydantic.datetime_parse import time_re
# TudatPy modules
from tudatpy.interface import spice
import tudatpy.data as data
from tudatpy.astro import time_representation, element_conversion
from tudatpy.math import interpolators
from tudatpy.dynamics import environment_setup, environment
import tudatpy.estimation as estimation
from tudatpy.estimation.observations_setup.ancillary_settings import FrequencyBands  # type:ignore
from tudatpy.estimation.observable_models_setup import links
from tudatpy.estimation import observable_models_setup,observable_models, observations_setup, observations, estimation_analysis

from tudatpy import util
import tudatpy.data as data

### 1.2 User Configuration

# =============================================================================
# USER CONFIGURATION - UPDATE THESE PATHS
# =============================================================================

# Main data directory (update this to your base path)
BASE_DIR = '/Users/lgisolfi/Desktop/mex_phobos_flyby'

# Subdirectories
KERNELS_FOLDER = os.path.join(BASE_DIR, 'kernels')
FDETS_FOLDER = os.path.join(BASE_DIR, 'fdets/complete')
IFMS_FOLDER = os.path.join(BASE_DIR, 'ifms/filtered')
ODF_FOLDER = os.path.join(BASE_DIR, 'odf')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Weather and VMF data paths (update these)
WEATHER_DATA_DIR = '/Users/lgisolfi/Desktop/data_archiving-1.0/dataset/mex/gr035/downloaded/met/NWNORCIA_362'
#weather_file_dscc60 = os.path.join(WEATHER_DATA_DIR, 'M60ODF0L1B_MET_130010000_00.TAB') # for complex 60 (e.g. DSS63)
#weather_file_dscc10 = os.path.join(WEATHER_DATA_DIR, 'M10ODF0L1B_MET_130010000_00.TAB') # for complex 10 (e.g. DSS14)
VMF_FILE = '/Users/lgisolfi/Desktop/mex_phobos_flyby/VMF/y2013.vmf3_r.txt'


# Analysis parameters
ANALYSIS_START = datetime(2013, 12, 28, 0,0,0)
ANALYSIS_END = datetime(2013, 12, 31, 0,0,0)
ANALYSIS_TO_AVOID_END_SCAN = datetime(2013, 12, 29, 0,0,0)

SPACECRAFT_NAME = "MEX"
SPACECRAFT_CENTRAL_BODY = "Mars"

# FDETS files to process (modify as needed)
FDETS_FILES_TO_PROCESS = os.listdir(FDETS_FOLDER)

#FDETS_FILES_TO_PROCESS = ['Fdets.mex2013.12.28.Ur.complete.r2i.txt']  # URUMQI ONLY

print("Configuration loaded successfully!")
print(f"Base directory: {BASE_DIR}")
print(f"Analysis period: {ANALYSIS_START.date()} to {ANALYSIS_END.date()}")

# +
### 1.3 Utility Functions

def generate_random_color():
    """Generate a random color in hexadecimal format."""
    return "#{:02x}{:02x}{:02x}".format(
        random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
    )

def ID_to_site(site_ID):
    """Maps a site ID to its corresponding ground station name."""
    id_mapping = {
        'Cd': 'CEDUNA', 'Hb': 'HOBART12', 'Yg': 'YARRA12M', 'Ke': 'KATH12M',
        'Ww': 'WARK', 'Ym': 'YAMAGU32', 'T6': 'TIANMA65', 'Km': 'KUNMING',
        'Ku': 'KVNUS', 'Bd': 'BADARY', 'Ur': 'URUMQI', 'Zc': 'ZELENCHK',
        'Hh': 'HARTRAO', 'Ht': 'HART15M', 'Mh': 'METSAHOV', 'Wz': 'WETTZELL',
        'Sv': 'SVETLOE', 'Mc': 'MEDICINA', 'Wb': 'WSTRBORK', 'On': 'ONSALA60',
        'Ys': 'YEBES40M', 'Sc': 'SC-VLBA', 'Hn': 'HN-VLBA', 'Nl': 'NL-VLBA',
        'Fd': 'FD-VLBA', 'La': 'LA-VLBA', 'Kp': 'KP-VLBA', 'Br': 'BR-VLBA',
        'Ov': 'OV-VLBA', 'Mk': 'MK-VLBA', 'Pt': 'PIETOWN',
    }
    return id_mapping.get(site_ID, None)

def get_fdets_receiving_station_name(fdets_file):
    """Extract receiving station name from FDETS filename."""
    site_id = os.path.basename(fdets_file).split('.')[4]
    return ID_to_site(site_id)

import os
import pandas as pd
import matplotlib.pyplot as plt
import itertools

def plot_weather_and_ifms_coverage(weather_dict, ifms_dict, offset=0.3):
    """
    Plot coverage intervals (col3 = TDB seconds) for weather files and IFMS files,
    shown on slightly different y-positions for the same station.

    - Weather coverage: thick solid lines (shifted down)
    - IFMS coverage: thin dashed lines (shifted up)
    - Filenames are shown next to each coverage bar

    Parameters
    ----------
    weather_dict : dict
        Dict mapping station codes -> list of weather files
    ifms_dict : dict
        Dict mapping station codes -> list of IFMS files
    offset : float
        Vertical offset between weather and IFMS bars for the same station
    """

    plt.figure(figsize=(14, 7))
    colors = itertools.cycle(plt.cm.tab10.colors)

    for i, station_code in enumerate(sorted(set(weather_dict.keys()) | set(ifms_dict.keys()))):
        color = next(colors)

        # --- Weather coverage (solid lines, shifted down) ---
        for f in weather_dict.get(station_code, []):
            if not os.path.exists(f):
                continue
            df = pd.read_csv(f, delim_whitespace=True, header=None)
            start, end = df[3].min(), df[3].max()

            # Plot coverage
            plt.hlines(y=i - offset, xmin=start, xmax=end,
                       linewidth=3, color=color)

            # Add filename text (just the basename)
            #plt.text(end, i - offset, os.path.basename(f),
            #         va="center", ha="left", fontsize=8, color=color)

        # --- IFMS coverage (dashed lines, shifted up) ---
        for f in ifms_dict.get(station_code, []):
            if not os.path.exists(f):
                continue
            df = pd.read_csv(f, delim_whitespace=True, header=None)
            start, end = df[3].min(), df[3].max()

            # Plot coverage
            plt.hlines(y=i + offset, xmin=start, xmax=end,
                       linewidth=1.2, color=color, linestyle="--", alpha=0.7)

            # Add filename text
            plt.text(end, i + offset, os.path.basename(f),
                     va="center", ha="left", fontsize=8, color=color)

        # Legend entry once per station
        plt.plot([], [], color=color, label=station_code)

    plt.xlabel("TDB seconds (col3)")
    plt.ylabel("Station code")
    plt.yticks(range(len(weather_dict)), sorted(weather_dict.keys()))
    plt.title("Weather & IFMS Coverage per Station")
    plt.legend(title="Stations")
    plt.grid(True, axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


print("Utility functions loaded!")
# -

# ## 2. Environment and Body Setup
#
# ### 2.1 Load SPICE Kernels

# +
print("Loading SPICE kernels...")

# Load standard kernels
spice.load_standard_kernels()

# Load mission-specific kernels
kernel_count = 0
for kernel in os.listdir(KERNELS_FOLDER):
    if not kernel.startswith('.'):  # Skip hidden files
        kernel_path = os.path.join(KERNELS_FOLDER, kernel)
        spice.load_kernel(kernel_path)
        kernel_count += 1


print(f"Loaded {kernel_count} mission-specific kernels")
# -

# ### 2.2 Define Time Window

# +
# Add buffer time for interpolation
start_time_seconds = time_representation.DateTime.to_epoch(time_representation.DateTime.from_python_datetime(ANALYSIS_START))
end_time_seconds = time_representation.DateTime.to_epoch(time_representation.DateTime.from_python_datetime(ANALYSIS_END))
start_time = time_representation.Time(time_representation.DateTime.to_epoch(time_representation.DateTime.from_python_datetime(ANALYSIS_START)))
end_time = time_representation.Time(time_representation.DateTime.to_epoch(time_representation.DateTime.from_python_datetime(ANALYSIS_END)))
end_time_to_avoid_end_scan = time_representation.Time(time_representation.DateTime.to_epoch(time_representation.DateTime.from_python_datetime(ANALYSIS_TO_AVOID_END_SCAN)))

print(f"Simulation time window:")
print(f"Start: {start_time_seconds} seconds since J2000")
print(f"End: {end_time_seconds} seconds since J2000")

# +
### 2.3 Create Body Settings and System

print("Setting up celestial body environment...")

# Define celestial bodies to include
bodies_to_create = ["Earth", "Sun", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Moon"]
global_frame_origin = "SSB"
global_frame_orientation = "J2000"

# Get default body settings
body_settings = environment_setup.get_default_body_settings_time_limited(
    bodies_to_create, start_time, end_time, global_frame_origin, global_frame_orientation
)

# Configure Earth with detailed rotation model
body_settings.get('Earth').shape_settings = environment_setup.shape.oblate_spherical_spice()
body_settings.get('Earth').rotation_model_settings = environment_setup.rotation_model.gcrs_to_itrs(
    environment_setup.rotation_model.iau_2006, global_frame_orientation,
    interpolators.interpolator_generation_settings(
        interpolators.cubic_spline_interpolation(), start_time, end_time, 3600.0),
    interpolators.interpolator_generation_settings(
        interpolators.cubic_spline_interpolation(), start_time, end_time, 3600.0),
    interpolators.interpolator_generation_settings(
        interpolators.cubic_spline_interpolation(), start_time, end_time, 60.0)
)
body_settings.get('Earth').gravity_field_settings.associated_reference_frame = "ITRS"

# Configure spacecraft
body_settings.add_empty_settings(SPACECRAFT_NAME)
body_settings.get(SPACECRAFT_NAME).ephemeris_settings = environment_setup.ephemeris.interpolated_spice(
    start_time, end_time, 10.0, SPACECRAFT_CENTRAL_BODY, global_frame_orientation
)
body_settings.get(SPACECRAFT_NAME).rotation_model_settings = environment_setup.rotation_model.spice(
    global_frame_orientation, SPACECRAFT_NAME + "_SPACECRAFT", ""
)

# Add ground stations
body_settings.get("Earth").ground_station_settings = environment_setup.ground_station.radio_telescope_stations()

# Create system of bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

print("Body system created successfully!")

### 2.4 Configure Atmospheric Corrections and Spacecraft Systems


print("Configuring atmospheric corrections and spacecraft systems...")

import os
from collections import defaultdict

def get_weather_files_by_station(weather_data_dir):
    """
    Retrieve weather files grouped by station codes.

    Parameters
    ----------
    weather_data_dir : str
        Path to the directory containing weather data files.

    Returns
    -------
    dict
        Dictionary where keys are station codes (e.g., '14', '63', '32')
        and values are lists of weather files corresponding to that station.
    """
    if not os.path.exists(weather_data_dir):
        print(f"Warning: Weather data directory not found at {weather_data_dir}")
        return {}

    station_files = defaultdict(list)

    for f in os.listdir(weather_data_dir):
        if f.startswith('.'):  # skip hidden/system files
            continue

        filename = os.path.basename(f)
        if len(filename) <= 3:
            continue

        station_code = filename[1:3]  # extract station code

        full_path = os.path.join(weather_data_dir, f)

        if station_code == '60':
            station_files['63'].append(full_path) # hardcoded for now
        elif station_code == '10':
            station_files['14'].append(full_path) # hardcoded for now
        elif station_code == '32':
            station_files['32'].append(full_path) # hardcoded for now

    return dict(station_files)

def get_ifms_files_by_station(ifms_data_dir):
    """
    Retrieve weather files grouped by station codes.

    Parameters
    ----------
    weather_data_dir : str
        Path to the directory containing weather data files.

    Returns
    -------
    dict
        Dictionary where keys are station codes (e.g., '14', '63', '32')
        and values are lists of weather files corresponding to that station.
    """
    if not os.path.exists(ifms_data_dir):
        print(f"Warning: Weather data directory not found at {ifms_data_dir}")
        return {}

    station_files = defaultdict(list)

    for f in os.listdir(ifms_data_dir):
        if f.startswith('.'):  # skip hidden/system files
            continue

        filename = os.path.basename(f)
        if len(filename) <= 3:
            continue

        station_code = filename[1:3]  # extract station code

        full_path = os.path.join(ifms_data_dir, f)

        if station_code == '63':
            station_files['63'].append(full_path) # hardcoded for now
        elif station_code == '14':
            station_files['14'].append(full_path) # hardcoded for now
        elif station_code == '32':
            station_files['32'].append(full_path) # hardcoded for now

    return dict(station_files)

# Load weather data for meteorological corrections
weather_dict = get_weather_files_by_station(WEATHER_DATA_DIR)
ifms_dict = get_ifms_files_by_station(IFMS_FOLDER)

# Configure spacecraft transponder
vehicle_sys = environment.VehicleSystems()
vehicle_sys.set_default_transponder_turnaround_ratio_function()
bodies.get_body(SPACECRAFT_NAME).system_models = vehicle_sys

print("Configuration complete!")
print(f"Available ground stations: {len(environment_setup.get_ground_station_list(bodies.get_body('Earth')))}")
# -

# ## 3. IFMS Data Processing
#
# ### 3.1 Process IFMS Files

# +
print("Processing IFMS data...")

# Initialize storage for IFMS results
ifms_station_residuals = {}

reception_band = FrequencyBands.x_band
transmission_band = FrequencyBands.x_band

# Get list of IFMS files
ifms_files = []
if os.path.exists(IFMS_FOLDER):
    for file in os.listdir(IFMS_FOLDER):
        if not file.startswith('.'):
            ifms_files.append(os.path.join(IFMS_FOLDER, file))

print(f"Found {len(ifms_files)} IFMS files")


# Set up antenna ephemeris
com_position = [-1.3, 0.0, 0.0]  # MEX center of mass offset
antenna_state = np.zeros((6, 1))
antenna_state[:3, 0] = spice.get_body_cartesian_position_at_epoch(
    "-41020", "-41000", "MEX_SPACECRAFT", "none", start_time
)
antenna_state[:3, 0] = antenna_state[:3, 0] - com_position

antenna_ephemeris_settings = environment_setup.ephemeris.constant(
    antenna_state, "-41000", "MEX_SPACECRAFT"
)
antenna_ephemeris = environment_setup.ephemeris.create_ephemeris(
    antenna_ephemeris_settings, "Antenna"
)

time_scale_converter = time_representation.default_time_scale_converter()
# Process each IFMS file
processed_count = 0
for ifms_file in ifms_files:
    filename = os.path.basename(ifms_file)
    
    # Extract station code from filename
    if len(filename) > 3:
        station_code = filename[1:3]
    else:
        continue
        
    # Map station codes to names
    if station_code == '14':
        transmitting_station_name = 'DSS14'
    elif station_code == '63':
        transmitting_station_name = 'DSS63'
    elif station_code == '32':
        transmitting_station_name = 'NWNORCIA'
    else:
        continue

    print(f"Processing {filename} -> {transmitting_station_name}")

    #print(weather_dict)
    weather_files = weather_dict[station_code]
    #print(ifms_dict)
    #ifms_files = ifms_dict[station_code]
    #plot_weather_and_ifms_coverage({'NWNORCIA': weather_files}, {'NWNORCIA': ifms_files})
    #print(station_code, weather_files)
    data.set_estrack_weather_data_in_ground_stations(bodies, weather_files, transmitting_station_name)


    # Load IFMS observations
    ifms_collection = observations_setup.observations_wrapper.observations_from_ifms_files(
        [ifms_file], bodies, SPACECRAFT_NAME, transmitting_station_name,
        reception_band, transmission_band, apply_troposphere_correction=False
    )

    time_filter = observations.observations_processing.observation_filter(
        observations.observations_processing.ObservationFilterType.time_bounds_filtering, start_time, end_time,
        use_opposite_condition=True
    )

    len_unfiltered_collection = len(ifms_collection.get_concatenated_observations())
    ifms_collection.filter_observations(time_filter)

    len_filtered_collection = len(ifms_collection.get_concatenated_observations())
    if len_filtered_collection == 0:
        continue

    ifms_collection.set_reference_point(
        bodies, antenna_ephemeris, "Antenna", "MEX", links.retransmitter
    )

    # Configure observation models
    light_time_corrections = [
        observable_models_setup.light_time_corrections.first_order_relativistic_light_time_correction(["Sun"]),
        observable_models_setup.light_time_corrections.saastamoinen_tropospheric_light_time_correction()
    ]

    doppler_link_ends = ifms_collection.link_definitions_per_observable[
        observable_models_setup.model_settings.dsn_n_way_averaged_doppler_type
    ]

    observation_model_settings = []
    for link_definition in doppler_link_ends:
        observation_model_settings.append(
            observable_models_setup.model_settings.dsn_n_way_doppler_averaged(
                link_definition, light_time_corrections, subtract_doppler_signature=False
            )
        )

    # Create simulators and compute residuals
    observation_simulators = observations_setup.observations_simulation_settings.create_observation_simulators(
        observation_model_settings, bodies
    )

    # Compute residuals
    observations.compute_residuals_and_dependent_variables(
        ifms_collection, observation_simulators, bodies
    )

    residual_filter = observations.observations_processing.observation_filter(observations.observations_processing.ObservationFilterType.residual_filtering, 0.1)
    ifms_collection.filter_observations(residual_filter)
    ifms_collection.remove_empty_observation_sets()

    times = ifms_collection.get_concatenated_observation_times() # in tdb
    # Convert time to UTC datetime
    times_utc = [time_scale_converter.convert_time(
        input_scale=time_representation.tdb_scale,
        output_scale=time_representation.utc_scale,
        input_value=t) for t in times]
    utc_times = [time_representation.DateTime.to_python_datetime(time_representation.DateTime.from_epoch(t)) for t in times_utc]

    # Extract results
    residuals = ifms_collection.get_concatenated_residuals()

    mean_residuals = ifms_collection.get_mean_residuals()
    rms_residuals = ifms_collection.get_rms_residuals()

    # Store results
    if transmitting_station_name not in ifms_station_residuals:
        ifms_station_residuals[transmitting_station_name] = []


    ifms_station_residuals[transmitting_station_name].append({
        'times': times,
        'utc_times': utc_times,
        'residuals': residuals,
        'mean': mean_residuals,
        'rms': rms_residuals
    })

    processed_count += 1

print(f"Successfully processed {processed_count} IFMS files")

# -

# ## 4. FDETS Data Processing
#
# ### 4.1 FDETS Processing Functions

# +
def get_filtered_fdets_collection(fdets_file, ifms_files, receiving_station_name):
    """
    Process FDETS file and filter observations based on IFMS time windows.
    
    Parameters:
    - fdets_file: Path to FDETS file
    - ifms_files: List of IFMS files for time filtering
    - receiving_station_name: Name of receiving station
    
    Returns:
    - Tuple of (collections_list, transmitting_stations_list, start_time, end_time)
    """
    base_frequency = 8412e6
    column_types = [
        "utc_datetime_string", "signal_to_noise_ratio", "normalised_spectral_max",
        "doppler_measured_frequency_hz", "doppler_noise_hz"
    ]
    reception_band = FrequencyBands.x_band
    transmission_band = FrequencyBands.x_band
    
    collections_list = []
    stations_list = []
    
    for ifms_file in ifms_files:
        if os.path.basename(ifms_file).startswith('.'):
            continue
            
        # Extract transmitting station from IFMS filename
        filename = os.path.basename(ifms_file)
        if len(filename) > 3:
            station_code = filename[1:3]
        else:
            continue
            
        if station_code == '14':
            transmitting_station_name = 'DSS14'
        elif station_code == '63':
            transmitting_station_name = 'DSS63'
        elif station_code == '32':
            transmitting_station_name = 'NWNORCIA'
        else:
            continue

        # Load IFMS observations
        ifms_collection = observations_setup.observations_wrapper.observations_from_ifms_files(
            [ifms_file], bodies, SPACECRAFT_NAME, transmitting_station_name,
            reception_band, transmission_band
        )

        ifms_times = ifms_collection.get_observation_times()
        flat_times = [time for sublist in ifms_times for time in sublist]
        start_ifms_time = min(flat_times)
        end_ifms_time = max(flat_times)

        # Convert to UTC for output
        start_mjd = time_representation.seconds_since_epoch_to_julian_day(start_ifms_time)
        end_mjd = time_representation.seconds_since_epoch_to_julian_day(end_ifms_time)
        start_utc = Time(start_mjd, format='jd', scale='utc').datetime
        end_utc = Time(end_mjd, format='jd', scale='utc').datetime

        # Load FDETS observations
        fdets_collection = observations_setup.observations_wrapper.observations_from_fdets_files(
            fdets_file, base_frequency, column_types, SPACECRAFT_NAME,
            transmitting_station_name, receiving_station_name, reception_band, transmission_band
        )

        # Apply time filtering
        time_filter = observations.observations_processing.observation_filter(
            observations.observations_processing.ObservationFilterType.time_bounds_filtering, start_ifms_time, end_ifms_time,
            use_opposite_condition=True
        )


        fdets_collection.filter_observations(time_filter)
        fdets_collection.filter_observations(residual_filter)

        # Check if observations remain after filtering
        if len(fdets_collection.get_observation_times()[0]) > 1:
            collections_list.append(fdets_collection)
            stations_list.append(transmitting_station_name)


    return collections_list, stations_list, start_utc, end_utc

print("FDETS processing functions defined!")
# -

# ### 4.2 Process FDETS Files

# +
print("Processing FDETS data...")

# Initialize storage for FDETS results  
fdets_station_residuals = {}

# Get IFMS files for time filtering
ifms_files = []
if os.path.exists(IFMS_FOLDER):
    for file in os.listdir(IFMS_FOLDER):
        if not file.startswith('.'):
            ifms_files.append(os.path.join(IFMS_FOLDER, file))

processed_fdets_count = 0

for fdets_filename in FDETS_FILES_TO_PROCESS:
    fdets_file = os.path.join(FDETS_FOLDER, fdets_filename)
    
    if not os.path.exists(fdets_file):
        print(f"Warning: FDETS file not found: {fdets_file}")
        continue
    
    # Get receiving station name
    site_name = get_fdets_receiving_station_name(fdets_file)
    if site_name is None:
        print(f"Could not determine station name for {fdets_filename}")
        continue
        
    # Check if station is available in ground station list
    available_stations = [station[1] for station in environment_setup.get_ground_station_list(bodies.get_body("Earth"))]
    if site_name not in available_stations:
        print(f"Station {site_name} not available in ground station list")
        continue
    
    print(f"Processing FDETS file: {fdets_filename} -> {site_name}")

    # Get filtered collections
    filtered_collections, transmitting_stations, start_time, end_time = get_filtered_fdets_collection(
        fdets_file, ifms_files, site_name
    )

    for filtered_collection, transmitting_station in zip(filtered_collections, transmitting_stations):
        print(f"  Processing link: {transmitting_station} -> {site_name}")

        filtered_collection.set_reference_point(
            bodies, antenna_ephemeris, "Antenna", "MEX", links.reflector1
        )

        # Define link ends
        link_ends = {
            links.receiver: links.body_reference_point_link_end_id('Earth', site_name),
            links.retransmitter: links.body_reference_point_link_end_id('MEX', 'Antenna'),
            links.transmitter: links.body_reference_point_link_end_id('Earth', transmitting_station),
        }
        link_definition = links.LinkDefinition(link_ends)

        observable_models_setup.light_time_corrections.set_vmf_troposphere_data(
            [ "/Users/lgisolfi/Desktop/mex_phobos_flyby/VMF/y2013.vmf3_r.txt" ], True, False, bodies, False, True )

        # Configure light time corrections
        light_time_corrections = [
            observable_models_setup.light_time_corrections.first_order_relativistic_light_time_correction(["Sun"]),
            observable_models_setup.light_time_corrections.saastamoinen_tropospheric_light_time_correction(),
        ]


    # Define observation model
        observation_model_settings = [
            observable_models_setup.model_settings.doppler_measured_frequency(
                link_definition, light_time_corrections
            )
        ]

        # Create simulators
        observation_simulators = observations_setup.observations_simulation_settings.create_observation_simulators(
            observation_model_settings, bodies
        )

        # Compute residuals
        observations.compute_residuals_and_dependent_variables(
            filtered_collection, observation_simulators, bodies
        )


        residual_filter = observations.observations_processing.observation_filter(observations.observations_processing.ObservationFilterType.residual_filtering, 0.1)
        filtered_collection.filter_observations(residual_filter)
        filtered_collection.remove_empty_observation_sets()

        times = filtered_collection.get_concatenated_observation_times() # in tdb
        # Convert time to UTC datetime
        times_utc = [time_scale_converter.convert_time(
            input_scale=time_representation.tdb_scale,
            output_scale=time_representation.utc_scale,
            input_value=t) for t in times]
        utc_times = [time_representation.DateTime.to_python_datetime(time_representation.DateTime.from_epoch(t)) for t in times_utc]

        # Extract results
        residuals = filtered_collection.get_concatenated_residuals()
        mean_residuals = filtered_collection.get_mean_residuals()
        rms_residuals = filtered_collection.get_rms_residuals()

        print(f"    Mean residuals (Hz): {mean_residuals}")
        print(f"    RMS residuals (Hz): {rms_residuals}")

        # Store results
        if site_name not in fdets_station_residuals:
            fdets_station_residuals[site_name] = []

        fdets_station_residuals[site_name].append({
            'times': times,
            'utc_times': utc_times,
            'residuals': residuals,
            'mean': mean_residuals,
            'rms': rms_residuals,
            'start_time': start_time,
            'end_time': end_time,
            'transmitting_station': transmitting_station
        })

    processed_fdets_count += 1


print(f"Successfully processed {processed_fdets_count} FDETS files")
# -

# ## 5. Results Visualization
#
# ### 5.1 Combined IFMS and FDETS Residuals Plot

# +
print("Creating combined residuals visualization...")

plt.figure(figsize=(14, 8))

# Color scheme for consistency
station_colors = {
    'NWNORCIA': 'lightgray',
    'HART15M': 'orchid', 
    'URUMQI': 'royalblue',
    'DSS14': 'red',
    'DSS63': 'green',
    'ONSALA60': 'orange'
}

added_labels = set()
all_residuals = []

# Plot IFMS residuals
for station_name, data_list in ifms_station_residuals.items():
    color = station_colors.get(station_name, generate_random_color())
    
    for data in data_list:
        utc_times = data['utc_times']
        residuals = data['residuals']
        mean_res = data['mean']
        rms_res = data['rms']

        all_residuals.extend(residuals)
        
        plt.scatter(
            utc_times, residuals,
            color=color, marker='o', s=15, alpha=0.7,
            label=f"{station_name} (IFMS) - RMS: {np.mean(rms_res)*1000:.2f} mHz"
                  if f"{station_name}_IFMS" not in added_labels else None
        )
        added_labels.add(f"{station_name}_IFMS")

# Plot FDETS residuals
for station_name, data_list in fdets_station_residuals.items():
    color = station_colors.get(station_name, generate_random_color())
    
    for data in data_list:
        utc_times = data['utc_times']
        residuals = data['residuals']
        mean_res = data['mean']
        rms_res = data['rms']

        
        plt.scatter(
            utc_times, residuals,
            color=color, marker='+', s=20, alpha=0.8,
            label=f"{station_name} (FDETS) - RMS: {np.mean(rms_res)*1000:.2f} mHz"
                  if f"{station_name}_FDETS" not in added_labels else None
        )
        added_labels.add(f"{station_name}_FDETS")

# Calculate overall statistics
overall_rms = np.sqrt(np.mean(np.array(all_residuals)**2))

# Format plot
plt.xlabel("UTC Time", fontsize=14)
plt.ylabel("Residuals (Hz)", fontsize=14)
plt.title(f"Mars Express Residual Analysis - Overall RMS: {overall_rms*1000:.2f} mHz", fontsize=16)
plt.grid(True, linestyle="--", alpha=0.3)

# Format time axis
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=6))
plt.gcf().autofmt_xdate()

# Add legend
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
# -

# ### 5.2 Subset Analysis (Specific Time Window) with Distribution Histogram

# Define subset time window for detailed analysis
subset_start = datetime(2013, 12, 28, 0, 0, 0)
subset_end = ANALYSIS_TO_AVOID_END_SCAN

print(f"Creating subset analysis for {subset_start.date()} to {subset_end.date()}")

# Create figure with two subplots: main plot and histogram
fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [4, 1]})
ax_main, ax_hist = axes

added_labels = set()
subset_residuals_all = []

# Combine data from both IFMS and FDETS for subset analysis
combined_data = {}

# Add IFMS data
for station_name, data_list in ifms_station_residuals.items():
    if station_name not in combined_data:
        combined_data[station_name] = {'IFMS': [], 'FDETS': []}
    combined_data[station_name]['IFMS'].extend(data_list)

# Add FDETS data
for station_name, data_list in fdets_station_residuals.items():
    if station_name not in combined_data:
        combined_data[station_name] = {'IFMS': [], 'FDETS': []}
    combined_data[station_name]['FDETS'].extend(data_list)

# Plot subset data
for station_name, data_types in combined_data.items():
    color = station_colors.get(station_name, generate_random_color())

    # Collect all residuals for this station for histogram
    station_residuals_for_hist = []

    for data_type, data_list in data_types.items():
        if not data_list:
            continue

        for data in data_list:
            utc_times_arr = np.array(data['utc_times'])
            residuals_arr = np.array(data['residuals'])

            # Apply subset time filter
            time_mask = (utc_times_arr >= subset_start) & (utc_times_arr <= subset_end)

            if data_type == 'FDETS':
                # Additional outlier filter for FDETS
                outlier_mask = np.abs(residuals_arr) <= 50
                time_mask = time_mask & outlier_mask

            subset_times = utc_times_arr[time_mask]
            subset_residuals = residuals_arr[time_mask]

            if len(subset_residuals) == 0:
                continue

            subset_residuals_all.extend(subset_residuals)
            station_residuals_for_hist.extend(subset_residuals)
            subset_rms = np.sqrt(np.mean(subset_residuals**2))

            marker = 'o' if data_type == 'IFMS' else '+'
            size = 15 if data_type == 'IFMS' else 20
            alpha = 0.7 if data_type == 'IFMS' else 0.8

            label_key = f"{station_name}_{data_type}"
            label = f"{station_name} ({data_type}) - RMS: {subset_rms*1000:.2f} mHz" if label_key not in added_labels else None

            ax_main.scatter(
                subset_times, subset_residuals,
                color=color, marker=marker, s=size, alpha=alpha,
                label=label
            )
            added_labels.add(label_key)

    # Create histogram for this station (combine all data for this station)
    if station_residuals_for_hist:
        station_residuals_combined = np.array(station_residuals_for_hist)
        ax_hist.hist(
            station_residuals_combined, bins=30,
            orientation='horizontal', alpha=0.6,
            color=color
        )

# Calculate subset statistics
if subset_residuals_all:
    subset_overall_rms = np.sqrt(np.mean(np.array(subset_residuals_all)**2))
else:
    subset_overall_rms = 0

# Format main plot
ax_main.set_xlabel("UTC Time", fontsize=15)
ax_main.set_ylabel("Residuals (Hz)", fontsize=15)
ax_main.set_title(f"Overall RMS: {subset_overall_rms*1000:.2f} mHz", fontsize=15)
ax_main.grid(True, linestyle="--", alpha=0.5)
ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
ax_main.xaxis.set_major_locator(mdates.AutoDateLocator())
ax_main.tick_params(axis='x', labelsize=11)
ax_main.tick_params(axis='y', labelsize=11)

# Format histogram
ax_hist.set_xlabel("Count", fontsize=15)
ax_hist.grid(True, linestyle='--', alpha=0.3)
ax_hist.tick_params(left=False, labelleft=False)

# Format x-axis dates for main plot
fig.autofmt_xdate()

# Add legend
ax_main.legend(fontsize=12)
plt.tight_layout()
plt.show()

# ### 5.3 Statistical Summary

# +
print("=== RESIDUAL ANALYSIS SUMMARY ===\n")

# IFMS Statistics
print("IFMS (Closed-Loop) Results:")
print("-" * 50)
ifms_stats = []
for station, data_list in ifms_station_residuals.items():
    for data in data_list:
        rms = data['rms']
        mean = data['mean']
        n_obs = len(data['residuals'])
        ifms_stats.append({'station': station, 'rms': rms, 'mean': mean, 'n_obs': n_obs})
        print(f"{station:>12}: RMS = {np.mean(rms)*1000:.2f} mHz, Mean = {np.mean(mean)*1000:.2f} mHz, N = {n_obs:4d}")

if ifms_stats:
    print(ifms_stats)
    # Extract scalar RMS values from your dicts
    rms_values = [float(s['rms'][0]) for s in ifms_stats]

    # Compute average RMS
    overall_ifms_rms = np.mean(rms_values)

    print("Per-station RMS values:", rms_values)
    print("Overall average RMS:", overall_ifms_rms)

print("\nFDETS (Open-Loop) Results:")
print("-" * 50)
fdets_stats = []
for station, data_list in fdets_station_residuals.items():
    for data in data_list:
        # Filter outliers for statistics
        residuals = np.array(data['residuals'])
        mask = np.abs(residuals) <= 50
        filtered_residuals = residuals[mask]
        
        if len(filtered_residuals) > 0:
            rms = np.sqrt(np.mean(filtered_residuals**2))
            mean = np.mean(filtered_residuals)
            n_obs = len(filtered_residuals)
            fdets_stats.append({'station': station, 'rms': rms, 'mean': mean, 'n_obs': n_obs})
            print(f"{station:>12}: RMS = {np.mean(rms)*1000:.2f} mHz, Mean = {np.mean(mean)*1000:.2f} mHz, N = {n_obs:4d}")

if fdets_stats:
    overall_fdets_rms = np.sqrt(np.mean([s['rms']**2 for s in fdets_stats]))
    print(f"{'Overall FDETS':>12}: RMS = {overall_fdets_rms*1000:6.2f} mHz")

# ## 6. Data Export
#
# ### 6.1 Export Residuals to CSV Files

# +
print("Exporting results to CSV files...")

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
ifms_output_dir = os.path.join(OUTPUT_DIR, 'ifms_residuals')
fdets_output_dir = os.path.join(OUTPUT_DIR, 'fdets_residuals')
os.makedirs(ifms_output_dir, exist_ok=True)
os.makedirs(fdets_output_dir, exist_ok=True)

# Export IFMS residuals
ifms_files_created = 0
for station_name, data_list in ifms_station_residuals.items():
    filename = f"{station_name}_residuals_newtest.csv"
    file_path = os.path.join(ifms_output_dir, filename)
    
    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        
        # Write header
        writer.writerow([f'# Station: {station_name}'])
        writer.writerow(['# Data Type: IFMS (Closed-Loop)'])
        writer.writerow(['# Time (seconds since J2000)', 'UTC Time', 'Residuals (Hz)'])
        
        # Write data
        for data in data_list:
            times = data['times']
            utc_times = data['utc_times'] 
            residuals = data['residuals']
            
            for time, utc_time, residual in zip(times, utc_times, residuals):
                writer.writerow([
                    f"{time:.6f}",
                    utc_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],  # millisecond precision
                    f"{residual:.8e}"
                ])
    
    print(f"Created IFMS file: {file_path}")
    ifms_files_created += 1

# Export FDETS residuals
fdets_files_created = 0
for station_name, data_list in fdets_station_residuals.items():
    filename = f"{station_name}_residuals_newtest.csv"
    file_path = os.path.join(fdets_output_dir, filename)
    
    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        
        # Write header
        writer.writerow([f'# Station: {station_name}'])
        writer.writerow(['# Data Type: FDETS (Open-Loop)'])
        writer.writerow(['# Time (seconds since J2000)', 'UTC Time', 'Residuals (Hz)', 'Transmitting Station'])
        
        # Write data
        for data in data_list:
            times = data['times']
            utc_times = data['utc_times']
            residuals = data['residuals']
            transmitting_station = data['transmitting_station']
            
            for time, utc_time, residual in zip(times, utc_times, residuals):
                writer.writerow([
                    f"{time:.6f}",
                    utc_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    f"{residual:.8e}",
                    transmitting_station
                ])
    
    print(f"Created FDETS file: {file_path}")
    fdets_files_created += 1

print(f"\nExport complete!")
print(f"IFMS files created: {ifms_files_created}")
print(f"FDETS files created: {fdets_files_created}")
print(f"Output directory: {OUTPUT_DIR}")