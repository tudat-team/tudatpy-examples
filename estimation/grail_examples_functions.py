# This file contains a few functions jointly used by several GRAIL estimation examples.
"""
Copyright (c) 2010-2022, Delft University of Technology. All rights reserved. This file is part of the Tudat. Redistribution and use in source and binary forms, with or without modification, are permitted exclusively under the terms of the Modified BSD license. You should have received a copy of the license with this file. If not, please or visit: http://tudat.tudelft.nl/LICENSE.
"""

# Load required standard modules
import os
import numpy as np
import pandas as pd

# Load required tudatpy modules
from tudatpy.interface import spice
from tudatpy.astro import frame_conversion
from tudatpy.numerical_simulation import environment_setup

from load_pds_files import download_url_files_time, download_url_files_time_interval
from urllib.request import urlretrieve


# This function retrieves all relevant files necessary to run the example over the time interval of interest
# (and automatically downloads them if they cannot be found locally). It returns a tuple containing the lists of
# relevant clock file, orientation kernels, tropospheric correction files, ionospheric correction files, manoeuvre file,
# antenna switch files and odf files that should be loaded.
def get_grail_files(local_path, start_date, end_date):

    # Check if local_path designates an existing directory and creates the directory is not
    if not os.path.isdir(local_path):
        os.mkdir(local_path)


    # Clock file
    print('---------------------------------------------')
    print('Download GRAIL clock file')
    clock_file = "gra_sclkscet_00014.tsc"
    # Define url where clock files can be downloaded for GRAIL
    url_clock_files = "https://naif.jpl.nasa.gov/pub/naif/pds/data/grail-l-spice-6-v1.0/grlsp_1000/data/sclk/"
    # Check if the GRAIL clock file exists locally
    if (os.path.exists(local_path + clock_file) == False):
        print('download', local_path + clock_file)
        # If not, download it
        urlretrieve(url_clock_files + clock_file, local_path + clock_file)

    # Add local path to clock file name
    clock_file = local_path + clock_file

    # Print and store all relevant clock file names
    print('relevant clock files')
    print(clock_file)


    # Orientation files (multiple orientation kernels are required, typically covering intervals of a few days)
    print('---------------------------------------------')
    print('Download GRAIL orientation kernels')
    # Define url where orientation kernels can be downloaded for GRAIL
    url_orientation_files = "https://naif.jpl.nasa.gov/pub/naif/pds/data/grail-l-spice-6-v1.0/grlsp_1000/data/ck/"
    # Retrieve the names of all spacecraft orientation kernels required to cover the time interval of interest, and download them if they
    # do not exist locally yet.
    grail_orientation_files = download_url_files_time_interval(
        local_path=local_path, filename_format='gra_rec_*.bc', start_date=start_date, end_date=end_date,
        url=url_orientation_files, time_interval_format='%y%m%d_%y%m%d')

    # Print and store all relevant orientation file names
    print('relevant orientation files')
    for f in grail_orientation_files:
        print(f)


    # Tropospheric corrections (multiple tropospheric correction files are required, typically covering intervals of a few days)
    print('---------------------------------------------')
    print('Download GRAIL tropospheric corrections files')
    # Define url where tropospheric correction files can be downloaded for GRAIL
    url_tro_files = "https://pds-geosciences.wustl.edu/grail/grail-l-rss-2-edr-v1/grail_0201/ancillary/tro/"
    # Retrieve the names of all tropospheric correction files required to cover the time interval of interest, and download them if they
    # do not exist locally yet
    tro_files = download_url_files_time_interval(
        local_path=local_path, filename_format='grxlugf*.tro', start_date=start_date,
        end_date=end_date, url=url_tro_files, time_interval_format='%Y_%j_%Y_%j')

    # Print all relevant tropospheric correction file names
    print('relevant tropospheric corrections files')
    for f in tro_files:
        print(f)


    # Ionospheric corrections (multiple ionospheric correction files are required, typically covering intervals of a few days)
    print('---------------------------------------------')
    print('Download GRAIL ionospheric corrections files')
    # Define url where ionospheric correction files can be downloaded for GRAIL
    url_ion_files = "https://pds-geosciences.wustl.edu/grail/grail-l-rss-2-edr-v1/grail_0201/ancillary/ion/"
    # Retrieve the names of all ionospheric correction files required to cover the time interval of interest, and download them if they
    # do not exist locally yet
    ion_files = download_url_files_time_interval(
        local_path=local_path, filename_format='gralugf*.ion', start_date=start_date,
        end_date=end_date, url=url_ion_files, time_interval_format='%Y_%j_%Y_%j')

    # Print all relevant ionospheric correction file names
    print('relevant ionospheric corrections files')
    for f in ion_files:
        print(f)


    # Manoeuvres file (a single file is necessary, as this file is identical for all dates)
    print('---------------------------------------------')
    print('Download GRAIL manoeuvres file')
    # Define filename and url where a manoeuvre file can be downloaded for GRAIL (the specific date is here arbitrarily chosen
    # since the manoeuvre files are identical for all dates and cover the full mission time span)
    manoeuvres_file = "mas00_2012_04_06_a_04.asc"
    url_manoeuvres_files = "https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-2-edr-v1/grail_0001/level_0/2012_04_06/"
    # Check if the manoeuvre file already exists locally
    if (os.path.exists(local_path + manoeuvres_file) == False):
        print('download', local_path + manoeuvres_file)
        # If not, download it
        urlretrieve(url_manoeuvres_files + manoeuvres_file, local_path + manoeuvres_file)

    # Add local path to manoeuvres file name
    manoeuvres_file = local_path + manoeuvres_file

    # Print the name of the manoeuvre file
    print('relevant manoeuvres files')
    print(manoeuvres_file)


    # Antenna switch files (multiple antenna switch files are required, typically one per day)
    print('---------------------------------------------')
    print('Download antenna switch files')
    # Define url where antenna switch files can be downloaded for GRAIL
    url_antenna_files = ("https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-3-cdr-v1/grail_0101/level_1b/")
    # Retrieve the names of all antenna switch files within the time interval of interest, and download them if they do not exist locally yet
    antenna_files = download_url_files_time(local_path=local_path, filename_format='*/vgs1b_*_a_04.asc', start_date=start_date,
                            end_date=end_date, url=url_antenna_files, time_format='%Y_%m_%d', indices_date_filename=[0,8])

    # Print the name of all relevant antenna switch files that have been identified over the time interval of interest
    print('relevant antenna files')
    for f in antenna_files:
        print(f)


    # ODF files (multiple ODF files are required, typically one per day)
    print('---------------------------------------------')
    print('Download GRAIL ODF files')
    # Define url where ODF files can be downloaded for GRAIL
    url_odf = ("https://pds-geosciences.wustl.edu/grail/grail-l-rss-2-edr-v1/grail_0201/odf/")
    # Retrieve the names of all existing ODF files within the time interval of interest, and download them if they do not exist locally yet
    odf_files = download_url_files_time(
        local_path=local_path, filename_format='gralugf*_\w\w\w\wsmmmv1.odf', start_date=start_date,
        end_date=end_date, url=url_odf, time_format='%Y_%j', indices_date_filename=[7])

    # Print the name of all relevant ODF files that have been identified over the time interval of interest
    print('relevant odf files')
    for f in odf_files:
        print(f)


    # GRAIL trajectory files (a single file is necessary to cover the period of interest).
    # Note that the file name hard coded below only covers the period from 01/03/2012 to 29/05/2012.
    # This should be adapted in case the time interval of the example is modified.
    print('---------------------------------------------')
    print('Download GRAIL trajectory file')
    trajectory_files = ["grail_120301_120529_sci_v02.bsp"]
    # Define url where trajectory files can be downloaded for GRAIL
    url_trajectory_files = "https://naif.jpl.nasa.gov/pub/naif/pds/data/grail-l-spice-6-v1.0/grlsp_1000/data/spk/"
    for file in trajectory_files:
        # Check if the relevant trajectory file exists locally
        if (os.path.exists(local_path + file) == False):
            print('download', local_path + file)
            # If not, download it
            urlretrieve(url_trajectory_files + file, local_path + file)

    # Print and store all relevant trajectory file names
    print('relevant trajectory files')
    for k in range(len(trajectory_files)):
        trajectory_files[k] = local_path + trajectory_files[k]
        print(trajectory_files[k])


    # Frames definition file for the GRAIL spacecraft (only one file is necessary). This is useful for
    # spacecraft-fixed frames definition.
    print('---------------------------------------------')
    print('Download GRAIL frames definition file')
    grail_frames_def_file = "grail_v07.tf"
    # Define url where the frames definition file can be downloaded for GRAIL
    url_frames_def_file = "https://naif.jpl.nasa.gov/pub/naif/pds/data/grail-l-spice-6-v1.0/grlsp_1000/data/fk/"
    # Check if the relevant frames definition file exists locally
    if (os.path.exists(local_path + grail_frames_def_file) == False):
        print('download', local_path + grail_frames_def_file)
        # If not, download it
        urlretrieve(url_frames_def_file + grail_frames_def_file, local_path + grail_frames_def_file)

    # Print and store the frames definition file name
    print('relevant GRAIL frames definition file')
    grail_frames_def_file = local_path + grail_frames_def_file
    print(grail_frames_def_file)


    # Orientation kernel for the Moon
    print('---------------------------------------------')
    print('Download Moon orientation kernel')
    moon_orientation_file = "moon_pa_de440_200625.bpc"
    # Define url where the Moon's orientation kernel can be downloaded
    url_moon_orientation_file = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/"
    # Check if the relevant orientation kernel exists locally
    if (os.path.exists(local_path + moon_orientation_file) == False):
        print('download', local_path + moon_orientation_file)
        # If not, download it
        urlretrieve(url_moon_orientation_file + moon_orientation_file, local_path + moon_orientation_file)

    # Print and store the Moon orientation file name
    print('relevant Moon orientation file')
    moon_orientation_file = local_path + moon_orientation_file
    print(moon_orientation_file)


    # Lunar reference frame kernel
    print('---------------------------------------------')
    print('Download lunar reference frame kernel')
    lunar_frame_file = ("moon_de440_250416.tf")
    # Define url where the lunar reference frame kernel can be downloaded
    url_lunar_frame_file = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/fk/satellites/"
    # Check if the reference frame kernel exists locally
    if (os.path.exists(local_path + lunar_frame_file) == False):
        print('download', local_path + lunar_frame_file)
        # If not, download it
        urlretrieve(url_lunar_frame_file + lunar_frame_file, local_path + lunar_frame_file)

    # Print and store the lunar reference frame file name
    print('relevant lunar reference frame file')
    lunar_frame_file = local_path + lunar_frame_file
    print(lunar_frame_file)


    # Retrieve filenames lists for clock file, GRAIL orientation kernels, tropospheric and ionospheric corrections, manoeuvre files,
    # antenna switch files, odf files, GRAIL trajectory files, GRAIL reference frames file, lunar orientation kernels, and
    # lunar reference frame kernel.
    return (clock_file, grail_orientation_files, tro_files, ion_files, manoeuvres_file, antenna_files,
            odf_files, trajectory_files, grail_frames_def_file, moon_orientation_file, lunar_frame_file)


# Function returning the complete settings for the panel model of the GRAIL spacecraft
def get_grail_panel_geometry():

    # First read the panel data and material metadata from input file
    this_file_path = os.path.dirname(os.path.abspath(__file__))
    panel_data = pd.read_csv(this_file_path + "/grail_inputs/grail_macromodel.txt", delimiter=", ", engine="python")
    material_data = pd.read_csv(this_file_path + "/grail_inputs/grail_materials.txt", delimiter=", ", engine="python")

    # Initialize list to store all panel settings
    all_panel_settings = []

    # Parse all panels
    for i, row in panel_data.iterrows():

        # Create panel geometry settings
        # Options are: frame_fixed_panel_geometry, time_varying_panel_geometry, body_tracking_panel_geometry
        panel_geometry_settings = environment_setup.vehicle_systems.frame_fixed_panel_geometry(
            np.array([row["x"], row["y"], row["z"]]),  # panel position in body reference frame
            row["area"])  # panel area

        # Retrieve panel material data
        panel_material_data = material_data[material_data["material"] == row["material"]]

        # Create panel radiation settings (for specular and diffuse reflection)
        specular_diffuse_body_panel_reflection_settings = environment_setup.radiation_pressure.specular_diffuse_body_panel_reflection(
            specular_reflectivity=float(panel_material_data["Cs"].iloc[0]),
            diffuse_reflectivity=float(panel_material_data["Cd"].iloc[0]), with_instantaneous_reradiation=True)

        # Create settings for complete panel (combining geometry and material properties relevant for radiation pressure calculations)
        complete_panel_settings = environment_setup.vehicle_systems.body_panel_settings(
            panel_geometry_settings,
            specular_diffuse_body_panel_reflection_settings)

        # Add panel settings to list of all panel settings
        all_panel_settings.append(complete_panel_settings)

    # Create settings object for complete vehicle shape
    full_panelled_body_settings = environment_setup.vehicle_systems.full_panelled_body_settings(all_panel_settings)

    # Return complete panel settings
    return full_panelled_body_settings


# Function that converts the estimated state history of a given body from inertial to RSW frame (radial, along-track, and cross-track),
# and returns the difference with respect to its reference SPICE trajectory. This allows to compare the body's estimated and reference trajectories.
def get_rsw_state_difference(estimated_state_history, body_name, central_body, global_frame_orientation):

    rsw_state_difference = np.zeros((len(estimated_state_history), 7))
    counter = 0

    # Parse all epochs in estimated state history
    for time in estimated_state_history:
        current_estimated_state = estimated_state_history[time]

        # Retrieve reference state from spice
        current_spice_state = spice.get_body_cartesian_state_at_epoch(body_name, central_body,
                                                                      global_frame_orientation, "None", time)

        # Compute difference in the inertial frame
        current_state_difference = current_estimated_state - current_spice_state
        current_position_difference = current_state_difference[0:3]
        current_velocity_difference = current_state_difference[3:6]

        # Compute the rotation matrix from inertial to RSW frames
        rotation_to_rsw = frame_conversion.inertial_to_rsw_rotation_matrix(current_estimated_state)

        # Convert the state difference from inertial to RSW frames
        rsw_state_difference[counter, 0] = time
        rsw_state_difference[counter, 1:4] = rotation_to_rsw @ current_position_difference
        rsw_state_difference[counter, 4:7] = rotation_to_rsw @ current_velocity_difference

        counter = counter + 1

    # Return state difference history in RSW frame
    return rsw_state_difference








