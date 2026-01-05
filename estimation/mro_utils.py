# %%
import os
import numpy as np
from tudatpy.dynamics import environment_setup

import requests
from bs4 import BeautifulSoup
from urllib.request import urlretrieve
from datetime import datetime, timedelta
from urllib.parse import urljoin
import glob
import re
import pandas as pd
from tudatpy.interface import spice
from tudatpy.astro import frame_conversion


# Function checking whether a given date is contained with a specific time interval
def is_date_in_intervals(date, intervals):
    in_intervals = False
    for interval in intervals:
        if (date.date() >= interval[0].date()) and (date.date() <= interval[1].date()):
            in_intervals = True
    return in_intervals


# Function that retrieves the names of all files at a given url that:
# 1) follow the specific filename format given as input (filename_format)
# 2) correspond to dates contained with the time interval defined by the start_date and end_date input arguments
# (with the format of the time interval specified by the time_interval_format input)
#
# This function returns a list of all files found at the given url that match the above requirements. It also looks
# whether each of these files already locally exists (at the location given by the local_path input argument). If not,
# it automatically downloads the missing file.
#
# This function is designed to handle files that cover time intervals of several days, rather than a specific date
# (see alternative download_url_files_time function).
#
# About the filename_format and time_interval_format inputs: the symbol '*' should be used to indicate where the time interval is specified
# in the filename. This will be used as a wildcard for the function to look for all pattern-matching filenames at the specified url.
# Similarly, the time_interval_format input makes use of the time format strings of the python datetime module
# (see https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes)
# As an example, the following filename_format = 'mro_sc_psp_*.bc' and time_interval_format = '%y%m%d_%y%m%d' will allow the function
# to look for files with names like 'mro_sc_psp_201006_201012.bc'


def download_url_files_time_interval(
    local_path, filename_format, start_date, end_date, url, time_interval_format
):

    # Split the file name where the time interval is specified (should be indicated by the symbol *)
    split_filename = filename_format.split("*")
    start_filename = filename_format[: len(split_filename[0])]
    end_filename = filename_format[-len(split_filename[1]) :]
    size_start_filename = len(start_filename)
    size_end_filename = len(end_filename)

    # Retrieve time format (the division by two accounts for the fact that the input is a time interval and thus contains
    # two dates)
    time_format = time_interval_format[: len(time_interval_format) // 2]

    # Compute the size of the time string for this specific time format
    size_time_format = 0
    for k in range(len(time_interval_format) // 2):
        if time_interval_format[k] == "%":
            size_time_format += 0
        elif time_interval_format[k] == "Y":
            size_time_format += 4
        elif time_interval_format[k] == "y":
            size_time_format += 2
        elif time_interval_format[k] == "m":
            size_time_format += 2
        elif time_interval_format[k] == "d":
            size_time_format += 2
        elif time_interval_format[k] == "j":
            size_time_format += 3
        else:
            size_time_format += 1
    size_interval_format = size_time_format * 2 + len(time_interval_format) % 2

    # Re-compute the full size of the filename, by adding the size of the time string to the rest of the filename
    size_filename = size_start_filename + size_interval_format + size_end_filename

    # print("size_time_format", size_time_format)
    # print("size_interval_format", size_interval_format)

    # Retrieve all dates contained within the time interval of interest
    all_dates = [
        start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)
    ]

    # Retrieve all filenames present at the "local_path" location that match the specified filename format
    existing_files = glob.glob(local_path + filename_format)
    # print("nb existing files", len(existing_files))
    relevant_intervals = []
    relevant_files = []
    for file in existing_files:
        # For each of these existing files, retrieve the time interval that it covers
        time = file[-(size_end_filename + size_interval_format) : -size_end_filename]
        current_interval = (
            datetime.strptime(time[:size_time_format], time_format),
            datetime.strptime(time[-size_time_format:], time_format),
        )

        # Check whether this time interval is contained within our interval of interest
        is_relevant = False
        date_cnt = 0
        while is_relevant == False and date_cnt < len(all_dates):
            # If so, store the name of the already existing file, as well as the time interval that it covers
            if is_date_in_intervals(all_dates[date_cnt], [current_interval]):
                is_relevant = True
                relevant_intervals.append(current_interval)
                relevant_files.append(file)
            date_cnt = date_cnt + 1

    # Identify dates of interest that are not covered by existing files
    dates_without_file = []
    for date in all_dates:
        # Check whether the current date is contained within one of the time intervals covered by existing files
        if is_date_in_intervals(date, relevant_intervals) == False:
            # If not, store the current date as non-covered by any file yet (i.e., date for which a file is missing)
            dates_without_file.append(date)

    # print("dates_without_file", dates_without_file)

    # Retrieve the missing files from the specified url
    if len(dates_without_file) > 0:

        # Dictionary containing the names of the files to be downloaded, with the time intervals that those files cover as keys and
        # the corresponding filenames as values
        files_url_dict = dict()

        # Parse all files contained at the targeted url
        reqs = requests.get(url)
        for link in BeautifulSoup(reqs.text, "html.parser").find_all("a"):

            # Retrieve full url link for each of these files
            full_link = link.get("href")

            # Only proceed if the full url link is a string object
            if isinstance(full_link, str):

                # Retrieve the filename, assuming it is of the required size (if not, the following checks will automatically fail anyway)
                file = full_link[-size_filename:]

                # Check whether the start of the filename (preceding the time string) and the end of the filename (following the time string)
                # both match with the filename format provided as input
                if (
                    file[:size_start_filename] == start_filename
                    and file[-size_end_filename:] == end_filename
                ):

                    # If so, retrieve the time interval part of the filename and derive the start and end dates of the interval
                    interval_filename = file[size_start_filename:-size_end_filename]
                    start_interval = datetime.strptime(
                        interval_filename[:size_time_format], time_format
                    )
                    end_interval = datetime.strptime(
                        interval_filename[-size_time_format:], time_format
                    )

                    # Store the name of the file to be downloaded (and the time interval that the file covers)
                    files_url_dict[(start_interval, end_interval)] = file

    # Parse all dates for which a file was originally missing. This needs to be an iterative process, as more files are progressively downloaded.
    # Since each file that gets downloaded covers a time interval of several days, it might be that a date originally not covered by any existing file
    # is now taken care of by a newly downloaded file (this is an important difference w.r.t. when loading files that only cover a single date, see
    # alternative download_url_files_time function).
    for date in dates_without_file:

        # Check if the current date is still not covered by any of the existing files
        if is_date_in_intervals(date, relevant_intervals) == False:
            # Retrieve the suitable interval and filename from the list of files to be downloaded and proceed to the download
            for new_interval in files_url_dict:
                if is_date_in_intervals(date, [new_interval]):
                    # print("download ", files_url_dict[new_interval])
                    urlretrieve(
                        url + files_url_dict[new_interval],
                        local_path + files_url_dict[new_interval],
                    )

                    # Add both the filename and the time interval that the file covers to the corresponding lists keeping track of
                    # what is now covered by existing files
                    relevant_files.append(local_path + files_url_dict[new_interval])
                    relevant_intervals.append(new_interval)

    # Return the list of all relevant files that should be loaded to cover the time interval of interest
    return relevant_files


# Function that retrieves the names of all files at a given url that:
# 1) follow the specific filename format given as input (filename_format)
# 2) correspond to dates contained with the time interval defined by the start_date and end_date input arguments
# (with the format of the time interval specified by the time_interval_format input)
#
# This function returns a list of all files found at the given url that match the above requirements. It also looks
# whether each of these files already locally exists (at the location given by the local_path input argument). If not,
# it automatically downloads the missing file.
#
# This function is designed to handle files that only cover one specific date, rather than time intervals of several days
# (see alternative download_url_files_time_interval function).
#
# About the filename_format and time_format inputs: the filename_format is a string where the symbol '*' should be used to indicate where
# the date of interest is specified. The symbol '\w' can also be included to denote any part of the filename that is unknown or unspecified
# by the user (e.g., the exact time at which a file is created). Both '*' and '\w' are then used as a wildcard for the function to look for
# all pattern-matching filenames at the specified url.
# The input indices_date_filename should moreover indicate where the date starts in the filename string.
# The time_format input is a string exploiting the time format strings of the python datetime module to specify the formatting used
# for the time tag in the filename (see https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes)
# As an example, the following filename_format = 'mromagr*_\w\w\w\wxmmmv1.odf' and time_format = '%Y_%j' will allow the function
# to look for files with names like 'mromagr2016_217_0840xmmmv1.odf'


def download_url_files_time(
    local_path,
    filename_format,
    start_date,
    end_date,
    url,
    time_format,
    indices_date_filename,
):

    # Retrieve all dates contained within the time interval defined by the start and end dates provided as inputs
    all_dates = [
        start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)
    ]

    # Split the file name at the symbol "/", to separate between the name of the file and the folder in which it might be contained
    filename_split = filename_format.split("/")
    folder = ""
    if len(filename_split) > 2:
        raise Exception(
            "in the current implementation, the filename format cannot contain more than one folder."
        )
    elif len(filename_split) == 2:
        folder = filename_split[0]

    # In the reduced file name (after removing the folder part), replace the wildcards - indicated by "\w" - by "*", which will later allow the
    # BeautifulSoup package to look for all pattern-matching names at the targeted url (without any a priori information on the date and/or
    # wildcard present in the file name)
    reduced_filename = filename_split[-1]
    reduced_filename = reduced_filename.replace("\w", "*")

    # Retrieve all filenames present at the "local_path" location that match the specified filename format
    existing_files = glob.glob(local_path + reduced_filename)
    print("nb existing files", len(existing_files))

    # Identify dates of interest that are not covered by existing files
    relevant_files = []
    dates_without_file = []
    for date in all_dates:
        # Create string object corresponding to the date under investigation
        date_str = date.strftime(time_format)

        # Reconstruct the expected filename for this particular date
        current_filename = ""
        index = 0
        for ind in indices_date_filename:
            current_filename += filename_format[index:ind] + date_str
            index = ind + 1
        current_filename += filename_format[index:]

        # Parse existing files and check whether the current file name can be found
        current_files = [
            x
            for x in existing_files
            if re.match(local_path + current_filename.split("/")[-1], x)
        ]
        # If so, add the identified file to the list of relevant files to be loaded
        if len(current_files) > 0:
            for file in current_files:
                relevant_files.append(current_files[0])
        # If not, mark the current date as non-covered by any file yet (i.e., date for which a file is missing)
        else:
            dates_without_file.append(date)

    # Retrieve the missing files from the specified url
    if len(dates_without_file) > 0:

        # List containing the names of the files to be downloaded
        files_url = []

        # Parse all files contained at the targeted url
        reqs = requests.get(url)
        for link in BeautifulSoup(reqs.text, "html.parser").find_all("a"):

            # Retrieve full url link for each of these files
            full_link = link.get("href")

            # Check whether the file of interest is nested within an additional folder
            if len(folder) == 0:
                current_filename = full_link.split("/")[-1]
            else:
                current_filename = full_link.split("/")[-2]

            # Store the name of the file to be downloaded
            files_url.append(current_filename)

    # Parse all dates for which a file was originally missing and download missing files from the specified url.
    for date in dates_without_file:

        # List of the files to be downloaded
        files_to_download = []

        # Reconstruct expected filename for the date under consideration
        date_str = date.strftime(time_format)
        current_filename = ""
        index = 0
        for ind in indices_date_filename:
            current_filename += filename_format[index:ind] + date_str
            index = ind + 1
        current_filename += filename_format[index:]

        # Check whether a matching file was found at the targeted url for this specific date, and split the filename at "/"
        # to account for the possibility that the targeted file is stored in a nested folder
        file_to_download = [
            x for x in files_url if re.match(current_filename.split("/")[0], x)
        ]

        # If the file is directly stored at the specified url (no nested folder), then the filename can be stored directly
        if len(folder) == 0:
            files_to_download = file_to_download

        # Otherwise, explore additional folder layer
        if len(folder) > 0 and len(file_to_download) > 0:
            reqs2 = requests.get(url + file_to_download[0])

            # Parse all files within the current folder
            for nested_link in BeautifulSoup(reqs2.text, "html.parser").find_all("a"):
                nested_file = nested_link.get("href")

                # Retrieve all matching file names within the current folder
                relevant_link = [
                    x
                    for x in [nested_file]
                    if re.match(current_filename.split("/")[-1], x.split("/")[-1])
                ]

                # If a match is found, store the filename that should be downloaded (now including the extra folder layer)
                if len(relevant_link) == 1:
                    files_to_download.append(
                        file_to_download[0] + "/" + relevant_link[0].split("/")[-1]
                    )

        # Download all relevant files from the targeted url
        for file in files_to_download:
            # print("download ", url + file)
            urlretrieve(url + file, local_path + file.split("/")[-1])
            relevant_files.append(local_path + file.split("/")[-1])

    # Return the list of all relevant files that should be loaded to cover the time interval of interest
    return relevant_files


def get_mro_files(local_path, start_date, end_date):

    # Check if local_path designates an existing directory and creates the directory is not
    if not os.path.isdir(local_path):
        os.mkdir(local_path)

    # Clock file (a single file is necessary)
    print("---------------------------------------------")
    print("Download MRO clock ")
    clock_files = ["mro_sclkscet_00112_65536.tsc"]
    # Define url where clock files can be downloaded for MRO
    url_clock_files = "https://naif.jpl.nasa.gov/pub/naif/pds/data/mro-m-spice-6-v1.0/mrosp_1000/data/sclk/"
    for file in clock_files:
        # Check if the relevant clock file exists locally
        if os.path.exists(local_path + file) == False:
            print("download", local_path + file)
            # If not, download it
            urlretrieve(url_clock_files + file, local_path + file)

    # Print and store all relevant clock file names
    print("relevant clock files")
    for k in range(len(clock_files)):
        clock_files[k] = local_path + clock_files[k]
        print(clock_files[k])

    # Orientation files (multiple orientation kerneks are required, typically covering intervals of a few days)
    # For this MRO example, orientation kernels should be loaded both for the spacecraft and for the MRO antenna specifically.
    print("---------------------------------------------")
    print("Download MRO orientation kernels")
    # Define url where orientation kernels can be downloaded for MRO
    url_orientation_files = "https://naif.jpl.nasa.gov/pub/naif/pds/data/mro-m-spice-6-v1.0/mrosp_1000/data/ck/"
    # Retrieve the names of all spacecraft orientation kernels required to cover the time interval of interest, and download them if they
    # do not exist locally yet.
    orientation_files = download_url_files_time_interval(
        local_path=local_path,
        filename_format="mro_sc_psp_*.bc",
        start_date=start_date - timedelta(days=1),
        end_date=end_date + timedelta(days=1),
        url=url_orientation_files,
        time_interval_format="%y%m%d_%y%m%d",
    )

    # Retrieve the names of all antenna orientation kernels required to cover the time interval of interest, and download them if they
    # do not exist locally yet
    antenna_files = download_url_files_time_interval(
        local_path=local_path,
        filename_format="mro_hga_psp_*.bc",
        start_date=start_date - timedelta(days=1),
        end_date=end_date + timedelta(days=1),
        url=url_orientation_files,
        time_interval_format="%y%m%d_%y%m%d",
    )

    panel_files = download_url_files_time_interval(
        local_path=local_path,
        filename_format="mro_sa_psp_*.bc",
        start_date=start_date - timedelta(days=1),
        end_date=end_date + timedelta(days=1),
        url=url_orientation_files,
        time_interval_format="%y%m%d_%y%m%d",
    )

    # Print and store all relevant orientation file names (both for the MRO spacecraft and antenna)
    orientation_files.extend(antenna_files)
    orientation_files.extend(panel_files)

    print("relevant orientation files")
    for f in orientation_files:
        print(f)

    # Tropospheric corrections (multiple tropospheric correction files are required, typically covering intervals of a few days)
    print("---------------------------------------------")
    print("Download MRO tropospheric corrections files")
    # Define url where tropospheric correction files can be downloaded for MRO
    url_tro_files = "https://pds-geosciences.wustl.edu/mro/mro-m-rss-1-magr-v1/mrors_0xxx/ancillary/tro/"
    # Retrieve the names of all tropospheric correction files required to cover the time interval of interest, and download them if they
    # do not exist locally yet
    tro_files = download_url_files_time_interval(
        local_path=local_path,
        filename_format="mromagr*.tro",
        start_date=start_date,
        end_date=end_date,
        url=url_tro_files,
        time_interval_format="%Y_%j_%Y_%j",
    )

    # Print all relevant tropospheric correction file names
    print("relevant tropospheric corrections files")
    for f in tro_files:
        print(f)

    # Ionospheric corrections (multiple ionospheric correction files are required, typically covering intervals of a few days)
    print("---------------------------------------------")
    print("Download MRO ionospheric corrections files")
    # Define url where ionospheric correction files can be downloaded for MRO
    url_ion_files = "https://pds-geosciences.wustl.edu/mro/mro-m-rss-1-magr-v1/mrors_0xxx/ancillary/ion/"
    # Retrieve the names of all ionospheric correction files required to cover the time interval of interest, and download them if they
    # do not exist locally yet
    ion_files = download_url_files_time_interval(
        local_path=local_path,
        filename_format="mromagr*.ion",
        start_date=start_date,
        end_date=end_date,
        url=url_ion_files,
        time_interval_format="%Y_%j_%Y_%j",
    )

    # Print all relevant ionospheric correction file names
    print("relevant ionospheric corrections files")
    for f in ion_files:
        print(f)

    # TNF files (multiple TNF files are required, typically one per day)
    print("---------------------------------------------")
    print("Download MRO TNF files")
    # Define url where TNF files can be downloaded for MRO
    url_odf = (
        "https://pds-geosciences.wustl.edu/mro/mro-m-rss-1-magr-v1/mrors_0xxx/tnf/"
    )
    # Retrieve the names of all existing TNF files within the time interval of interest, and download them if they do not exist locally yet
    tnf_files = download_url_files_time(
        local_path=local_path,
        filename_format="mromagr*_\w\w\w\wxmmmv1.tnf",
        start_date=start_date,
        end_date=end_date,
        url=url_odf,
        time_format="%Y_%j",
        indices_date_filename=[7],
    )

    # Print the name of all relevant TNF files that have been identified over the time interval of interest
    print("relevant TNF files")
    for f in tnf_files:
        print(f)

    # MRO trajectory files (multiple files are necessary to cover one entire year, typically each file covers ~ 3-4 months)
    # Note that the file names hard coded below cover calendar year 2012. This should be modified in case the time interval
    # of the example is modified.
    print("---------------------------------------------")
    print("Download MRO trajectory files")
    trajectory_files = [
        "mro_psp21.bsp",
        "mro_psp22.bsp",
        "mro_psp23.bsp",
        "mro_psp24.bsp",
        "mro_psp25.bsp",
    ]
    # Define url where trajectory files can be downloaded for MRO
    url_trajectory_files = "https://naif.jpl.nasa.gov/pub/naif/pds/data/mro-m-spice-6-v1.0/mrosp_1000/data/spk/"
    for file in trajectory_files:
        # Check if the relevant trajectory file exists locally
        if os.path.exists(local_path + file) == False:
            print("download", local_path + file)
            # If not, download it
            urlretrieve(url_trajectory_files + file, local_path + file)

    # Print and store all relevant trajectory file names
    print("relevant trajectory files")
    for k in range(len(trajectory_files)):
        trajectory_files[k] = local_path + trajectory_files[k]
        print(trajectory_files[k])

    # Frames definition file for the MRO spacecraft (only one file is necessary). This is useful for HGA and
    # spacecraft-fixed frames definition.
    print("---------------------------------------------")
    print("Download MRO frames definition file")
    frames_def_file = "mro_v16.tf"
    # Define url where the frames definition file can be downloaded for MRO
    url_frames_def_file = "https://naif.jpl.nasa.gov/pub/naif/pds/data/mro-m-spice-6-v1.0/mrosp_1000/data/fk/"
    # Check if the relevant frames definition file exists locally
    if os.path.exists(local_path + frames_def_file) == False:
        print("download", local_path + frames_def_file)
        # If not, download it
        urlretrieve(url_frames_def_file + frames_def_file, local_path + frames_def_file)

    # Print and store the frames definition file name
    print("relevant MRO frames definition file")
    frames_def_file = local_path + frames_def_file
    print(frames_def_file)

    # Structure file for the MRO spacecraft (only one file is necessary). This is useful to retrieve the antenna
    # position in spacecraft-fixed frame.
    print("---------------------------------------------")
    print("Download MRO structure file")
    structure_file = "mro_struct_v10.bsp"
    # Define url where the MRO structure file can be downloaded
    url_structure_file = "https://naif.jpl.nasa.gov/pub/naif/pds/data/mro-m-spice-6-v1.0/mrosp_1000/data/spk/"
    # Check if the relevant structure file exists locally
    if os.path.exists(local_path + structure_file) == False:
        print("download", local_path + structure_file)
        # If not, download it
        urlretrieve(url_structure_file + structure_file, local_path + structure_file)

    # Print and store the structure file name
    print("relevant MRO structure file")
    structure_file = local_path + structure_file
    print(structure_file)

    # Return filenames lists for clock files, orientation kernels, tropospheric and ionospheric corrections, odf files,
    # trajectory files, MRO reference frames file, and MRO structure file.
    return (
        clock_files,
        orientation_files,
        tro_files,
        ion_files,
        tnf_files,
        trajectory_files,
        frames_def_file,
        structure_file,
    )


def macromodel_mro():

    dae_path = "mro_macromodel/"

    # BUS
    bus_material_properties = {
        "INS": environment_setup.vehicle_systems.material_properties(
            specular_reflectivity=0.03,
            diffuse_reflectivity=0.12,
            normal_accomodation_coefficient=1.0,
            tangential_accomodation_coefficient=1.0,
            normal_velocity_at_wall_ratio=0.1,
        ),
        "BUS": environment_setup.vehicle_systems.material_properties(
            specular_reflectivity=0.05,
            diffuse_reflectivity=0.21,
            normal_accomodation_coefficient=1.0,
            tangential_accomodation_coefficient=1.0,
            normal_velocity_at_wall_ratio=0.1,
        ),
    }
    bus_reradiation_settings = {"INS": True, "BUS": True}
    bus_frame_origin = np.zeros(3)
    bus_panels = environment_setup.vehicle_systems.body_panel_settings_list_from_dae(
        dae_path + "MRO_bus_lowfidelity.dae",
        bus_frame_origin,
        bus_material_properties,
        bus_reradiation_settings,
    )
    # HGA
    hga_material_properties = {
        "HGA_front": environment_setup.vehicle_systems.material_properties(
            specular_reflectivity=0.55,
            diffuse_reflectivity=0.25,
            tangential_accomodation_coefficient=1.0,
            normal_accomodation_coefficient=1.0,
            normal_velocity_at_wall_ratio=0.1,
        ),
        "HGA_back": environment_setup.vehicle_systems.material_properties(
            specular_reflectivity=0.05,
            diffuse_reflectivity=0.8,
            tangential_accomodation_coefficient=1.0,
            normal_accomodation_coefficient=1.0,
            normal_velocity_at_wall_ratio=0.1,
        ),
    }
    hga_reradiation_settings = {"HGA_front": True, "HGA_back": True}
    hga_frame_origin = np.array([0.0, -3.15, -1.52])
    hga_panels = environment_setup.vehicle_systems.body_panel_settings_list_from_dae(
        dae_path + "MRO_hga_8.dae",
        hga_frame_origin,
        hga_material_properties,
        hga_reradiation_settings,
        "MRO_HGA_OUTER_GIMBAL",
    )
    hga_rotation_settings = environment_setup.rotation_model.spice(
        "MRO_SPACECRAFT", "MRO_HGA_OUTER_GIMBAL", ""
    )
    # SAPX/SAMX
    sa_material_properties = {
        "SA_front": environment_setup.vehicle_systems.material_properties(
            specular_reflectivity=0.03,
            diffuse_reflectivity=0.07,
            tangential_accomodation_coefficient=1.0,
            normal_accomodation_coefficient=1.0,
            normal_velocity_at_wall_ratio=0.1,
        ),
        "SA_back": environment_setup.vehicle_systems.material_properties(
            specular_reflectivity=0.02,
            diffuse_reflectivity=0.2,
            tangential_accomodation_coefficient=1.0,
            normal_accomodation_coefficient=1.0,
            normal_velocity_at_wall_ratio=0.1,
        ),
    }
    sa_reradiation_settings = {"SA_front": True, "SA_back": True}
    sapx_frame_origin = np.array([1.144, -2.5354, 0.113])
    sapx_panels = environment_setup.vehicle_systems.body_panel_settings_list_from_dae(
        dae_path + "MRO_sa.dae",
        sapx_frame_origin,
        sa_material_properties,
        sa_reradiation_settings,
        "MRO_SAPX",
    )
    sapx_rotation_settings = environment_setup.rotation_model.spice(
        "MRO_SPACECRAFT", "MRO_SAPX", ""
    )
    samx_frame_origin = np.array([-1.144, -2.5354, 0.113])
    samx_panels = environment_setup.vehicle_systems.body_panel_settings_list_from_dae(
        dae_path + "MRO_sa.dae",
        samx_frame_origin,
        sa_material_properties,
        sa_reradiation_settings,
        "MRO_SAMX",
    )
    samx_rotation_settings = environment_setup.rotation_model.spice(
        "MRO_SPACECRAFT", "MRO_SAMX", ""
    )
    # merge body panels
    list_panel_body_settings = (
        environment_setup.vehicle_systems.merge_body_panel_setting_lists(
            [bus_panels, hga_panels, sapx_panels, samx_panels]
        )
    )
    # merge rotation settings
    dict_rotation_settings = {
        "MRO_HGA_OUTER_GIMBAL": hga_rotation_settings,
        "MRO_SAPX": sapx_rotation_settings,
        "MRO_SAMX": samx_rotation_settings,
    }
    full_panelled_body = environment_setup.vehicle_systems.full_panelled_body_settings(
        list_panel_body_settings, dict_rotation_settings
    )

    return full_panelled_body


def get_rsw_state_difference(
    estimated_state_history, body_name, central_body, global_frame_orientation
):

    rsw_state_difference = np.zeros((len(estimated_state_history), 7))
    counter = 0

    # Parse all epochs in estimated state history
    for time in estimated_state_history:
        current_estimated_state = estimated_state_history[time]

        # Retrieve reference state from spice
        current_spice_state = spice.get_body_cartesian_state_at_epoch(
            body_name, central_body, global_frame_orientation, "None", time
        )

        # Compute difference in the inertial frame
        current_state_difference = current_estimated_state - current_spice_state
        current_position_difference = current_state_difference[0:3]
        current_velocity_difference = current_state_difference[3:6]

        # Compute the rotation matrix from inertial to RSW frames
        rotation_to_rsw = frame_conversion.inertial_to_rsw_rotation_matrix(
            current_estimated_state
        )

        # Convert the state difference from inertial to RSW frames
        rsw_state_difference[counter, 0] = time
        rsw_state_difference[counter, 1:4] = (
            rotation_to_rsw @ current_position_difference
        )
        rsw_state_difference[counter, 4:7] = (
            rotation_to_rsw @ current_velocity_difference
        )

        counter = counter + 1

    # Return state difference history in RSW frame
    return rsw_state_difference
