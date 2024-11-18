import requests
from bs4 import BeautifulSoup
from urllib.request import urlretrieve
from datetime import datetime, timedelta
import glob
import re

# Function checking whether a given date is contained with a specific time interval
def is_date_in_intervals(date, intervals):
    in_intervals = False
    for interval in intervals:
        if ( ( date.date() >= interval[0].date() ) and ( date.date() <= interval[1].date() ) ):
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

def download_url_files_time_interval(local_path, filename_format, start_date, end_date, url, time_interval_format ):

    # Split the file name where the time interval is specified (should be indicated by the symbol *)
    split_filename = filename_format.split("*")
    start_filename = filename_format[:len(split_filename[0])]
    end_filename = filename_format[-len(split_filename[1]):]
    size_start_filename = len(start_filename)
    size_end_filename = len(end_filename)

    # Retrieve time format (the division by two accounts for the fact that the input is a time interval and thus contains
    # two dates)
    time_format = time_interval_format[:len(time_interval_format)//2]

    # Compute the size of the time string for this specific time format
    size_time_format = 0
    for k in range(len(time_interval_format)//2):
        if time_interval_format[k] == '%':
            size_time_format += 0
        elif time_interval_format[k] == 'Y':
            size_time_format += 4
        elif time_interval_format[k] == 'y':
            size_time_format += 2
        elif time_interval_format[k] == 'm':
            size_time_format += 2
        elif time_interval_format[k] == 'd':
            size_time_format += 2
        elif time_interval_format[k] == 'j':
            size_time_format += 3
        else:
            size_time_format += 1
    size_interval_format = size_time_format*2 + len(time_interval_format)%2

    # Re-compute the full size of the filename, by adding the size of the time string to the rest of the filename
    size_filename = size_start_filename + size_interval_format + size_end_filename

    print('size_time_format', size_time_format)
    print('size_interval_format', size_interval_format)

    # Retrieve all dates contained within the time interval of interest
    all_dates = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]

    # Retrieve all filenames present at the "local_path" location that match the specified filename format
    existing_files = glob.glob(local_path + filename_format)
    print('nb existing files', len(existing_files))
    relevant_intervals = []
    relevant_files = []
    for file in existing_files:
        # For each of these existing files, retrieve the time interval that it covers
        time = file[-(size_end_filename+size_interval_format):-size_end_filename]
        current_interval = (datetime.strptime(time[:size_time_format], time_format),
                            datetime.strptime(time[-size_time_format:], time_format))

        # Check whether this time interval is contained within our interval of interest
        is_relevant = False
        date_cnt = 0
        while (is_relevant == False and date_cnt < len(all_dates)):
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
        if (is_date_in_intervals(date, relevant_intervals) == False):
            # If not, store the current date as non-covered by any file yet (i.e., date for which a file is missing)
            dates_without_file.append(date)

    print("dates_without_file", dates_without_file)

    # Retrieve the missing files from the specified url
    if len(dates_without_file) > 0:

        # Dictionary containing the names of the files to be downloaded, with the time intervals that those files cover as keys and
        # the corresponding filenames as values
        files_url_dict = dict()

        # Parse all files contained at the targeted url
        reqs = requests.get(url)
        for link in BeautifulSoup(reqs.text, 'html.parser').find_all('a'):

            # Retrieve full url link for each of these files
            full_link = link.get('href')

            # Only proceed if the full url link is a string object
            if isinstance(full_link, str):

                # Retrieve the filename, assuming it is of the required size (if not, the following checks will automatically fail anyway)
                file = full_link[-size_filename:]

                # Check whether the start of the filename (preceding the time string) and the end of the filename (following the time string)
                # both match with the filename format provided as input
                if (file[:size_start_filename] == start_filename and file[-size_end_filename:] == end_filename):

                    # If so, retrieve the time interval part of the filename and derive the start and end dates of the interval
                    interval_filename = file[size_start_filename:-size_end_filename]
                    start_interval = datetime.strptime(interval_filename[:size_time_format], time_format)
                    end_interval = datetime.strptime(interval_filename[-size_time_format:], time_format)

                    # Store the name of the file to be downloaded (and the time interval that the file covers)
                    files_url_dict[(start_interval, end_interval)] = file


    # Parse all dates for which a file was originally missing. This needs to be an iterative process, as more files are progressively downloaded.
    # Since each file that gets downloaded covers a time interval of several days, it might be that a date originally not covered by any existing file
    # is now taken care of by a newly downloaded file (this is an important difference w.r.t. when loading files that only cover a single date, see
    # alternative download_url_files_time function).
    for date in dates_without_file:

        # Check if the current date is still not covered by any of the existing files
        if (is_date_in_intervals(date, relevant_intervals) == False):
            # Retrieve the suitable interval and filename from the list of files to be downloaded and proceed to the download
            for new_interval in files_url_dict:
                if is_date_in_intervals(date, [new_interval]):
                    print('download ', files_url_dict[new_interval])
                    urlretrieve(url + files_url_dict[new_interval], local_path + files_url_dict[new_interval])

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

def download_url_files_time(local_path, filename_format, start_date, end_date, url, time_format,
                            indices_date_filename ):

    # Retrieve all dates contained within the time interval defined by the start and end dates provided as inputs
    all_dates = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]

    # Split the file name at the symbol "/", to separate between the name of the file and the folder in which it might be contained
    filename_split = filename_format.split('/')
    folder = ''
    if (len(filename_split)>2):
        raise Exception("in the current implementation, the filename format cannot contain more than one folder.")
    elif (len(filename_split) == 2):
        folder = filename_split[0]

    # In the reduced file name (after removing the folder part), replace the wildcards - indicated by "\w" - by "*", which will later allow the
    # BeautifulSoup package to look for all pattern-matching names at the targeted url (without any a priori information on the date and/or
    # wildcard present in the file name)
    reduced_filename = filename_split[-1]
    reduced_filename = reduced_filename.replace('\w', '*')

    # Retrieve all filenames present at the "local_path" location that match the specified filename format
    existing_files = glob.glob(local_path + reduced_filename)
    print('nb existing files', len(existing_files))

    # Identify dates of interest that are not covered by existing files
    relevant_files = []
    dates_without_file = []
    for date in all_dates:
        # Create string object corresponding to the date under investigation
        date_str = date.strftime(time_format)

        # Reconstruct the expected filename for this particular date
        current_filename = ''
        index = 0
        for ind in indices_date_filename:
            current_filename += filename_format[index:ind] + date_str
            index = ind+1
        current_filename += filename_format[index:]

        # Parse existing files and check whether the current file name can be found
        current_files = [x for x in existing_files if re.match(local_path + current_filename.split('/')[-1], x)]
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
        for link in BeautifulSoup(reqs.text, 'html.parser').find_all('a'):

            # Retrieve full url link for each of these files
            full_link = link.get('href')

            # Check whether the file of interest is nested within an additional folder
            if len(folder)==0:
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
        current_filename = ''
        index = 0
        for ind in indices_date_filename:
            current_filename += filename_format[index:ind] + date_str
            index = ind + 1
        current_filename += filename_format[index:]

        # Check whether a matching file was found at the targeted url for this specific date, and split the filename at "/"
        # to account for the possibility that the targeted file is stored in a nested folder
        file_to_download = [x for x in files_url if re.match(current_filename.split('/')[0], x)]

        # If the file is directly stored at the specified url (no nested folder), then the filename can be stored directly
        if (len(folder)==0):
            files_to_download = file_to_download

        # Otherwise, explore additional folder layer
        if (len(folder)>0 and len(file_to_download)>0):
            reqs2 = requests.get(url + file_to_download[0])

            # Parse all files within the current folder
            for nested_link in BeautifulSoup(reqs2.text, 'html.parser').find_all('a'):
                nested_file = nested_link.get('href')

                # Retrieve all matching file names within the current folder
                relevant_link = [x for x in [nested_file] if re.match(current_filename.split('/')[-1], x.split('/')[-1])]

                # If a match is found, store the filename that should be downloaded (now including the extra folder layer)
                if (len(relevant_link) == 1):
                    files_to_download.append(file_to_download[0] + "/" + relevant_link[0].split("/")[-1])


        # Download all relevant files from the targeted url
        for file in files_to_download:
            print('download ', url + file)
            urlretrieve(url + file, local_path + file.split("/")[-1])
            relevant_files.append(local_path + file.split("/")[-1])

    # Return the list of all relevant files that should be loaded to cover the time interval of interest
    return relevant_files




