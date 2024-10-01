import sys

import numpy as np

sys.path.insert(0, "/home/mfayolle/Tudat/tudat-bundle/cmake-build-release/tudatpy")

import requests
from bs4 import BeautifulSoup
import os
from urllib.request import urlretrieve
from datetime import datetime, timedelta
import glob
import re


def is_date_in_intervals(date, intervals):
    in_intervals = False
    for interval in intervals:
        if ( ( date.date() >= interval[0].date() ) and ( date.date() <= interval[1].date() ) ):
            in_intervals = True
    return in_intervals


def download_url_files_time_interval(local_path, filename_format, start_date, end_date, url, time_interval_format ):

    split_filename = filename_format.split("*")
    start_filename = filename_format[:len(split_filename[0])]
    end_filename = filename_format[-len(split_filename[1]):]
    size_start_filename = len(start_filename)
    size_end_filename = len(end_filename)

    time_format = time_interval_format[:len(time_interval_format)//2]

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

    size_filename = size_start_filename + size_interval_format + size_end_filename

    print('size_time_format', size_time_format)
    print('size_interval_format', size_interval_format)

    all_dates = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]

    existing_files = glob.glob(local_path + filename_format)
    print('nb existing files', len(existing_files))
    relevant_intervals = []
    relevant_files = []
    for file in existing_files:
        time = file[-(size_end_filename+size_interval_format):-size_end_filename]
        current_interval = (datetime.strptime(time[:size_time_format], time_format),
                            datetime.strptime(time[-size_time_format:], time_format))

        is_relevant = False
        date_cnt = 0
        while (is_relevant == False and date_cnt < len(all_dates)):
            if is_date_in_intervals(all_dates[date_cnt], [current_interval]):
                is_relevant = True
                relevant_intervals.append(current_interval)
                relevant_files.append(file)
            date_cnt = date_cnt + 1

    # Identify dates not covered by existing files
    dates_without_file = []
    for date in all_dates:
        if (is_date_in_intervals(date, relevant_intervals) == False):
            dates_without_file.append(date)

    print("dates_without_file", dates_without_file)

    # Retrieve files from PDS
    if len(dates_without_file) > 0:

        reqs = requests.get(url)

        files_url_dict = dict()
        for link in BeautifulSoup(reqs.text, 'html.parser').find_all('a'):
            full_link = link.get('href')
            if isinstance(full_link, str):
                file = full_link[-size_filename:]
                if (file[:size_start_filename] == start_filename and file[-size_end_filename:] == end_filename):

                    interval_filename = file[size_start_filename:-size_end_filename]
                    start_interval = datetime.strptime(interval_filename[:size_time_format], time_format)
                    end_interval = datetime.strptime(interval_filename[-size_time_format:], time_format)

                    files_url_dict[(start_interval, end_interval)] = file


    for date in dates_without_file:

        if (is_date_in_intervals(date, relevant_intervals) == False):
            for new_interval in files_url_dict:
                if is_date_in_intervals(date, [new_interval]):
                    print('download ', files_url_dict[new_interval])
                    urlretrieve(url + files_url_dict[new_interval], local_path + files_url_dict[new_interval])
                    relevant_files.append(local_path + files_url_dict[new_interval])
                    relevant_intervals.append(new_interval)

    # print('relevant intervals')
    # for interval in relevant_intervals:
    #     print(interval)
    #
    # print('relevant files')
    # for f in relevant_files:
    #     print(f)

    return relevant_files


def download_url_files_time(local_path, filename_format, start_date, end_date, url, time_format, filename_size,
                            indices_date_filename ):

    all_dates = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]

    filename_split = filename_format.split('/')
    nested_folder = ''
    if (len(filename_split)>2):
        raise Exception("the filename format cannot contain more than one nested folder.")
    elif (len(filename_split) == 2):
        nested_folder = filename_split[0]

    reduced_filename = filename_split[-1]
    reduced_filename = reduced_filename.replace('\w', '*')
    # print('reduced_filename', reduced_filename)


    existing_files = glob.glob(local_path + reduced_filename)
    print('nb existing files', len(existing_files))
    # print(existing_files)

    relevant_files = []
    dates_without_file = []
    for date in all_dates:
        date_str = date.strftime(time_format)

        current_filename = ''
        index = 0
        for ind in indices_date_filename:
            current_filename += filename_format[index:ind] + date_str
            index = ind+1
        current_filename += filename_format[index:]

        # print('current filename for existing file', local_path + current_filename.split('/')[-1])

        current_files = [x for x in existing_files if re.match(local_path + current_filename.split('/')[-1], x)]
        if len(current_files) > 0:
            for file in current_files:
                relevant_files.append(current_files[0])
        else:
            dates_without_file.append(date)

    # print("dates_without_file", dates_without_file)

    # Retrieve files from PDS
    if len(dates_without_file) > 0:
        reqs = requests.get(url)
        files_url = []
        for link in BeautifulSoup(reqs.text, 'html.parser').find_all('a'):
            full_link = link.get('href')
            if len(nested_folder)==0:
                current_filename = full_link.split("/")[-1]
            else:
                current_filename = full_link.split("/")[-2]
            files_url.append(current_filename)

        # print('files url', files_url)

    # Download missing files from PDS
    for date in dates_without_file:

        files_to_download = []

        date_str = date.strftime(time_format)
        current_filename = ''
        index = 0
        for ind in indices_date_filename:
            current_filename += filename_format[index:ind] + date_str
            index = ind + 1
        current_filename += filename_format[index:]

        file_to_download = [x for x in files_url if re.match(current_filename.split('/')[0], x)]

        if (len(nested_folder)==0):
            files_to_download = file_to_download

        # Explore nested folder if any
        if (len(nested_folder)>0 and len(file_to_download)>0):
            reqs2 = requests.get(url + file_to_download[0])
            for nested_link in BeautifulSoup(reqs2.text, 'html.parser').find_all('a'):
                nested_file = nested_link.get('href')
                # print('nested_link', nested_file)

                relevant_link = [x for x in [nested_file] if re.match(current_filename.split('/')[-1], x.split('/')[-1])]
                # print('relevant link', relevant_link)

                if (len(relevant_link) == 1):
                    files_to_download.append(file_to_download[0] + "/" + relevant_link[0].split("/")[-1])

        # print('files_to_download', files_to_download)

        for file in files_to_download:
            print('download ', url + file)
            urlretrieve(url + file, local_path + file.split("/")[-1])
            relevant_files.append(local_path + file.split("/")[-1])

    # print('relevant files')
    # for f in relevant_files:
    #     print(f)

    return relevant_files




