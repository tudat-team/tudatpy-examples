#!/usr/bin/env python
# coding: utf-8

# ## Objectives
# 
# This script allows to download:
# 1) Spice (clock, frame, orientation, planetary) Kernels
# 2) Closed Loop Doppler Data (DSN and IFMS)
# 3) Ancillary Ionospheric and Tropospheric Data
# 
# **NOTE 1 (Supported Missions and Relative Downloaded Data).** 
# 
# The following spice kernels:
# 
# 1) **existing spice kernels** in the mission folder
# 2) (new) **downloaded spice kernels**
# 3) **standard TUDAT kernels** from [`spice.load_standard_kernels()`](https://py.api.tudat.space/en/latest/spice.html#tudatpy.interface.spice.load_standard_kernels) 
# 
# are loaded at the end of a run (some information is printed as output).
# 
# List of currently supported input_missions:
# - **Mars Reconnnaissance Orbiter (MRO)** [Available: Spice Kernels, Doppler Data, Ancillary]
# - **Mars Express (MEX)** [Available: Spice Kernels, Doppler Data, Ancillary]
# - **Jupiter Icy Moons Explorer (JUICE)** [Available: Spice Kernels, Not Available (yet): Doppler Data, Ancillary]
# - **Cassini**
# 
# Foreseen supported missions (in descending priority order, code yet to be developed):
# 
# - Insight
# - VEX
# - GRAIL
# - LRO
# 
# **NOTE 2. (Default and Custom Outputs)**
# 
# The downloaded data for a given mission is stored (by default) in a folder named `<mission_name>_archive`. Subfolders are created automatically where Kernels, Radio Science Data and Ancillary Data are stored. 
# The user can still define a custom output path passing the flag `custom_output` in the `get_mission_files` function. The output files will then be found in: `custom_path`
# 
# **NOTE 3. (About DSN-TNF)**
# 
# No DSN-TNF download is implemented, since for many applications the TNF is too cumbersome. We download the DSN-ODF files instead. The ODF is an edited and partially processed version of the TNF. It is a smaller file, often issued in daily increments of about 0.2 MB. It contains the most important information (range and Doppler) 
# 
# **Note 4: About spiceypy**
# 
# A compiled version of TUDAT containing the `spiceypy` dependence is needed (if spiceypy is not present, you won't be able to download files for the Cassini mission)

# ## Load required standard modules
# The required modules and dependencies are taken from the python file: `Mission_Data_Retriever_Class.py` present in the`tudatpy-examples/estimation` folder.

# In[7]:


from mission_data_downloader_class import *


# ## Create the LoadPDS Object
# First, we create the LoadPDS() object.

# In[2]:


object = LoadPDS()


# ## MRO Downloader (with Default Output)
# 
# ### Set Time Interval(s) for Downloading
# 
# Then we select the `start_date` and `end_date`, and we do so for each mission we wish to download (Cassini will be an exception due to the peculiar mission concept, see **Cassini Downloader** below). Of course, each mission has to come with its own dates, as operations are carried out over different periods. 
# 
# ### Download Mission Files (MRO, MEX, JUICE)
# Finally, we can call the function "get_mission_files
# get_mro_files will download: clock kernels, orientation kernels, radio science (odf) files for mro
# get_mex_files will download: clock kernels, orientation kernels, radio science (ifms) files for mex 
# get_juice_files will download: clock kernels, orientation kernels, for juice. No radio science data, cause none is yet available on the server (fdets retrieval is needed).

# In[3]:


start_date_juice = datetime(2023, 7, 1)
end_date_juice = datetime(2023, 8, 10)

start_date_mro = datetime(2007, 1, 3)
end_date_mro = datetime(2007, 1, 5)

start_date_mex = datetime(2004, 1, 3)
end_date_mex = datetime(2004,2, 7) 

# Download Mission Files with default output folder
# (only MRO is shown here. Uncomment the corresponding lines to download data for MEX and JUICE!)

kernel_files_mro, radio_science_files_mro, ancillary_files_mro = object.get_mission_files(input_mission = 'mro', start_date = start_date_mro, end_date = end_date_mro, custom_output = None)         
print(f'Total number of loaded kernels: {spice.get_total_count_of_kernels_loaded()}')

#kernel_files_mex, radio_science_files_mex, ancillary_files_mex = object.get_mission_files(input_mission = 'mex', start_date = start_date_mex, end_date = end_date_mex)         
#print(f'Total number of loaded kernels: {spice.get_total_count_of_kernels_loaded()}')

#kernel_files_juice, radio_science_files_juice, ancillary_files_juice = object.get_mission_files(input_mission = 'juice', start_date = start_date_juice, end_date = end_date_juice) 
#print(f'Total number of loaded kernels: {spice.get_total_count_of_kernels_loaded()}')


# ## Loaded Kernels for MRO, MEX and JUICE (Existing + Downloaded)
# Last, you can print the list of existing + downloaded files. 

# In[4]:


print(f'MRO Kernels: {kernel_files_mro}\n')
print(f'MRO Radio Science: {radio_science_files_mro}\n')
print(f'MRO Ancillary: {ancillary_files_mro}\n')

#print(f'MEX Kernels: {kernel_files_mex}\n')
#print(f'MEX Radio Science: {radio_science_files_mex}\n')
#print(f'MEX Ancillary: {ancillary_files_mex}\n')

#print(f'JUICE Kernels: {kernel_files_juice}\n')
#print(f'JUICE Radio Science: {radio_science_files_juice}\n') # it will be empty for now... (no Radio Science yet available on the server)
#print(f'JUICE Ancillary: {ancillary_files_juice}\n') # it will be empty for now... (no Ancillary files yet available on the server)


# ## Cassini Downloader (with Custom Output)
# 
# ### Set Flybys for Downloading
# 
# The most valuable data collected for Cassini is probably the one related to the various flybys of the Moons of Saturn.
# For this reason, in order to retrieve Cassini data, we will require the name of the flyby data the user wishes to download, rather than a start and end date (see **Notes 5 and 6** for info on the supported flybys). 
# 
# ### Download Cassini Flyby Files
# We can call the function "get_mission_files" with 'Cassini' as input_mission. This call will print a comprehensive table of supported flybys to choose from, and the user will be asked to manually input the name of one of them.
# 
# ### Flyby Data Output Division
# 
# The default output folder will be named `cassini_archive/MOON_NAME/FLYBY_ID/`, where `MOON_NAME` is the flyby moon, and `FLYBY_ID` is the denomination of each downloaded flyby (e.g. T011, T022). Each subfolder will contain kernel ancillary and radio science subdirectories. In this example, we will download files from the flyby T011, and we will store them in the custom path: `'CASSINI_CUSTOM_ARCHIVE/'`
# 
# **Note 5: Supported Flyby Moons**
# 
# For now, only Titan Flybys are supported.
# 
# **Note 6: flyby_IDs types**
# 
# `flyby_IDs` can be:
# 1) a list made of single flybys like: `flyby_IDs = ['T011', 'T022']` or `flyby_IDs = ['T011']`
# 2) a single string object like: `flyby_IDs = 'T011'` (not a list)
# 3) a list made of all flybys performed at a given moon: `flyby_IDs = ['ALL_TITAN']`
# 4) a single string object like: `flyby_IDs = 'ALL_TITAN'`
# 5) a mixed list like: `flyby_IDs = ['T011', 'ALL_TITAN']`
# 
# **Note 7: Custom Output**
# 
# As we mentioned above, the **default** output folder will be: `cassini_archive/MOON_NAME/FLYBY_ID/`
# The user can still define a custom output path using the flag `custom_output` in the `get_mission_files` function. The output files will then be found in: `custom_path/MOON_NAME/FLYBY_ID/`

# In[5]:


# Download Cassini Titan Flyby T011 Files specifying './CASSINI_ARCHIVE/' as custom output
flyby_IDs = 'T011'
kernel_files_cassini, radio_science_files_cassini, ancillary_files_cassini = object.get_mission_files(input_mission = 'cassini', flyby_IDs = flyby_IDs, custom_output = 'CASSINI_CUSTOM_ARCHIVE/')
print(f'Total number of loaded kernels: {spice.get_total_count_of_kernels_loaded()}')


# ## Loaded Kernels for Cassini Titan Flyby (Existing + Downloaded)
# Last, you can print the list of existing + downloaded files. 

# In[6]:


print(f'CASSINI Kernels: {kernel_files_cassini}\n')
print(f'CASSINI Radio Science: {radio_science_files_cassini}\n')
print(f'CASSINI Ancillary: {ancillary_files_cassini}\n')

