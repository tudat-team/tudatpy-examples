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
# 
# Foreseen supported missions (in descending priority order, code yet to be developed):
# 
# - Cassini
# - Insight
# - VEX
# - GRAIL
# - LRO
# 
# **NOTE 2. (Outputs)**
# 
# The downloaded data for a given mission is stored in a folder named `<mission_name>_archive`. Subfolders are created automatically where Kernels, Radio Science Data and Ancillary Data are stored.
# 
# **NOTE 3. (about DSN-TNF)**
# 
# No DSN-TNF download is implemented, since for many applications the TNF is too cumbersome. We download the DSN-ODF files instead. The ODF is an edited and partially processed version of the TNF. It is a smaller file, often issued in daily increments of about 0.2 MB. It contains the most important information (range and Doppler) 
# 
# 
# 

# ## Load required standard modules
# The required modules and dependencies are taken from the python file: `Mission_Data_Retriever_Class.py` present in the`tudatpy-examples/estimation` folder.

# In[2]:


from tudatpy.data.mission_data_downloader import *


# ## Create the LoadPDS Object
# First, we create the LoadPDS() object.

# In[2]:


object = LoadPDS()


# ## Downloader
# 
# ### Set Time Interval(s) for Downloading
# 
# Then we select the `start_date` and `end_date`, and we do so for each mission we wish to download. Of course, each mission has to come with its own dates, as operations are carried out over different periods. 
# 
# ### Download Mission Files
# Finally, we can call the function "get_mission_files
# get_mro_files will download: clock kernels, orientation kernels, radio science (odf) files for mro
# get_mex_files will download: clock kernels, orientation kernels, radio science (ifms) files for mex 
# get_juice_files will download: clock kernels, orientation kernels, for juice. No radio science data, cause none is yet available on the server (fdets retrieval is needed).
# 
# **NOTE 4. (about the following cell output)**
# 
# Please note that, although the output main cell for the MRO mission is visible in this Jupyter Notebook file, the `mro_archive` folder won't be present in the repo. You will need to run this cell yourself in order to automatically create the folder.

# In[3]:


if __name__ == "__main__":

    # Set Time Interval(s) for Downloading

    start_date_juice = datetime(2023, 7, 1)
    end_date_juice = datetime(2023, 8, 10)
    
    start_date_mro = datetime(2007, 1, 3)
    end_date_mro = datetime(2007, 1, 5)
    
    start_date_mex = datetime(2004, 1, 3)
    end_date_mex = datetime(2004,2, 7) 

    # Download Mission Files 
    # (only MRO is shown here. Uncomment the corresponding lines to download data for MEX and JUICE!)
    
    kernel_files_mro, radio_science_files_mro, ancillary_files_mro = object.get_mission_files(input_mission = 'mro', start_date = start_date_mro, end_date = end_date_mro)         
    print(f'Total number of loaded kernels: {spice.get_total_count_of_kernels_loaded()}')
    
    #kernel_files_mex, radio_science_files_mex, ancillary_files_mex = object.get_mission_files(input_mission = 'mex', start_date = start_date_mex, end_date = end_date_mex)         
    #print(f'Total number of loaded kernels: {spice.get_total_count_of_kernels_loaded()}')
    
    #kernel_files_juice, radio_science_files_juice, ancillary_files_juice = object.get_mission_files(input_mission = 'juice', start_date = start_date_juice, end_date = end_date_juice) 
    #print(f'Total number of loaded kernels: {spice.get_total_count_of_kernels_loaded()}')


# ## Loaded Kernels (Existing + Downloaded)
# Last, you can print the list of existing + downloaded files. 

# In[5]:


print(f'MRO Kernels: {kernel_files_mro}\n')
print(f'MRO Radio Science: {radio_science_files_mro}\n')
print(f'MRO Ancillary: {ancillary_files_mro}\n')

#print(f'MEX Kernels: {kernel_files_mex}\n')
#print(f'MEX Radio Science: {radio_science_files_mex}\n')
#print(f'MEX Ancillary: {ancillary_files_mex}\n')

#print(f'JUICE Kernels: {kernel_files_juice}\n')
#print(f'JUICE Radio Science: {radio_science_files_juice}\n') # it will be empty for now... (no Radio Science yet available on the server)
#print(f'JUICE Ancillary: {ancillary_files_juice}\n') # it will be empty for now... (no Ancillary files yet available on the server)

