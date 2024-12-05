#!/usr/bin/env python
# coding: utf-8

# ## Objectives
# 
# - This script shows how to use the TudatPy `mission_data_downloader` functionality  to download:
#     1) **SPICE** (clock, frame, orientation, planetary) **Kernels**
#     2) **Closed Loop Doppler Data** (DSN and IFMS)
#     3) **Ancillary** Ionospheric and Tropospheric **Data**
# 
#    for some supported missions (see **NOTE 1** for a list of supported missions). 
# 
# - Moreover, we also show how to use the functionality to download the same type of files for a **custom, user-defined mission**.
# 
# - Last but not least, we show how to use tudatpy to download:
#   **all files listed in a given SPICE Meta-Kernel**.
# 
#   The mission archive is **cleaned up (namely, empty folders are removed)** at the end of each run.
# 
# ### NOTES
# **NOTE 1. (Supported Missions and Relative Downloaded Data).**
# 
# The following spice kernels can be loaded automatically at the end of a run, if  the `load_kernels` flag in the `get_mission_files` function is set to **True** (see MRO example in the next cells):
# 
# 1) **existing spice kernels** in the mission folder
# 2) (new) **downloaded spice kernels**
# 3) **standard TUDAT kernels** from [`spice.load_standard_kernels()`](https://py.api.tudat.space/en/latest/spice.html#tudatpy.interface.spice.load_standard_kernels) 
# 
# List of currently supported input_missions:
# - **Mars Reconnnaissance Orbiter (MRO)** [Available: Spice Kernels, Doppler Data, Ancillary]
# - **Mars Express (MEX)** [Available: Spice Kernels, Doppler Data, Ancillary]
# - **Jupiter Icy Moons Explorer (JUICE)** [Available: Spice Kernels, Not Available (yet): Doppler Data, Ancillary]
# - **Cassini** [Available: Spice Kernels, Doppler Data, Ancillary for all **Titan Flybys**]
# - **GRAIL (both Grail-A and Grail-B)** [Available: Spice Kernels, Doppler Data, Ancillary]
# 
# Foreseen supported missions (in descending priority order, code yet to be developed):
# 
# - Insight
# - VEX
# - LRO
# 
# **NOTE 2. (Default and Custom Outputs)**
# 
# The downloaded data for a given mission is stored (by default) in a folder named `<mission_name>_archive`. Subfolders are created automatically where Kernels, Radio Science Data and Ancillary Data are stored. 
# The user can still define a custom output path passing the flag `custom_output` in the `get_mission_files` function. The output files will then be found in: `custom_path`
# 
# **NOTE 3. (About DSN-TNF)**
# 
# No DSN-TNF download functionality has been implemented (at least, not yet!)

# ## Load required standard modules
# The required modules and dependencies are taken from the `mission_data_downloader` class.

# In[1]:


from tudatpy.data.mission_data_downloader import *


# ## Create the LoadPDS Object
# First, we create the LoadPDS() object.

# In[2]:


object = LoadPDS()
spice.clear_kernels() #lets clear the kernels to avoid duplicates,since we will load all standard + Downloaded + existing kernels


# ## MRO, MEX and JUICE Downloaders (Default Output)
# 
# ### Set Time Interval(s) for Downloading
# 
# Then we select the `start_date` and `end_date`, and we do so for each mission we wish to download (Cassini will be an exception due to the peculiar mission concept, see **Cassini Downloader** below). Of course, each mission has to come with its own dates, as operations are carried out over different periods.
# 
# ### Download Mission Files (MRO, MEX, JUICE)
# 
# 
# Finally, we can call the function `get_mission_files`, specifying the above-defined start and end dates.
# 
# Only for this first example, we set the `load_kernels` flag to **True**. This way, we allow for automatic loading of SPICE kernels. If you do not wish to load them, you can simply remove the flag.
# 
# **NOTE 5. (About MEX and JUICE Files Download)**
# 
# Here, we will only showcase the downloading of MRO files. However, **if you wish to download files for MEX or JUICE**, you can simply **uncomment the corresponding lines** in the cell below! We have included those **just for you**!
# 

# In[3]:


start_date_mro = datetime(2007, 1, 3)
end_date_mro = datetime(2007, 1, 5)

#start_date_mex = datetime(2004, 1, 3)
#end_date_mex = datetime(2004,2, 7) 

#start_date_juice = datetime(2023, 7, 1)
#end_date_juice = datetime(2023, 8, 10)

# Download Mission Files with default output folder
# (only MRO is shown here. Uncomment the corresponding lines to download data for MEX and JUICE!)

kernel_files_mro, radio_science_files_mro, ancillary_files_mro = object.get_mission_files(
    input_mission = 'mro', 
    start_date = start_date_mro, 
    end_date = end_date_mro, 
    custom_output = None,
    load_kernels = True)      

print(f'Total number of loaded kernels: {spice.get_total_count_of_kernels_loaded()}')

#kernel_files_mex, radio_science_files_mex, ancillary_files_mex = object.get_mission_files(
#    input_mission = 'mex', 
#    start_date = start_date_mex, 
#    end_date = end_date_mex)         
#print(f'Total number of loaded kernels: {spice.get_total_count_of_kernels_loaded()}')

#kernel_files_juice, radio_science_files_juice, ancillary_files_juice = object.get_mission_files(
#    input_mission = 'juice',
#    start_date = start_date_juice,
#    end_date = end_date_juice) 
#print(f'Total number of loaded kernels: {spice.get_total_count_of_kernels_loaded()}')


# ### Loaded Kernels for MRO, MEX and JUICE (Existing + Downloaded)
# Let's print the list of existing + downloaded files. 

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
# For this reason, in order to retrieve Cassini data, we will require the **name of the flyby** data the user wishes to download, rather than a start and end date (see **Notes 5 and 6** for info on the supported flybys). 
# 
# ### How to pick a Cassini flyby?
# - If you already know the name of the flyby you want to download (e.g. T011, T022, etc...), you can simply call the function `get_mission_files` with `input_mission = 'Cassini'`, also adding the flag `flyby_IDs = T011` (or `flyby_IDs = T022`, ec...). 
# 
# - Alternatively, calling `get_mission_files` with 'Cassini' as `input_mission` and no `flyby_IDs` flag **will print a comprehensive table of supported flybys to choose from**, and you will be asked to **manually input** the name of one (or more) of them. This table is constructed based on the [Cassini Titan Gravity Science Table](https://pds-atmospheres.nmsu.edu/data_and_services/atmospheres_data/Cassini/Cassini/RSS%20PDS%20page%202019-01-23/rss/TI_12.html).
# 
# ### Flyby Data Output Division
# 
# The default output folder will be named `cassini_archive/MOON_NAME/FLYBY_ID/`, where `MOON_NAME` is the flyby moon, and `FLYBY_ID` is the denomination of each downloaded flyby (e.g. T011, T022). Each subfolder will contain kernel ancillary and radio science subdirectories. In this example, we will download files from the **flyby T011**, and we will store them in the custom path: `'CASSINI_CUSTOM_ARCHIVE/'`
# 
# 
# ### NOTES
# **NOTE 5. Supported Flyby Moons**
# 
# For now, only Titan Flybys are supported.
# 
# **NOTE 6. All flyby_IDs types**
# 
# `flyby_IDs` can be:
# 1) a list made of single flybys like: `flyby_IDs = ['T011', 'T022']` or `flyby_IDs = ['T011']`
# 2) a single string object like: `flyby_IDs = 'T011'` (not a list)
# 3) a list made of all flybys performed at a given moon: `flyby_IDs = ['ALL_TITAN']`
# 4) a single string object like: `flyby_IDs = 'ALL_TITAN'`
# 5) a mixed list like: `flyby_IDs = ['T011', 'ALL_TITAN']`
# 
# As mentioned above in *Download Cassini Flyby Files*, **you can also decide not to specify any flyby_ID**. 
# In this case, **a table will be printed out** from which you will be able to interactively select the flyby you're interested in. 
# 
# **NOTE 7. Custom Output**
# 
# As we mentioned above, the **default** output folder will be: `cassini_archive/MOON_NAME/FLYBY_ID/`
# The user can still define a custom output path using the flag `custom_output` in the `get_mission_files` function. The output files will then be found in: `custom_path/MOON_NAME/FLYBY_ID/`

# In[5]:


# Download Cassini Titan Flyby T011 Files specifying './CASSINI_ARCHIVE/' as custom output
flyby_IDs = 'T011'
kernel_files_cassini, radio_science_files_cassini, ancillary_files_cassini = object.get_mission_files(
    input_mission = 'cassini', 
    flyby_IDs = flyby_IDs, 
    custom_output = 'CASSINI_CUSTOM_ARCHIVE/')
print(f'Total number of loaded kernels: {spice.get_total_count_of_kernels_loaded()}')


# ### Loaded Kernels for Cassini Titan Flyby (Existing + Downloaded)
# Last, you can print the list of existing + downloaded files. 

# In[6]:


print(f'CASSINI Kernels: {kernel_files_cassini}\n')
print(f'CASSINI Radio Science: {radio_science_files_cassini}\n')
print(f'CASSINI Ancillary: {ancillary_files_cassini}\n')


# ## GRAIL-A and GRAIL-B Downloaders (Custom Output)
# You can also download **GRAIL data** for both **GRAIL-A and GRAIL-B** spacecraft. These can be downloaded as usual via the command: `get_mission_files`, specifying either: 'grail-a' or 'grail_b' as `input_mission` (we add the lines relative to **GRAIL-B as comments**, but users are definitely encouraged to **uncomment them and try them out!**) Two folders will be created by default: `grail_archive/grail-a` and `grail_archive/grail-b`. However, you can still choose your own custom output folder. Let's call it `'GRAIL_ARCHIVE'`.

# In[7]:


start_date_grail_a = datetime(2012, 4, 6)
end_date_grail_a = datetime(2012, 4, 12)
start_date_grail_b = datetime(2012, 5, 6)
end_date_grail_b = datetime(2012, 6, 12)

kernel_files_grail_a, radio_science_files_grail_a, ancillary_files_grail_a = object.get_mission_files(
    input_mission = 'grail-a', 
    start_date = start_date_grail_a, 
    end_date = end_date_grail_a, 
    custom_output = './GRAIL_ARCHIVE'
)         
print(f'Total number of loaded kernels: {spice.get_total_count_of_kernels_loaded()}\n\n')

print(f'GRAIL-A Kernels: {kernel_files_grail_a}\n')
print(f'GRAIL-A Radio Science: {radio_science_files_grail_a}\n')
print(f'GRAIL-A Ancillary: {ancillary_files_grail_a}\n\n')

#kernel_files_grail_b, radio_science_files_grail_b, ancillary_files_grail_b = object.get_mission_files(
#    input_mission = 'grail-b', start_date = start_date_grail_b, end_date = end_date_grail_b, custom_output = './GRAIL_ARCHIVE'
#)         
#print(f'Total number of loaded kernels: {spice.get_total_count_of_kernels_loaded()}\n\n')

#print(f'GRAIL-B Kernels: {kernel_files_grail_b}\n')
#print(f'GRAIL-B Radio Science: {radio_science_files_grail_b}\n')
#print(f'GRAIL-B Ancillary: {ancillary_files_grail_b}\n')


# ## Downloading Kernels For Non-Supported Missions 
# As we mentioned in the beginning, users can also download **SPICE kernels** and **Ancillary Data** for non-supported missions (**Doppler data download** for non-supported missions is **not supported yet**!). 
# 
# ### Customizing URLs and Patterns
# The folllowing cell shows how to download SPICE Orientation Kernels (ck) for **LRO** (not supported mission) specifying **custom url** and **output folder**.
# 
# The steps to be performed are:
# 
# 1) Define the name of your `custom_input_mission`;
# 2) Add a `custom_ck_url` (namely, the NAIF link where ck kernels for the mission are stored);
# 3) Define a `placeholders` list.
# 
#       ***IMPORTANT!***
#    
#       *`placeholders` is a list containing the single pieces that make up a "template filename", selected by the user.
#       For instance, **let's suppose you want to download all files with the same structure as**: `lrodv_2012060_2012092_v01.bc`. This will then be considered as your **template filename**.*
#       *The `placeholders` list defines the keys to a vocabulary, plus the underscores positions.*
#       
#       *In this case:*
#       
#       *mission = lro,*
#       *instrument =  dv,*
#       *start_date = 2012060,*
#       *end_date = 2012092,*
#       *version = v01,*
#       *extension = bc*
#       
#       *and corresponding placeholders list will be:*
#       *placeholders = ['mission', 'instrument', '_', 'start_date_file', '_', 'end_date_file', '_', 'version', 'extension'],*
#       
#       *also encoding the underscores positions.*
#       
# 5) Create the pattern corresponding to the `placeholders` list;
# 6) Add the corresponding pattern to the list of supported patterns;
# 7) Define the `custom_local_path`  where you want to store your data;
# 8) Call the `get_mission_files` function as done for other missions already, and do not forget the **flag**: `all_meta_kernel_files`.
# 9) Specify the start and end date
# 10) Call the `dynamic_download_url_files_time_interval` function.
# 
# **NOTE 8. About Start and End Dates** 
# 
# As it will be shown later in the example, you can also **[download all files of a certain type](#Download_all_Files_of_a_Certain_Type)** (ck, spk, fk, odf, etc...) present at a given url, **regardless of the start and end date**.

# In[8]:


custom_input_mission = 'lro'
custom_ck_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/lro-l-spice-6-v1.0/lrosp_1000/data/ck/'

local_path = './lro_archive'
start_date_lro = datetime(2009, 7, 1)
end_date_lro = datetime(2009, 7, 10)

object.add_custom_mission_kernel_url(custom_input_mission, custom_ck_url)

placeholders_ck = ['mission', 'instrument', '_', 'start_date_file', '_', 'end_date_file', '_', 'version', 'extension']
custom_ck_pattern = object.create_pattern(placeholders_ck)

object.add_custom_mission_kernel_pattern(custom_input_mission, 'ck', custom_ck_pattern)
object.dynamic_download_url_files_time_interval(custom_input_mission, local_path, start_date_lro, end_date_lro, custom_ck_url)

custom_spk_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/lro-l-spice-6-v1.0/lrosp_1000/data/spk/'
placeholders_spk = ['mission', 'instrument', '_', 'start_date_file', '_', 'end_date_file', '_', 'version', 'extension']
custom_spk_pattern = object.create_pattern(placeholders_spk)

object.add_custom_mission_kernel_url(custom_input_mission, custom_spk_url)
object.add_custom_mission_kernel_pattern(custom_input_mission, 'spk', custom_spk_pattern)
object.dynamic_download_url_files_time_interval(custom_input_mission, local_path, start_date_lro, end_date_lro, custom_spk_url)


# ## Download all Files of a Certain Type
# 
# Users might also be interested in downloaing all files of a certain type, say all frame kernels for lro, all starting in `lro` and ending in `*.tf`. 
# Here's how it's done:
# 
# 1) Define the name of your `custom_input_mission`;
# 2) Define a `custom_fk_url` (namely, the NAIF link where fk kernels for the mission are stored);
# 3) call the `get_kernels` function.
#    Its inputs are:
#    - input_mission [required]
#    - url (from which the files will be retrieved) [required]
#    - wanted_files (list of files present at a given url for which you already know the name) [optional]
#    - wanted_files_pattern (list of patterns of files to be downloaded, for instance: `wanted_files_patterns = ['lro*.tf']`) [optional]
#    - custom_output [optional]
# <a id='Download_all_Files_of_a_Certain_Type'></a>

# In[9]:


custom_fk_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/lro-l-spice-6-v1.0/lrosp_1000/data/fk/'
object.get_kernels(input_mission = 'lro', url = custom_fk_url, wanted_files_patterns = ['lro*.tf'])


# ## Downloading Custom Meta-Kernel Files for Non-Supported Missions
# The following cell shows how to download all relevant files specified in a custom mission [Meta-Kernel](https://naif.jpl.nasa.gov/naif/furnsh_details.html#:~:text=A%20meta%2Dkernel%2C%20also%20known,one%20or%20more%20meta%2Dkernels.) using custom inputs.
# The steps to be performed are:
# 
# 1) Define the name of your `custom_input_mission`;
# 2) Add a `custom_meta_kernel_url` associated to the `custom_input_mission`;
# 3) Add a `custom_meta_kernel_pattern`  (or name, if you already know what's the exact name of the **meta-kernel** to be downloaded!);
# 4) Add the `custom_kernels_url` where kernels for your custom input mission are stored;
# 5) Define the `custom_local_path` where you want to store your data;
# 6) Call the `get_mission_files` function as done for other missions already, and do not forget the **flag**: `all_meta_kernel_files`;
# 
# Get popcorns, it's gonna take a (long) while! 🍿
# 
# **NOTE 9. About the Output of the Following Cell** 
# 
# As for the following cell, **we avoid plotting its output**, for two main reasons:
# - outputs are essentially the same as the previous ones, so it's easy to understand them;
# - the full downloading of all files listed in a Meta-Kernel typically takes a long time.
# 
# Neverthless, we encourage users to play with this cool feature!

# In[ ]:


custom_input_mission = 'lro'
custom_meta_kernel_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/lro-l-spice-6-v1.0/lrosp_1000/extras/mk/'
custom_meta_kernel_pattern = 'lro_2024_v02.tm'
custom_kernels_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/lro-l-spice-6-v1.0/lrosp_1000/data/'

object.add_custom_mission_meta_kernel_url(custom_input_mission , custom_meta_kernel_url)
object.add_custom_mission_meta_kernel_pattern(custom_input_mission , custom_meta_kernel_pattern)
object.add_custom_mission_kernel_url(custom_input_mission, custom_kernels_url)
custom_local_path = './lro_archive'

object.get_mission_files(
    input_mission = custom_input_mission, 
    custom_output = custom_local_path, 
    all_meta_kernel_files = True)

