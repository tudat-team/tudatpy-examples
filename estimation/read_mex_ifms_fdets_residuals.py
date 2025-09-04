'''
This script allows to plot the fdets and ifms residuals on the same plot, from the residuals txt files
(after running new_mex_residuals_ifms for the ifms, and mex_residuals_fdets for the fdets).
The residual files are found in mex_phobos_flyby/output/fdets_residuals or mex_phobos_flyby/output/ifms_residuals
'''
import csv
import matplotlib.pyplot as plt
from datetime import datetime
import os
import matplotlib.dates as mdates
import numpy as np
import random
from tudatpy.astro import time_representation
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import estimation
def generate_random_color():
    """Generate a random color in hexadecimal format."""
    return "#{:02x}{:02x}{:02x}".format(
        random.randint(0, 255),  # Red
        random.randint(0, 255),  # Green
        random.randint(0, 255)   # Blue
    )

# File path
ifms_residuals_folder = '/Users/lgisolfi/Desktop/mex_phobos_flyby/output/ifms_residuals'
fdets_residuals_folder = '/Users/lgisolfi/Desktop/mex_phobos_flyby/output/fdets_residuals'

station_residuals = dict()


####################################################################################################################################
# Here you can set the residuals to be plotted. I think these three make a good example (they can be compared to Tatiana's).
fdets_files_trial = ['NWNORCIA_residuals.csv','URUMQI_residuals.csv', 'HART15M_residuals.csv']
####################################################################################################################################
# Plotting the residuals
plt.figure(figsize=(10, 6))


for fdets_file in fdets_files_trial:

    # Initialize lists to store UTC times and residuals
    fdets_utc_times = []
    fdets_residuals = []

    # Read the file
    with open(os.path.join(fdets_residuals_folder, fdets_file), mode="r") as file:
        reader = csv.reader(file)

        # Skip the header lines (starting with #)
        for row in reader:
            if row[0].startswith("#"):
                continue
            # Parse UTC Time and Residuals
            fdets_utc_times.append(datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S")) if np.abs(float(row[2])) <= 50 else None
            fdets_residuals.append(float(row[2])) if np.abs(float(row[2])) <= 50 else None

    #Populate Station Residuals Dictionary
    site_name = fdets_file.split('_')[0]
    if site_name not in station_residuals.keys():
        station_residuals[site_name] = [(fdets_utc_times, fdets_residuals)]
    else:
        station_residuals[site_name].append((fdets_utc_times, fdets_residuals))

#tudat_fdets_utc_times = [time_representation.DateTime.from_python_datetime(fdets_utc_time) for fdets_utc_time in fdets_utc_times[]]
#day_arc_filter = estimation.observations_processing.observation_filter(
#    estimation.observations_processing.ObservationFilterType.time_bounds_filtering,
#    time_representation.DateTime.to_epoch(tudat_fdets_utc_times[0]),
#    time_representation.DateTime.to_epoch(tudat_fdets_utc_times[0]) + 86400.0,
#    use_opposite_condition=True,
#    )
#original_odf_observations.filter_observations(day_arc_filter)
#original_odf_observations.remove_empty_observation_sets()

########################################################################################################################
# If uncommented, the following  allows to plot the IFMS residuals (after running the mex_residuals_ifms.py script)

#plt.scatter(fdets_utc_times, fdets_residuals, marker="o",  color="b", label="Fdets Residuals", s = 10)
#plt.scatter(np.array(ifms_utc_times)[np.array(ifms_residuals) <= 1000], np.array(ifms_residuals)[np.array(ifms_residuals) <= 1000], marker="+",  color="r", label="IFMS Residuals", s = 10)

#plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
#plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
#plt.gcf().autofmt_xdate()  # Auto-rotate date labels for better readability
#plt.xlabel("UTC Time", fontsize=12)
#plt.ylabel("Residuals", fontsize=12)
#plt.title("Mars Express GR035 Experiment", fontsize=14)
#plt.grid(True, linestyle="--", alpha=0.6)
#plt.legend()
#plt.tight_layout()
# Show the plot
#plt.show()
########################################################################################################################

# Plot Residuals
added_labels = set()
label_colors = dict()
# Plot residuals for each station
for site_name, data_list in station_residuals.items():
    if site_name not in label_colors:
        label_colors[site_name] = generate_random_color()

    if site_name == 'NWNORCIA':
        label_colors[site_name] = 'lightgray'
    elif site_name == 'HART15M':
        label_colors[site_name] = 'orchid'
    elif site_name == 'URUMQI':
        label_colors[site_name] = 'royalblue'

    for utc_times, residuals in data_list:
        # Plot all stations' residuals on the same figure
        plt.scatter(
            utc_times, residuals,
            color = label_colors[site_name],
            marker = '+', s=7,
            label=f"{site_name}"
            if site_name not in added_labels else None
        )
        added_labels.add(site_name)  # Avoid duplicate labels in the legend


# Flatten only the residual values, ignore datetime
residuals_array = np.concatenate([
    np.ravel([val[1] for val in v]) if isinstance(v[0], (list, tuple, np.ndarray)) else np.ravel(v)
    for v in station_residuals.values()
])

# Compute RMS
print(residuals_array)
overall_rms = np.sqrt(np.mean(residuals_array.astype(float)**2))
print("Overall RMS:", overall_rms)
plt.xlabel("UTC Time", fontsize=15)
plt.ylabel("Residuals (Hz)", fontsize=15)
plt.ylim(-0.045,0.045)
plt.title(f"Overall RMS: {overall_rms*1000:.2f} mHz", fontsize=15)
plt.grid(True, linestyle="--", alpha=0.2)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gcf().autofmt_xdate()  # Auto-rotate date labels for better readability
lgnd = plt.legend(loc='upper left', bbox_to_anchor=(1.00, 1.0), borderaxespad=0.)
#change the marker size manually for both lines
lgnd.legend_handles[0].set_sizes([20])
#lgnd.legend_handles[1].set_sizes([20])

# Adjust layout to make room for the legend
plt.show()

# Define your subset time window
start_time = datetime(2013, 12, 28, 0, 0, 0)  # replace with your desired start
end_time   = datetime(2013, 12, 29, 0, 0, 0) # replace with your desired end

# Plot residuals for each station, filtered by subset time window
plt.figure(figsize=(12, 6))
added_labels = set()
subset_residuals_all = []

for site_name, data_list in station_residuals.items():

    if site_name == 'NWNORCIA':
        color = 'lightgray'
    elif site_name == 'HART15M':
        color = 'orchid'
    elif site_name == 'URUMQI':
        color = 'royalblue'

    for utc_times, residuals in data_list:
        # Convert to NumPy arrays for easy boolean indexing
        utc_times_arr = np.array(utc_times)
        residuals_arr = np.array(residuals)

        # Filter by subset time window
        mask = (utc_times_arr >= start_time) & (utc_times_arr <= end_time)
        utc_subset = utc_times_arr[mask]
        residuals_subset = residuals_arr[mask]

        # Collect residuals for RMS calculation
        subset_residuals_all.append(residuals_subset)

        # Plot subset
        rms_subset = np.sqrt(np.mean(residuals_subset**2))
        plt.scatter(
            utc_subset, residuals_subset,
            label=site_name + f', RMS: {rms_subset*1000:.2f} mHz' if site_name not in added_labels else None,
            s=7, color = color, alpha = 0.5
        )
        added_labels.add(site_name)

# Flatten subset residuals and compute RMS
subset_residuals_flat = np.concatenate(subset_residuals_all)
subset_rms = np.sqrt(np.mean(subset_residuals_flat**2))

# Formatting plot
plt.xlabel("UTC Time", fontsize=15)
plt.ylabel("Residuals (Hz)", fontsize=15)
plt.title(f"Overall RMS: {subset_rms*1000:.2f} mHz", fontsize=15)
plt.grid(True, linestyle="--", alpha=0.5)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gcf().autofmt_xdate()
plt.tick_params(axis='x', labelsize=11)  # x-axis ticks
plt.tick_params(axis='y', labelsize=11)  # y-axis ticks
plt.legend(fontsize = 12)
plt.tight_layout()
plt.show()

plt.close('all')



