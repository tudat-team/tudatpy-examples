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
from tudatpy.astro import time_conversion
from tudatpy.numerical_simulation.estimation_setup import observation
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


fdets_files_trial = ['METSAHOV_residuals.csv', 'MK-VLBA_residuals.csv']
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

# Plot Residuals
added_labels = set()
label_colors = dict()
# Plot residuals for each station
for site_name, data_list in station_residuals.items():
    if site_name not in label_colors:
        label_colors[site_name] = generate_random_color()

    print(label_colors)
    print(data_list)
    for utc_times, residuals in data_list:
        # Plot all stations' residuals on the same figure
        plt.scatter(
            utc_times, residuals,
            color = label_colors[site_name],
            marker = '+', s=10,
            label=f"{site_name}"
            if site_name not in added_labels else None
        )
        added_labels.add(site_name)  # Avoid duplicate labels in the legend

# Format the x-axis for dates
plt.xlabel("UTC Time", fontsize=12)
plt.ylabel("Residuals", fontsize=12)
plt.ylim(-1,1)
plt.title("Mars Express GR035 Experiment", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gcf().autofmt_xdate()  # Auto-rotate date labels for better readability
lgnd = plt.legend(loc='upper left', bbox_to_anchor=(1.00, 1.0), borderaxespad=0.)
#change the marker size manually for both lines
lgnd.legend_handles[0].set_sizes([20])
lgnd.legend_handles[1].set_sizes([20])

# Adjust layout to make room for the legend
plt.show()
plt.close('all')
exit()
