import numpy as np
import matplotlib.pyplot as plt
import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from tudatpy.interface import spice
from tudatpy.dynamics import environment_setup
from tudatpy.astro import element_conversion

# -----------------------------------------------------------------------------
# 0. CONFIGURATION & CLEANUP
# -----------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning, module="tudatpy.estimation")

def get_station_geodetic_coordinates(station_settings_list):
    """
    Extract geodetic coordinates (lon, lat) from Tudat ground stations.
    """
    equatorial_radius = 6378.137e3  # WGS84
    flattening = 1.0 / 298.257223563

    lons, lats, names = [], [], []

    for station in station_settings_list:
        cartesian_pos = station.station_position

        geodetic_pos = element_conversion.convert_cartesian_to_geodetic_coordinates(
            cartesian_coordinates=cartesian_pos,
            equatorial_radius=equatorial_radius,
            flattening=flattening,
            tolerance=1.0e-4
        )

        lats.append(np.rad2deg(geodetic_pos[1]))
        lons.append(np.rad2deg(geodetic_pos[2]))
        names.append(station.station_name)

    return lons, lats, names

# -----------------------------------------------------------------------------
# 1. RETRIEVAL
# -----------------------------------------------------------------------------
spice.load_standard_kernels()

print("Retrieving station data...")
radio_stations = environment_setup.ground_station.radio_telescope_stations()
optical_stations = environment_setup.ground_station.optical_telescope_stations()

radio_lons, radio_lats, _ = get_station_geodetic_coordinates(radio_stations)
optical_lons, optical_lats, _ = get_station_geodetic_coordinates(optical_stations)

# -----------------------------------------------------------------------------
# 2. PLOTTING (ROBUST, NO IMAGE DOWNLOAD)
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(15, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

# Background
ax.add_feature(cfeature.OCEAN, facecolor='#A6CAE0', zorder=0)
ax.add_feature(cfeature.LAND, facecolor='#EFE8D8', zorder=0)

# Geographic features
ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=1)
ax.add_feature(cfeature.BORDERS, linewidth=0.6, zorder=1)
ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor='gray', zorder=1)


ax.scatter(
    optical_lons, optical_lats,
    c='orange', edgecolors='black', s=50, marker='^',
    label='Optical Telescopes', linewidth=1, zorder=4,
    transform=ccrs.PlateCarree(), alpha = 0.5
)


# Stations
ax.scatter(
    radio_lons, radio_lats,
    c='cyan', edgecolors='white', s=50, marker='o',
    label='Radio Telescopes', linewidth=1.5, zorder=3,
    transform=ccrs.PlateCarree()
)
# -----------------------------------------------------------------------------
# 3. FORMATTING
# -----------------------------------------------------------------------------
ax.set_title(
    "Tudat-Available Ground Stations",
    fontsize=18, fontweight='bold', pad=15
)

ax.set_global()
ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
gl.top_labels = False
gl.right_labels = False

ax.legend(
    loc='upper right',
    frameon=True,
    facecolor='white',
    framealpha=0.9,
    fontsize=12,
    shadow=True
)

ax.text(
    0, -100,
    "Source: Tudatpy Ground Station Database",
    ha='center',
    fontsize=10,
    style='italic',
    transform=ccrs.PlateCarree()
)

plt.tight_layout()
plt.show()
