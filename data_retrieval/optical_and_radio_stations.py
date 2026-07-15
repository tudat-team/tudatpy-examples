"""
# Available Optical and Radio Stations in Tudat

## Objectives
With this short example, you will see how easy it is, within Tudatpy, to load pre-existing optical and radio stations coordinates. The mapping of cartesian coordinates to the telescope codes and names can be found in two files, named [`glo.sit`](https://gitlab.com/gofrito/pysctrack/-/blob/master/cats/glo.sit?ref_type=heads) and [`mpc.sit`](https://www.projectpluto.com/mpc_stat.txt) (click on their names to be redirected to the links where they are hosted). These come packaged with Tudatpy. 

## Import Statements
Here, we load the needed Tudatpy dependencies, such as `environment_setup` to load the stations, and `element_conversion`, used to convert cartesian positions into geodetic coordinates. Moreover, we will use the `cartopy` dependency to show nice maps to get an idea on where exactly the ground stations are located in the world. 
"""


import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tudatpy.dynamics import environment_setup
from tudatpy.astro import element_conversion
from  tudatpy import constants

"""
## Helper Function
We define an helper function that takes the station settings list as input, and returns arrays of geodetic latitudes and longitudes, and the corresponding station codes. We need such a function, because the original files (`glo.sit`, `mpc.sit`) store coordinates in cartesian elements, yet we want to plot geodetic coordinates. We will use the [WGS84](https://share.google/d2p2b46f8FWXm3DJc) values to define Earth's flattening and equatorial radius. 
"""


def get_station_geodetic_coordinates(station_settings_list):
    """
    Extract geodetic coordinates (lon, lat) from Tudat ground stations.
    """
    equatorial_radius = 6378.137e3  # WGS84
    flattening = 1.0 / 298.257223563 # WGS84

    lons, lats, codes = [], [], []

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
        codes.append(station.station_name)

    return np.array(lons), np.array(lats), np.array(codes)


def get_single_geodetic_coordinate(cartesian_pos):
    """Helper to convert a single cartesian coordinate to geodetic."""
    equatorial_radius = 6378.137e3
    flattening = 1.0 / 298.257223563
    geodetic_pos = element_conversion.convert_cartesian_to_geodetic_coordinates(
        cartesian_coordinates=cartesian_pos,
        equatorial_radius=equatorial_radius,
        flattening=flattening,
        tolerance=1.0e-4
    )
    return np.rad2deg(geodetic_pos[2]), np.rad2deg(geodetic_pos[1])  # lon, lat

"""
## Load ground station settings

The next snippet shows how easy it is to load both radio and optical telescope settings within Tudatpy. You just need two lines to unlock a huge number of stations, whose coordinates are ready to be used within a tudat observation simulation or estimation script. 
"""


radio_stations_settings = environment_setup.ground_station.radio_telescope_stations()
optical_stations_settings = environment_setup.ground_station.optical_telescope_stations()

# Let's find "DWINGELO" in our list to manipulate it
dwingelo_station = None
for station in radio_stations_settings:
    if station.station_name == "DWINGELO":
        dwingelo_station = station
        break

## --- Define and Apply Linear Station Motion ---
# 1. Define a reference epoch (e.g., J2000 = 0.0 seconds)
ref_epoch = 0.0

# 2. Define a linear velocity vector [Vx, Vy, Vz] in m/s.
# Realistic tectonic drift is ~few cm/year. For visualization, let's exaggerate
# the drift significantly, or simulate a long timespan.
# Let's assign a velocity of ~0.1 meters per year (~3.17e-9 m/s) eastward/northward
linear_velocity = np.array([0.05, 0.05, 0.02]) # m/s in Cartesian body-fixed frame

# 3. Create the motion settings and append it to our station
motion_settings = environment_setup.ground_station.linear_station_motion(
    linear_velocity=linear_velocity,
    reference_epoch=ref_epoch
)

# Assign the motion model to the station settings object
dwingelo_station.station_motion_settings.append(motion_settings)

# Calculate a future position manually to plot the shift
# Let's look 50,000 Julian Years into the future
years_future = 50000
seconds_in_future = years_future * constants.JULIAN_YEAR

# New Cartesian position = Reference Position + (Velocity * Delta_t)
initial_cartesian = dwingelo_station.station_position
future_cartesian = initial_cartesian + (linear_velocity * seconds_in_future)

# Convert both to geodetic for plotting
orig_lon, orig_lat = get_single_geodetic_coordinate(initial_cartesian)
future_lon, future_lat = get_single_geodetic_coordinate(future_cartesian)

"""
## Explore Available Ground Stations

We are now finally ready to give both radio and optcial stations settings to the helper function we defined above. We can also take a look at the vlbi denominations and mpc codes by printing their names. 
"""


radio_lons, radio_lats, radio_names = get_station_geodetic_coordinates(radio_stations_settings)
optical_lons, optical_lats, optical_names = get_station_geodetic_coordinates(optical_stations_settings)

print("Slice of Radio Codes:", radio_names[20:30])
print("Slice of Optical Codes:", optical_names[20:30])


"""
## Plot Ground Stations
Last but not least, we can plot the geodetic coordinates of all our Tudat-available ground stations. As an example, we plot `DWINGELOO` and the `VERA RUBIN OBSERVATORY` with different colors and shapes, to make them stand out. Play around with this code to make your favourite ground stations stand out!
"""


fig, axes = plt.subplots(2, 1, figsize=(15, 12), subplot_kw={'projection': ccrs.PlateCarree()})
titles = ["Radio Telescopes", "Optical Telescopes"]
data = [(radio_lons, radio_lats, 'orange', 'o', 'Radio'), 
        (optical_lons, optical_lats, 'cyan', 'o', 'Optical')]
for i, ax in enumerate(axes):
    # Background
    ax.add_feature(cfeature.OCEAN, facecolor='#A6CAE0', zorder=0)
    ax.add_feature(cfeature.LAND, facecolor='#EFE8D8', zorder=0)

    # Geographic features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6, zorder=1)
    ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor='gray', zorder=1)

    lons, lats, color, marker, label = data[i]

    ax.scatter(
        lons, lats,
        c=color, edgecolors='black', s=20, marker=marker,
        label=f'{label} Stations', linewidth=1, zorder=3,
        transform=ccrs.PlateCarree()
    )

    if label == "Radio":
        dwingeloo_index = np.where(np.array(radio_names) == "DWINGELO")[0]
        ax.scatter(
            lons[dwingeloo_index], lats[dwingeloo_index],
            c="red", edgecolors='yellow', s=100, marker="*",
            label=f'DWINGELOO', linewidth=1, zorder=3,
            transform=ccrs.PlateCarree()
        )
        ax.scatter(future_lon, future_lat, c="red", edgecolors='black', s=80, marker="X", label=f'DWINGELOO (+{years_future} Years)', zorder=3)

    elif label == "Optical":
        vera_rubin_index = np.where(np.array(optical_names) == "X05")[0]
        ax.scatter(
            lons[vera_rubin_index], lats[vera_rubin_index],
            c="red", edgecolors='yellow', s=100, marker="*",
            label=f'VERA RUBIN', linewidth=1, zorder=3,
            transform=ccrs.PlateCarree()
        )

    ax.set_title(titles[i], fontsize=14, fontweight='bold')
    ax.set_global()
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

fig.suptitle("Tudat-Available Ground Stations", fontsize=18, fontweight='bold', y=0.95)
plt.figtext(
    0.5, 0.02,
    "Source: Tudatpy Ground Station Database",
    ha='center',
    fontsize=10,
    style='italic'
)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


plt.show()