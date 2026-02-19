from tudatpy.data.spacetrack import SpaceTrackQuery, OMMUtils
from tudatpy.dynamics import environment_setup, propagation_setup, simulator
from tudatpy.astro import time_representation
from tudatpy.util import result2array
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from tudatpy.interface import spice

# Load Spice kernels
spice.load_standard_kernels()

# Initialize SpaceTrack
username = os.getenv("SPACETRACK_USER")
password = os.getenv("SPACETRACK_PASS")

spacetrack_query = SpaceTrackQuery(
    username=username,
    password=password,
    spacetrack_url='https://for-testing-only.space-track.org'
)

custom_catalog_file = "my_polar_satellites.json"
filepath = os.path.join(spacetrack_query.tle_data_folder, custom_catalog_file)

# --- Conditional Fetching ---
# Logic: Only hit the API if the file doesn't exist locally

# --- STEP 1 & 2: Fetch Data for Polar Satellites ---
spacetrack_query.filtered_by_oe_dict(
    filter_oe_dict={'INCLINATION': (97.0, 99.0)},
    limit=5,
    output_file=custom_catalog_file,
    update_existing=False
)

# --- STEP 1 & 2: Fetch Data for Equatorial Satellites, Use update_existing = True ---
spacetrack_query.filtered_by_oe_dict(
    filter_oe_dict={'INCLINATION': (-10,10)},
    limit=10,
    output_file=custom_catalog_file,
    update_existing=True
)

# --- Load Data ---
filepath = os.path.join(spacetrack_query.tle_data_folder, custom_catalog_file)
with open(filepath, 'r') as f:
    content = json.load(f)
    data_list = content['data'] if isinstance(content, dict) else content

# --- ENVIRONMENT SETUP (Required for Lat/Lon) ---
bodies_to_create = ["Earth"]
global_frame_origin = "Earth"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(bodies_to_create, global_frame_origin, global_frame_orientation)

# Set Earth rotation and shape (essential for geodetic calculations)
body_settings.get("Earth").rotation_model_settings = environment_setup.rotation_model.gcrs_to_itrs(
    environment_setup.rotation_model.iau_2006, global_frame_orientation)
body_settings.get("Earth").shape_settings = environment_setup.shape.oblate_spherical(6378137.0, 1.0 / 298.257)

bodies = environment_setup.create_system_of_bodies(body_settings)
body_settings.add_empty_settings("sat")
system_of_bodies = environment_setup.create_system_of_bodies(body_settings)

# --- PROPAGATION & PLOTTING ---
plt.figure(figsize=(15, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.stock_img()
ax.coastlines()

unique_sats = {sat['NORAD_CAT_ID']: sat for sat in data_list}

for norad_id, sat_data in unique_sats.items():
    name = sat_data.get('OBJECT_NAME', f"ID-{norad_id}")
    print(f"Propagating ground track for: {name}")

    # Initial State from TLE
    line1, line2 = sat_data['TLE_LINE1'], sat_data['TLE_LINE2']
    tle_epoch = OMMUtils.get_tle_reference_epoch(line1)

    # Convert Epoch to TDB
    time_converter = time_representation.default_time_scale_converter()
    initial_time = time_converter.convert_time(
        time_representation.utc_scale, time_representation.tdb_scale,
        time_representation.DateTime.from_python_datetime(tle_epoch).to_epoch()
    )

    # Setup Ephemeris for Initial State
    tle_obj = environment_setup.ephemeris.sgp4(line1, line2)
    initial_state = environment_setup.create_body_ephemeris(tle_obj, "Earth").cartesian_state(initial_time)

    # Accelerations (Simplified Point Mass)
    accel_settings = {"sat": {"Earth": [propagation_setup.acceleration.point_mass_gravity()]}}
    accel_models = propagation_setup.create_acceleration_models(system_of_bodies, accel_settings, ["sat"], ["Earth"])

    # Dependent Variables: Latitude and Longitude
    dep_vars = [
        propagation_setup.dependent_variable.geodetic_latitude("sat", "Earth"),
        propagation_setup.dependent_variable.longitude("sat", "Earth")
    ]

    # Termination and Integrator
    termination = propagation_setup.propagator.time_termination(initial_time + 72000) # 20 hours
    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(initial_time_step=60.0,
                                    coefficient_set=propagation_setup.integrator.CoefficientSets.rkdp_87)


    # Propagate
    prop_settings = propagation_setup.propagator.translational(
        ["Earth"], accel_models, ["sat"], initial_state, initial_time,
        integrator_settings, termination, output_variables=dep_vars
    )

    results = simulator.create_dynamics_simulator(system_of_bodies, prop_settings)
    dep_var_array = result2array(results.propagation_results.dependent_variable_history)

    # Plotting Ground Track
    lats = np.degrees(dep_var_array[:, 1])
    lons = (np.degrees(dep_var_array[:, 2]) + 180) % 360 - 180

    # Use scatter to avoid lines jumping across the map wrap-around
    ax.scatter(lons, lats, s=5, label=name, transform=ccrs.Geodetic())

plt.title("Satellite Ground Tracks (2-Hour Propagation)")
plt.legend(loc='lower left', markerscale=5)
plt.show()