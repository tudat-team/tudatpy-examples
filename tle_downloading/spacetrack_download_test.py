from tudatpy.dynamics import environment
from tudatpy.astro.time_representation import DateTime
import numpy as np
import matplotlib.pyplot as plt
from tudatpy.data.spacetrack import SpaceTrackQuery
import math
from collections import defaultdict



def cartesian2radec(vec):
    """
    Convert 3D Cartesian vector to (range, RA, Dec).

    Parameters
    ----------
    vec : array_like
        Cartesian vector [x, y, z] in km (or any consistent unit)

    Returns
    -------
    r : float
        Range (magnitude) of the vector
    ra_deg : float
        Right ascension in degrees (0 to 360)
    dec_deg : float
        Declination in degrees (-90 to 90)
    """
    x, y, z = vec
    r = np.sqrt(x**2 + y**2 + z**2)
    ra = np.arctan2(y, x)           # radians
    if ra < 0:
        ra += 2 * np.pi             # Normalize to [0, 2π)
    dec = np.arcsin(z / r)          # radians

    ra_deg = np.degrees(ra)
    dec_deg = np.degrees(dec)

    return r, ra_deg, dec_deg


DEMO_FLAG = True
username = 'l.gisolfi@tudelft.nl'
password = 'l.gisolfi*tudelft.nl'

if DEMO_FLAG:
    q = SpaceTrackQuery(username, password)
    json_dict_5001 = q.download_tle.single_norad_id(5001, 5)
    json_dict_5000 = q.download_tle.single_norad_id(5000, 5)
    json_dict_descending = q.download_tle.descending_epoch(5)

    json_dict_list = [json_dict_5001,json_dict_descending, json_dict_5000]

    # Time span (e.g., 12 hours sampled every minute)
    simulation_start_epoch = DateTime(2025, 7, 15, 0).epoch()
    simulation_end_epoch = DateTime(2025, 7, 18, 15).epoch()
    list_of_times = np.arange(simulation_start_epoch, simulation_end_epoch, 120)

    # Determine subplot grid size (square or close to square)
    n = len(json_dict_list)
    cols = math.ceil(np.sqrt(n))
    rows = math.ceil(n / cols)

    # Create figure
    for idx, json_dict in enumerate(json_dict_list):
        tle_dict = q.OMMUtils.get_tles(SpaceTrackQuery, json_dict)
        trajectory_dict = defaultdict(list)
        radec_dict = defaultdict(list)
        for key, value in tle_dict.items():
            if len(value) > 1:
                print(f"Multiple TLEs for object {key}. Plotting the most recent one.")
                tle_line_1, tle_line_2 = value[0], value[1] # most recent tle (or single tle if len(value) = 1)]
            elif len(value) == 1:
                tle_line_1, tle_line_2 = value[0][0], value[0][1] # most recent tle (or single tle if len(value) = 1)]

            # Consider switching to sgp4 propagator
            object_tle = environment.Tle(tle_line_1, tle_line_2)
            object_ephemeris = environment.TleEphemeris("Earth", "J2000", object_tle, False)
            for t in list_of_times:
                state = object_ephemeris.cartesian_state(t)
                radec = cartesian2radec(state[:3])[1:]
                trajectory_dict[key].append(state[:3])
                radec_dict[key].append(list(radec))

        # Create 3D figure and axis
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        q.OMMUtils.plot_earth(SpaceTrackQuery, ax) #plot Earth as sphere of radius 6378 km

        # Plot each trajectory
        for key, value in trajectory_dict.items():
            trajectory_array = np.vstack(value)  # shape (N, 3)
            x = trajectory_array[:, 0] / 1000  # km
            y = trajectory_array[:, 1] / 1000
            z = trajectory_array[:, 2] / 1000
            ax.plot(x, y, z, label=f'NORAD {key}', linewidth=1)

        # Plot Earth center
        ax.scatter(0, 0, 0, color='blue', label='Earth Center', s=30)

        # Labels and formatting
        ax.set_xlabel("X [km]")
        ax.set_ylabel("Y [km]")
        ax.set_zlabel("Z [km]")
        ax.legend(fontsize=8)
        ax.grid(True)

        # Set equal aspect ratio manually
        max_range = max(
            max(abs(x).max(), abs(y).max(), abs(z).max())
            for x, y, z in zip(
                (value[0] / 1000 for value in trajectory_dict.values()),
                (value[1] / 1000 for value in trajectory_dict.values()),
                (value[2] / 1000 for value in trajectory_dict.values())
            )
        )
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        plt.tight_layout()
        plt.show()


        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        # Plot each trajectory
        for key, value in radec_dict.items():
            radec_dict = np.vstack(value)  # shape (N, 3)
            x = radec_dict[:, 0] # km
            y = radec_dict[:, 1]
            ax.scatter(x, y, label=f'NORAD {key}', s=0.5)

        # Plot Earth center
        #ax.scatter(0, 0, 0, color='blue', label='Earth Center', s=30)

        # Labels and formatting
        ax.set_xlabel("X [km]")
        ax.set_ylabel("Y [km]")
        ax.legend(fontsize=8)
        ax.grid(True)
        plt.title('Visibility Plot')
        plt.tight_layout()
        plt.show()

else:
    q = SpaceTrackQuery(username, password)
    filter_oe_dict = {'SEMIMAJOR_AXIS': [10000,100000]}
    q.download_tle.filtered_by_oe_dict(filter_oe_dict, 100)
    # List of JSON TLE files
    json_filenames = ['filtered_results.json']

    # Time span (e.g., 12 hours sampled every 2 minutes)
    simulation_start_epoch = DateTime(2025, 7, 15, 0).epoch()
    simulation_end_epoch = DateTime(2025, 7, 18, 15).epoch()
    list_of_times = np.arange(simulation_start_epoch, simulation_end_epoch, 120)

    # Determine subplot grid size (square or close to square)
    n = len(json_filenames)
    cols = math.ceil(np.sqrt(n))
    rows = math.ceil(n / cols)

    # Create figure
    for idx, json_dict in enumerate(json_dict_list):
        tle_dict = q.OMMUtils.get_tles(SpaceTrackQuery, json_dict)
        trajectory_dict = defaultdict(list)
        radec_dict = defaultdict(list)
        for key, value in tle_dict.items():
            if len(value) > 1:
                print(f"Multiple TLEs for object {key}. Plotting the most recent one.")
            tle_line_1, tle_line_2 = value[0][0], value[0][1] # most recent tle (or single tle if len(value) = 1)
            object_tle = environment.Tle(tle_line_1, tle_line_2)
            object_ephemeris = environment.TleEphemeris("Earth", "J2000", object_tle, False)
            for t in list_of_times:
                # if this is used, it retrieves the state from spice according to TleEphemeris class.
                # check whether spice uses sgp4 or not
                state = object_ephemeris.cartesian_state(t)
                radec = cartesian2radec(state[:3])[1:]
                trajectory_dict[key].append(state[:3])
                radec_dict[key].append(list(radec))

        # Create 3D figure and axis
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        q.OMMUtils.plot_earth(SpaceTrackQuery, ax) #plot Earth as sphere of radius 6378 km

        # Plot each trajectory
        for key, value in trajectory_dict.items():
            trajectory_array = np.vstack(value)  # shape (N, 3)
            x = trajectory_array[:, 0] / 1000  # km
            y = trajectory_array[:, 1] / 1000
            z = trajectory_array[:, 2] / 1000
            ax.plot(x, y, z, label=f'NORAD {key}', linewidth=1)

        # Plot Earth center
        ax.scatter(0, 0, 0, color='blue', label='Earth Center', s=30)

        # Labels and formatting
        ax.set_xlabel("X [km]")
        ax.set_ylabel("Y [km]")
        ax.set_zlabel("Z [km]")
        ax.legend(fontsize=8)
        ax.grid(True)

        # Set equal aspect ratio manually
        max_range = max(
            max(abs(x).max(), abs(y).max(), abs(z).max())
            for x, y, z in zip(
                (value[0] / 1000 for value in trajectory_dict.values()),
                (value[1] / 1000 for value in trajectory_dict.values()),
                (value[2] / 1000 for value in trajectory_dict.values())
            )
        )
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        plt.tight_layout()
        plt.show()


        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        # Plot each trajectory
        for key, value in radec_dict.items():
            radec_dict = np.vstack(value)  # shape (N, 3)
            x = radec_dict[:, 0] # km
            y = radec_dict[:, 1]
            ax.scatter(x, y, label=f'NORAD {key}', s=0.5)

        # Plot Earth center
        #ax.scatter(0, 0, 0, color='blue', label='Earth Center', s=30)

        # Labels and formatting
        ax.set_xlabel("Ra [Deg]")
        ax.set_ylabel("Dec [Deg]")
        ax.legend(fontsize=8)
        ax.grid(True)
        plt.title('Visibility Plot')
        plt.tight_layout()
        plt.show()