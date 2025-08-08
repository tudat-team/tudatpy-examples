from tudatpy.data.spacetrack import SpaceTrackQuery
from tudatpy.data.discos import DiscosQuery
from collections import defaultdict
from tudatpy.dynamics import environment_setup, propagation_setup

from tudatpy.dynamics.propagation_setup.acceleration import orbital_regimes
from tudatpy.dynamics.environment_setup.ephemeris.create_ephemeris_settings import *

username = 'l.gisolfi@tudelft.nl'
password = 'l.gisolfi*tudelft.nl'
spactrack_request = SpaceTrackQuery(username, password)
tle_query = spactrack_request.DownloadTle(spactrack_request)
omm_utils =spactrack_request.OMMUtils(tle_query)

GetAccelerationSettingsPerRegime = orbital_regimes.GetAccelerationSettingsPerRegime()

filter_oe_dict = {'SEMIMAJOR_AXIS': [10000,100000]}
json_dict = tle_query.filtered_by_oe_dict(filter_oe_dict, 100)
# List of JSON TLE files
json_filenames = ['filtered_results.json']

tabulated_sgp4_ephemeris = dict()
for idx, json_filename in enumerate(json_filenames):
    tle_dict =omm_utils.get_tles(json_dict)
    trajectory_dict = defaultdict(list)
    radec_dict = defaultdict(list)
    for key, value in tle_dict.items():
        print(value)
        tle_line_1, tle_line_2 = value[0], value[0] # most recent tle (or single tle if len(value) = 1)
        tabulated_sgp4_ephemeris[key] = omm_utils.tle_to_sgp4_ephemeris_object(tle_line_1, tle_line_2)

    print(tabulated_sgp4_ephemeris)






