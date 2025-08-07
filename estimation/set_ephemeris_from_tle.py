from tudatpy.data.spacetrack import SpaceTrackQuery
from tudatpy.data.discos import DiscosQuery
from collections import defaultdict
from tudatpy.dynamics import environment_setup, propagation_setup

from tudatpy.dynamics.propagation_setup.acceleration import orbital_regimes
from tudatpy.dynamics.environment_setup.ephemeris.create_ephemeris_settings import *

username = 'l.gisolfi@tudelft.nl'
password = 'l.gisolfi*tudelft.nl'
SpaceTrackQuery = SpaceTrackQuery(username, password)
GetAccelerationSettingsPerRegime = orbital_regimes.GetAccelerationSettingsPerRegime()
CreateEphemerisSettings = environment_setup.ephemeris.CreateEphemerisSettings(SpaceTrackQuery, GetAccelerationSettingsPerRegime)

filter_oe_dict = {'SEMIMAJOR_AXIS': [10000,100000]}
SpaceTrackQuery.download_tle.filtered_by_oe_dict(filter_oe_dict, 100)
# List of JSON TLE files
json_filenames = ['filtered_results.json']

for idx, json_filename in enumerate(json_filenames):
    tle_dict = SpaceTrackQuery.TleUtils.get_tle_dict_from_json(SpaceTrackQuery, json_filename)
    trajectory_dict = defaultdict(list)
    radec_dict = defaultdict(list)
    for key, value in tle_dict.items():
        if len(value) > 1:
            print(f"[{json_filename}] Multiple TLEs for object {key}. Plotting the most recent one.")
        tle_line_1, tle_line_2 = value[0][0], value[0][1] # most recent tle (or single tle if len(value) = 1)
        ephemeris_object = SpaceTrackQuery.TleUtils.tle_to_TleEphemeris_object(SpaceTrackQuery.TleUtils,tle_line_1, tle_line_2)
        print(f'Created ephemeris for NORAD_CAT_ID object {key}.')
        print(ephemeris_object)
        tle_reference_epoch = SpaceTrackQuery.TleUtils.get_tle_reference_epoch(SpaceTrackQuery.TleUtils, tle_line_1)
        print(tle_reference_epoch)
        set_ephemeris_object = CreateEphemerisSettings.set_object_ephemeris_settings(key, tle_line_1, tle_line_2, dynamical_model = 'LEO-REGIME')
        print(f'Set ephemeris for NORAD_CAT_ID object {key}.')
        print(set_ephemeris_object)




