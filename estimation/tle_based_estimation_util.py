
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# This script makes use of the spacetrack-module for python (https://spacetrack.readthedocs.io)

# General imports
import numpy as np
import spacetrack
import requests
import tempfile
import getpass
import sys
import os

# tudatpy imports
from tudatpy.kernel import constants

def get_space_track_client():
    # General URLs
    uriBase = "https://www.space-track.org/"
    requestLogin = "ajaxauth/login/"
    # Begin of log-in procedure
    log_in_procedure = True
    while log_in_procedure:
        # Log in to personal space-track.org account
        print('\nEnter your personal space-track.org username (usually your email address for registration):')
        configUsr = input()
        # Exit program if user does not wish to continue, otherwise ask for password
        if configUsr == 'exit':
            sys.exit()
        else:
            print('Username capture complete.\n')
            # configPwd = getpass.getpass(prompt='Securely enter your space-track.org password '
            #                                    '(minimum of 15 characters): ')

            ###########################################################################
            ### REMOVE BEFORE PUSH ####################################################
            ###########################################################################

            configPwd = 'nEwdov-jyhfi2-wozqym'

            ###########################################################################
            ### REMOVE BEFORE PUSH ####################################################
            ###########################################################################

            siteCred = {'identity': configUsr, 'password': configPwd}

        # Running the session in a with block additionally force-closes it at the end if not done manually
        with requests.Session() as session:
            # Initial log in attempt. Upon success website returns a value of 200. Otherwise, prompt for new password.
            resp = session.post(uriBase + requestLogin, data=siteCred)
            if resp.status_code != 200:
                session.close()
                print('\nSomething went wrong - your username and password are not known to the system. '
                      '\nPlease try again. If you wish to end this program, type exit.')
            else:
                log_in_procedure = False
                space_track_client = spacetrack.SpaceTrackClient(configUsr, configPwd)

    return space_track_client


if __name__ == '__main__':
    # Dictionary path
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Establish connection with space-track.org
    st = get_space_track_client()
    # NORAD ID of the satellite of interest
    norad_cat_id = 25544

    # Get past TLE(s)
    raw_data = st.tle(norad_cat_id=norad_cat_id, orderby='epoch desc', limit=2, format='tle')

    # Either create or clear folder to save TLEs into
    if not os.path.exists(os.path.join(dir_path, 'TLEs')):
        os.makedirs(os.path.join(dir_path, 'TLEs'))
    else:
        for f in os.listdir(os.path.join(dir_path, 'TLEs')):
            os.remove(os.path.join(os.path.join(dir_path, 'TLEs'), f))

    # Open concatenated list of all TLEs as a temporary file
    with tempfile.TemporaryFile(mode='w+') as f:
        # Read individual lines of all TLEs
        f.write(raw_data)
        f.seek(0)
        lines = f.read().splitlines()

        # Check that correct file (format) has been loaded
        if len(lines) % 2 == 0:
            # Calculate epoch of TLE
            for i in range(int(len(lines) / 2)):
                tle_year = float(lines[i * 2][18:20])
                tle_day_of_year = float(lines[i * 2][20:32])
                if tle_year >= 60.0:
                    epoch_year = 1900 + tle_year
                else:
                    epoch_year = 2000 + tle_year

                tle_epoch = int(epoch_year * constants.JULIAN_YEAR + tle_day_of_year * constants.JULIAN_DAY)

                # Save one TLE per file
                file_path = 'TLEs/{0}_epoch_{1}.dat'.format(norad_cat_id, tle_epoch)
                designatedPath = os.path.join(dir_path, file_path)
                try:
                    with open(designatedPath, 'w') as new_file:
                        new_file.writelines([lines[i * 2], lines[i * 2 + 1]])
                except ():
                    print('An error occurred while trying to write files to disk.')
        else:
            print('Something is wrong with your file structure, please check the way TLEs are retrieved.')
            sys.exit()

