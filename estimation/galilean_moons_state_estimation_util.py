
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import math
import numpy as np

# tudatpy imports
from tudatpy.kernel.astro import element_conversion

###########################################################################
# LAPLACE UTILITIES #######################################################
###########################################################################

def calculate_mean_longitude(kepler_elements: dict):
    # Calculate dictionary for moon-wise longitudes
    mean_longitude_dict = dict()
    # Loop over every moon of interest (Io, Europa, Ganymede)
    for moon in kepler_elements.keys():
        mean_anomaly_per_moon = list()
        kepler_elements_per_moon = kepler_elements[moon]
        # For every epoch get the mean anomaly of the moon
        for i in range(len(kepler_elements[moon])):
            mean_anomaly_per_moon.append(element_conversion.true_to_mean_anomaly(
                eccentricity=kepler_elements_per_moon[i, 1],
                true_anomaly=kepler_elements_per_moon[i, 5]))
        mean_anomaly_per_moon = np.array(mean_anomaly_per_moon)
        mean_anomaly_per_moon[mean_anomaly_per_moon < 0] = mean_anomaly_per_moon[mean_anomaly_per_moon < 0] \
                                                           + 2 * math.pi
        # Calculate the mean longitude as
        # (longitude of the ascending node) + (argument of the pericenter) + (mean anomaly)
        longitude_of_the_ascending_node = kepler_elements_per_moon[:, 4]
        argument_of_the_pericenter = kepler_elements_per_moon[:, 3]

        mean_longitude_per_moon = longitude_of_the_ascending_node + argument_of_the_pericenter + mean_anomaly_per_moon
        # Include epoch-wise mean longitude in dictionary
        mean_longitude_per_moon = np.mod(mean_longitude_per_moon, 2*math.pi)
        mean_longitude_dict[moon] = mean_longitude_per_moon

    return mean_longitude_dict