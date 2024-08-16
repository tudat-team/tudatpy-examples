import sys

sys.path.insert(0, '/home/mfayolle/Tudat/tudat-bundle/build/tudatpy')

# Load required standard modules
import os
import numpy as np
from matplotlib import pyplot as plt

# Load required tudatpy modules
from tudatpy import constants
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation, propagation_setup
from tudatpy.numerical_simulation import estimation, estimation_setup
from tudatpy.numerical_simulation.estimation_setup import observation
from tudatpy.astro.time_conversion import DateTime
from tudatpy.astro import element_conversion
from tudatpy.util import result2array


# Function retrieving JUICE's position wrt the flyby moon from their SPICE ephemerides, at a given time.
def get_juice_position_wrt_moon(time):
    return spice.get_body_cartesian_position_at_epoch(
        "-28", flyby_moon, "J2000", aberration_corrections="none", ephemeris_time=time)


# Function identifying JUICE's closest approaches wrt the flyby moon occurring within the specified time bounds, based on their SPICE trajectories.
# Only events where the closest distance meets the threshold are selected.
def find_closest_approaches(lower_time_bound, upper_time_bound, threshold):
    flyby_times = []

    tolerance = 1.0
    step = 100.0

    lower_time = lower_time_bound
    mid_time = lower_time_bound + step
    upper_time = lower_time_bound + 2.0 * step

    while upper_time <= upper_time_bound:
        upper_value = np.linalg.norm(get_juice_position_wrt_moon(upper_time))
        mid_value = np.linalg.norm(get_juice_position_wrt_moon(mid_time))
        lower_value = np.linalg.norm(get_juice_position_wrt_moon(lower_time))

        if (upper_value - mid_value) > 0 and (mid_value - lower_value) < 0:

            current_lower_time = lower_time
            current_upper_time = upper_time
            current_mid_time = (current_lower_time + current_upper_time) / 2.0
            current_test_time = current_mid_time
            counter = 0

            while np.abs(current_mid_time - current_test_time) > tolerance:

                current_test_time = current_mid_time

                current_lower_distance = np.linalg.norm(get_juice_position_wrt_moon(current_lower_time))
                current_upper_distance = np.linalg.norm(get_juice_position_wrt_moon(current_upper_time))
                current_test_distance = np.linalg.norm(get_juice_position_wrt_moon(current_test_time))

                sign_upper_derivative = np.sign((current_upper_distance - np.linalg.norm(
                    get_juice_position_wrt_moon(current_upper_time - tolerance))) / tolerance)
                sign_lower_derivative = np.sign((current_lower_distance - np.linalg.norm(
                    get_juice_position_wrt_moon(current_lower_time - tolerance))) / tolerance)
                sign_test_derivative = np.sign((current_test_distance - np.linalg.norm(
                    get_juice_position_wrt_moon(current_test_time - tolerance))) / tolerance)

                if sign_upper_derivative > 0 and sign_test_derivative < 0:
                    current_mid_time = (current_upper_time + current_test_time) / 2.0
                    current_lower_time = current_test_time
                elif sign_lower_derivative < 0 and sign_test_derivative > 0:
                    current_mid_time = (current_test_time + current_lower_time) / 2.0
                    current_upper_bound = current_test_time

                counter += 1
                if counter > 1000:
                    raise Exception("no minimum identified")

            possible_time = current_mid_time

            if np.linalg.norm(get_juice_position_wrt_moon(possible_time)) <= threshold:
                flyby_times.append(possible_time)

        lower_time = lower_time + step
        mid_time = mid_time + step
        upper_time = upper_time + step

    return flyby_times


# Specify which moon is to be considered for the JUICE flybys (can be set to Europa, Ganymede, or Callisto)
flyby_moon = "Callisto"
if flyby_moon != "Europa" and flyby_moon != "Ganymede" and flyby_moon != "Callisto":
    raise NameError('flyby_moon should be set to Europa, Ganymede, or Callisto.')

# Load spice kernels
path = os.path.dirname(__file__)
kernels = [path + '/../kernels/kernel_juice.bsp', path + '/../kernels/kernel_noe.bsp']
spice.load_standard_kernels(kernels)

# Set simulation start and end epochs according to JUICE mission timeline
start_epoch = 32.0 * constants.JULIAN_YEAR
end_epoch = 34.5 * constants.JULIAN_YEAR

# Define default body settings
bodies_to_create = ["Europa", "Ganymede", "Callisto", "Jupiter", "Sun"]
global_frame_origin = "Jupiter"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(bodies_to_create, global_frame_origin,
                                                            global_frame_orientation)

# Set rotation of flyby moon to synchronous
body_settings.get(flyby_moon).rotation_model_settings = environment_setup.rotation_model.synchronous(
    "Jupiter", global_frame_orientation, "IAU_" + flyby_moon)

# Create empty settings for JUICE spacecraft
body_settings.add_empty_settings("JUICE")

# Set empty ephemeris for JUICE
empty_ephemeris_dict = dict()
juice_ephemeris = environment_setup.ephemeris.tabulated(
    empty_ephemeris_dict,
    global_frame_origin,
    global_frame_orientation)
body_settings.get("JUICE").ephemeris_settings = juice_ephemeris

# Create system of bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

# Add JUICE spacecraft to system of bodies
bodies.get("JUICE").mass = 5.0e3

# Create radiation pressure settings
ref_area = 100.0
srp_coef = 1.2
occulting_bodies = {"Sun": [flyby_moon]}
juice_srp_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
    ref_area, srp_coef, occulting_bodies)
environment_setup.add_radiation_pressure_target_model(bodies, "JUICE", juice_srp_settings)

# Find all JUICE flybys around the specified flyby moon.
# This finding algorithm identifies all closest approaches of the JUICE spacecraft with respect to the flyby moon,
# based on the JUICE's SPICE trajectory. Only those with a closest distance smaller or equal to 2.0e7 m are kept.
closest_approaches_juice = find_closest_approaches(start_epoch, end_epoch, 2.0e7)

# Extract number of JUICE flybys
nb_flybys = len(closest_approaches_juice)
print("nb_flybys ", nb_flybys)

# Define accelerations acting on JUICE
accelerations_settings_juice = dict(
    Europa=[
        propagation_setup.acceleration.spherical_harmonic_gravity(2, 2),
    ],
    Ganymede=[
        propagation_setup.acceleration.spherical_harmonic_gravity(2, 2),
    ],
    Callisto=[
        propagation_setup.acceleration.spherical_harmonic_gravity(2, 2),
    ],
    Jupiter=[
        propagation_setup.acceleration.spherical_harmonic_gravity(2, 0)
    ],
    Sun=[
        propagation_setup.acceleration.radiation_pressure(),
        propagation_setup.acceleration.point_mass_gravity()
    ])

acceleration_settings = {"JUICE": accelerations_settings_juice}

body_to_propagate = ["JUICE"]
central_body = [flyby_moon]

acceleration_models = propagation_setup.create_acceleration_models(
    bodies, acceleration_settings, body_to_propagate, central_body)

# Define integrator settings
integrator = propagation_setup.integrator.runge_kutta_fixed_step_size(
    initial_time_step=10.0, coefficient_set=propagation_setup.integrator.rkf_78)

# Define dependent variables
dependent_variables_names = [
    propagation_setup.dependent_variable.latitude("JUICE", flyby_moon),
    propagation_setup.dependent_variable.longitude("JUICE", flyby_moon),
    propagation_setup.dependent_variable.altitude("JUICE", flyby_moon)
]

# Define propagator settings for each arc (i.e., each flyby)
propagator_settings_list = []
for k in range(nb_flybys):
    # The initial time of the propagation is set at the time of closest approach (obtained from SPICE), and the JUICE spacecraft is then
    # propagated backwards and forwards from that point.
    flyby_time = closest_approaches_juice[k]

    # Get initial state of JUICE wrt the flyby moon from SPICE (JUICE's SPICE ID: -28)
    initial_state = spice.get_body_cartesian_state_at_epoch("-28", flyby_moon, "J2000", "None", flyby_time)
    print('flyby altitude ',
          (np.linalg.norm(initial_state[:3]) - bodies.get(flyby_moon).shape_model.average_radius) / 1e3, 'km')

    # Create termination settings (propagate between 30 min before and after closest approach)
    time_around_closest_approach = 0.5 * 3600.0
    termination_condition = propagation_setup.propagator.non_sequential_termination(
        propagation_setup.propagator.time_termination(flyby_time + time_around_closest_approach),
        propagation_setup.propagator.time_termination(flyby_time - time_around_closest_approach))

    # Define arc-wise propagator settings
    propagator_settings_list.append(propagation_setup.propagator.translational(
        central_body, acceleration_models, body_to_propagate, initial_state, flyby_time, integrator,
        termination_condition,
        propagation_setup.propagator.cowell, dependent_variables_names))

# Concatenate all arc-wise propagator settings into multi-arc propagator settings
propagator_settings = propagation_setup.propagator.multi_arc(propagator_settings_list)

# Propagate dynamics
simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings)
propagation_results = simulator.propagation_results.single_arc_results

# Plot flybys
moon_map = flyby_moon.lower() + '_map.jpg'
img = plt.imread(moon_map)

fig, ax = plt.subplots()
ax.imshow(img, extent=[0, 360, -90, 90])
for k in range(nb_flybys):
    dependent_variables = result2array(propagation_results[k].dependent_variable_history)

    # Resolve 2pi ambiguity for longitude
    for i in range(len(dependent_variables)):
        if dependent_variables[i, 2] < 0:
            dependent_variables[i, 2] = dependent_variables[i, 2] + 2.0 * np.pi

    plot = ax.scatter(dependent_variables[:, 2] * 180 / np.pi, dependent_variables[:, 1] * 180 / np.pi, s=2,
                      c=dependent_variables[:, 3] / 1e3, cmap='rainbow_r', vmin=0, vmax=5000)
cb = plt.colorbar(plot)

plt.xlabel('Longitude [deg]')
plt.ylabel('Latitude [deg]')
plt.xticks(np.arange(0, 361, 40))
plt.yticks(np.arange(-90, 91, 30))
cb.set_label('Altitude [km]')
plt.title('JUICE flybys at ' + flyby_moon)
plt.show()
