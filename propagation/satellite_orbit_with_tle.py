from tudatpy.interface import spice
from tudatpy.dynamics import environment_setup, environment, propagation_setup, simulator
from tudatpy.astro import element_conversion, time_representation
import numpy as np
import matplotlib.pyplot as plt
from tudatpy.data.spacetrack import SpaceTrackQuery
from tudatpy.dynamics import parameters_setup, parameters, propagation, propagation_setup
from tudatpy.dynamics.propagation_setup.acceleration import orbital_regimes
from tudatpy.dynamics.simulator import create_dynamics_simulator
from tudatpy.util import result2array


# Load spice standard kernels
spice.load_standard_kernels()

norad_id = str(32789) # NORAD ID for delfi-c3
username = 'l.gisolfi@tudelft.nl'
password = 'l.gisolfi*tudelft.nl'

# Initialize SpaceTrackQuery
spacetrack_request = SpaceTrackQuery(username, password)
tle_query = spacetrack_request.DownloadTle(spacetrack_request)
omm_utils = spacetrack_request.OMMUtils(tle_query)

# OMM Dict
json_dict_list = tle_query.single_norad_id(norad_id)

print(f"OBJECT NAME CORRESPONDING TO NORAD ID {norad_id}: {json_dict_list[0]['OBJECT_NAME']}")
# Retrieve TLEs
tle_dict = omm_utils.get_tles(json_dict_list)
tle_line1, tle_line2 = tle_dict[norad_id][0], tle_dict[norad_id][1]

# Retrieve TLE Reference epoch, this will be start epoch of simulation
tle_reference_epoch = omm_utils.get_tle_reference_epoch(tle_line1)

timestep_global = 5 #seconds

# Define Simulation Start and End (Date)Times
propagation_start_epoch = time_representation.DateTime.from_python_datetime(tle_reference_epoch).to_epoch()
propagation_end_epoch =  propagation_start_epoch + 86400/2 # one day propagation

# define the frame origin and orientation.
global_frame_origin = "Earth"
global_frame_orientation = "J2000"

# List the bodies for our environment
bodies_to_create = [
    "Sun",
    "Earth",
    "Moon",
]

# Create default body settings
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation
)

body_settings.get("Earth").rotation_model_settings = environment_setup.rotation_model.gcrs_to_itrs(
    environment_setup.rotation_model.iau_2006,
    global_frame_orientation )
body_settings.get("Earth").gravity_field_settings.associated_reference_frame = "ITRS"
# create atmosphere settings and add to body settings of body "Earth"
body_settings.get( "Earth" ).atmosphere_settings = environment_setup.atmosphere.nrlmsise00()

# create empty settings for norad_id
mass = 2.2
body_settings.add_empty_settings(norad_id)
body_settings.get(norad_id).constant_mass = mass

# Create aerodynamic coefficient interface settings
reference_area_drag = (4*0.3*0.1+2*0.1*0.1)/4  # Average projection area of a 3U CubeSat
drag_coefficient = 1.2
aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
    reference_area_drag, [drag_coefficient, 0.0, 0.0]
)

# Add the aerodynamic interface to the environment
body_settings.get(norad_id).aerodynamic_coefficient_settings = aero_coefficient_settings

# Create radiation pressure settings
reference_area_radiation = (4*0.3*0.1+2*0.1*0.1)/4  # Average projection area of a 3U CubeSat
radiation_pressure_coefficient = 1.2
occulting_bodies_dict = dict()
occulting_bodies_dict["Sun"] = ["Earth"]
radiation_pressure_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
    reference_area_radiation, radiation_pressure_coefficient, occulting_bodies_dict )

# create ephemeris for the object via sgp4 ephemeris
original_sgp4_ephemeris =  environment_setup.ephemeris.sgp4(
    tle_line1,
    tle_line2,
    frame_origin = global_frame_origin,
    frame_orientation = global_frame_orientation)

# The following ephemeris creation DOES NOT add the ephemeris to the environment.
# In this case, this is a desired behaviour because in the end, we would like to set up
# the actual norad_id ephemeris with the state_history of the propagation,
# which is in turn based on a selected (LEO, MEO, GEO, etc...) dynamical model.
# We use create_body_ephemeris here just to retrieve the intial_state to initiate the propagation.
original_sgp4_ephemeris = environment_setup.create_body_ephemeris(original_sgp4_ephemeris, norad_id)
initial_state = original_sgp4_ephemeris.cartesian_state(propagation_start_epoch)

bodies = environment_setup.create_system_of_bodies(body_settings)
bodies_to_propagate = [norad_id]
central_bodies = [global_frame_origin]

# Create propagator settings
integrator_settings = propagation_setup.integrator. \
    runge_kutta_fixed_step_size(initial_time_step= time_representation.Time(timestep_global),
                                coefficient_set=propagation_setup.integrator.CoefficientSets.rkdp_87)

# Terminate at the time of oldest observation
termination_condition = propagation_setup.propagator.time_termination(propagation_end_epoch)

# get acceleration model by orbital regime
orbital_regime, orbital_regime_definition = omm_utils.get_orbital_regime(json_dict_list[0])

GetAccelerationSettingsPerRegime = orbital_regimes.GetAccelerationSettingsPerRegime()
acceleration_settings, bodies = GetAccelerationSettingsPerRegime.get_acceleration_settings(
    bodies,
    body_settings,
    orbital_regime,
    aerodynamics = False,
    radiation_pressure = False
)

acceleration_settings = {norad_id: acceleration_settings}
acceleration_models = propagation_setup.create_acceleration_models(
    bodies, acceleration_settings, bodies_to_propagate, central_bodies
)

propagator_settings = propagation_setup.propagator.translational(
    central_bodies=['Earth'],
    acceleration_models=acceleration_models,
    bodies_to_integrate=bodies_to_propagate,
    initial_states=initial_state,
    initial_time=propagation_start_epoch,
    integrator_settings=integrator_settings,
    termination_settings=termination_condition
)

#this line is crucial, as it allows to set ephemeris automatically as a result of the propagation
propagator_settings.processing_settings.set_integrated_result = True
dynamics_simulator = create_dynamics_simulator(bodies, propagator_settings)

# Extract the resulting state history and convert it to an ndarray
states = dynamics_simulator.propagation_results.state_history
states_array = result2array(states)

times_list = np.arange(propagation_start_epoch, propagation_end_epoch, 5)
sgp4_states_array = np.array([original_sgp4_ephemeris.cartesian_state(t) for t in times_list])

"""
## Post-process the propagation results
The results of the propagation are then processed to a more user-friendly form.

### Print initial and final states
First, let's print the initial and final position and velocity vector of `Delfi-C3`.
"""


print(
    f"""
Single Earth-Orbiting Satellite Example.
The initial position vector of Delfi-C3 is [km]: \n{
    states[propagation_start_epoch][:3] / 1E3} 
The initial velocity vector of Delfi-C3 is [km/s]: \n{
    states[propagation_start_epoch][3:] / 1E3}
\nAfter {propagation_end_epoch - propagation_start_epoch} seconds the position vector of Delfi-C3 is [km]: \n{
    states[propagation_end_epoch][:3] / 1E3}
And the velocity vector of Delfi-C3 is [km/s]: \n{
    states[propagation_end_epoch][3:] / 1E3}
    """
)

"""
### Visualise the trajectory
Finally, let's plot the trajectory of `Delfi-C3` around Earth in 3D.
"""

# Define a 3D figure using pyplot
fig = plt.figure(figsize=(6,6), dpi=125)
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f'Delfi-C3 trajectory around Earth')

# Plot the positional state history
ax.plot(states_array[:, 1], states_array[:, 2], states_array[:, 3], label='LEO Acceleration Regime', linestyle='-.')
ax.plot(sgp4_states_array[:, 0], sgp4_states_array[:, 1], sgp4_states_array[:, 2], label='SGP4', linestyle='--', c = 'red', alpha = 0.3)
ax.scatter(0.0, 0.0, 0.0, label="Earth", marker='o', color='blue')

# Add the legend and labels, then show the plot
ax.legend()
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
plt.show()

plt.show()

# Difference plot
fig = plt.figure(figsize=(6,6), dpi=125)
ax = fig.add_subplot(111)
ax.set_title(f'Delfi-C3 trajectory around Earth')

x_diff = states_array[:-1, 1] - sgp4_states_array[:, 0]
y_diff = states_array[:-1, 2] - sgp4_states_array[:, 1]
z_diff = states_array[:-1, 3] - sgp4_states_array[:, 2]

new_times_list = np.array([time - times_list[0] for time in times_list])
# Plot main differences
ax.plot(new_times_list / 3600, x_diff / 1000, label='$x_{LEO}$ - $x_{SGP4}$')
ax.plot(new_times_list / 3600, y_diff / 1000, label='$y_{LEO}$ - $y_{SGP4}$')
ax.plot(new_times_list / 3600, z_diff / 1000, label='$z_{LEO}$ - $z_{SGP4}$')

ax.legend()
ax.set_xlabel('Time [hours]')
ax.set_ylabel('Difference [km]')

plt.show()