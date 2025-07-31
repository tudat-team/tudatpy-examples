# Tudat imports for propagation and estimation
from tudatpy.interface import spice
from tudatpy.dynamics import environment_setup, environment, propagation_setup, simulator
from tudatpy.estimation import observable_models_setup,observable_models, observations_setup, observations, estimation_analysis
from tudatpy.astro import element_conversion, time_representation
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tudatpy.data.spacetrack import SpaceTrackQuery
from tudatpy.dynamics import parameters_setup, parameters, propagation, propagation_setup

# Load spice standard kernels
spice.load_standard_kernels()

norad_id = 64865
username = 'l.gisolfi@tudelft.nl'
password = 'l.gisolfi*tudelft.nl'

# Initialize SpaceTrackQuery
SpaceTrackQuery = SpaceTrackQuery(username, password)

# OMM Dict
json_dict = SpaceTrackQuery.DownloadTle.single_norad_id(SpaceTrackQuery, norad_id)

# retrieve orbital elements from json_dict

a,e,i,omega,raan,true_anomaly = SpaceTrackQuery.OMMUtils.get_tudat_keplerian_element_set(SpaceTrackQuery.OMMUtils,json_dict)

# Retrieve TLEs
tle_dict = SpaceTrackQuery.OMMUtils.get_tles(SpaceTrackQuery,json_dict)

tle_line1, tle_line2 = tle_dict[str(norad_id)][0], tle_dict[str(norad_id)][1]

# Retrieve TLE Reference epoch
tle_reference_epoch = SpaceTrackQuery.OMMUtils.get_tle_reference_epoch(SpaceTrackQuery,tle_line1)

tle_ephemeris_object = SpaceTrackQuery.OMMUtils.tle_to_TleEphemeris_object(SpaceTrackQuery, tle_line1, tle_line2)

number_of_pod_iterations = 6 # number of iterations for our estimation
timestep_global = 5 # timestep of 120 seconds for our estimation
time_buffer = 60 # uncomment only if needed.

# Define Simulation Start and End (Date)Times
observations_start = tle_reference_epoch # set as tle reference epoch for now, but can vary
observations_end = tle_reference_epoch + timedelta(seconds=60*30) # one hour after observation start
float_observations_start = time_representation.DateTime.from_python_datetime(observations_start).to_epoch()
float_observations_end = time_representation.DateTime.from_python_datetime(observations_end).to_epoch()

# define the frame origin and orientation.
global_frame_origin = "Earth"
global_frame_orientation = "J2000"

# List the bodies for our environment
bodies_to_create = [
    "Sun",
    "Earth",
    "Moon",
]

# Create system of bodies
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation
)

body_settings.add_empty_settings(str(norad_id))


original_sgp4_ephemeris =  environment_setup.ephemeris.sgp4(
    tle_line1,
    tle_line2,
    frame_origin = global_frame_origin,
    frame_orientation = global_frame_orientation)

body_settings.get(str(norad_id)).ephemeris_settings =  environment_setup.ephemeris.tabulated_from_existing(
    original_sgp4_ephemeris,
    float_observations_start,
    float_observations_end,
    timestep_global)

bodies = environment_setup.create_system_of_bodies(body_settings)

earth_gravitational_parameter = bodies.get_body('Earth').gravitational_parameter
initial_state = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=earth_gravitational_parameter,
    semi_major_axis= a,
    eccentricity=e,
    inclination=i,
    argument_of_periapsis=omega,
    longitude_of_ascending_node=raan,
    true_anomaly=true_anomaly,
)

bodies_to_propagate = [str(norad_id)]
central_bodies = [global_frame_origin]


# Create propagator settings
integrator_settings = propagation_setup.integrator. \
    runge_kutta_fixed_step_size(initial_time_step= time_representation.Time(timestep_global),
                                coefficient_set=propagation_setup.integrator.CoefficientSets.rkdp_87)
# Terminate at the time of oldest observation
termination_condition = propagation_setup.propagator.time_termination(float_observations_end)

# Define acceleration model
acceleration_settings_on_spacecraft = dict()
accelerations = {
    "Sun": [
        propagation_setup.acceleration.point_mass_gravity(),
    ],
    "Earth": [propagation_setup.acceleration.spherical_harmonic_gravity(2,2)],
    "Moon": [propagation_setup.acceleration.point_mass_gravity()],
}

# Set up the accelerations settings for each body, in this case only for our norad_id
acceleration_settings = {}
acceleration_settings[str(norad_id)] = accelerations

# create the acceleration models
acceleration_models = propagation_setup.create_acceleration_models(
    bodies, acceleration_settings, bodies_to_propagate, central_bodies
)

propagator_settings = propagation_setup.propagator.translational(
    central_bodies=['Earth'],
    acceleration_models=acceleration_models,
    bodies_to_integrate=bodies_to_propagate,
    initial_states=initial_state,
    initial_time=float_observations_start,
    integrator_settings=integrator_settings,
    termination_settings=termination_condition
)

# Setup parameters settings to propagate the state transition matrix
parameter_settings = parameters_setup.initial_states(
    propagator_settings, bodies
)
parameters_to_estimate = parameters_setup.create_parameter_set(
    parameter_settings, bodies, propagator_settings
)

# Save the true initial state to later analyse the error
truth_parameters = parameters_to_estimate.parameter_vector
print(f'Truth Parameters: \n {truth_parameters}')

# Create light-time corrections list
light_time_correction_list = list()
light_time_correction_list.append(
    observable_models_setup.light_time_corrections.first_order_relativistic_light_time_correction(["Sun"]))

# Define TU Delft Rooftop station and coordinates
reference_point = "TU_DELFT_ROOFTOP"
reference_point_backup = "ARUBA_ROOFTOP"
stations_altitude = 0.0
delft_latitude = np.deg2rad(52.00667)
delft_longitude = np.deg2rad(4.35556)

aruba_latitude = np.deg2rad(12.517572)
aruba_longitude = np.deg2rad(-69.9649462)

environment_setup.add_ground_station(
    bodies.get_body("Earth"),
    reference_point,
    [stations_altitude, delft_latitude, delft_longitude],
    element_conversion.geodetic_position_type)

environment_setup.add_ground_station(
    bodies.get_body("Earth"),
    reference_point_backup,
    [stations_altitude, aruba_latitude, aruba_longitude],
    element_conversion.geodetic_position_type)

link_ends_list = [{observable_models_setup.links.receiver: observable_models_setup.links.body_reference_point_link_end_id('Earth', reference_point),
                   observable_models_setup.links.transmitter: observable_models_setup.links.body_origin_link_end_id(str(norad_id))},
                  {observable_models_setup.links.receiver: observable_models_setup.links.body_reference_point_link_end_id('Earth', reference_point_backup),
                   observable_models_setup.links.transmitter: observable_models_setup.links.body_origin_link_end_id(str(norad_id))}, ]

link_definition_list = []
for link_ends in link_ends_list:
    link_definition_list.append(observable_models_setup.links.link_definition(link_ends))


observation_model_settings = list()

for link_definition in link_definition_list:
    observation_model_settings.append(observable_models_setup.model_settings.angular_position(
        link_definition, light_time_correction_list))

observation_times = np.arange(float_observations_start + time_buffer, float_observations_end - time_buffer, timestep_global)
observation_simulation_settings = list()
body_state_history = dict()

for link_definition in link_definition_list:
    observation_simulation_settings.append(observations_setup.observations_simulation_settings.tabulated_simulation_settings(
        observable_models_setup.model_settings.angular_position_type,
        link_definition,
        observation_times))

noise_level = 1e-6 # typical astrometry error for detection is around 1 arcsec, corresponding to 1e-6 radians.
observations_setup.random_noise.add_gaussian_noise_to_observable(
    observation_simulation_settings,
    noise_level,
    observable_models_setup.model_settings.angular_position_type
)

# Create observation simulators
observation_simulators = observations_setup.observations_simulation_settings.create_observation_simulators(observation_model_settings, bodies)

norad_id_simulated_observations = observations_setup.observations_wrapper.simulate_observations(
    observation_simulation_settings,
    observation_simulators,
    bodies)

norad_id_simulated_observations.set_constant_weight(1e-6)
# Compute and set residuals in the compressed observation collection
observations.compute_residuals_and_dependent_variables(norad_id_simulated_observations, observation_simulators, bodies)

# Filter residual outliers
norad_id_simulated_observations.filter_observations(
    observations.observations_processing.observation_filter(observations.observations_processing.residual_filtering, 0.1))

weights = norad_id_simulated_observations.concatenated_weights

concatenated_observations = norad_id_simulated_observations.concatenated_observations
concatenated_residuals = norad_id_simulated_observations.get_concatenated_residuals()
concatenated_times= norad_id_simulated_observations.concatenated_times

concatenated_ra = concatenated_observations[::2]
concatenated_dec = concatenated_observations[1::2]
concatenated_ra_deg = [np.rad2deg(ra) for ra in concatenated_ra]
concatenated_dec_deg = [np.rad2deg(dec) for dec in concatenated_dec]
# Plot observations
fig, axs = plt.subplots(
    1,
    1,
    figsize=(9, 7),
    sharex=True,
    sharey=False,
)
# plot the residuals, split between RA and DEC types
for idx, ax in enumerate(fig.get_axes()):
    ax.grid()
    # we take every second
    ax.scatter(
        concatenated_times[::2],
        concatenated_ra_deg,
        marker="+",
        s=60,
        label="Right Ascension",
    )

    ax.scatter(
        concatenated_times[::2],
        concatenated_dec_deg,
        marker="+",
        s=60,
        label="Declination",
    )


plt.legend()
plt.tight_layout()
plt.show()

plt.scatter(concatenated_times,concatenated_observations, s = 5)
plt.show()
plt.xlabel('Observation Times [s]')
plt.ylabel('Observations')

# Plot PRE-FIT Residuals
plt.scatter(concatenated_times,concatenated_residuals, s = 5)
plt.show()
plt.xlabel('Observation Times [s]')
plt.ylabel('Residuals')

# Set up the estimator
estimator = estimation_analysis.Estimator(
    bodies=bodies,
    estimated_parameters=parameters_to_estimate,
    observation_settings=observation_model_settings,
    propagator_settings=propagator_settings,
)

# provide the observation collection as input, and limit number of iterations for estimation.
estimation_input = estimation_analysis.EstimationInput(
    observations_and_times=norad_id_simulated_observations,
    convergence_checker=estimation_analysis.estimation_convergence_checker(
        maximum_iterations=10,
    ),
)

# Set methodological options
estimation_input.define_estimation_settings(save_state_history_per_iteration = True, reintegrate_variational_equations=True)
# Define weighting of the observations in the inversion
estimation_input.set_weights_from_observation_collection()

# Perform the estimation
estimation_output = estimator.perform_estimation(estimation_input)

# retrieve the estimated initial state.


initial_states_updated = parameters_to_estimate.parameter_vector
print('Done with the estimation...')
print(f'Updated initial states: {initial_states_updated}')

vector_error_final = (np.array(initial_states_updated) - truth_parameters)[0:3]
error_magnitude_final = np.linalg.norm(vector_error_final)/1000

print(
    f"{norad_id} final error to TLE initial state: {round(error_magnitude_final, 2)} km"
)


simulator_object = estimation_output.simulation_results_per_iteration[-1]
state_history = simulator_object.dynamics_results.state_history
dependent_variable_history = simulator_object.dynamics_results.dependent_variable_history

ephemeris_state = list()
for epoch in state_history.keys():
    ephemeris_state.append(state_history[epoch])


print(ephemeris_state)
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

# Extract x, y, z components
x_values = [state[0]/1000 for state in ephemeris_state]
y_values = [state[1]/1000 for state in ephemeris_state]
z_values = [state[2]/1000 for state in ephemeris_state]

# Create 3D figure and plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_values, y_values, z_values)
ax.scatter(0,0,0, label ='Earth', s = 2)

# (Optional) Label axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Ephemeris Trajectory')

plt.show()

"""
## Visualising the results

### Change in residuals per iteration
We want to visualise the residuals, splitting them between Right Ascension and Declination. Internally, `concatenated_observations` orders the observations alternating RA, DEC, RA, DEC,... This allows us to map the colors accordingly by taking every other item in the `residual_history`/`concatenated_observations`, i.e. by slicing [::2].
"""

residual_history = estimation_output.residual_history

# Number of columns and rows for our plot
number_of_columns = 2

number_of_rows = (
    int(number_of_pod_iterations / number_of_columns)
    if number_of_pod_iterations % number_of_columns == 0
    else int((number_of_pod_iterations + 1) / number_of_columns)
)

fig, axs = plt.subplots(
    number_of_rows,
    number_of_columns,
    figsize=(9, 3.5 * number_of_rows),
    sharex=True,
    sharey=False,
)

# We cheat a little to get an approximate year out of our times (which are in seconds since J2000)
residual_times = (
        np.array(norad_id_simulated_observations.concatenated_times) / (86400 * 365.25) + 2000
)


# plot the residuals, split between RA and DEC types
for idx, ax in enumerate(fig.get_axes()):
    ax.grid()
    # we take every second
    ax.scatter(
        residual_times[::2],
        residual_history[
        ::2,
        idx,
        ],
        marker="+",
        s=60,
        label="Right Ascension",
    )
    ax.scatter(
        residual_times[1::2],
        residual_history[
        1::2,
        idx,
        ],
        marker="+",
        s=60,
        label="Declination",
    )
    ax.set_ylabel("Observation Residual [rad]")
    ax.set_title("Iteration " + str(idx + 1))

plt.tight_layout()

# add the year label for the x-axis
for col in range(number_of_columns):
    axs[int(number_of_rows - 1), col].set_xlabel("Year")

axs[0, 0].legend()

plt.show()

"""
### Residuals Correlations Matrix
Lets check out the correlation of the estimated parameters.
"""

# Correlation can be retrieved using the CovarianceAnalysisInput class:
covariance_input = estimation_analysis.CovarianceAnalysisInput(norad_id_simulated_observations)
covariance_output = estimator.compute_covariance(covariance_input)

correlations = covariance_output.correlations
estimated_param_names = ["x", "y", "z", "vx", "vy", "vz"]


fig, ax = plt.subplots(1, 1, figsize=(9, 7))

im = ax.imshow(correlations, cmap=cm.RdYlBu_r, vmin=-1, vmax=1)

ax.set_xticks(np.arange(len(estimated_param_names)), labels=estimated_param_names)
ax.set_yticks(np.arange(len(estimated_param_names)), labels=estimated_param_names)

# add numbers to each of the boxes
for i in range(len(estimated_param_names)):
    for j in range(len(estimated_param_names)):
        text = ax.text(
            j, i, round(correlations[i, j], 2), ha="center", va="center", color="w"
        )

cb = plt.colorbar(im)

ax.set_xlabel("Estimated Parameter")
ax.set_ylabel("Estimated Parameter")

fig.suptitle(f"Correlations for estimated parameters for {norad_id}")

fig.set_tight_layout(True)
plt.show()

