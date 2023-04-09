# General imports
import math
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
# tudatpy imports
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.astro import time_conversion
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import estimation, estimation_setup


###########################################################################
# DEFINE SIMULATION SETTINGS ##############################################
###########################################################################

# Load spice kernels
spice.load_standard_kernels()

# Define temporal scope of the simulation - equal to the time JUICE will spend in orbit around Jupiter
simulation_start_epoch = 31.0 * constants.JULIAN_YEAR + 182.0 * constants.JULIAN_DAY
simulation_end_epoch = 35.73 * constants.JULIAN_YEAR
simulation_duration = simulation_end_epoch - simulation_start_epoch


###########################################################################
# CREATE ENVIRONMENT ######################################################
###########################################################################

# Create default body settings for selected celestial bodies
jovian_moons_to_create = ['Io', 'Europa', 'Ganymede', 'Callisto']
planets_to_create = ['Jupiter', 'Saturn']
stars_to_create = ['Sun']
bodies_to_create = np.concatenate((jovian_moons_to_create, planets_to_create, stars_to_create))

# Create default body settings for bodies_to_create, with 'Jupiter'/'J2000'
# as global frame origin and orientation.
global_frame_origin = 'Jupiter'
global_frame_orientation = 'ECLIPJ2000'
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation)

### Ephemeris Settings Moons ###
# Get the original ephemeris settings
original_io_ephemeris_settings = body_settings.get('Io').ephemeris_settings
original_europa_ephemeris_settings = body_settings.get('Europa').ephemeris_settings
original_ganymede_ephemeris_settings = body_settings.get('Ganymede').ephemeris_settings
original_callisto_ephemeris_settings = body_settings.get('Callisto').ephemeris_settings
# Apply new tabulated ephemeris settings
body_settings.get('Io').ephemeris_settings = environment_setup.ephemeris.tabulated_from_existing(
    original_io_ephemeris_settings,
    simulation_start_epoch,
    simulation_end_epoch,
    time_step=5.0 * 60.0)
body_settings.get('Europa').ephemeris_settings = environment_setup.ephemeris.tabulated_from_existing(
    original_europa_ephemeris_settings,
    simulation_start_epoch,
    simulation_end_epoch,
    time_step=5.0 * 60.0)
body_settings.get('Ganymede').ephemeris_settings = environment_setup.ephemeris.tabulated_from_existing(
    original_ganymede_ephemeris_settings,
    simulation_start_epoch,
    simulation_end_epoch,
    time_step=5.0 * 60.0)
body_settings.get('Callisto').ephemeris_settings = environment_setup.ephemeris.tabulated_from_existing(
    original_callisto_ephemeris_settings,
    simulation_start_epoch,
    simulation_end_epoch,
    time_step=5.0 * 60.0)

### Rotational Models ###
# Define overall parameters describing the synchronous rotation model
central_body_name = "Jupiter"
original_frame = "ECLIPJ2000"
# Define satellite specific parameters and change rotation model settings
target_frame = 'IAU_IO'
body_settings.get('Io').rotation_model_settings = environment_setup.rotation_model.synchronous(
    central_body_name, original_frame, target_frame)
target_frame = 'IAU_Europa'
body_settings.get('Europa').rotation_model_settings = environment_setup.rotation_model.synchronous(
    central_body_name, original_frame, target_frame)
target_frame = 'IAU_Ganymede'
body_settings.get('Ganymede').rotation_model_settings = environment_setup.rotation_model.synchronous(
    central_body_name, original_frame, target_frame)
target_frame = 'IAU_Callisto'
body_settings.get('Callisto').rotation_model_settings = environment_setup.rotation_model.synchronous(
    central_body_name, original_frame, target_frame)

# Create system of selected bodies
bodies = environment_setup.create_system_of_bodies(body_settings)


###########################################################################
# CREATE PROPAGATION SETTINGS #############################################
###########################################################################

# Define bodies that are propagated, and their central bodies of propagation
bodies_to_propagate = ['Io', 'Europa', 'Ganymede', 'Callisto']
central_bodies = ['Jupiter', 'Jupiter', 'Jupiter', 'Jupiter']

### Acceleration Settings ###
# Dirkx et al. (2016) - restricted to second degree
love_number_moons = 0.3
dissipation_parameter_moons = 0.015
q_moons = love_number_moons / dissipation_parameter_moons
# Lari (2018)
mean_motion_io = 203.49 * (math.pi / 180) * 1 / constants.JULIAN_DAY
mean_motion_europa = 101.37 * (math.pi / 180) * 1 / constants.JULIAN_DAY
mean_motion_ganymede = 50.32 * (math.pi / 180) * 1 / constants.JULIAN_DAY
mean_motion_callisto = 21.57 * (math.pi / 180) * 1 / constants.JULIAN_DAY

# Dirkx et al. (2016) - restricted to second degree
love_number_jupiter = 0.38
dissipation_parameter_jupiter= 1.1E-5
q_jupiter = love_number_jupiter / dissipation_parameter_jupiter

# Lainey et al. (2009)
tidal_frequency_io = 23.3 # rad.day-1
spin_frequency_jupiter = math.pi/tidal_frequency_io + mean_motion_io

acceleration_settings_moons = dict()

### Io ###
time_lag_io = 1 / mean_motion_io * np.arctan(1 / q_moons)
time_lag_jupiter_io = 1/(spin_frequency_jupiter - mean_motion_io) * np.arctan(1 / q_jupiter)
acceleration_settings_io = dict(
    Jupiter=[propagation_setup.acceleration.mutual_spherical_harmonic_gravity(8, 0, 2, 2),
             propagation_setup.acceleration.direct_tidal_dissipation_acceleration(love_number_moons,
                                                                                  time_lag_io,
                                                                                  True, False),
             propagation_setup.acceleration.direct_tidal_dissipation_acceleration(love_number_jupiter,
                                                                                  time_lag_jupiter_io,
                                                                                  True, True),
             propagation_setup.acceleration.relativistic_correction(use_schwarzschild=True)],
    Europa=[propagation_setup.acceleration.mutual_spherical_harmonic_gravity(2, 2, 2, 2)],
    Ganymede=[propagation_setup.acceleration.mutual_spherical_harmonic_gravity(2, 2, 2, 2)],
    Callisto=[propagation_setup.acceleration.mutual_spherical_harmonic_gravity(2, 2, 2, 2)],
    Sun=[propagation_setup.acceleration.point_mass_gravity()],
    Saturn=[propagation_setup.acceleration.point_mass_gravity()]
)
acceleration_settings_moons['Io'] = acceleration_settings_io

### Europa ###
time_lag_europa = 1 / mean_motion_europa * np.arctan(1 / q_moons)
time_lag_jupiter_europa = 1 / (spin_frequency_jupiter - mean_motion_europa) * np.arctan(1 / q_jupiter)
acceleration_settings_europa = dict(
    Jupiter=[propagation_setup.acceleration.mutual_spherical_harmonic_gravity(8, 0, 2, 2),
             propagation_setup.acceleration.direct_tidal_dissipation_acceleration(love_number_moons,
                                                                                  time_lag_europa,
                                                                                  True, False),
             propagation_setup.acceleration.direct_tidal_dissipation_acceleration(love_number_jupiter,
                                                                                  time_lag_jupiter_europa,
                                                                                  True, True),
             propagation_setup.acceleration.relativistic_correction(use_schwarzschild=True)],
    Io=[propagation_setup.acceleration.mutual_spherical_harmonic_gravity(2, 2, 2, 2)],
    Ganymede=[propagation_setup.acceleration.mutual_spherical_harmonic_gravity(2, 2, 2, 2)],
    Callisto=[propagation_setup.acceleration.mutual_spherical_harmonic_gravity(2, 2, 2, 2)],
    Sun=[propagation_setup.acceleration.point_mass_gravity()],
    Saturn=[propagation_setup.acceleration.point_mass_gravity()]
)
acceleration_settings_moons['Europa'] = acceleration_settings_europa

### Ganymede ###
time_lag_ganymede = 1 / mean_motion_ganymede * np.arctan(1 / q_moons)
time_lag_jupiter_ganymede = 1 / (spin_frequency_jupiter - mean_motion_ganymede) * np.arctan(1 / q_jupiter)
acceleration_settings_ganymede = dict(
    Jupiter=[propagation_setup.acceleration.mutual_spherical_harmonic_gravity(8, 0, 2, 2),
             propagation_setup.acceleration.direct_tidal_dissipation_acceleration(love_number_moons,
                                                                                  time_lag_ganymede,
                                                                                  True, False),
             propagation_setup.acceleration.direct_tidal_dissipation_acceleration(love_number_jupiter,
                                                                                  time_lag_jupiter_ganymede,
                                                                                  True, True),
             propagation_setup.acceleration.relativistic_correction(use_schwarzschild=True)],
    Io=[propagation_setup.acceleration.mutual_spherical_harmonic_gravity(2, 2, 2, 2)],
    Europa=[propagation_setup.acceleration.mutual_spherical_harmonic_gravity(2, 2, 2, 2)],
    Callisto=[propagation_setup.acceleration.mutual_spherical_harmonic_gravity(2, 2, 2, 2)],
    Sun=[propagation_setup.acceleration.point_mass_gravity()],
    Saturn=[propagation_setup.acceleration.point_mass_gravity()]
)
acceleration_settings_moons['Ganymede'] = acceleration_settings_ganymede

### Callisto ###
time_lag_callisto = 1 / mean_motion_callisto * np.arctan(1 / q_moons)
time_lag_jupiter_callisto = 1 / (spin_frequency_jupiter - mean_motion_callisto) * np.arctan(1 / q_jupiter)
acceleration_settings_callisto = dict(
    Jupiter=[propagation_setup.acceleration.mutual_spherical_harmonic_gravity(8, 0, 2, 2),
             propagation_setup.acceleration.direct_tidal_dissipation_acceleration(love_number_moons,
                                                                                  time_lag_callisto,
                                                                                  True, False),
             propagation_setup.acceleration.direct_tidal_dissipation_acceleration(love_number_jupiter,
                                                                                  time_lag_jupiter_callisto,
                                                                                  True, True),
             propagation_setup.acceleration.relativistic_correction(use_schwarzschild=True)],
    Io=[propagation_setup.acceleration.mutual_spherical_harmonic_gravity(2, 2, 2, 2)],
    Europa=[propagation_setup.acceleration.mutual_spherical_harmonic_gravity(2, 2, 2, 2)],
    Ganymede=[propagation_setup.acceleration.mutual_spherical_harmonic_gravity(2, 2, 2, 2)],
    Sun=[propagation_setup.acceleration.point_mass_gravity()],
    Saturn=[propagation_setup.acceleration.point_mass_gravity()]
)
acceleration_settings_moons['Callisto'] = acceleration_settings_callisto

acceleration_settings = acceleration_settings_moons
# Create acceleration models
acceleration_models = propagation_setup.create_acceleration_models(
    bodies, acceleration_settings, bodies_to_propagate, central_bodies)

# Define initial state
initial_states = list()
for body in bodies_to_propagate:
    initial_states.append(spice.get_body_cartesian_state_at_epoch(
        target_body_name=body,
        observer_body_name='Jupiter',
        reference_frame_name='ECLIPJ2000',
        aberration_corrections='none',
        ephemeris_time=simulation_start_epoch))
initial_states = np.concatenate(initial_states)

### Integrator Settings ###
# Use fixed step-size integrator (DP8) with fixed time-step of 30 minutes
# Create integrator settings
time_step_sec = 30.0 * 60.0
integrator_settings = propagation_setup.integrator. \
    runge_kutta_fixed_step_size(initial_time_step=time_step_sec,
                                coefficient_set=propagation_setup.integrator.CoefficientSets.rkdp_87)

### Termination Settings ###
termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch)

### Propagator Settings ###
propagator_settings = propagation_setup.propagator. \
    translational(central_bodies=central_bodies,
                  acceleration_models=acceleration_models,
                  bodies_to_integrate=bodies_to_propagate,
                  initial_states=initial_states,
                  initial_time=simulation_start_epoch,
                  integrator_settings=integrator_settings,
                  termination_settings=termination_condition)


### CREATE LINK ENDS FOR MOONS ###
link_ends_io = dict()
link_ends_io[estimation_setup.observation.observed_body] = estimation_setup.observation.\
    body_origin_link_end_id('Io')
link_definition_io = estimation_setup.observation.LinkDefinition(link_ends_io)

link_ends_europa = dict()
link_ends_europa[estimation_setup.observation.observed_body] = estimation_setup.observation.\
    body_origin_link_end_id('Europa')
link_definition_europa = estimation_setup.observation.LinkDefinition(link_ends_europa)

link_ends_ganymede = dict()
link_ends_ganymede[estimation_setup.observation.observed_body] = estimation_setup.observation.\
    body_origin_link_end_id('Ganymede')
link_definition_ganymede = estimation_setup.observation.LinkDefinition(link_ends_ganymede)

link_ends_callisto = dict()
link_ends_callisto[estimation_setup.observation.observed_body] = estimation_setup.observation.\
    body_origin_link_end_id('Callisto')
link_definition_callisto = estimation_setup.observation.LinkDefinition(link_ends_callisto)

### OBSERVATION MODEL SETTINGS ###
position_observation_settings = [estimation_setup.observation.cartesian_position(link_definition_io),
                                 estimation_setup.observation.cartesian_position(link_definition_europa),
                                 estimation_setup.observation.cartesian_position(link_definition_ganymede),
                                 estimation_setup.observation.cartesian_position(link_definition_callisto)]

### OBSERVATIONS SIMULATION SETTINGS ###
# Define epochs at which the ephemerides shall be checked
observation_times = np.arange(simulation_start_epoch, simulation_end_epoch, 3.0 * 3600)
# Create the observation simulation settings per moon
observation_simulation_settings_io = estimation_setup.observation.tabulated_simulation_settings(
    estimation_setup.observation.position_observable_type,
    link_definition_io,
    observation_times,
    reference_link_end_type=estimation_setup.observation.observed_body)
observation_simulation_settings_europa = estimation_setup.observation.tabulated_simulation_settings(
    estimation_setup.observation.position_observable_type,
    link_definition_europa,
    observation_times,
    reference_link_end_type=estimation_setup.observation.observed_body)
observation_simulation_settings_ganymede = estimation_setup.observation.tabulated_simulation_settings(
    estimation_setup.observation.position_observable_type,
    link_definition_ganymede,
    observation_times,
    reference_link_end_type=estimation_setup.observation.observed_body)
observation_simulation_settings_callisto = estimation_setup.observation.tabulated_simulation_settings(
    estimation_setup.observation.position_observable_type,
    link_definition_callisto,
    observation_times,
    reference_link_end_type=estimation_setup.observation.observed_body)
# Create conclusive list of observation simulation settings
observation_simulation_settings = [observation_simulation_settings_io,
                                   observation_simulation_settings_europa,
                                   observation_simulation_settings_ganymede,
                                   observation_simulation_settings_callisto]

### "SIMULATE" EPHEMERIS STATES OF SATELLITES ###
# Create observation simulators
ephemeris_observation_simulators = estimation_setup.create_observation_simulators(
    position_observation_settings, bodies)
# Get ephemeris states as ObservationCollection
print('Checking ephemerides...')
ephemeris_satellite_states = estimation.simulate_observations(
    observation_simulation_settings,
    ephemeris_observation_simulators,
    bodies)

### PARAMETERS TO ESTIMATE ###
parameters_to_estimate_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)
parameters_to_estimate = estimation_setup.create_parameter_set(parameters_to_estimate_settings, bodies)
original_parameter_vector = parameters_to_estimate.parameter_vector

### PERFORM THE ESTIMATION ###
# Create the estimator
print('Running propagation...')
estimator = numerical_simulation.Estimator(bodies, parameters_to_estimate,
                                           position_observation_settings, propagator_settings)
# Create input object for the estimation
estimation_input = estimation.EstimationInput(ephemeris_satellite_states)
# Set methodological options
estimation_input.define_estimation_settings(save_state_history_per_iteration=True)
# Perform the estimation
print('Performing the estimation...')
print(f'Original initial states: {original_parameter_vector}')
estimation_output = estimator.perform_estimation(estimation_input)
initial_states_updated = parameters_to_estimate.parameter_vector
print('Done with the estimation...')
print(f'Updated initial states: {initial_states_updated}')


### LOAD DATA ###
simulator_object = estimation_output.simulation_results_per_iteration[-1]
state_history = simulator_object.dynamics_results.state_history
### Ephemeris Kepler elements ####
# Initialize containers
ephemeris_state_history = dict()
jupiter_gravitational_parameter = bodies.get('Jupiter').gravitational_parameter
# Loop over the propagated states and use the IMCEE ephemeris as benchmark solution
for epoch in state_history.keys():
    io_from_ephemeris = spice.get_body_cartesian_state_at_epoch(
    target_body_name='Io',
    observer_body_name='Jupiter',
    reference_frame_name='ECLIPJ2000',
    aberration_corrections='none',
    ephemeris_time=epoch)

    europa_from_ephemeris = spice.get_body_cartesian_state_at_epoch(
        target_body_name='Europa',
        observer_body_name='Jupiter',
        reference_frame_name='ECLIPJ2000',
        aberration_corrections='none',
        ephemeris_time=epoch)

    ganymede_from_ephemeris = spice.get_body_cartesian_state_at_epoch(
        target_body_name='Ganymede',
        observer_body_name='Jupiter',
        reference_frame_name='ECLIPJ2000',
        aberration_corrections='none',
        ephemeris_time=epoch)

    callisto_from_ephemeris = spice.get_body_cartesian_state_at_epoch(
        target_body_name='Callisto',
        observer_body_name='Jupiter',
        reference_frame_name='ECLIPJ2000',
        aberration_corrections='none',
        ephemeris_time=epoch)

    ephemeris_state = np.concatenate((io_from_ephemeris, europa_from_ephemeris,
                                      ganymede_from_ephemeris, callisto_from_ephemeris))
    ephemeris_state_history[epoch] = ephemeris_state



state_history_difference = np.vstack(list(state_history.values())) - np.vstack(list(ephemeris_state_history.values()))
position_difference = {'Io': state_history_difference[:, 0:3],
                       'Europa': state_history_difference[:, 6:9],
                       'Ganymede': state_history_difference[:, 12:15],
                       'Callisto': state_history_difference[:, 18:21]}

### PLOTTING ###
time2plt = list()
epochs_julian_seconds = np.vstack(list(state_history.keys()))
for epoch in epochs_julian_seconds:
    epoch_days = constants.JULIAN_DAY_ON_J2000 + epoch / constants.JULIAN_DAY
    time2plt.append(time_conversion.julian_day_to_calendar_date(epoch_days))

fig, ax1 = plt.subplots(1, 1, figsize=(9, 6))

ax1.plot(time2plt, np.linalg.norm(position_difference['Io'], axis=1) * 1E-3,
         label=r'Io ($i=1$)', c='#A50034')
ax1.plot(time2plt, np.linalg.norm(position_difference['Europa'], axis=1) * 1E-3,
         label=r'Europa ($i=2$)', c='#0076C2')
ax1.plot(time2plt, np.linalg.norm(position_difference['Ganymede'], axis=1) * 1E-3,
         label=r'Ganymede ($i=3$)', c='#EC6842')
ax1.plot(time2plt, np.linalg.norm(position_difference['Callisto'], axis=1) * 1E-3,
         label=r'Callisto ($i=4$)', c='#009B77')
ax1.set_title(r'Difference in Position (C-M)')
ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=1))
ax1.xaxis.set_minor_locator(mdates.MonthLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
ax1.set_ylabel(r'Difference [km]')
ax1.legend()
